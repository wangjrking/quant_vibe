"""Lightweight factor extraction from STOCK_DAILY_DATA.

This path is for broad-universe experiments where rebuilding the full 900+
column factor parquet is too expensive. It uses raw SQL columns plus a small
set of deterministic derived fields needed by the existing training pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from leakage_guard import validate_no_leakage
from stock_pool_module import filter_frame_by_stock_pool, load_stock_pool


META_COLUMNS = [
    "trade_date",
    "name",
    "stock_code",
    "st_type",
    "limit_times",
    "close",
    "pre_close",
    "open",
    "high",
    "low",
    "industry",
    "industry_encode",
    "stock_encode",
    "atr_qfq",
]

DERIVED_FEATURES = {
    "close_rate",
    "open_rate",
    "high_rate",
    "low_rate",
    "high_open_rate",
    "high_close_rate",
    "low_open_rate",
    "low_close_rate",
    "diff_close_low",
    "diff_close_high",
    "diff_high_low",
    "industry_encode",
    "stock_encode",
}

FORWARD_COLUMNS = [
    "post_open",
    "post2_open",
    "post3_open",
    "post4_open",
    "post5_open",
    "post6_open",
    "post12_open",
    "post_close",
    "post2_close",
    "post2_high",
    "post_high",
    "post_close_1d",
]


def load_feature_list(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", payload if isinstance(payload, list) else [])
    return [str(feature) for feature in features]


def find_unsupported_features(features: list[str], available_columns: set[str]) -> list[str]:
    unsupported = []
    for feature in features:
        if feature.startswith("legacy_"):
            unsupported.append(feature)
        elif feature in available_columns or feature in DERIVED_FEATURES:
            continue
        else:
            unsupported.append(feature)
    return unsupported


def _read_sql_columns(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(STOCK_DAILY_DATA)").fetchall()}


def read_raw_frame(
    db_path: str | Path,
    start: str,
    end: str | None,
    needed_columns: list[str],
    stock_pool_path: str | Path | None = None,
) -> pd.DataFrame:
    db_path = Path(db_path)
    available = _read_sql_columns(db_path)
    columns = [col for col in needed_columns if col in available]
    if "stock_code" not in columns:
        columns.insert(0, "stock_code")
    if "trade_date" not in columns:
        columns.insert(1, "trade_date")
    quoted = ", ".join(f'"{col}"' for col in dict.fromkeys(columns))
    query = f'SELECT {quoted} FROM STOCK_DAILY_DATA WHERE trade_date >= ?'
    params: list[str] = [start]
    if end:
        query += " AND trade_date <= ?"
        params.append(end)
    query += " ORDER BY stock_code, trade_date"
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql(query, conn, params=params)
    if stock_pool_path:
        stock_pool = load_stock_pool(stock_pool_path)
        frame = filter_frame_by_stock_pool(frame, stock_pool)
    return frame


def _add_forward_columns(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("stock_code", sort=False)
    frame["post_open"] = grouped["open"].shift(-1)
    frame["post2_open"] = grouped["open"].shift(-2)
    frame["post3_open"] = grouped["open"].shift(-3)
    frame["post4_open"] = grouped["open"].shift(-4)
    frame["post5_open"] = grouped["open"].shift(-5)
    frame["post6_open"] = grouped["open"].shift(-6)
    frame["post12_open"] = grouped["open"].shift(-12)
    frame["post_close_1d"] = grouped["close"].shift(-1)
    frame["post_close"] = grouped["close"].shift(-10)
    frame["post2_close"] = grouped["close"].shift(-2)
    frame["post_high"] = grouped["high"].shift(-10)
    frame["post2_high"] = grouped["high"].shift(-2)
    frame["10d_yield_rate"] = (frame["post_close"] - frame["close"]) / frame["close"]
    frame["2d_yield_rate"] = (frame["post2_close"] - frame["close"]) / frame["close"]
    return frame


def build_light_factor_frame(raw_frame: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    grouped = frame.groupby("stock_code", sort=False)
    for col in ["open", "close", "high", "low", "pre_close", "atr_qfq"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if {"open", "close", "high", "low"}.issubset(frame.columns):
        # Use the prior qfq close from the same adjusted price series so every
        # price-derived factor stays on a single front-adjusted basis.
        frame["pre_close"] = grouped["close"].shift(1)
        frame["close_rate"] = frame["close"] / frame["pre_close"]
        frame["open_rate"] = frame["open"] / frame["pre_close"]
        frame["high_rate"] = frame["high"] / frame["pre_close"]
        frame["low_rate"] = frame["low"] / frame["pre_close"]
        frame["high_open_rate"] = frame["high_rate"] - frame["open_rate"]
        frame["high_close_rate"] = frame["high_rate"] - frame["close_rate"]
        frame["low_open_rate"] = frame["low_rate"] - frame["open_rate"]
        frame["low_close_rate"] = frame["low_rate"] - frame["close_rate"]
        frame["diff_close_low"] = frame["close"] - frame["low"]
        frame["diff_close_high"] = frame["close"] - frame["high"]
        frame["diff_high_low"] = frame["high"] - frame["low"]
        frame = _add_forward_columns(frame)

    if "industry" in frame.columns:
        frame["industry_encode"] = pd.factorize(frame["industry"].fillna("").astype(str), sort=True)[0]
    if "stock_code" in frame.columns:
        frame["stock_encode"] = pd.factorize(frame["stock_code"].fillna("").astype(str), sort=True)[0]

    if label == "executable_10d_open_return":
        entry_cash = frame["post_open"] * (1.0 + 0.0003 + 0.001)
        exit_cash = frame["post12_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_5d_open_return":
        entry_cash = frame["post_open"] * (1.0 + 0.0003 + 0.001)
        exit_cash = frame["post6_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_3d_open_return":
        entry_cash = frame["post_open"] * (1.0 + 0.0003 + 0.001)
        exit_cash = frame["post4_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_1d_open_return":
        entry_cash = frame["post_open"] * (1.0 + 0.0003 + 0.001)
        exit_cash = frame["post2_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_2d_open_return":
        entry_cash = frame["post_open"] * (1.0 + 0.0003 + 0.001)
        exit_cash = frame["post2_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label not in {"10d_yield_rate", "2d_yield_rate"} and label not in frame.columns:
        raise ValueError(f"unsupported light label: {label}")

    keep = list(dict.fromkeys(META_COLUMNS + FORWARD_COLUMNS + features + ["10d_yield_rate", "2d_yield_rate", label]))
    for col in keep:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[keep]


def split_light_factor_data(
    factor_data: pd.DataFrame,
    features: list[str],
    label: str,
    train_start: str,
    test_start: str,
):
    frame = factor_data[factor_data["trade_date"].astype(str) >= train_start].copy()
    frame = frame[frame["name"].notna()]
    frame = frame[~frame["name"].astype(str).str.contains("ST", na=False)]
    if "st_type" in frame.columns:
        frame = frame[frame["st_type"].fillna("").astype(str) != "ST"]
    if "limit_times" in frame.columns:
        frame = frame[frame["limit_times"].isna()]
    train_data = frame[frame["trade_date"].astype(str) < test_start].dropna(subset=[label])
    test_data = frame[frame["trade_date"].astype(str) >= test_start]
    train_factor_data = train_data[features + [label]]
    test_factor_data = test_data[features + [label]]
    for col in train_factor_data.select_dtypes(include=["object"]).columns:
        train_factor_data.loc[:, col] = pd.to_numeric(train_factor_data[col], errors="coerce")
        test_factor_data.loc[:, col] = pd.to_numeric(test_factor_data[col], errors="coerce")
    train_y = train_factor_data[label]
    train_x = train_factor_data.drop(columns=[label])
    test_y = test_factor_data[label]
    test_x = test_factor_data.drop(columns=[label])
    train_index = pd.MultiIndex.from_frame(train_data[["stock_code", "trade_date"]])
    test_index = pd.MultiIndex.from_frame(test_data[["stock_code", "trade_date"]])
    train_x.index = train_index
    train_y.index = train_index
    test_x.index = test_index
    test_y.index = test_index
    validate_no_leakage(train_x.columns, label=label)
    validate_no_leakage(test_x.columns, label=label)
    return train_x, train_y, test_x, test_y, train_data, test_data


def get_light_factor_data(
    data_dir: str | Path,
    features_path: str | Path,
    train_start: str,
    test_start: str,
    label: str,
    end: str | None = None,
    stock_pool_path: str | Path | None = None,
    allow_missing_features: bool = False,
):
    data_dir = Path(data_dir)
    db_path = data_dir / "odb.db"
    features = load_feature_list(features_path)
    available = _read_sql_columns(db_path)
    unsupported = find_unsupported_features(features, available)
    if unsupported and not allow_missing_features:
        preview = ", ".join(unsupported[:20])
        if len(unsupported) > 20:
            preview += f", ... (+{len(unsupported) - 20} more)"
        raise ValueError(f"Unsupported light features: {preview}")
    usable_features = [feature for feature in features if feature not in unsupported]
    needed_columns = list(dict.fromkeys(META_COLUMNS + usable_features + ["open", "high", "low", "close", "pre_close"]))
    raw = read_raw_frame(db_path, start=train_start, end=end, needed_columns=needed_columns, stock_pool_path=stock_pool_path)
    factor = build_light_factor_frame(raw, usable_features, label)
    return split_light_factor_data(factor, usable_features, label, train_start, test_start)
