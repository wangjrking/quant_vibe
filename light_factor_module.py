"""Lightweight factor extraction from the daily factor table.

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
from stock_daily_data_route import resolve_stock_daily_db_path
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
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "open_gap",
    "intraday_ret",
    "amplitude",
    "upper_shadow",
    "lower_shadow",
    "ma5_ratio",
    "ma10_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "vol5_ratio",
    "vol10_ratio",
    "vol20_ratio",
    "amount5_ratio",
    "amount10_ratio",
    "amount20_ratio",
    "turnover5_mean",
    "turnover10_mean",
    "turnover20_mean",
    "volatility5",
    "volatility10",
    "volatility20",
    "rsi6",
    "rsi12",
    "amount_rank",
    "turnover_rate_rank",
    "volume_ratio_rank",
    "circ_mv_rank",
    "total_mv_rank",
    "pb_rank",
    "pe_rank",
    "ret_5_rank",
    "ret_20_rank",
    "close_rate_rank",
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


def _resolve_daily_factor_table(db_path: Path) -> str:
    candidates = ("STOCK_DAILY_DATA", "STK_FACTOR", "stk_factor", "daily_data")
    with sqlite3.connect(db_path) as conn:
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    for table in candidates:
        if table in existing:
            return table
    raise ValueError(f"No daily factor source table found in {db_path}")


def _read_sql_columns(db_path: Path, table_name: str | None = None) -> set[str]:
    table_name = table_name or _resolve_daily_factor_table(db_path)
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()}


def _ts_code_to_gm(code: str) -> str:
    value = str(code)
    if value.startswith(("SHSE.", "SZSE.", "BJSE.")):
        return value
    if value.endswith(".SH"):
        return f"SHSE.{value[:6]}"
    if value.endswith(".SZ"):
        return f"SZSE.{value[:6]}"
    if value.endswith(".BJ"):
        return f"BJSE.{value[:6]}"
    return value


def _attach_stock_basic(db_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in ("name", "industry") if col not in frame.columns or frame[col].isna().all()]
    if not missing or "ts_code" not in frame.columns:
        return frame
    with sqlite3.connect(db_path) as conn:
        try:
            basic = pd.read_sql(
                "SELECT ts_code, name, industry FROM stock_basic_data",
                conn,
            )
        except Exception:
            return frame
    if basic.empty:
        return frame
    drop_cols = [col for col in ("name", "industry") if col in frame.columns]
    frame = frame.drop(columns=drop_cols)
    return frame.merge(basic, on="ts_code", how="left")


def read_raw_frame(
    db_path: str | Path,
    start: str,
    end: str | None,
    needed_columns: list[str],
    stock_pool_path: str | Path | None = None,
) -> pd.DataFrame:
    db_path = Path(db_path)
    table_name = _resolve_daily_factor_table(db_path)
    available = _read_sql_columns(db_path, table_name)
    columns = [col for col in needed_columns if col in available]
    has_stock_code = "stock_code" in available
    if not has_stock_code and "ts_code" in available:
        columns.insert(0, "ts_code")
    if has_stock_code and "stock_code" not in columns:
        columns.insert(0, "stock_code")
    if "trade_date" not in columns:
        columns.insert(1, "trade_date")
    quoted = ", ".join(f'"{col}"' for col in dict.fromkeys(columns))
    query = f'SELECT {quoted} FROM "{table_name}" WHERE trade_date >= ?'
    params: list[str] = [start]
    if end:
        query += " AND trade_date <= ?"
        params.append(end)
    order_code = "stock_code" if has_stock_code else "ts_code"
    query += f' ORDER BY "{order_code}", trade_date'
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql(query, conn, params=params)
    if "stock_code" not in frame.columns and "ts_code" in frame.columns:
        frame["stock_code"] = frame["ts_code"].map(_ts_code_to_gm)
    frame = _attach_stock_basic(db_path, frame)
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
        frame["ret_1"] = frame["close"] / grouped["close"].shift(1) - 1.0
        for window in [2, 3, 5, 10, 20]:
            frame[f"ret_{window}"] = frame["close"] / grouped["close"].shift(window) - 1.0
        frame["open_gap"] = frame["open_rate"] - 1.0
        frame["intraday_ret"] = frame["close"] / frame["open"] - 1.0
        frame["amplitude"] = (frame["high"] - frame["low"]) / frame["pre_close"]
        frame["upper_shadow"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / frame["pre_close"]
        frame["lower_shadow"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / frame["pre_close"]
        for window in [5, 10, 20, 60]:
            ma = grouped["close"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean())
            frame[f"ma{window}_ratio"] = frame["close"] / ma - 1.0
        if "vol" in frame.columns:
            frame["vol"] = pd.to_numeric(frame["vol"], errors="coerce")
            for window in [5, 10, 20]:
                vol_ma = grouped["vol"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean())
                frame[f"vol{window}_ratio"] = frame["vol"] / vol_ma
        if "amount" in frame.columns:
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
            for window in [5, 10, 20]:
                amount_ma = grouped["amount"].transform(lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean())
                frame[f"amount{window}_ratio"] = frame["amount"] / amount_ma
        if "turnover_rate" in frame.columns:
            frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
            for window in [5, 10, 20]:
                frame[f"turnover{window}_mean"] = grouped["turnover_rate"].transform(
                    lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean()
                )
        for window in [5, 10, 20]:
            frame[f"volatility{window}"] = grouped["ret_1"].transform(
                lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).std()
            )
        delta = grouped["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        for window in [6, 12]:
            avg_gain = gain.groupby(frame["stock_code"], sort=False).transform(
                lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean()
            )
            avg_loss = loss.groupby(frame["stock_code"], sort=False).transform(
                lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean()
            )
            rs = avg_gain / avg_loss.replace(0, np.nan)
            frame[f"rsi{window}"] = 100.0 - 100.0 / (1.0 + rs)
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

    rank_sources = {
        "amount_rank": "amount",
        "turnover_rate_rank": "turnover_rate",
        "volume_ratio_rank": "volume_ratio",
        "circ_mv_rank": "circ_mv",
        "total_mv_rank": "total_mv",
        "pb_rank": "pb",
        "pe_rank": "pe",
        "ret_5_rank": "ret_5",
        "ret_20_rank": "ret_20",
        "close_rate_rank": "close_rate",
    }
    for rank_col, source_col in rank_sources.items():
        if source_col in frame.columns:
            frame[rank_col] = frame.groupby("trade_date", sort=False)[source_col].rank(pct=True)

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
    elif label.startswith("executable_") and ("_top" in label):
        from ai_module import prepare_training_label

        frame = prepare_training_label(frame, label)
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
    names = frame["name"].astype(str)
    valid_name = (
        ~names.str.contains("ST", na=False)
        & ~names.str.contains("\u9000\u5e02", na=False)
        & ~names.str.startswith("\u9000", na=False)
    )
    frame = frame[valid_name]
    if "st_type" in frame.columns:
        frame = frame[frame["st_type"].fillna("").astype(str) != "ST"]
    if "limit_times" in frame.columns:
        frame = frame[frame["limit_times"].isna()]
    train_data = frame[frame["trade_date"].astype(str) < test_start].dropna(subset=[label])
    test_data = frame[frame["trade_date"].astype(str) >= test_start]
    train_factor_data = train_data[features + [label]].copy()
    test_factor_data = test_data[features + [label]].copy()
    train_factor_data = train_factor_data.apply(pd.to_numeric, errors="coerce")
    test_factor_data = test_factor_data.apply(pd.to_numeric, errors="coerce")
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
    db_path = resolve_stock_daily_db_path(data_dir=data_dir)
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
