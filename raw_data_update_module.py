"""Incremental raw-data update helpers.

This module keeps the raw parquet layer independent from strategy stock pools.
It appends newly downloaded rows and deduplicates by natural keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from l1_universe_rules import filter_stock_codes_no_bj


@dataclass(frozen=True)
class RawTableSpec:
    name: str
    file_name: str
    key_columns: tuple[str, ...]
    date_column: str = "trade_date"
    per_stock: bool = True


RAW_TABLE_SPECS: dict[str, RawTableSpec] = {
    "daily": RawTableSpec("daily", "daily_data.parquet", ("ts_code", "trade_date")),
    "daily_basic": RawTableSpec("daily_basic", "daily_index_data.parquet", ("ts_code", "trade_date")),
    "moneyflow": RawTableSpec("moneyflow", "moneyflow.parquet", ("ts_code", "trade_date")),
    "stk_factor": RawTableSpec("stk_factor", "stk_factor.parquet", ("ts_code", "trade_date")),
    "limit_list": RawTableSpec("limit_list", "limit_list_data.parquet", ("ts_code", "trade_date")),
    "cyq_perf": RawTableSpec("cyq_perf", "cyq_perf.parquet", ("ts_code", "trade_date")),
    "adj_factor": RawTableSpec("adj_factor", "adj_factor.parquet", ("ts_code", "trade_date"), per_stock=False),
    "top_list": RawTableSpec("top_list", "top_list.parquet", ("ts_code", "trade_date"), per_stock=False),
    "ths_hot": RawTableSpec("ths_hot", "ths_hot.parquet", ("ts_code", "trade_date"), per_stock=False),
    "dc_hot": RawTableSpec("dc_hot", "dc_hot.parquet", ("ts_code", "trade_date"), per_stock=False),
    "stock_st": RawTableSpec("stock_st", "stock_st.parquet", ("ts_code", "trade_date"), per_stock=False),
    "index_daily": RawTableSpec("index_daily", "index_daily.parquet", ("ts_code", "trade_date"), per_stock=False),
    "stk_shock": RawTableSpec(
        "stk_shock",
        "stk_shock.parquet",
        ("ts_code", "trade_date", "reason", "period"),
        per_stock=False,
    ),
    "stk_high_shock": RawTableSpec(
        "stk_high_shock",
        "stk_high_shock.parquet",
        ("ts_code", "trade_date", "reason", "period"),
        per_stock=False,
    ),
    "stk_alert": RawTableSpec(
        "stk_alert",
        "stk_alert.parquet",
        ("ts_code", "start_date", "end_date", "type"),
        date_column="start_date",
        per_stock=False,
    ),
}


def next_start_date(path: str | Path, default_start: str, date_column: str = "trade_date") -> str:
    path = Path(path)
    if not path.exists():
        return default_start
    frame = pd.read_parquet(path, columns=[date_column])
    if frame.empty:
        return default_start
    max_date = str(frame[date_column].max())
    return (datetime.strptime(max_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


def append_parquet_dedup(path: str | Path, new_rows: pd.DataFrame, subset: list[str] | tuple[str, ...]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_rows is None or new_rows.empty:
        return 0
    if path.exists():
        old_rows = pd.read_parquet(path)
        merged = pd.concat([old_rows, new_rows], ignore_index=True)
    else:
        merged = new_rows.copy()
    merged = merged.replace({"None": pd.NA, "none": pd.NA, "nan": pd.NA, "NaN": pd.NA, "": pd.NA})
    key_set = set(subset)
    text_columns = key_set | {
        "name",
        "area",
        "industry",
        "cnspell",
        "market",
        "list_date",
        "act_name",
        "act_ent_type",
        "limit_type",
        "reason",
        "period",
        "trade_market",
        "type",
        "source_api",
        "source_row_sha256",
        "fetched_at",
        "start_date",
        "end_date",
        "first_time",
        "last_time",
    }
    for col in merged.columns:
        if col in text_columns:
            continue
        if merged[col].dtype == "object":
            converted = pd.to_numeric(merged[col], errors="coerce")
            if converted.notna().sum() >= merged[col].notna().sum() * 0.8:
                merged[col] = converted
    merged = merged.drop_duplicates(subset=list(subset), keep="last")
    temp_path = path.with_suffix(".tmp.parquet")
    merged.to_parquet(temp_path, index=False)
    temp_path.replace(path)
    return int(new_rows.shape[0])


def load_stock_codes(path: str | Path) -> list[str]:
    frame = pd.read_csv(path)
    col = "stock_code" if "stock_code" in frame.columns else "ts_code"
    return filter_stock_codes_no_bj(frame[col].dropna())


def load_existing_stock_codes(path: str | Path, code_column: str = "ts_code") -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    frame = pd.read_parquet(path, columns=[code_column])
    if frame.empty:
        return set()
    return {
        str(value).strip().upper()
        for value in frame[code_column].dropna()
        if str(value).strip()
    }


def download_by_stock(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    fetcher: Callable[[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    frames = []
    for stock_code in stock_codes:
        frame = fetcher(stock_code, start_date, end_date)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
