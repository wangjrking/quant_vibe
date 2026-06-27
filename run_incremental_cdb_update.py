from pathlib import Path
import gc

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_process_module import get_factor_data
from project_paths import resolve_data_dir
from stock_daily_data_route import connect_stock_daily_readonly


DATA_DIR = resolve_data_dir()
FACTOR_PATH = DATA_DIR / "stock_factor_data.parquet"
WINDOW_START = "20250101"


def _load_recent_integ_data() -> pd.DataFrame:
    conn = connect_stock_daily_readonly(data_dir=DATA_DIR)
    try:
        frame = pd.read_sql(
            """
            SELECT *
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= ?
            ORDER BY stock_code, trade_date
            """,
            conn,
            params=(WINDOW_START,),
        )
    finally:
        conn.close()
    frame.columns = frame.columns.str.lower()
    return frame


def _latest_mapping(frame: pd.DataFrame, key: str, value: str) -> dict:
    cols = [key, value, "trade_date"]
    available = frame[cols].dropna(subset=[key, value])
    if available.empty:
        return {}
    latest = available.sort_values("trade_date").drop_duplicates(key, keep="last")
    return latest.set_index(key)[value].to_dict()


def _normalize_to_old_schema(frame: pd.DataFrame) -> pd.DataFrame:
    schema = pq.read_schema(FACTOR_PATH)
    for field in schema:
        col = field.name
        if col not in frame.columns:
            continue
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            frame[col] = frame[col].astype("string")
        elif pa.types.is_integer(field.type):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        elif pa.types.is_floating(field.type):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def main() -> None:
    print("incremental_cdb_start", flush=True)
    old_meta_cols = [
        "stock_code",
        "trade_date",
        "stock_encode",
        "industry",
        "industry_encode",
        "act_ent_type",
        "act_ent_type_encode",
    ]
    old_meta = pd.read_parquet(FACTOR_PATH, columns=old_meta_cols)
    old_meta["trade_date"] = old_meta["trade_date"].astype(str)
    old_max_date = old_meta["trade_date"].max()
    old_columns = pq.read_schema(FACTOR_PATH).names
    print(f"old_factor_meta_loaded old_max={old_max_date}", flush=True)

    integ_data = _load_recent_integ_data()
    integ_data["trade_date"] = integ_data["trade_date"].astype(str)
    latest_date = integ_data["trade_date"].max()
    print(f"integ_loaded rows={integ_data.shape[0]} max={latest_date}", flush=True)
    if latest_date <= old_max_date:
        print(f"incremental_cdb_skip old_max={old_max_date} integ_max={latest_date}", flush=True)
        return

    print(f"recent_window rows={integ_data.shape[0]} start={WINDOW_START}", flush=True)
    factor_recent = get_factor_data(integ_data)
    del integ_data
    gc.collect()
    print(f"recent_factor_done rows={factor_recent.shape[0]}", flush=True)
    factor_recent["trade_date"] = factor_recent["trade_date"].astype(str)
    new_rows = factor_recent[factor_recent["trade_date"] > old_max_date].copy()
    del factor_recent
    gc.collect()

    if new_rows.empty:
        print(f"incremental_cdb_no_rows old_max={old_max_date} integ_max={latest_date}", flush=True)
        return

    code_map = _latest_mapping(old_meta, "stock_code", "stock_encode")
    industry_map = _latest_mapping(old_meta, "industry", "industry_encode")
    ent_type_map = _latest_mapping(old_meta, "act_ent_type", "act_ent_type_encode")
    del old_meta
    gc.collect()

    if code_map:
        new_rows["stock_encode"] = new_rows["stock_code"].map(code_map).fillna(new_rows["stock_encode"])
    if industry_map:
        new_rows["industry_encode"] = new_rows["industry"].map(industry_map).fillna(new_rows["industry_encode"])
    if ent_type_map:
        new_rows["act_ent_type_encode"] = (
            new_rows["act_ent_type"].map(ent_type_map).fillna(new_rows["act_ent_type_encode"])
        )

    for col in old_columns:
        if col not in new_rows.columns:
            new_rows[col] = pd.NA
    new_rows = new_rows[old_columns]

    print(f"new_rows_ready rows={new_rows.shape[0]} max={new_rows['trade_date'].max()}", flush=True)
    old_factor = pd.read_parquet(FACTOR_PATH)
    old_factor["trade_date"] = old_factor["trade_date"].astype(str)
    print(f"old_factor_full_loaded rows={old_factor.shape[0]}", flush=True)
    added_count = new_rows.shape[0]
    merged = pd.concat(
        [old_factor[old_factor["trade_date"] <= old_max_date], new_rows],
        ignore_index=True,
    )
    del old_factor, new_rows
    gc.collect()
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged = _normalize_to_old_schema(merged)
    print(f"merged_ready rows={merged.shape[0]} max={merged['trade_date'].max()}", flush=True)
    merged.to_parquet(FACTOR_PATH)
    print(
        f"incremental_cdb_done rows={merged.shape[0]} old_max={old_max_date} "
        f"new_max={merged['trade_date'].max()} added={added_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
