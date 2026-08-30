from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

from duckdb_asset_route import connect_duckdb_readonly, require_duckdb_installed
from l1_raw_data_route import resolve_l1_raw_duckdb_path, resolve_raw_table_split_db_path


ENV_L1_DUCKDB_SYNC = "QUANT_L1_DUCKDB_SYNC"


def l1_duckdb_sync_enabled() -> bool:
    return str(os.environ.get(ENV_L1_DUCKDB_SYNC, "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _connect_writable(path: Path):
    require_duckdb_installed()
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def sync_raw_table_full_to_duckdb(
    data_dir: str | Path,
    table_name: str,
    frame: pd.DataFrame,
    *,
    duckdb_path: str | Path | None = None,
) -> Path:
    path = resolve_l1_raw_duckdb_path(table_name, data_dir=data_dir, duckdb_path=duckdb_path)
    if frame is None:
        return path
    conn = _connect_writable(path)
    try:
        conn.register("__raw_frame", frame)
        conn.execute(
            f"CREATE OR REPLACE TABLE {_quote_ident(table_name)} AS SELECT * FROM __raw_frame"
        )
        conn.unregister("__raw_frame")
    finally:
        conn.close()
    return path


def sync_raw_table_split_full_to_duckdb(
    data_dir: str | Path,
    table_name: str,
    *,
    sqlite_db_path: str | Path | None = None,
    filter_bj: bool = True,
    duckdb_path: str | Path | None = None,
) -> Path:
    split_path = (
        Path(sqlite_db_path)
        if sqlite_db_path is not None
        else resolve_raw_table_split_db_path(data_dir, table_name, require_exists=True)
    )
    uri = split_path.resolve().as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True, timeout=60) as conn:
        frame = pd.read_sql_query(f"SELECT * FROM {_quote_ident(table_name)}", conn)
    if filter_bj and "ts_code" in frame.columns:
        ts_codes = frame["ts_code"].astype(str).str.upper().str.strip()
        frame = frame[~ts_codes.str.endswith(".BJ")].copy()
    return sync_raw_table_full_to_duckdb(
        data_dir,
        table_name,
        frame,
        duckdb_path=duckdb_path,
    )


def sync_raw_table_trade_range_to_duckdb(
    data_dir: str | Path,
    table_name: str,
    frame: pd.DataFrame,
    start: str,
    end: str,
    *,
    trade_col: str = "trade_date",
    duckdb_path: str | Path | None = None,
) -> Path:
    path = resolve_l1_raw_duckdb_path(table_name, data_dir=data_dir, duckdb_path=duckdb_path)
    if frame is None or frame.empty:
        return path
    conn = _connect_writable(path)
    try:
        exists = (
            conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
            > 0
        )
        conn.register("__raw_frame", frame)
        if not exists:
            conn.execute(
                f"CREATE TABLE {_quote_ident(table_name)} AS SELECT * FROM __raw_frame"
            )
        else:
            conn.execute(
                f"DELETE FROM {_quote_ident(table_name)} "
                f"WHERE {_quote_ident(trade_col)} >= ? AND {_quote_ident(trade_col)} <= ?",
                [start, end],
            )
            conn.execute(f"INSERT INTO {_quote_ident(table_name)} SELECT * FROM __raw_frame")
        conn.unregister("__raw_frame")
    finally:
        conn.close()
    return path


def duckdb_table_row_count(path: str | Path, table_name: str) -> int:
    conn = connect_duckdb_readonly(path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])
    finally:
        conn.close()
