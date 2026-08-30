from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from l1_duckdb_sync import (
    l1_duckdb_sync_enabled,
    sync_raw_table_full_to_duckdb,
    sync_raw_table_trade_range_to_duckdb,
)
from l1_raw_data_route import (
    BACKEND_DUCKDB,
    BACKEND_LEGACY_ODB,
    BACKEND_SPLIT,
    resolve_l1_raw_duckdb_path,
    resolve_legacy_odb_path,
    resolve_raw_table_split_db_path,
)


RAW_DB_MODE_ENV = "QUANT_RAW_DB_MODE"
ALLOW_LEGACY_RAW_SQLITE_ENV = "QUANT_ALLOW_LEGACY_RAW_SQLITE"
RAW_DB_MODE_DUCKDB = BACKEND_DUCKDB
RAW_DB_MODE_SPLIT = BACKEND_SPLIT
RAW_DB_MODE_LEGACY = BACKEND_LEGACY_ODB


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def raw_table_db_mode() -> str:
    mode = os.environ.get(RAW_DB_MODE_ENV, RAW_DB_MODE_DUCKDB).strip().lower()
    aliases = {
        "duckdb": RAW_DB_MODE_DUCKDB,
        "split": RAW_DB_MODE_SPLIT,
        "raw_split": RAW_DB_MODE_SPLIT,
        "legacy": RAW_DB_MODE_LEGACY,
        "odb": RAW_DB_MODE_LEGACY,
    }
    normalized = aliases.get(mode, mode)
    if normalized not in {RAW_DB_MODE_DUCKDB, RAW_DB_MODE_SPLIT, RAW_DB_MODE_LEGACY}:
        raise ValueError(f"unsupported {RAW_DB_MODE_ENV}: {mode}")
    return normalized


def _assert_legacy_sqlite_opt_in(mode: str) -> None:
    if mode == RAW_DB_MODE_DUCKDB:
        return
    if os.environ.get(ALLOW_LEGACY_RAW_SQLITE_ENV) == "1":
        return
    raise ValueError(
        "legacy L1 SQLite write modes are disabled in the DuckDB-only workspace; "
        f"set {ALLOW_LEGACY_RAW_SQLITE_ENV}=1 only for explicit historical rollback work"
    )


def raw_table_db_path(data_dir: str | Path, table_name: str) -> Path:
    mode = raw_table_db_mode()
    if mode == RAW_DB_MODE_DUCKDB:
        return resolve_l1_raw_duckdb_path(table_name, data_dir=data_dir)
    _assert_legacy_sqlite_opt_in(mode)
    if mode == RAW_DB_MODE_LEGACY:
        return resolve_legacy_odb_path(data_dir)
    return resolve_raw_table_split_db_path(data_dir, table_name)


def connect_raw_table_db(data_dir: str | Path, table_name: str, timeout: int = 60) -> sqlite3.Connection:
    if raw_table_db_mode() == RAW_DB_MODE_DUCKDB:
        raise ValueError("connect_raw_table_db only supports explicit SQLite rollback modes")
    path = raw_table_db_path(data_dir, table_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout)
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def replace_raw_table_trade_range(
    data_dir: str | Path,
    table_name: str,
    frame: pd.DataFrame,
    start: str,
    end: str,
    *,
    trade_col: str = "trade_date",
) -> Path:
    target_db = raw_table_db_path(data_dir, table_name)
    if frame is None or frame.empty:
        return target_db
    if raw_table_db_mode() == RAW_DB_MODE_DUCKDB:
        sync_raw_table_trade_range_to_duckdb(
            data_dir,
            table_name,
            frame,
            start,
            end,
            trade_col=trade_col,
        )
        return target_db
    with closing(connect_raw_table_db(data_dir, table_name)) as conn:
        exists = (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            is not None
        )
        if not exists:
            frame.to_sql(table_name, con=conn, if_exists="replace", index=False)
            conn.commit()
        else:
            conn.execute(
                f"DELETE FROM {quote_ident(table_name)} WHERE {quote_ident(trade_col)} >= ? AND {quote_ident(trade_col)} <= ?",
                (start, end),
            )
            frame.to_sql(table_name, con=conn, if_exists="append", index=False)
            conn.commit()
    if l1_duckdb_sync_enabled():
        sync_raw_table_trade_range_to_duckdb(
            data_dir,
            table_name,
            frame,
            start,
            end,
            trade_col=trade_col,
        )
    return target_db


def replace_raw_table_full(data_dir: str | Path, table_name: str, frame: pd.DataFrame) -> Path:
    target_db = raw_table_db_path(data_dir, table_name)
    if raw_table_db_mode() == RAW_DB_MODE_DUCKDB:
        sync_raw_table_full_to_duckdb(data_dir, table_name, frame)
        return target_db
    with closing(connect_raw_table_db(data_dir, table_name)) as conn:
        frame.to_sql(table_name, con=conn, if_exists="replace", index=False)
        conn.commit()
    if l1_duckdb_sync_enabled():
        sync_raw_table_full_to_duckdb(data_dir, table_name, frame)
    return target_db
