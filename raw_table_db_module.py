from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd


RAW_DB_MODE_ENV = "QUANT_RAW_DB_MODE"
RAW_DB_MODE_SPLIT = "split"
RAW_DB_MODE_LEGACY = "legacy_odb"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def raw_table_db_path(data_dir: str | Path, table_name: str) -> Path:
    data_path = Path(data_dir)
    mode = os.environ.get(RAW_DB_MODE_ENV, RAW_DB_MODE_SPLIT).strip().lower()
    if mode == RAW_DB_MODE_LEGACY:
        return data_path / "odb.db"
    return data_path / "raw_table_dbs" / f"{table_name}.DB"


def connect_raw_table_db(data_dir: str | Path, table_name: str, timeout: int = 60) -> sqlite3.Connection:
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
            return target_db
        conn.execute(
            f"DELETE FROM {quote_ident(table_name)} WHERE {quote_ident(trade_col)} >= ? AND {quote_ident(trade_col)} <= ?",
            (start, end),
        )
        frame.to_sql(table_name, con=conn, if_exists="append", index=False)
        conn.commit()
    return target_db


def replace_raw_table_full(data_dir: str | Path, table_name: str, frame: pd.DataFrame) -> Path:
    target_db = raw_table_db_path(data_dir, table_name)
    with closing(connect_raw_table_db(data_dir, table_name)) as conn:
        frame.to_sql(table_name, con=conn, if_exists="replace", index=False)
        conn.commit()
    return target_db
