from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from model_asset_route import resolve_model_prediction_db_path
from model_asset_route import resolve_model_prediction_duckdb_path
from project_paths import resolve_data_dir


ENV_L4_DUCKDB_SYNC = "QUANT_L4_DUCKDB_SYNC"


def l4_duckdb_sync_enabled() -> bool:
    value = str(os.environ.get(ENV_L4_DUCKDB_SYNC, "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _connect_writable(path: Path):
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def sqlite_table_row_count(path: Path, table_name: str) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=120) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0])


def sync_prediction_table_full_to_duckdb(
    data_dir: str | Path | None,
    table_name: str,
    *,
    sqlite_db_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    target_table: str | None = None,
) -> dict[str, int | str]:
    sqlite_path = (
        Path(sqlite_db_path)
        if sqlite_db_path
        else resolve_model_prediction_db_path(data_dir=data_dir, create_parent=False)
    )
    target_path = (
        Path(duckdb_path)
        if duckdb_path
        else resolve_model_prediction_duckdb_path(data_dir=data_dir, require_exists=False)
    )
    target_name = str(target_table or table_name).strip()
    source_rows = sqlite_table_row_count(sqlite_path, table_name)

    with _connect_writable(target_path) as conn:
        try:
            conn.execute("LOAD sqlite")
        except Exception:
            conn.execute("INSTALL sqlite")
            conn.execute("LOAD sqlite")
        conn.execute(
            f"CREATE OR REPLACE TABLE {_quote_ident(target_name)} AS "
            "SELECT * FROM sqlite_scan(?, ?)",
            [str(sqlite_path), table_name],
        )
        target_rows = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(target_name)}").fetchone()[0])

    return {
        "sqlite_db_path": str(sqlite_path),
        "duckdb_path": str(target_path),
        "source_table": table_name,
        "target_table": target_name,
        "source_rows": source_rows,
        "target_rows": target_rows,
    }
