from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd

from duckdb_asset_route import connect_duckdb_readonly
from production_asset_registry import active_main_workflow_asset, split_asset_path


ENV_L1_RAW_BACKEND = "QUANT_L1_RAW_BACKEND"
ENV_L1_RAW_DUCKDB = "QUANT_L1_RAW_DUCKDB"
ENV_DATA_DIR = "QUANT_DATA_DIR"

BACKEND_SPLIT = "split"
BACKEND_DUCKDB = "duckdb"
BACKEND_LEGACY_ODB = "legacy_odb"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data_file"


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else value.resolve()


def _default_l1_duckdb_root(data_dir: str | Path | None = None) -> Path:
    base = data_dir or os.environ.get(ENV_DATA_DIR) or default_data_dir()
    return _resolve_path(base) / "production_assets" / "duckdb" / "l1_raw_tables"


def resolve_l1_raw_backend(value: str | None = None, *, data_dir: str | Path | None = None) -> str:
    raw_value = value if value is not None else os.environ.get(ENV_L1_RAW_BACKEND)
    if raw_value not in (None, ""):
        backend = str(raw_value).strip().lower()
    else:
        asset = active_main_workflow_asset("L1_raw_data", data_dir=data_dir)
        asset_type = str(asset.get("asset_type", "")).lower() if asset else ""
        asset_path = str(asset.get("asset_path", "")).lower() if asset else ""
        if "duckdb" in asset_type or asset_path.endswith(".duckdb") or ".duckdb::" in asset_path or "l1_raw_tables" in asset_path:
            return BACKEND_DUCKDB
        if "raw_split" in asset_type or "raw_table_dbs" in asset_path:
            return BACKEND_SPLIT
        if asset_path.endswith("odb.db") or asset_path.endswith("odb.db/"):
            return BACKEND_LEGACY_ODB
        return BACKEND_DUCKDB
    aliases = {
        "raw_split": BACKEND_SPLIT,
        "split_db": BACKEND_SPLIT,
        "sqlite_split": BACKEND_SPLIT,
        "odb": BACKEND_LEGACY_ODB,
        "legacy": BACKEND_LEGACY_ODB,
    }
    backend = aliases.get(backend, backend)
    if backend not in {BACKEND_SPLIT, BACKEND_DUCKDB, BACKEND_LEGACY_ODB}:
        raise ValueError(f"unsupported L1 raw backend: {backend}")
    return backend


def resolve_raw_split_db_root(data_dir: str | Path | None = None) -> Path:
    asset = active_main_workflow_asset("L1_raw_data", data_dir=data_dir)
    if asset:
        asset_path, _ = split_asset_path(asset.get("asset_path"))
        if asset_path and asset_path.name.lower() == "raw_table_dbs":
            return asset_path
    base = data_dir or os.environ.get(ENV_DATA_DIR) or default_data_dir()
    return _resolve_path(base) / "raw_table_dbs"


def resolve_raw_table_split_db_path(
    data_dir: str | Path | None,
    table_name: str,
    *,
    require_exists: bool = False,
) -> Path:
    path = resolve_raw_split_db_root(data_dir) / f"{table_name}.DB"
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L1 split raw DB not found for {table_name}: {path}")
    return path


def resolve_legacy_odb_path(data_dir: str | Path | None = None, *, require_exists: bool = False) -> Path:
    base = data_dir or os.environ.get(ENV_DATA_DIR) or default_data_dir()
    path = _resolve_path(base) / "odb.db"
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L1 legacy odb.db not found: {path}")
    return path


def resolve_l1_raw_duckdb_path(
    table_name: str,
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    explicit = duckdb_path or os.environ.get(ENV_L1_RAW_DUCKDB)
    if explicit:
        path = _resolve_path(explicit)
        if path.suffix.lower() != ".duckdb":
            path = path / f"{table_name}.duckdb"
        if require_exists and not path.is_file():
            raise FileNotFoundError(f"L1 DuckDB asset not found: {path}")
        return path
    asset = active_main_workflow_asset("L1_raw_data", data_dir=data_dir)
    if asset:
        asset_path, _ = split_asset_path(asset.get("asset_path"))
        if asset_path:
            path = asset_path if asset_path.suffix.lower() == ".duckdb" else asset_path / f"{table_name}.duckdb"
            if require_exists and not path.is_file():
                raise FileNotFoundError(f"L1 DuckDB asset not found: {path}")
            return path
    path = _default_l1_duckdb_root(data_dir) / f"{table_name}.duckdb"
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"L1 DuckDB asset not found: {path}")
    return path


def read_raw_table_frame(
    data_dir: str | Path | None,
    table_name: str,
    *,
    where_sql: str | None = None,
    params: Sequence[object] | None = None,
    backend: str | None = None,
    duckdb_path: str | Path | None = None,
) -> pd.DataFrame:
    selected = resolve_l1_raw_backend(backend, data_dir=data_dir)
    sql = f"SELECT * FROM {quote_ident(table_name)}"
    if where_sql:
        sql += f" WHERE {where_sql}"

    if selected == BACKEND_DUCKDB:
        path = resolve_l1_raw_duckdb_path(table_name, data_dir=data_dir, duckdb_path=duckdb_path, require_exists=True)
        conn = connect_duckdb_readonly(path)
        try:
            return conn.execute(sql, params or []).df()
        finally:
            conn.close()

    if selected == BACKEND_LEGACY_ODB:
        path = resolve_legacy_odb_path(data_dir=data_dir, require_exists=True)
    else:
        path = resolve_raw_table_split_db_path(data_dir, table_name, require_exists=True)

    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    try:
        return pd.read_sql_query(sql, conn, params=params or ())
    finally:
        conn.close()


def materialize_sqlite_temp_raw_table(
    conn: sqlite3.Connection,
    *,
    temp_name: str,
    table_name: str,
    data_dir: str | Path | None,
    where_sql: str | None = None,
    params: Sequence[object] | None = None,
    backend: str | None = None,
    duckdb_path: str | Path | None = None,
) -> None:
    frame = read_raw_table_frame(
        data_dir,
        table_name,
        where_sql=where_sql,
        params=params,
        backend=backend,
        duckdb_path=duckdb_path,
    )
    staging_name = f"__raw_stage_{temp_name}"
    conn.execute(f"DROP TABLE IF EXISTS {quote_ident(staging_name)}")
    conn.execute(f"DROP TABLE IF EXISTS temp.{quote_ident(temp_name)}")
    frame.to_sql(staging_name, con=conn, if_exists="replace", index=False)
    conn.execute(
        f"CREATE TEMP TABLE {quote_ident(temp_name)} AS SELECT * FROM {quote_ident(staging_name)}"
    )
    conn.execute(f"DROP TABLE {quote_ident(staging_name)}")
