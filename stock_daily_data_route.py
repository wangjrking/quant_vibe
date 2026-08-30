from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from duckdb_asset_route import (
    BACKEND_DUCKDB,
    BACKEND_LEGACY,
    ENV_ALLOW_LEGACY_ROLLBACK,
    connect_duckdb_readonly,
    resolve_duckdb_path,
)
from production_asset_registry import active_main_workflow_asset, split_asset_path


STOCK_DAILY_TABLE = "STOCK_DAILY_DATA"
STOCK_DAILY_DB_NAME = "STOCK_DAILY_DATA.db"
LEGACY_MIXED_DB_NAME = "odb.db"
ENV_STOCK_DAILY_DB = "QUANT_STOCK_DAILY_DB"
ENV_DATA_DIR = "QUANT_DATA_DIR"
ENV_STOCK_DAILY_BACKEND = "QUANT_STOCK_DAILY_BACKEND"
ENV_STOCK_DAILY_DUCKDB = "QUANT_STOCK_DAILY_DUCKDB"


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data_file"


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else value.resolve()


def resolve_stock_daily_db_path(
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    """Return an explicit legacy SQLite L2 STOCK_DAILY_DATA path.

    Current mainline L2 is DuckDB-only. This helper exists only for explicit
    legacy rollback/reproduction flows and no longer provides an implicit
    STOCK_DAILY_DATA.db fallback.
    """
    explicit = db_path or os.environ.get(ENV_STOCK_DAILY_DB)
    if explicit:
        path = _resolve_path(explicit)
    else:
        asset = active_main_workflow_asset("L2_stock_daily_base", data_dir=data_dir)
        if asset:
            asset_path, _ = split_asset_path(asset.get("asset_path"))
            if asset_path and asset_path.suffix.lower() == ".db":
                path = asset_path
            else:
                raise RuntimeError(
                    "L2 current mainline is DuckDB-only; implicit STOCK_DAILY_DATA.db fallback is disabled. "
                    "Use resolve_stock_daily_duckdb_path() for the active route or pass an explicit legacy db_path."
                )
        else:
            raise RuntimeError(
                "Implicit STOCK_DAILY_DATA.db fallback is disabled in the current DuckDB-only production architecture. "
                "Use resolve_stock_daily_duckdb_path() for active production reads or pass an explicit legacy db_path."
            )
    if require_exists and not path.exists():
        raise FileNotFoundError(f"L2 stock daily database not found: {path}")
    return path


def resolve_stock_daily_duckdb_path(
    data_dir: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    explicit = duckdb_path or os.environ.get(ENV_STOCK_DAILY_DUCKDB)
    if explicit:
        path = _resolve_path(explicit)
        if require_exists and not path.exists():
            raise FileNotFoundError(f"L2 stock daily DuckDB asset not found: {path}")
        return path
    asset = active_main_workflow_asset("L2_stock_daily_base", data_dir=data_dir)
    if asset:
        asset_path, _ = split_asset_path(asset.get("asset_path"))
        if asset_path and asset_path.suffix.lower() == ".duckdb":
            if require_exists and not asset_path.exists():
                raise FileNotFoundError(f"L2 stock daily DuckDB asset not found: {asset_path}")
            return asset_path
    return resolve_duckdb_path("L2", data_dir=data_dir, require_exists=require_exists)


def resolve_legacy_mixed_db_path(data_dir: str | Path | None = None) -> Path:
    base = data_dir or os.environ.get(ENV_DATA_DIR) or default_data_dir()
    return _resolve_path(base) / LEGACY_MIXED_DB_NAME


def connect_stock_daily_readonly(
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    *,
    backend: str | None = None,
    timeout: int = 120,
):
    selected = resolve_stock_daily_backend(backend, data_dir=data_dir)
    if selected == BACKEND_DUCKDB:
        path = resolve_stock_daily_duckdb_path(
            data_dir=data_dir,
            duckdb_path=duckdb_path,
            require_exists=True,
        )
        return connect_duckdb_readonly(path)
    if selected == BACKEND_LEGACY:
        path = resolve_stock_daily_db_path(data_dir=data_dir, db_path=db_path, require_exists=True)
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    raise ValueError(f"unsupported stock daily backend: {selected}")


def resolve_stock_daily_backend(
    value: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> str:
    explicit = value if value is not None else os.environ.get(ENV_STOCK_DAILY_BACKEND)
    if explicit not in (None, ""):
        selected = str(explicit).strip().lower()
        if selected in {"sqlite", "parquet", "sqlite_parquet", "legacy_sqlite_parquet"}:
            raise ValueError(
                "deprecated stock daily backend alias is not allowed: "
                f"{selected}. Use 'duckdb' for current production or explicit 'legacy' only for approved rollback flows."
            )
        if selected not in {BACKEND_DUCKDB, BACKEND_LEGACY}:
            raise ValueError(f"unsupported stock daily backend: {selected}")
        if selected == BACKEND_LEGACY and not str(os.environ.get(ENV_ALLOW_LEGACY_ROLLBACK, "")).strip() == "1":
            raise RuntimeError(
                "explicit legacy SQLite/Parquet backend selection requires opt-in via "
                f"{ENV_ALLOW_LEGACY_ROLLBACK}=1"
            )
        return selected

    asset = active_main_workflow_asset("L2_stock_daily_base", data_dir=data_dir)
    asset_type = str(asset.get("asset_type", "")).lower() if asset else ""
    asset_path = str(asset.get("asset_path", "")).lower() if asset else ""
    if "duckdb" in asset_type or asset_path.endswith(".duckdb") or ".duckdb::" in asset_path:
        return BACKEND_DUCKDB
    return BACKEND_DUCKDB


def connect_stock_daily_sqlite_rollback_readonly(
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    *,
    timeout: int = 120,
) -> sqlite3.Connection:
    path = resolve_stock_daily_db_path(data_dir=data_dir, db_path=db_path, require_exists=True)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
