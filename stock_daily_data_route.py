from __future__ import annotations

import os
import sqlite3
from pathlib import Path


STOCK_DAILY_TABLE = "STOCK_DAILY_DATA"
STOCK_DAILY_DB_NAME = "STOCK_DAILY_DATA.db"
LEGACY_MIXED_DB_NAME = "odb.db"
ENV_STOCK_DAILY_DB = "QUANT_STOCK_DAILY_DB"
ENV_DATA_DIR = "QUANT_DATA_DIR"


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
    """Return the canonical L2 STOCK_DAILY_DATA database path.

    Default L2 read contract:
    quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA

    The old quant/data_file/odb.db::STOCK_DAILY_DATA copy is legacy-only and
    should be used only for rollback or historical reproduction.
    """
    explicit = db_path or os.environ.get(ENV_STOCK_DAILY_DB)
    if explicit:
        path = _resolve_path(explicit)
    else:
        base = data_dir or os.environ.get(ENV_DATA_DIR) or default_data_dir()
        path = _resolve_path(base) / STOCK_DAILY_DB_NAME
    if require_exists and not path.exists():
        raise FileNotFoundError(f"L2 stock daily database not found: {path}")
    return path


def resolve_legacy_mixed_db_path(data_dir: str | Path | None = None) -> Path:
    base = data_dir or os.environ.get(ENV_DATA_DIR) or default_data_dir()
    return _resolve_path(base) / LEGACY_MIXED_DB_NAME


def connect_stock_daily_readonly(
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    *,
    timeout: int = 120,
) -> sqlite3.Connection:
    path = resolve_stock_daily_db_path(data_dir=data_dir, db_path=db_path, require_exists=True)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
