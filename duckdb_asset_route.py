from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from project_paths import resolve_data_dir


ENV_DUCKDB_ROOT = "QUANT_DUCKDB_ASSET_ROOT"
ENV_DUCKDB_ZONE = "QUANT_DUCKDB_ASSET_ZONE"
ENV_ASSET_BACKEND = "QUANT_LOCAL_ASSET_BACKEND"
ENV_ALLOW_LEGACY_ROLLBACK = "QUANT_ALLOW_LEGACY_SQLITE_PARQUET_ROLLBACK"

BACKEND_DUCKDB = "duckdb"
BACKEND_LEGACY = "legacy"
ZONE_PRODUCTION = "production"
ZONE_EXPERIMENT = "experiment"
ZONE_ARCHIVE = "archive"

DUCKDB_DIR = "production_assets/duckdb"
DUCKDB_EXPERIMENT_FILE = "quant_experiment.duckdb"
DUCKDB_ARCHIVE_FILE = "quant_archive.duckdb"
DUCKDB_L2_FILE = "l2_stock_daily_data.duckdb"
DUCKDB_L3_FEATURE_FILE = "l3_feature_current.duckdb"
DUCKDB_L3_LABEL_FILE = "l3_label_current.duckdb"
DUCKDB_L4_FILE = "l4_predictions_current.duckdb"
DUCKDB_ZONE_FILES = {
    ZONE_EXPERIMENT: DUCKDB_EXPERIMENT_FILE,
    ZONE_ARCHIVE: DUCKDB_ARCHIVE_FILE,
}
DUCKDB_LAYER_FILES = {
    "L2": DUCKDB_L2_FILE,
    "L3_FEATURES": DUCKDB_L3_FEATURE_FILE,
    "L3_LABELS": DUCKDB_L3_LABEL_FILE,
    "L4": DUCKDB_L4_FILE,
}

LAYER_ALIASES = {
    "l1": "L1",
    "l1_raw": "L1",
    "l1_raw_data": "L1",
    "l2": "L2",
    "l2_base": "L2",
    "l2_stock_daily_base": "L2",
    "l3": "L3_FEATURES",
    "l3_features": "L3_FEATURES",
    "l3_feature": "L3_FEATURES",
    "l3_labels": "L3_LABELS",
    "l3_label": "L3_LABELS",
    "l4": "L4",
    "l4_model": "L4",
    "l4_model_prediction": "L4",
    "l5": "L5",
    "l5_strategy": "L5",
    "l5_strategy_signal": "L5",
    "l6": "L6",
    "l6_backtest": "L6",
    "l7": "L7",
    "l7_trading": "L7",
    "l7_trading_delivery": "L7",
}
SPLIT_ONLY_CURRENT_LAYERS = {"L1", "L5", "L6", "L7"}


class DuckDBRouteError(RuntimeError):
    pass


def duckdb_available() -> bool:
    return importlib.util.find_spec("duckdb") is not None


def require_duckdb_installed() -> None:
    if not duckdb_available():
        raise DuckDBRouteError(
            "duckdb is not installed in the unified Python environment; "
            "install requirements.txt before using DuckDB routes."
        )


def normalize_duckdb_layer(layer: str) -> str:
    key = str(layer).strip().lower()
    normalized = LAYER_ALIASES.get(key)
    if not normalized:
        allowed = ", ".join(sorted(DUCKDB_LAYER_FILES))
        raise ValueError(f"unsupported DuckDB layer: {layer}; allowed: {allowed}")
    return normalized


def resolve_duckdb_root(data_dir: str | Path | None = None) -> Path:
    explicit = os.environ.get(ENV_DUCKDB_ROOT)
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else path.resolve()
    return resolve_data_dir(data_dir) / DUCKDB_DIR


def normalize_duckdb_zone(zone: str | None = None) -> str:
    value = str(zone or os.environ.get(ENV_DUCKDB_ZONE) or ZONE_PRODUCTION).strip().lower()
    aliases = {
        "prod": ZONE_PRODUCTION,
        "production": ZONE_PRODUCTION,
        "exp": ZONE_EXPERIMENT,
        "experimental": ZONE_EXPERIMENT,
        "experiment": ZONE_EXPERIMENT,
        "archive": ZONE_ARCHIVE,
        "archival": ZONE_ARCHIVE,
        "history": ZONE_ARCHIVE,
    }
    normalized = aliases.get(value, value)
    if normalized == ZONE_PRODUCTION:
        raise DuckDBRouteError(
            "production DuckDB assets are isolated per table. Use layer-specific helpers, "
            "table-file helpers, or active registry bindings instead of a production bundle route."
        )
    if normalized not in DUCKDB_ZONE_FILES:
        allowed = ", ".join(sorted([ZONE_PRODUCTION, *DUCKDB_ZONE_FILES]))
        raise ValueError(f"unsupported DuckDB zone: {zone}; allowed: {allowed}")
    return normalized


def resolve_duckdb_zone_path(
    zone: str | None = None,
    data_dir: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    normalized = normalize_duckdb_zone(zone)
    path = resolve_duckdb_root(data_dir) / DUCKDB_ZONE_FILES[normalized]
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"DuckDB {normalized} asset not found: {path}")
    return path


def resolve_duckdb_path(
    layer: str,
    data_dir: str | Path | None = None,
    *,
    require_exists: bool = False,
    zone: str | None = None,
) -> Path:
    normalized = normalize_duckdb_layer(layer)
    root = resolve_duckdb_root(data_dir)
    requested_zone = str(zone or os.environ.get(ENV_DUCKDB_ZONE) or ZONE_PRODUCTION).strip().lower()
    is_default_production = requested_zone in {"", "prod", "production"}
    if normalized in SPLIT_ONLY_CURRENT_LAYERS and is_default_production:
        raise DuckDBRouteError(
            f"{normalized} current production assets are split per-table DuckDB files. "
            "Use table-specific current asset helpers or active registry bindings instead of the removed legacy "
            "bundle route."
        )
    if normalized in DUCKDB_LAYER_FILES and is_default_production:
        filename = DUCKDB_LAYER_FILES[normalized]
    else:
        normalized_zone = normalize_duckdb_zone(zone)
        filename = DUCKDB_LAYER_FILES.get(normalized, DUCKDB_ZONE_FILES[normalized_zone])
    path = root / filename
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"DuckDB production asset not found for {layer}: {path}")
    return path


def resolve_asset_backend(value: str | None = None) -> str:
    raw_value = value
    if raw_value is None:
        raw_value = os.environ.get(ENV_ASSET_BACKEND)
    explicit_selection = raw_value is not None and str(raw_value).strip() != ""
    backend = str(raw_value or BACKEND_DUCKDB).strip().lower()
    if backend in {"sqlite", "parquet", "sqlite_parquet", "legacy_sqlite_parquet"}:
        raise ValueError(
            "deprecated local asset backend alias is not allowed: "
            f"{backend}. Use 'duckdb' for current production or explicit 'legacy' only for approved rollback flows."
        )
    if backend not in {BACKEND_DUCKDB, BACKEND_LEGACY}:
        raise ValueError(f"unsupported local asset backend: {backend}")
    if backend == BACKEND_LEGACY and explicit_selection and not legacy_rollback_opted_in():
        raise DuckDBRouteError(
            "explicit legacy SQLite/Parquet backend selection requires opt-in via "
            f"{ENV_ALLOW_LEGACY_ROLLBACK}=1"
        )
    return backend


def legacy_rollback_opted_in() -> bool:
    return str(os.environ.get(ENV_ALLOW_LEGACY_ROLLBACK, "")).strip() == "1"


def connect_duckdb_readonly(path: str | Path):
    require_duckdb_installed()
    import duckdb

    return duckdb.connect(str(Path(path)), read_only=True)


def connect_layer_duckdb_readonly(layer: str, data_dir: str | Path | None = None):
    path = resolve_duckdb_path(layer, data_dir=data_dir, require_exists=True)
    return connect_duckdb_readonly(path)


def duckdb_manifest_entry(
    *,
    asset_id: str,
    layer: str,
    table: str,
    asset_path: str | Path,
    source_asset_id: str,
    audit_record: str,
    migration_source_asset: str,
) -> dict[str, Any]:
    normalized = normalize_duckdb_layer(layer)
    return {
        "asset_id": asset_id,
        "track": "production",
        "layer": normalized,
        "asset_type": "duckdb_table",
        "owner_agent": "mcp-agent",
        "status": "production_active",
        "allowed_for_main_workflow": True,
        "asset_path": f"{Path(asset_path).as_posix()}::{table}",
        "source_assets": [source_asset_id],
        "audit_record": audit_record,
        "migration_source_asset": migration_source_asset,
    }

