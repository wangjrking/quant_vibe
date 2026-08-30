from __future__ import annotations

import json
import os
from pathlib import Path

from duckdb_asset_route import resolve_duckdb_path
from production_asset_registry import active_main_workflow_asset, active_main_workflow_assets, split_asset_path
from project_paths import resolve_data_dir


MODEL_FEATURE_MODE_ENV = "QUANT_MODEL_FEATURE_MODE"
MODEL_FEATURE_MODE_SPLIT = "production_split"
MODEL_FEATURE_MODE_LEGACY = "legacy_mixed"

MODEL_PREDICTION_MODE_ENV = "QUANT_MODEL_PREDICTION_MODE"
MODEL_PREDICTION_MODE_INDEPENDENT = "independent"
MODEL_PREDICTION_MODE_LEGACY = "legacy_odb"
LEGACY_MODEL_ASSET_CHAIN_ENV = "QUANT_ALLOW_LEGACY_MODEL_ASSET_CHAIN"

ENV_MODEL_FEATURE_PATH = "QUANT_MODEL_FEATURE_PATH"
ENV_MODEL_LABEL_PATH = "QUANT_MODEL_LABEL_PATH"
ENV_MODEL_PREDICTION_ROOT = "QUANT_MODEL_PREDICTION_ROOT"
ENV_MODEL_PREDICTION_DB = "QUANT_MODEL_PREDICTION_DB"
ENV_MODEL_FEATURE_DUCKDB = "QUANT_MODEL_FEATURE_DUCKDB"
ENV_MODEL_LABEL_DUCKDB = "QUANT_MODEL_LABEL_DUCKDB"
ENV_MODEL_PREDICTION_DUCKDB = "QUANT_MODEL_PREDICTION_DUCKDB"
ENV_MODEL_FEATURE_DUCKDB_TABLE = "QUANT_MODEL_FEATURE_DUCKDB_TABLE"
ENV_MODEL_LABEL_DUCKDB_TABLE = "QUANT_MODEL_LABEL_DUCKDB_TABLE"
ENV_MODEL_PREDICTION_DUCKDB_TABLE = "QUANT_MODEL_PREDICTION_DUCKDB_TABLE"

DEFAULT_FEATURE_DIR_NAME = "production_factor_parts"
DEFAULT_LABEL_DIR_NAME = "prediction_label_parts"
DEFAULT_LEGACY_FACTOR_FILE = "stock_factor_data.parquet"
DEFAULT_PREDICTION_ROOT_DIR = "model_predictions"
DEFAULT_PREDICTION_DB_NAME = "MODEL_PREDICTIONS.db"
DEFAULT_LEGACY_DB_NAME = "odb.db"
RESEARCH_PREDICTION_ASSET_ROLE = "l4_research_prediction_asset"
RESEARCH_PREDICTION_APPROVAL_STATUS = "research_only_not_for_l5"
LEGACY_MODEL_ASSET_CHAIN_NOTICE = (
    "legacy model asset chain is archived and disabled by default. "
    "Use production_factor_parts/ + prediction_label_parts/ + model_predictions/ instead. "
    f"Set {LEGACY_MODEL_ASSET_CHAIN_ENV}=1 only for explicit rollback, migration, or legacy reproduction."
)


def _base_data_dir(data_dir: str | Path | None = None) -> Path:
    return resolve_data_dir(data_dir)


def _resolve_explicit_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else path.resolve()


def _asset_is_active_sqlite_route(asset: dict | None) -> bool:
    if not asset:
        return False
    asset_type = str(asset.get("asset_type", "")).strip().lower()
    asset_path = str(asset.get("asset_path", "")).strip().lower()
    return (
        asset_type.startswith("sqlite")
        or asset_path.endswith(".db")
        or ".db::" in asset_path
    )


def _registry_asset_path(layer: str, *, data_dir: str | Path | None = None) -> tuple[Path | None, str | None, dict | None]:
    if str(layer).strip().lower() == "l4_model_prediction":
        assets = active_main_workflow_assets(layer, data_dir=data_dir)
        preferred_order = {
            "duckdb": 0,
            "sqlite_table_with_formal_manifests": 1,
            "sqlite_table": 2,
            "formal_prediction_manifest": 3,
        }

        def _rank(asset: dict) -> int:
            asset_type = str(asset.get("asset_type", "")).lower()
            asset_path = str(asset.get("asset_path", "")).lower()
            if "duckdb" in asset_type or asset_path.endswith(".duckdb") or ".duckdb::" in asset_path:
                return preferred_order["duckdb"]
            return preferred_order.get(asset_type, len(preferred_order))

        ranked_assets = sorted(
            assets,
            key=_rank,
        )
        asset = ranked_assets[0] if ranked_assets else None
        if _asset_is_active_sqlite_route(asset):
            raise RuntimeError(
                "active L4 prediction mainline must not route to SQLite in the current DuckDB-only production architecture. "
                "Use approved formal DuckDB manifests or DuckDB-backed prediction assets instead."
            )
    else:
        asset = active_main_workflow_asset(layer, data_dir=data_dir)
        if _asset_is_active_sqlite_route(asset):
            raise RuntimeError(
                f"active {layer} mainline must not route to SQLite in the current DuckDB-only production architecture. "
                "Use split DuckDB assets instead."
            )
    if not asset:
        return None, None, None
    path, table = split_asset_path(asset.get("asset_path"))
    return path, table, asset


def _active_l4_formal_manifest_assets(*, data_dir: str | Path | None = None) -> list[dict]:
    return [
        asset
        for asset in active_main_workflow_assets("L4_model_prediction", data_dir=data_dir)
        if str(asset.get("asset_type", "")).lower() == "formal_prediction_manifest"
    ]


def model_feature_mode(value: str | None = None) -> str:
    mode = str(value or os.environ.get(MODEL_FEATURE_MODE_ENV) or MODEL_FEATURE_MODE_SPLIT).strip().lower()
    if mode not in {MODEL_FEATURE_MODE_SPLIT, MODEL_FEATURE_MODE_LEGACY}:
        raise ValueError(f"unsupported model feature mode: {mode}")
    return mode


def use_legacy_mixed_features(value: str | None = None) -> bool:
    return model_feature_mode(value) == MODEL_FEATURE_MODE_LEGACY


def model_prediction_mode(value: str | None = None) -> str:
    mode = str(value or os.environ.get(MODEL_PREDICTION_MODE_ENV) or MODEL_PREDICTION_MODE_INDEPENDENT).strip().lower()
    if mode not in {MODEL_PREDICTION_MODE_INDEPENDENT, MODEL_PREDICTION_MODE_LEGACY}:
        raise ValueError(f"unsupported model prediction mode: {mode}")
    return mode


def use_legacy_prediction_db(value: str | None = None) -> bool:
    return model_prediction_mode(value) == MODEL_PREDICTION_MODE_LEGACY


def legacy_model_asset_chain_opted_in() -> bool:
    return str(os.environ.get(LEGACY_MODEL_ASSET_CHAIN_ENV, "")).strip() == "1"


def require_legacy_model_asset_chain_opt_in(*, reason: str | None = None) -> None:
    if legacy_model_asset_chain_opted_in():
        return
    suffix = f" Requested legacy usage: {reason}." if reason else ""
    raise RuntimeError(LEGACY_MODEL_ASSET_CHAIN_NOTICE + suffix)


def resolve_model_feature_path(
    data_dir: str | Path | None = None,
    *,
    feature_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(feature_path or os.environ.get(ENV_MODEL_FEATURE_PATH))
    if explicit:
        path = explicit
    else:
        registry_path, _registry_table, _asset = _registry_asset_path("L3_features", data_dir=data_dir)
        if registry_path and registry_path.suffix.lower() == ".duckdb":
            raise RuntimeError(
                "L3 feature mainline is registered as DuckDB; use resolve_model_feature_duckdb_path() "
                "and resolve_model_feature_duckdb_table() instead of parquet directory routes."
            )
        path = registry_path or (_base_data_dir(data_dir) / DEFAULT_FEATURE_DIR_NAME)
    if require_exists and not path.exists():
        raise FileNotFoundError(f"model feature path not found: {path}")
    return path


def resolve_model_feature_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(duckdb_path or os.environ.get(ENV_MODEL_FEATURE_DUCKDB))
    if explicit:
        path = explicit
    else:
        registry_path, _registry_table, _asset = _registry_asset_path("L3_features", data_dir=data_dir)
        if registry_path and registry_path.suffix.lower() == ".duckdb":
            path = registry_path
        else:
            path = resolve_duckdb_path("L3_FEATURES", data_dir=data_dir)
    if require_exists and not path.exists():
        raise FileNotFoundError(f"model feature DuckDB asset not found: {path}")
    return path


def resolve_model_feature_duckdb_table(
    data_dir: str | Path | None = None,
    *,
    table_name: str | None = None,
) -> str:
    explicit = str(table_name or os.environ.get(ENV_MODEL_FEATURE_DUCKDB_TABLE) or "").strip()
    if explicit:
        return explicit
    _registry_path, registry_table, asset = _registry_asset_path("L3_features", data_dir=data_dir)
    if registry_table:
        return registry_table
    if asset and "duckdb" in str(asset.get("asset_type", "")).lower():
        raise RuntimeError("active L3 feature DuckDB asset is missing table binding in asset_path")
    return ""


def resolve_model_label_path(
    data_dir: str | Path | None = None,
    *,
    label_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(label_path or os.environ.get(ENV_MODEL_LABEL_PATH))
    if explicit:
        path = explicit
    else:
        registry_path, _registry_table, _asset = _registry_asset_path("L3_labels", data_dir=data_dir)
        if registry_path and registry_path.suffix.lower() == ".duckdb":
            raise RuntimeError(
                "L3 label mainline is registered as DuckDB; use resolve_model_label_duckdb_path() "
                "and resolve_model_label_duckdb_table() instead of parquet directory routes."
            )
        path = registry_path or (_base_data_dir(data_dir) / DEFAULT_LABEL_DIR_NAME)
    if require_exists and not path.exists():
        raise FileNotFoundError(f"model label path not found: {path}")
    return path


def resolve_model_label_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(duckdb_path or os.environ.get(ENV_MODEL_LABEL_DUCKDB))
    if explicit:
        path = explicit
    else:
        registry_path, _registry_table, _asset = _registry_asset_path("L3_labels", data_dir=data_dir)
        if registry_path and registry_path.suffix.lower() == ".duckdb":
            path = registry_path
        else:
            path = resolve_duckdb_path("L3_LABELS", data_dir=data_dir)
    if require_exists and not path.exists():
        raise FileNotFoundError(f"model label DuckDB asset not found: {path}")
    return path


def resolve_model_label_duckdb_table(
    data_dir: str | Path | None = None,
    *,
    table_name: str | None = None,
) -> str:
    explicit = str(table_name or os.environ.get(ENV_MODEL_LABEL_DUCKDB_TABLE) or "").strip()
    if explicit:
        return explicit
    _registry_path, registry_table, asset = _registry_asset_path("L3_labels", data_dir=data_dir)
    if registry_table:
        return registry_table
    if asset and "duckdb" in str(asset.get("asset_type", "")).lower():
        raise RuntimeError("active L3 label DuckDB asset is missing table binding in asset_path")
    return ""


def resolve_legacy_mixed_factor_path(
    data_dir: str | Path | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    path = _base_data_dir(data_dir) / DEFAULT_LEGACY_FACTOR_FILE
    if require_exists and not path.exists():
        raise FileNotFoundError(f"legacy mixed factor parquet not found: {path}")
    return path


def resolve_model_prediction_root(
    data_dir: str | Path | None = None,
    *,
    prediction_root: str | Path | None = None,
    create: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(prediction_root or os.environ.get(ENV_MODEL_PREDICTION_ROOT))
    path = explicit or (_base_data_dir(data_dir) / DEFAULT_PREDICTION_ROOT_DIR)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_prediction_db_path(
    data_dir: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    create_parent: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(db_path or os.environ.get(ENV_MODEL_PREDICTION_DB))
    if explicit:
        path = explicit
    else:
        registry_path, _registry_table, asset = _registry_asset_path("L4_model_prediction", data_dir=data_dir)
        if registry_path and registry_path.suffix.lower() == ".duckdb":
            raise RuntimeError(
                "L4 prediction mainline is registered as DuckDB; use prediction manifests or "
                "resolve_model_prediction_duckdb_path() for DuckDB-backed prediction assets."
            )
        if registry_path and registry_path.suffix.lower() == ".db":
            path = registry_path
        elif _active_l4_formal_manifest_assets(data_dir=data_dir):
            raise RuntimeError(
                "L4 prediction mainline is manifest-routed and DuckDB-only; "
                "use approved formal prediction manifests instead of resolve_model_prediction_db_path()."
            )
        else:
            raise RuntimeError(
                "Implicit MODEL_PREDICTIONS.db fallback is disabled in the current DuckDB-only production architecture. "
                "Use approved formal prediction manifests, resolve_model_prediction_duckdb_path(), or pass an explicit legacy db_path."
            )
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_prediction_duckdb_path(
    data_dir: str | Path | None = None,
    *,
    duckdb_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(duckdb_path or os.environ.get(ENV_MODEL_PREDICTION_DUCKDB))
    if explicit:
        path = explicit
    else:
        registry_path, _registry_table, _asset = _registry_asset_path("L4_model_prediction", data_dir=data_dir)
        if registry_path and registry_path.suffix.lower() == ".duckdb":
            path = registry_path
        elif _active_l4_formal_manifest_assets(data_dir=data_dir):
            raise RuntimeError(
                "L4 prediction mainline is manifest-routed across formal DuckDB assets; "
                "use approved formal prediction manifests instead of resolve_model_prediction_duckdb_path()."
            )
        else:
            path = resolve_duckdb_path("L4", data_dir=data_dir)
    if require_exists and not path.exists():
        raise FileNotFoundError(f"model prediction DuckDB asset not found: {path}")
    return path


def resolve_model_prediction_duckdb_table(
    data_dir: str | Path | None = None,
    *,
    table_name: str | None = None,
) -> str:
    explicit = str(table_name or os.environ.get(ENV_MODEL_PREDICTION_DUCKDB_TABLE) or "").strip()
    if explicit:
        return explicit
    _registry_path, registry_table, asset = _registry_asset_path("L4_model_prediction", data_dir=data_dir)
    if registry_table:
        return registry_table
    if _active_l4_formal_manifest_assets(data_dir=data_dir):
        raise RuntimeError(
            "active L4 prediction mainline is manifest-routed across multiple formal DuckDB assets; "
            "use formal prediction manifests instead of resolve_model_prediction_duckdb_table()."
        )
    if asset and "duckdb" in str(asset.get("asset_type", "")).lower():
        raise RuntimeError(
            "active L4 prediction DuckDB asset is registered as a multi-table bundle; "
            "use formal prediction manifests instead of resolve_model_prediction_duckdb_table()."
        )
    return ""


def resolve_legacy_prediction_db_path(data_dir: str | Path | None = None) -> Path:
    return _base_data_dir(data_dir) / DEFAULT_LEGACY_DB_NAME


def resolve_prediction_run_dir(
    data_dir: str | Path | None,
    *,
    label: str,
    output_table: str | None = None,
    create: bool = False,
) -> Path:
    table_name = output_table or f"stock_predict_data_{label}_rolling"
    path = resolve_model_prediction_root(data_dir) / table_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_prediction_manifest(path: str | Path, payload: dict) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def enrich_research_prediction_manifest(payload: dict) -> dict:
    manifest = dict(payload)
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("asset_role", RESEARCH_PREDICTION_ASSET_ROLE)
    manifest.setdefault("model_track", "research")
    manifest.setdefault("approval_status", RESEARCH_PREDICTION_APPROVAL_STATUS)
    manifest.setdefault("promotion_requires_user_confirmation", True)
    manifest.setdefault(
        "notes",
        "Research prediction asset generated from the standard model chain. Not approved for L5 production use.",
    )
    return manifest
