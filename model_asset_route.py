from __future__ import annotations

import json
import os
from pathlib import Path

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
    path = explicit or (_base_data_dir(data_dir) / DEFAULT_FEATURE_DIR_NAME)
    if require_exists and not path.exists():
        raise FileNotFoundError(f"model feature path not found: {path}")
    return path


def resolve_model_label_path(
    data_dir: str | Path | None = None,
    *,
    label_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    explicit = _resolve_explicit_path(label_path or os.environ.get(ENV_MODEL_LABEL_PATH))
    path = explicit or (_base_data_dir(data_dir) / DEFAULT_LABEL_DIR_NAME)
    if require_exists and not path.exists():
        raise FileNotFoundError(f"model label path not found: {path}")
    return path


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
    path = explicit or (resolve_model_prediction_root(data_dir) / DEFAULT_PREDICTION_DB_NAME)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
