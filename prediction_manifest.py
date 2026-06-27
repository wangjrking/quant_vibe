from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MARKET_DB = PROJECT_ROOT.parent / "data_file" / "STOCK_DAILY_DATA.db"


def _resolve_input_path(raw_path: str | Path | None) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _resolve_manifest_relative_path(raw_path: str | Path | None, manifest_path: Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def is_legacy_db_path(db_path: str | Path | None) -> bool:
    if db_path in (None, ""):
        return False
    return Path(str(db_path)).name.lower() == "odb.db"


def is_legacy_prediction_source(
    db_path: str | Path | None,
    manifest: dict[str, Any] | None = None,
) -> bool:
    manifest = manifest or {}
    if bool(manifest.get("legacy_source")):
        return True
    if str(manifest.get("asset_role", "")).lower().startswith("legacy"):
        return True
    return is_legacy_db_path(db_path)


def load_prediction_source_manifest(
    manifest_path: str | Path,
    *,
    require_approved: bool,
    allow_legacy: bool,
) -> dict[str, Any]:
    resolved_manifest_path = _resolve_input_path(manifest_path)
    if resolved_manifest_path is None:
        raise ValueError("prediction manifest path is required")
    with resolved_manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    approval_status = str(manifest.get("approval_status", "") or "")
    if require_approved and approval_status != "approved_for_l5":
        raise ValueError(
            f"prediction manifest must be approved_for_l5, got {approval_status or 'missing'}"
        )

    source_type = str(manifest.get("source_type", "sqlite_table") or "sqlite_table")
    if source_type != "sqlite_table":
        raise ValueError(f"unsupported prediction source_type: {source_type}")

    db_path = _resolve_manifest_relative_path(
        manifest.get("db_path") or manifest.get("database"),
        resolved_manifest_path,
    )
    table = str(manifest.get("table") or manifest.get("prediction_table") or "")
    if not db_path or not table:
        raise ValueError("prediction manifest must define db_path and table for sqlite_table sources")

    if not allow_legacy and is_legacy_prediction_source(db_path, manifest):
        raise ValueError("prediction manifest points to a legacy source; production mode blocks legacy odb.db inputs")

    market_db_path = _resolve_manifest_relative_path(manifest.get("market_db_path"), resolved_manifest_path)
    return {
        "manifest_path": resolved_manifest_path,
        "db_path": db_path,
        "table": table,
        "market_db_path": market_db_path,
        "approval_status": approval_status,
        "asset_role": manifest.get("asset_role"),
        "manifest": manifest,
    }


def resolve_prediction_source(
    *,
    prediction_manifest: str | Path | None,
    legacy_reproduction: bool,
    db_path: str | Path | None,
    table: str | None,
) -> dict[str, Any]:
    if prediction_manifest not in (None, ""):
        return load_prediction_source_manifest(
            prediction_manifest,
            require_approved=True,
            allow_legacy=False,
        )

    if not legacy_reproduction:
        raise ValueError(
            "production signal export requires a prediction manifest; use --legacy-reproduction with explicit --db and --table for legacy replay"
        )

    resolved_db_path = _resolve_input_path(db_path)
    normalized_table = str(table or "")
    if not resolved_db_path or not normalized_table:
        raise ValueError("legacy reproduction mode requires explicit db and table")

    return {
        "manifest_path": None,
        "db_path": resolved_db_path,
        "table": normalized_table,
        "market_db_path": None,
        "approval_status": "legacy_reproduction",
        "asset_role": "legacy_reproduction_prediction_asset",
        "manifest": {
            "legacy_source": True,
            "asset_role": "legacy_reproduction_prediction_asset",
        },
    }


def resolve_market_db_path(
    source: dict[str, Any],
    market_db_path: str | Path | None,
) -> Path:
    if market_db_path not in (None, ""):
        resolved = _resolve_input_path(market_db_path)
        assert resolved is not None
        return resolved
    manifest_market_db = source.get("market_db_path")
    if manifest_market_db:
        return Path(manifest_market_db)
    return DEFAULT_MARKET_DB.resolve()
