from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adjustment_semantics import (
    MARKET_FIELD_SCHEMA_KIND_STRATEGY,
    validate_adjustment_semantics,
    validate_market_field_semantics,
)
from stock_daily_data_route import resolve_stock_daily_backend, resolve_stock_daily_db_path, resolve_stock_daily_duckdb_path
PROJECT_ROOT = Path(__file__).resolve().parent


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


def _is_duckdb_path(path: str | Path | None) -> bool:
    if path in (None, ""):
        return False
    return Path(str(path)).suffix.lower() == ".duckdb"


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


def _resolve_required_manifest_path(
    manifest: dict[str, Any],
    key: str,
    manifest_path: Path,
    *,
    label: str,
) -> Path:
    resolved = _resolve_manifest_relative_path(manifest.get(key), manifest_path)
    if resolved is None or not resolved.is_file():
        raise ValueError(f"{label} is required for duckdb_table source and must exist: {manifest.get(key)}")
    return resolved


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

    source_type = str(manifest.get("source_type", "") or "").strip()
    if not source_type:
        raise ValueError("prediction manifest must define explicit source_type; DuckDB-only production mode does not allow implicit sqlite defaults")
    if source_type not in {"sqlite_table", "duckdb_table"}:
        raise ValueError(f"unsupported prediction source_type: {source_type}")
    if source_type == "sqlite_table" and not allow_legacy:
        raise ValueError(
            "prediction manifest points to sqlite_table; current production path is DuckDB-only. "
            "Use duckdb_table manifests or explicit legacy reproduction mode."
        )

    db_path = _resolve_manifest_relative_path(
        manifest.get("db_path") or manifest.get("database"),
        resolved_manifest_path,
    )
    table = str(manifest.get("table") or manifest.get("prediction_table") or "")
    if not db_path or not table:
        raise ValueError("prediction manifest must define db_path and table for table sources")

    if source_type == "sqlite_table" and not allow_legacy and is_legacy_prediction_source(db_path, manifest):
        raise ValueError("prediction manifest points to a legacy source; production mode blocks legacy odb.db inputs")

    audit_record = None
    duckdb_migration_manifest = None
    adjustment_semantics = None
    market_field_semantics = None
    if source_type == "duckdb_table":
        if require_approved or approval_status == "approved_for_l5":
            audit_record = _resolve_required_manifest_path(
                manifest,
                "audit_record",
                resolved_manifest_path,
                label="audit_record",
            )
            duckdb_migration_manifest = _resolve_required_manifest_path(
                manifest,
                "duckdb_migration_manifest",
                resolved_manifest_path,
                label="duckdb_migration_manifest",
            )
            adjustment_semantics = validate_adjustment_semantics(
                manifest.get("adjustment_semantics"),
                context=f"prediction manifest {resolved_manifest_path.name}",
            )
            market_field_semantics = validate_market_field_semantics(
                manifest.get("market_field_semantics"),
                context=f"prediction manifest {resolved_manifest_path.name}",
                expected_schema_kind=MARKET_FIELD_SCHEMA_KIND_STRATEGY,
            )
        else:
            audit_record = _resolve_manifest_relative_path(manifest.get("audit_record"), resolved_manifest_path)
            duckdb_migration_manifest = _resolve_manifest_relative_path(
                manifest.get("duckdb_migration_manifest"),
                resolved_manifest_path,
            )
            raw_semantics = manifest.get("adjustment_semantics")
            if raw_semantics is not None:
                adjustment_semantics = validate_adjustment_semantics(
                    raw_semantics,
                    context=f"prediction manifest {resolved_manifest_path.name}",
                )
            raw_market_field_semantics = manifest.get("market_field_semantics")
            if raw_market_field_semantics is not None:
                market_field_semantics = validate_market_field_semantics(
                    raw_market_field_semantics,
                    context=f"prediction manifest {resolved_manifest_path.name}",
                    expected_schema_kind=MARKET_FIELD_SCHEMA_KIND_STRATEGY,
                )
        if not db_path.is_file():
            raise ValueError(f"duckdb_table db_path does not exist: {db_path}")

    market_db_path = _resolve_manifest_relative_path(manifest.get("market_db_path"), resolved_manifest_path)
    return {
        "manifest_path": resolved_manifest_path,
        "source_type": source_type,
        "db_path": db_path,
        "table": table,
        "market_db_path": market_db_path,
        "approval_status": approval_status,
        "asset_role": manifest.get("asset_role"),
        "audit_record": audit_record,
        "duckdb_migration_manifest": duckdb_migration_manifest,
        "adjustment_semantics": adjustment_semantics,
        "market_field_semantics": market_field_semantics,
        "lineage": manifest.get("lineage"),
        "lineage_sources": manifest.get("lineage_sources"),
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
        "source_type": "sqlite_table",
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
    source_type = str(source.get("source_type") or "").strip().lower()
    if market_db_path not in (None, ""):
        resolved = _resolve_input_path(market_db_path)
        assert resolved is not None
        if source_type == "duckdb_table" and not _is_duckdb_path(resolved):
            raise ValueError(
                "production market_db_path override must point to a DuckDB file; "
                "DuckDB-only production mode blocks SQLite market routes"
            )
        return resolved
    manifest_market_db = source.get("market_db_path")
    if manifest_market_db:
        resolved = Path(manifest_market_db)
        if source_type == "duckdb_table" and not _is_duckdb_path(resolved):
            raise ValueError(
                "prediction manifest market_db_path must point to a DuckDB file; "
                "DuckDB-only production mode blocks SQLite market routes"
            )
        return resolved
    backend = resolve_stock_daily_backend()
    if backend == "duckdb":
        return resolve_stock_daily_duckdb_path(require_exists=False)
    if source_type == "sqlite_table":
        raise ValueError(
            "legacy sqlite_table prediction sources must provide explicit market_db_path; "
            "implicit SQLite market defaults are disabled in the current DuckDB-only production architecture"
        )
    return resolve_stock_daily_db_path(require_exists=False)
