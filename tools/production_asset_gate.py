"""Fail-closed guard for production asset changes.

The tool validates proposed production asset changes before they are allowed to
enter the main workflow. It is a governance guard only and does not mutate
business assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from adjustment_semantics import (
    MARKET_FIELD_SCHEMA_KIND_BASE,
    MARKET_FIELD_SCHEMA_KIND_STRATEGY,
    NAKED_MARKET_PRICE_COLUMNS,
    QFQ_SUFFIX,
    validate_adjustment_semantics,
    validate_market_field_semantics,
)

REQUIRED_CHANGE_FIELDS = {
    "change_id",
    "change_type",
    "layer",
    "owner_agent",
    "asset_after",
    "audit_required",
    "audit_status",
    "audit_record",
}

DUCKDB_ALLOWED_ACTIVE_STATUSES = {"production_active", "approved_for_l5"}
INACTIVE_ASSET_STATUSES = {"retired_legacy_reference", "superseded"}


def layer_requires_adjustment_semantics(layer: str) -> bool:
    normalized = str(layer or "").upper()
    if normalized in {"L2_STOCK_DAILY_BASE", "L3_FEATURES"}:
        return True
    return normalized.startswith(("L4", "L5", "L6", "L7"))


def layer_market_field_schema_kind(layer: str) -> str | None:
    normalized = str(layer or "").upper()
    if normalized == "L2_STOCK_DAILY_BASE":
        return MARKET_FIELD_SCHEMA_KIND_BASE
    if normalized.startswith(("L5", "L6", "L7")):
        return MARKET_FIELD_SCHEMA_KIND_STRATEGY
    return None


def _normalized_contract_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _has_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def check_adjustment_contract_rule_text(context: str, semantics: dict[str, Any] | None, errors: list[str]) -> None:
    if not isinstance(semantics, dict):
        return
    text = _normalized_contract_text(semantics.get("contract_rule"))
    if not text:
        return
    has_qfq = "qfq" in text
    has_columns = "open/high/low/close/pre_close" in text
    has_explicit_front_adjusted = _has_any_token(
        text,
        ("front-adjusted", "前复权", "explicitly marked", "*_qfq"),
    )
    if not (has_qfq and has_columns and has_explicit_front_adjusted):
        errors.append(
            f"{context} adjustment_semantics.contract_rule must explicitly describe qfq/front-adjusted semantics"
        )


def check_market_field_contract_rule_text(
    context: str,
    semantics: dict[str, Any] | None,
    expected_schema_kind: str | None,
    errors: list[str],
) -> None:
    if not isinstance(semantics, dict):
        return
    text = _normalized_contract_text(semantics.get("contract_rule"))
    if not text or expected_schema_kind is None:
        return
    if expected_schema_kind == MARKET_FIELD_SCHEMA_KIND_BASE:
        has_raw_market_base = _has_any_token(
            text,
            ("raw market prices", "raw market price", "原始行情"),
        )
        has_columns = "open/high/low/close/pre_close" in text
        has_qfq = "qfq" in text
        if not (has_raw_market_base and has_columns and has_qfq):
            errors.append(
                f"{context} market_field_semantics.contract_rule must explicitly describe raw market base columns and qfq semantics"
            )
        return
    if expected_schema_kind == MARKET_FIELD_SCHEMA_KIND_STRATEGY:
        has_raw_suffix = "*_raw" in text
        has_qfq_suffix = "*_qfq" in text or "qfq" in text
        if not (has_raw_suffix and has_qfq_suffix):
            errors.append(
                f"{context} market_field_semantics.contract_rule must explicitly describe *_raw and *_qfq strategy semantics"
            )


def project_root() -> Path:
    return PROJECT_ROOT


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value).split("::", 1)[0])
    if path.is_absolute():
        return path
    return root / path


def _asset_table_name(asset: dict[str, Any]) -> str | None:
    asset_path = str(asset.get("asset_path") or "")
    if "::" not in asset_path:
        return None
    return asset_path.split("::", 1)[1].strip() or None


def _duckdb_table_columns(path: Path, table: str) -> set[str] | None:
    import duckdb

    try:
        with duckdb.connect(str(path), read_only=True) as conn:
            return {str(row[1]) for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except duckdb.Error:
        return None


def check_duckdb_schema_adjustment_markers(
    context: str,
    layer: str,
    asset: dict[str, Any],
    root: Path,
    errors: list[str],
) -> None:
    if str(asset.get("status") or "") in INACTIVE_ASSET_STATUSES:
        return
    asset_type = str(asset.get("asset_type") or "").lower()
    if asset_type == "duckdb_table_files":
        return
    table = _asset_table_name(asset)
    path = resolve(root, asset.get("asset_path"))
    if table is None or path is None or not path.is_file():
        return

    normalized_layer = str(layer or "").upper()
    columns = _duckdb_table_columns(path, table)
    if columns is None:
        return
    qfq_columns = {column for column in columns if column.endswith("_qfq")}

    if normalized_layer.startswith(("L2", "L3_FEATURES")) and qfq_columns:
        missing_qfq_prices = [
            f"{column}_qfq"
            for column in NAKED_MARKET_PRICE_COLUMNS
            if column in columns and f"{column}_qfq" not in columns
        ]
        if missing_qfq_prices:
            errors.append(
                f"{context} schema contains naked market price columns without explicit qfq counterparts: "
                f"{missing_qfq_prices}"
            )

    if normalized_layer.startswith("L3_FEATURES") and qfq_columns:
        ambiguous_naked_factor_columns = sorted(
            column
            for column in columns
            if not column.endswith(QFQ_SUFFIX)
            and column not in NAKED_MARKET_PRICE_COLUMNS
            and f"{column}{QFQ_SUFFIX}" in columns
        )
        if ambiguous_naked_factor_columns:
            errors.append(
                f"{context} schema contains naked factor columns alongside explicit qfq counterparts: "
                f"{ambiguous_naked_factor_columns}"
            )

    if normalized_layer.startswith(("L5", "L6", "L7")):
        naked_market_columns = [column for column in NAKED_MARKET_PRICE_COLUMNS if column in columns]
        if naked_market_columns:
            errors.append(
                f"{context} strategy-side schema must not expose naked market price columns: {naked_market_columns}"
            )


def registry_path(root: Path) -> Path:
    return root / "quant" / "data_file" / "asset_registry" / "production_assets.json"


def check_change(root: Path, change: dict[str, Any], errors: list[str]) -> None:
    missing = REQUIRED_CHANGE_FIELDS - set(change)
    change_id = change.get("change_id", "<unknown>")
    if missing:
        errors.append(f"change {change_id} missing fields: {', '.join(sorted(missing))}")

    if change.get("audit_required") is not True:
        errors.append(f"change {change_id} must set audit_required=true")
    if change.get("audit_status") not in {"passed", "approved", "approved_for_production"}:
        errors.append(f"change {change_id} audit_status is not passed/approved")

    audit_record = resolve(root, change.get("audit_record"))
    if audit_record is None:
        errors.append(f"change {change_id} missing audit_record")
    elif not audit_record.is_file():
        errors.append(f"change {change_id} audit_record not found: {audit_record}")

    asset_after = change.get("asset_after")
    if isinstance(asset_after, dict):
        if asset_after.get("track") != "production":
            errors.append(f"change {change_id} asset_after.track must be production")
        if asset_after.get("allowed_for_main_workflow") is not True:
            errors.append(f"change {change_id} asset_after must be allowed_for_main_workflow")
        if not asset_after.get("audit_record"):
            errors.append(f"change {change_id} asset_after missing audit_record")
        check_duckdb_asset_after(root, change_id, asset_after, errors)
        check_duckdb_schema_adjustment_markers(
            f"change {change_id} asset_after",
            str(change.get("layer") or ""),
            asset_after,
            root,
            errors,
        )
    check_adjustment_semantics_requirement(change_id, change, errors)
    check_market_field_semantics_requirement(change_id, change, errors)


def _is_duckdb_asset(asset: dict[str, Any]) -> bool:
    asset_type = str(asset.get("asset_type") or "").lower()
    asset_path = str(asset.get("asset_path") or "").lower()
    return "duckdb" in asset_type or ".duckdb" in asset_path


def _requires_active_duckdb_status(asset: dict[str, Any]) -> bool:
    status = str(asset.get("status") or "")
    if status in INACTIVE_ASSET_STATUSES:
        return False
    return asset.get("allowed_for_main_workflow") is True


def check_duckdb_asset_after(
    root: Path,
    change_id: str,
    asset_after: dict[str, Any],
    errors: list[str],
) -> None:
    if not _is_duckdb_asset(asset_after):
        return
    if str(asset_after.get("status") or "") in INACTIVE_ASSET_STATUSES:
        return
    if _requires_active_duckdb_status(asset_after) and asset_after.get("status") not in DUCKDB_ALLOWED_ACTIVE_STATUSES:
        allowed = ", ".join(sorted(DUCKDB_ALLOWED_ACTIVE_STATUSES))
        errors.append(f"change {change_id} DuckDB asset_after.status must be one of: {allowed}")
    migration_manifest = resolve(root, asset_after.get("duckdb_migration_manifest"))
    if migration_manifest is None or not migration_manifest.is_file():
        errors.append(
            f"change {change_id} DuckDB asset missing duckdb_migration_manifest file"
        )
    if asset_after.get("rollback_source_asset"):
        errors.append(
            f"change {change_id} active DuckDB asset must not declare rollback_source_asset; "
            "use migration_source_asset for provenance only"
        )
    migration_source = asset_after.get("migration_source_asset")
    if not migration_source:
        errors.append(f"change {change_id} DuckDB asset missing migration_source_asset")
    asset_path = resolve(root, asset_after.get("asset_path"))
    asset_type = str(asset_after.get("asset_type") or "").lower()
    if asset_type == "duckdb_table_files":
        if asset_path is None or not asset_path.is_dir():
            errors.append(
                f"change {change_id} DuckDB asset directory not found: {asset_after.get('asset_path')}"
            )
        return
    if asset_path is None or not asset_path.is_file():
        errors.append(f"change {change_id} DuckDB asset file not found: {asset_after.get('asset_path')}")


def check_adjustment_semantics_requirement(
    change_id: str,
    change: dict[str, Any],
    errors: list[str],
) -> None:
    layer = str(change.get("layer") or "").upper()
    asset_after = change.get("asset_after")
    if not layer_requires_adjustment_semantics(layer) or not isinstance(asset_after, dict) or not _is_duckdb_asset(asset_after):
        return
    try:
        validate_adjustment_semantics(
            asset_after.get("adjustment_semantics"),
            context=f"change {change_id} asset_after",
        )
    except ValueError as exc:
        errors.append(str(exc))
        return
    check_adjustment_contract_rule_text(
        f"change {change_id} asset_after",
        asset_after.get("adjustment_semantics"),
        errors,
    )


def check_market_field_semantics_requirement(
    change_id: str,
    change: dict[str, Any],
    errors: list[str],
) -> None:
    layer = str(change.get("layer") or "").upper()
    expected_schema_kind = layer_market_field_schema_kind(layer)
    asset_after = change.get("asset_after")
    if expected_schema_kind is None or not isinstance(asset_after, dict) or not _is_duckdb_asset(asset_after):
        return
    try:
        validate_market_field_semantics(
            asset_after.get("market_field_semantics"),
            context=f"change {change_id} asset_after",
            expected_schema_kind=expected_schema_kind,
        )
    except ValueError as exc:
        errors.append(str(exc))
        return
    check_market_field_contract_rule_text(
        f"change {change_id} asset_after",
        asset_after.get("market_field_semantics"),
        expected_schema_kind,
        errors,
    )


def check_registry_asset_adjustment_semantics(asset_id: str, asset: dict[str, Any], errors: list[str]) -> None:
    layer = str(asset.get("layer") or "").upper()
    if not layer_requires_adjustment_semantics(layer) or not _is_duckdb_asset(asset):
        return
    if asset.get("status") in INACTIVE_ASSET_STATUSES:
        return
    try:
        validate_adjustment_semantics(
            asset.get("adjustment_semantics"),
            context=f"asset {asset_id}",
        )
    except ValueError as exc:
        errors.append(str(exc))
        return
    check_adjustment_contract_rule_text(
        f"asset {asset_id}",
        asset.get("adjustment_semantics"),
        errors,
    )


def check_registry_asset_market_field_semantics(asset_id: str, asset: dict[str, Any], errors: list[str]) -> None:
    layer = str(asset.get("layer") or "").upper()
    expected_schema_kind = layer_market_field_schema_kind(layer)
    if expected_schema_kind is None or not _is_duckdb_asset(asset):
        return
    if asset.get("status") in INACTIVE_ASSET_STATUSES:
        return
    try:
        validate_market_field_semantics(
            asset.get("market_field_semantics"),
            context=f"asset {asset_id}",
            expected_schema_kind=expected_schema_kind,
        )
    except ValueError as exc:
        errors.append(str(exc))
        return
    check_market_field_contract_rule_text(
        f"asset {asset_id}",
        asset.get("market_field_semantics"),
        expected_schema_kind,
        errors,
    )


def check_change_file(root: Path, path: Path) -> list[str]:
    data = load_json(path)
    changes = data if isinstance(data, list) else data.get("production_asset_changes", [data])
    errors: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            errors.append("change entry must be object")
            continue
        check_change(root, change, errors)
    return errors


def check_registry(root: Path) -> list[str]:
    data = load_json(registry_path(root))
    errors: list[str] = []
    for asset in data.get("assets", []):
        asset_id = asset.get("asset_id", "<unknown>")
        status = str(asset.get("status") or "")
        if asset.get("track") != "production":
            errors.append(f"asset {asset_id} is not production track")
        if status in INACTIVE_ASSET_STATUSES:
            if asset.get("allowed_for_main_workflow") is not False:
                errors.append(f"asset {asset_id} rollback asset must not be allowed for main workflow")
            if not asset.get("superseded_by"):
                errors.append(f"asset {asset_id} rollback asset missing superseded_by")
        elif asset.get("allowed_for_main_workflow") is not True:
            errors.append(f"asset {asset_id} not allowed for main workflow")
        audit_record = resolve(root, asset.get("audit_record"))
        if audit_record is None or not audit_record.is_file():
            errors.append(f"asset {asset_id} audit_record not found: {asset.get('audit_record')}")
        check_duckdb_asset_after(root, f"registry:{asset_id}", asset, errors)
        check_duckdb_schema_adjustment_markers(
            f"asset {asset_id}",
            str(asset.get("layer") or ""),
            asset,
            root,
            errors,
        )
        check_registry_asset_adjustment_semantics(asset_id, asset, errors)
        check_registry_asset_market_field_semantics(asset_id, asset, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production asset change gate.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    parser.add_argument("--change-file", default=None, help="JSON change request to validate.")
    parser.add_argument("--check-registry", action="store_true", help="Validate current production registry.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    errors: list[str] = []
    if args.check_registry:
        errors.extend(check_registry(root))
    if args.change_file:
        path = Path(args.change_file)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"change file not found: {path}")
        else:
            errors.extend(check_change_file(root, path))
    if not args.check_registry and not args.change_file:
        errors.append("must pass --check-registry or --change-file")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS production_asset_gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
