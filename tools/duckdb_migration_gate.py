"""Fail-closed gate for DuckDB production migration reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value.split("::", 1)[0])
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_audit_record(value: str | None, label: str, errors: list[str]) -> None:
    path = resolve_project_path(value)
    if path is None:
        errors.append(f"{label} missing audit_record")
    elif not path.is_file():
        errors.append(f"{label} audit_record not found: {value}")


def check_report(report: dict[str, Any], approval_audit_record: str | None, errors: list[str]) -> None:
    if report.get("dry_run") is True:
        errors.append("DuckDB migration report is dry_run; production gate requires materialized report")
    boundary = report.get("boundary", {})
    if boundary.get("edits_production_registry") is not False:
        errors.append("migration report must not edit production registry")
    if boundary.get("switches_mainline_routes") is not False:
        errors.append("migration report must not switch mainline routes")
    for layer, path_text in report.get("duckdb_files", {}).items():
        path = resolve_project_path(path_text)
        if path is None or not path.is_file():
            errors.append(f"DuckDB file for {layer} not found: {path_text}")

    if not approval_audit_record:
        errors.append("approval audit record is required before DuckDB production switch")
    else:
        check_audit_record(approval_audit_record, "approval", errors)

    for asset in report.get("assets", []):
        asset_id = asset.get("asset_id", "<unknown>")
        if asset.get("status") != "ok":
            errors.append(f"asset {asset_id} migration status is not ok: {asset.get('status')}")
        check_audit_record(asset.get("source_audit_record"), f"asset {asset_id}", errors)
        tables = asset.get("tables")
        if not isinstance(tables, list) or not tables:
            errors.append(f"asset {asset_id} has no materialized table checks")
            continue
        for table in tables:
            table_name = table.get("target_table", "<unknown>")
            if table.get("row_count_match") is not True:
                errors.append(f"asset {asset_id} table {table_name} row_count_match is not true")
            if not table.get("columns"):
                errors.append(f"asset {asset_id} table {table_name} has no columns")
            if not table.get("sample_sha256"):
                errors.append(f"asset {asset_id} table {table_name} missing sample_sha256")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DuckDB migration report before production switch.")
    parser.add_argument("--report", required=True, help="duckdb_migration_report.json")
    parser.add_argument("--approval-audit-record", help="Audit record approving DuckDB production switch.")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    errors: list[str] = []
    if not report_path.is_file():
        errors.append(f"migration report not found: {report_path}")
    else:
        check_report(load_json(report_path), args.approval_audit_record, errors)

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS duckdb_migration_gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
