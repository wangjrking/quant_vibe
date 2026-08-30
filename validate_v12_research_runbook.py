from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_BOUNDARIES = [
    "no_training",
    "no_prediction",
    "no_production_manifest_change",
    "no_signal",
    "no_backtest",
]


def _contains_legacy_path(value: str) -> bool:
    text = str(value).replace("\\", "/").lower()
    return any(
        marker in text
        for marker in [
            "stock_factor_data.parquet",
            "standard_factor_by_date_parts",
            "odb.db",
        ]
    )


def _job_spec_by_horizon(job_specs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("horizon")): item for item in job_specs.get("job_specs", [])}


def validate_runbook(runbook: dict[str, Any], job_specs: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if runbook.get("approval_status") != "research_plan_not_approved_for_execution":
        errors.append("approval_status must be research_plan_not_approved_for_execution")

    standard_chain = runbook.get("standard_chain", {})
    prediction_asset_root = str(standard_chain.get("prediction_asset_root", "")).replace("\\", "/")
    if "production_factor_parts" not in str(standard_chain.get("feature_input", "")):
        errors.append("standard_chain.feature_input must point to production_factor_parts")
    if "prediction_label_parts" not in str(standard_chain.get("label_input", "")):
        errors.append("standard_chain.label_input must point to prediction_label_parts")
    if "prediction_db" in standard_chain:
        errors.append("standard_chain.prediction_db is retired; use prediction_asset_root in DuckDB-only mode")
    if "production_assets/duckdb" not in prediction_asset_root:
        errors.append("standard_chain.prediction_asset_root must point to production_assets/duckdb")
    for key, value in standard_chain.items():
        if _contains_legacy_path(str(value)):
            errors.append(f"standard_chain.{key} must not point to legacy asset")

    entrypoints = runbook.get("entrypoints", {})
    for name, path in entrypoints.items():
        if not Path(path).exists():
            errors.append(f"entrypoint {name} does not exist: {path}")

    boundaries = runbook.get("boundaries", {})
    for key in REQUIRED_BOUNDARIES:
        if boundaries.get(key) is not True:
            errors.append(f"boundary {key} must be true")

    artifacts = set(str(item) for item in runbook.get("required_artifacts", []))
    for artifact in ["models/model_foldXX.json", "models/model_foldXX_metadata.json", "prediction_manifest.json"]:
        if artifact not in artifacts:
            errors.append(f"required_artifacts missing {artifact}")

    specs_by_horizon = _job_spec_by_horizon(job_specs)
    checked_horizons: list[str] = []
    for item in runbook.get("v12_execution_order", []):
        horizon = str(item.get("horizon"))
        checked_horizons.append(horizon)
        spec = specs_by_horizon.get(horizon)
        if spec is None:
            errors.append(f"{horizon} missing from job specs")
            continue
        for runbook_key, spec_key in [
            ("label", "label"),
            ("job_type", "job_type"),
            ("failed_constraint", "failed_constraints"),
            ("requires_training_authorization", "requires_training_authorization"),
        ]:
            if item.get(runbook_key) != spec.get(spec_key):
                errors.append(f"{horizon} {runbook_key} does not match job specs")

        if item.get("requires_training_authorization") is True:
            table_template = str(item.get("research_table_template", ""))
            if table_template and "_research" not in table_template:
                errors.append(f"{horizon} research_table_template must contain _research")
            if "formal" in table_template:
                errors.append(f"{horizon} research_table_template must not contain formal")
        else:
            if horizon != "10d":
                warnings.append(f"{horizon} is validation-only but is not 10d")
            if horizon == "10d":
                if item.get("formula_asset_validation_required") is not True:
                    errors.append("10d validation-only item must require formula asset validation")
                if "formula_asset_validation" not in entrypoints:
                    errors.append("entrypoint formula_asset_validation is required for validation-only 10d")

    if not checked_horizons:
        errors.append("v12_execution_order is empty")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked_horizons": checked_horizons,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate V12 research runbook against job specs.")
    parser.add_argument("--runbook-json", required=True)
    parser.add_argument("--job-specs-json", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runbook = json.loads(Path(args.runbook_json).read_text(encoding="utf-8"))
    job_specs = json.loads(Path(args.job_specs_json).read_text(encoding="utf-8"))
    result = validate_runbook(runbook, job_specs)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
