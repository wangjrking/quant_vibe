from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _find_objective_result(objective_score: dict[str, Any], horizon: str) -> dict[str, Any] | None:
    for item in objective_score.get("results", []):
        if str(item.get("horizon")) == str(horizon):
            return item
    return None


def validate_candidate_acceptance(
    *,
    horizon: str,
    artifact_validation: dict[str, Any],
    runbook_validation: dict[str, Any],
    objective_score: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if artifact_validation.get("ok") is not True:
        artifact_errors = artifact_validation.get("errors") or []
        errors.append("artifact validation failed")
        errors.extend(str(item) for item in artifact_errors)

    prediction_table_summary = artifact_validation.get("prediction_table_summary")
    if not isinstance(prediction_table_summary, dict):
        errors.append("artifact validation must include prediction_table_summary")
    else:
        row_count = int(prediction_table_summary.get("row_count") or 0)
        null_pred_prob = int(prediction_table_summary.get("null_pred_prob") or 0)
        duplicate_key_groups = int(prediction_table_summary.get("duplicate_key_groups") or 0)
        if row_count <= 0:
            errors.append("prediction table row_count must be positive")
        if null_pred_prob != 0:
            errors.append("prediction table null pred_prob count must be 0")
        if duplicate_key_groups != 0:
            errors.append("prediction table duplicate key groups must be 0")

    if runbook_validation.get("ok") is not True:
        runbook_errors = runbook_validation.get("errors") or []
        errors.append("runbook validation failed")
        errors.extend(str(item) for item in runbook_errors)

    result = _find_objective_result(objective_score, horizon)
    weighted_score = None
    if result is None:
        errors.append(f"objective result missing for {horizon}")
    else:
        weighted_score = result.get("weighted_score")
        failed_constraints = result.get("failed_constraints") or []
        missing_metrics = result.get("missing_metrics") or []
        if result.get("gate_status") != "passed":
            detail = ", ".join(str(item) for item in failed_constraints) or str(result.get("gate_status"))
            errors.append(f"objective gate failed: {detail}")
        if missing_metrics:
            errors.append("objective metrics missing: " + ", ".join(str(item) for item in missing_metrics))

    return {
        "ok": not errors,
        "status": "ready_for_model_side_audit" if not errors else "not_ready",
        "horizon": str(horizon),
        "weighted_score": weighted_score,
        "prediction_table_summary": prediction_table_summary,
        "errors": errors,
        "warnings": warnings,
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate whether a V12 research candidate is ready for model-side audit.")
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--artifact-validation-json", required=True)
    parser.add_argument("--runbook-validation-json", required=True)
    parser.add_argument("--objective-score-json", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = validate_candidate_acceptance(
        horizon=args.horizon,
        artifact_validation=_load_json(args.artifact_validation_json),
        runbook_validation=_load_json(args.runbook_validation_json),
        objective_score=_load_json(args.objective_score_json),
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
