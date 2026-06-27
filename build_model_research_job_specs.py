from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STOP_CONDITIONS = [
    "production_feature_or_label_input_missing",
    "legacy_feature_asset_detected",
    "future_or_label_column_detected_in_features",
    "fold_model_file_not_saved",
    "objective_gate_metrics_missing",
    "prediction_coverage_or_duplicate_key_failure",
]


def _artifact_requirements() -> dict[str, bool]:
    return {
        "fold_model_files": True,
        "fold_selected_features": True,
        "fold_train_test_window_meta": True,
        "model_params_snapshot": True,
        "evaluation_summary": True,
        "coverage_report": True,
    }


def build_job_specs(
    queue_payload: dict[str, Any],
    objective_config: dict[str, Any],
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    standard_chain = objective_config.get("standard_chain", {})
    objectives = objective_config.get("horizon_objectives", {})
    specs: list[dict[str, Any]] = []
    for item in queue_payload.get("queue", []):
        horizon = str(item.get("horizon"))
        objective = objectives.get(horizon, {})
        needs_training = item.get("status") != "passed_objective_gate"
        specs.append(
            {
                "job_id": f"{run_id}_{horizon}_{item.get('next_action_type')}",
                "queue_rank": item.get("queue_rank"),
                "horizon": horizon,
                "label": item.get("label") or objective.get("label"),
                "priority": item.get("priority"),
                "job_type": item.get("next_action_type"),
                "execution_status": "not_started_requires_authorization"
                if needs_training
                else "validation_only_not_training",
                "requires_training_authorization": bool(needs_training),
                "requires_audit_before_formal": True,
                "approval_status": "research_plan_not_approved_for_execution",
                "feature_input": standard_chain.get("feature_input"),
                "label_input": standard_chain.get("label_input"),
                "prediction_db": standard_chain.get("prediction_db"),
                "optimization_style": objective.get("optimization_style"),
                "failed_constraints": item.get("failed_constraints", ""),
                "objective_constraints": objective.get("additional_constraints", {}),
                "model_artifacts_required": _artifact_requirements(),
                "candidate_output_policy": objective_config.get("candidate_output_policy", {}),
                "stop_conditions": list(STOP_CONDITIONS),
                "boundaries": {
                    "no_training_executed_by_spec_generation": True,
                    "no_prediction_generated_by_spec_generation": True,
                    "no_production_manifest_change": True,
                    "no_signal": True,
                    "no_backtest": True,
                },
            }
        )
    return specs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-executing model research job specs from a queue.")
    parser.add_argument("--queue-json", required=True)
    parser.add_argument("--objective-config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_payload = json.loads(Path(args.queue_json).read_text(encoding="utf-8"))
    objective_config = json.loads(Path(args.objective_config).read_text(encoding="utf-8"))
    specs = build_job_specs(queue_payload, objective_config, run_id=args.run_id)
    payload = {
        "run_id": args.run_id,
        "queue_json": str(Path(args.queue_json).resolve()),
        "objective_config": str(Path(args.objective_config).resolve()),
        "job_specs": specs,
        "governance": {
            "execution_mode": "plan_only",
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output.resolve()), "jobs": len(specs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
