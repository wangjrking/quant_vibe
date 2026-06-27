from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _priority_rank(priority: str | None) -> int:
    return PRIORITY_ORDER.get(str(priority or "").upper(), 99)


def _next_action_type(result: dict[str, Any]) -> str:
    if result.get("gate_status") == "passed":
        return "audit_and_strategy_research_validation"
    constraints = set(result.get("failed_constraints") or [])
    if any("rank_ic" in item for item in constraints):
        return "retrain_or_feature_search"
    if any("top10" in item for item in constraints):
        return "topn_constrained_retraining"
    return "research_refinement"


def build_research_queue(
    score_payload: dict[str, Any],
    objective_config: dict[str, Any],
) -> list[dict[str, Any]]:
    objectives = objective_config.get("horizon_objectives", {})
    rows: list[dict[str, Any]] = []
    for result in score_payload.get("results", []):
        horizon = str(result.get("horizon"))
        objective = objectives.get(horizon, {})
        gate_status = str(result.get("gate_status"))
        failed_constraints = result.get("failed_constraints") or []
        missing_metrics = result.get("missing_metrics") or []
        rows.append(
            {
                "horizon": horizon,
                "label": objective.get("label") or result.get("label"),
                "priority": result.get("priority") or objective.get("priority"),
                "status": "passed_objective_gate" if gate_status == "passed" else "failed_objective_gate",
                "next_action_type": _next_action_type(result),
                "optimization_style": result.get("optimization_style") or objective.get("optimization_style"),
                "weighted_score": float(result.get("weighted_score") or 0.0),
                "failed_constraints": ";".join(str(item) for item in failed_constraints),
                "missing_metrics": ";".join(str(item) for item in missing_metrics),
                "search_recommendation": " | ".join(str(item) for item in objective.get("search_recommendation", [])),
                "current_best_research_table": result.get("current_best_research_table")
                or objective.get("current_best_research_table"),
                "requires_training_authorization": gate_status != "passed",
                "requires_audit_before_formal": True,
            }
        )
    rows.sort(
        key=lambda item: (
            0 if item["status"] == "failed_objective_gate" else 1,
            _priority_rank(item.get("priority")),
            item["horizon"],
        )
    )
    for index, item in enumerate(rows, start=1):
        item["queue_rank"] = index
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a model research queue from objective-gate scores.")
    parser.add_argument("--score-json", required=True)
    parser.add_argument("--objective-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    score_payload = json.loads(Path(args.score_json).read_text(encoding="utf-8"))
    objective_config = json.loads(Path(args.objective_config).read_text(encoding="utf-8"))
    queue = build_research_queue(score_payload, objective_config)
    payload = {
        "score_json": str(Path(args.score_json).resolve()),
        "objective_config": str(Path(args.objective_config).resolve()),
        "queue": queue,
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(queue).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(json.dumps({"output_json": str(output_json.resolve()), "output_csv": str(output_csv.resolve()), "rows": len(queue)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
