from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_equal(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    passed = actual == expected
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "reason": None if passed else f"{name} expected {expected}, got {actual}",
    }


def _check_floor(name: str, actual: Any, floor: Any) -> dict[str, Any]:
    actual_num = _as_float(actual)
    floor_num = _as_float(floor)
    passed = actual_num is not None and floor_num is not None and actual_num >= floor_num
    return {
        "name": name,
        "expected": {"min": floor_num},
        "actual": actual_num,
        "passed": passed,
        "reason": None if passed else f"{name} expected >= {floor_num}, got {actual_num}",
    }


def evaluate_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hard_results: list[dict[str, Any]] = []
    reminders: list[dict[str, Any]] = []

    standard_chain = config.get("standard_chain", {})
    hard = config.get("hard_constraints", {})
    governance = hard.get("governance", {})
    coverage = hard.get("coverage", {})
    reproducibility = hard.get("reproducibility", {})
    quality_common = hard.get("quality", {}).get("common", {})
    quality_by_label = hard.get("quality", {}).get("by_label", {})

    label = str(candidate.get("label") or "")
    label_quality = quality_by_label.get(label, {})

    hard_results.append(
        _check_equal(
            "approval_status",
            candidate.get("approval_status"),
            governance.get("required_candidate_approval_status"),
        )
    )
    hard_results.append(
        _check_equal(
            "feature_input",
            candidate.get("feature_input"),
            standard_chain.get("feature_input"),
        )
    )
    hard_results.append(
        _check_equal(
            "label_input",
            candidate.get("label_input"),
            standard_chain.get("label_input"),
        )
    )
    hard_results.append(
        _check_equal(
            "prediction_db",
            candidate.get("prediction_db"),
            standard_chain.get("prediction_db"),
        )
    )
    hard_results.append(
        _check_equal(
            "forbidden_inputs_present",
            _as_bool(candidate.get("forbidden_inputs_present")),
            False,
        )
    )
    hard_results.append(
        _check_floor(
            "min_trade_date",
            candidate.get("min_trade_date"),
            coverage.get("required_min_trade_date"),
        )
    )
    hard_results.append(
        _check_equal(
            "latest_trade_date",
            candidate.get("latest_trade_date"),
            candidate.get("expected_latest_trade_date"),
        )
    )
    if _as_bool(coverage.get("latest_day_rows_must_match_production_factor_rows")):
        hard_results.append(
            _check_equal(
                "latest_day_rows",
                candidate.get("latest_day_rows"),
                candidate.get("expected_latest_day_rows"),
            )
        )
    hard_results.append(
        _check_equal(
            "null_pred_prob",
            candidate.get("null_pred_prob"),
            coverage.get("null_pred_prob_must_equal"),
        )
    )
    hard_results.append(
        _check_equal(
            "duplicate_key_groups",
            candidate.get("duplicate_key_groups"),
            coverage.get("duplicate_key_groups_must_equal"),
        )
    )

    is_train_candidate = _as_bool(candidate.get("is_train_candidate"))
    if is_train_candidate and _as_bool(reproducibility.get("saved_model_files_required_for_train_candidates")):
        hard_results.append(_check_equal("saved_model_files_present", _as_bool(candidate.get("saved_model_files_present")), True))
    if is_train_candidate and _as_bool(reproducibility.get("model_params_required_for_train_candidates")):
        hard_results.append(_check_equal("model_params_present", _as_bool(candidate.get("model_params_present")), True))
    if is_train_candidate and _as_bool(reproducibility.get("selected_features_required_for_train_candidates")):
        hard_results.append(_check_equal("selected_features_present", _as_bool(candidate.get("selected_features_present")), True))
    if is_train_candidate and _as_bool(reproducibility.get("training_window_required_for_train_candidates")):
        hard_results.append(_check_equal("training_window_present", _as_bool(candidate.get("training_window_present")), True))
    if _as_bool(reproducibility.get("evaluation_report_required")):
        hard_results.append(_check_equal("evaluation_report_present", _as_bool(candidate.get("evaluation_report_present")), True))
    if _as_bool(reproducibility.get("candidate_manifest_required")):
        hard_results.append(_check_equal("candidate_manifest_present", _as_bool(candidate.get("candidate_manifest_present")), True))

    for name, floor in quality_common.items():
        if name.endswith("_floor"):
            metric_name = name[: -len("_floor")]
            hard_results.append(_check_floor(metric_name, candidate.get(metric_name), floor))
    for name, floor in label_quality.items():
        if name.endswith("_floor"):
            metric_name = name[: -len("_floor")]
            hard_results.append(_check_floor(metric_name, candidate.get(metric_name), floor))

    soft = config.get("soft_constraints", {})
    for metric in soft.get("recent_effectiveness", []):
        reminders.append({"type": "recent_effectiveness", "metric": metric, "value": candidate.get(metric)})
    for metric in soft.get("full_window_stability", []):
        reminders.append({"type": "full_window_stability", "metric": metric, "value": candidate.get(metric)})
    for item in soft.get("engineering", []):
        reminders.append({"type": "engineering", "metric": item, "value": candidate.get(item)})

    passed = all(item["passed"] for item in hard_results)
    status = "promotable_formal_candidate" if passed else "research_only"
    output_status = (
        config.get("promotion_states", {}).get("promotable_formal_candidate")
        if passed
        else config.get("promotion_states", {}).get("research_only")
    )

    return {
        "label": label,
        "asset": candidate.get("asset"),
        "table": candidate.get("table"),
        "hard_constraint_passed": passed,
        "promotion_decision": status,
        "target_approval_status": output_status,
        "hard_constraints": hard_results,
        "soft_reminders": reminders,
        "failed_hard_constraints": [item["name"] for item in hard_results if not item["passed"]],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score whether a research model candidate can be promoted into a formal candidate pool.")
    parser.add_argument("--candidate-json", required=True, help="JSON file describing a single candidate asset.")
    parser.add_argument("--constraint-config", required=True, help="JSON constraint config.")
    parser.add_argument("--output-json", help="Optional JSON output path.")
    parser.add_argument("--output-csv", help="Optional CSV output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    config = json.loads(Path(args.constraint_config).read_text(encoding="utf-8"))
    result = evaluate_candidate(candidate, config)
    payload = {
        "candidate_json": str(Path(args.candidate_json).resolve()),
        "constraint_config": str(Path(args.constraint_config).resolve()),
        "result": result,
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "label": result["label"],
                "asset": result["asset"],
                "table": result["table"],
                "hard_constraint_passed": result["hard_constraint_passed"],
                "promotion_decision": result["promotion_decision"],
                "target_approval_status": result["target_approval_status"],
                "failed_hard_constraints": ";".join(result["failed_hard_constraints"]),
            }
        ]
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
