from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _term_values(result: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in result.get("weighted_terms", []):
        key = f"{item.get('window')}.{item.get('metric')}"
        try:
            values[key] = float(item.get("value"))
        except (TypeError, ValueError):
            continue
    return values


def _score_by_horizon(objective_score: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("horizon")): item for item in objective_score.get("results", [])}


def validate_acceptance_matrix(*, matrix: dict[str, Any], objective_score: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    results_by_horizon = _score_by_horizon(objective_score)
    experiments: list[dict[str, Any]] = []

    for item in matrix.get("experiments", []):
        horizon = str(item.get("horizon"))
        variant = str(item.get("variant"))
        score = results_by_horizon.get(horizon)
        if score is None:
            errors.append(f"{horizon}/{variant} objective score missing")
            experiments.append({**item, "primary_actual": None, "primary_gap": None, "primary_passed": False})
            continue

        values = _term_values(score)
        metric_key = str(item.get("primary_metric_key"))
        actual = values.get(metric_key)
        floor = item.get("primary_floor")
        if actual is None:
            errors.append(f"{horizon}/{variant} primary metric {metric_key} missing")
            experiments.append({**item, "primary_actual": None, "primary_gap": None, "primary_passed": False})
            continue
        try:
            floor_value = float(floor)
        except (TypeError, ValueError):
            errors.append(f"{horizon}/{variant} primary floor missing")
            experiments.append({**item, "primary_actual": actual, "primary_gap": None, "primary_passed": False})
            continue

        gap = float(actual) - floor_value
        passed = gap >= 0
        if not passed:
            errors.append(f"{horizon}/{variant} primary metric {metric_key} below floor")
        experiments.append(
            {
                **item,
                "primary_actual": float(actual),
                "primary_floor": floor_value,
                "primary_gap": gap,
                "primary_passed": passed,
            }
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "experiments_checked": len(experiments),
        "experiments": experiments,
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
    parser = argparse.ArgumentParser(description="Validate V12 experiment acceptance matrix against objective score.")
    parser.add_argument("--matrix-json", required=True)
    parser.add_argument("--objective-score-json", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = validate_acceptance_matrix(
        matrix=_load_json(args.matrix_json),
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
