from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_ALIASES = {
    "daily_rank_ic_mean": "rank_ic",
    "daily_rank_ic": "rank_ic",
    "rank_ic": "rank_ic",
    "rank_ic_positive_ratio": "rank_ic_positive_ratio",
    "top1": "top1",
    "top3": "top3",
    "top5": "top5",
    "top10": "top10",
    "top20": "top20",
    "top50": "top50",
    "top_minus_bottom": "top_minus_bottom",
    "top_minus_bottom_mean": "top_minus_bottom",
}


def _metric_to_delta_columns(metric_name: str) -> list[str]:
    normalized = metric_name
    if normalized.endswith("_delta"):
        normalized = normalized[: -len("_delta")]
    normalized = METRIC_ALIASES.get(normalized, normalized)
    return [f"delta_vs_formal_{normalized}", f"{normalized}_delta"]


def _metric_value(
    metrics: pd.DataFrame,
    *,
    horizon: str,
    window: str,
    metric_name: str,
) -> float | None:
    data = metrics.copy()
    data["horizon"] = data["horizon"].astype(str).str.lower()
    data["window"] = data["window"].astype(str).str.lower()
    row = data[(data["horizon"] == horizon.lower()) & (data["window"] == window.lower())]
    if row.empty:
        return None
    value = None
    for column in _metric_to_delta_columns(metric_name):
        if column in row.columns:
            value = row.iloc[0][column]
            break
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _parse_weight_key(key: str) -> tuple[str, str]:
    pieces = key.split(".", 1)
    if len(pieces) != 2:
        raise ValueError(f"invalid score weight key: {key}")
    return pieces[0], pieces[1]


def _parse_floor_constraint(name: str) -> tuple[str, str] | None:
    suffix = "_delta_floor"
    if not name.endswith(suffix):
        return None
    body = name[: -len(suffix)]
    for window in ("recent252", "recent126", "recent63", "recent20", "full"):
        prefix = f"{window}_"
        if body.startswith(prefix):
            return window, body[len(prefix) :]
    return None


def score_horizon_candidate(
    metrics: pd.DataFrame,
    *,
    horizon: str,
    objective: dict[str, Any],
) -> dict[str, Any]:
    weighted_terms: list[dict[str, Any]] = []
    missing_metrics: list[str] = []
    weighted_score = 0.0

    for key, raw_weight in objective.get("score_weights", {}).items():
        window, metric_name = _parse_weight_key(str(key))
        value = _metric_value(metrics, horizon=horizon, window=window, metric_name=metric_name)
        if value is None:
            missing_metrics.append(str(key))
            continue
        weight = float(raw_weight)
        contribution = weight * value
        weighted_score += contribution
        weighted_terms.append(
            {
                "key": str(key),
                "window": window,
                "metric": metric_name,
                "value": value,
                "weight": weight,
                "contribution": contribution,
            }
        )

    failed_constraints: list[str] = []
    passed_constraints: list[str] = []
    ignored_constraints: list[str] = []
    for name, raw_floor in objective.get("additional_constraints", {}).items():
        parsed = _parse_floor_constraint(str(name))
        if parsed is None:
            ignored_constraints.append(str(name))
            continue
        window, metric_name = parsed
        value = _metric_value(metrics, horizon=horizon, window=window, metric_name=metric_name)
        if value is None or value < float(raw_floor):
            failed_constraints.append(str(name))
        else:
            passed_constraints.append(str(name))

    gate_status = "passed" if not failed_constraints and not missing_metrics else "failed"
    return {
        "horizon": horizon,
        "weighted_score": weighted_score,
        "gate_status": gate_status,
        "failed_constraints": failed_constraints,
        "passed_constraints": passed_constraints,
        "ignored_constraints": ignored_constraints,
        "missing_metrics": missing_metrics,
        "weighted_terms": weighted_terms,
    }


def score_all_horizons(metrics: pd.DataFrame, objective_config: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for horizon, objective in objective_config.get("horizon_objectives", {}).items():
        result = score_horizon_candidate(metrics, horizon=str(horizon), objective=objective)
        result["label"] = objective.get("label")
        result["priority"] = objective.get("priority")
        result["current_best_research_table"] = objective.get("current_best_research_table")
        result["optimization_style"] = objective.get("optimization_style")
        results.append(result)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model research candidates against configured objectives.")
    parser.add_argument("--metrics-csv", required=True, help="CSV with candidate metric deltas by horizon/window.")
    parser.add_argument("--objective-config", required=True, help="JSON objective config.")
    parser.add_argument("--horizon", help="Optional horizon to score, such as 1d, 3d, 5d, or 10d.")
    parser.add_argument("--output-json", help="Optional JSON output path.")
    parser.add_argument("--output-csv", help="Optional CSV output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = pd.read_csv(args.metrics_csv)
    objective_config = json.loads(Path(args.objective_config).read_text(encoding="utf-8"))
    if args.horizon:
        objective = objective_config["horizon_objectives"][args.horizon]
        results = [score_horizon_candidate(metrics, horizon=args.horizon, objective=objective)]
    else:
        results = score_all_horizons(metrics, objective_config)

    payload = {
        "metrics_csv": str(Path(args.metrics_csv).resolve()),
        "objective_config": str(Path(args.objective_config).resolve()),
        "results": results,
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
        rows = [
            {
                "horizon": item["horizon"],
                "gate_status": item["gate_status"],
                "weighted_score": item["weighted_score"],
                "failed_constraints": ";".join(item["failed_constraints"]),
                "missing_metrics": ";".join(item["missing_metrics"]),
                "ignored_constraints": ";".join(item["ignored_constraints"]),
            }
            for item in results
        ]
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
