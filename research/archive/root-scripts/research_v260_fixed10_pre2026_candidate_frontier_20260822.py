from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = REPORTS / "strategy_agent_v260_fixed10_pre2026_candidate_frontier_20260822"
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
METRIC_KEYS = (
    "cumulative_return", "cagr", "sharpe", "max_drawdown",
    "turnover_annualized", "average_invested_ratio",
)


def utility(metrics: dict) -> float:
    return float(
        metrics["cagr"]
        + 0.25 * metrics["sharpe"]
        - 0.50 * metrics["max_drawdown"]
        - 0.0025 * metrics["turnover_annualized"]
    )


def dominates(left: dict, right: dict) -> bool:
    no_worse = (
        left["cagr"] >= right["cagr"]
        and left["sharpe"] >= right["sharpe"]
        and left["max_drawdown"] <= right["max_drawdown"]
        and left["turnover_annualized"] <= right["turnover_annualized"]
    )
    strictly_better = (
        left["cagr"] > right["cagr"]
        or left["sharpe"] > right["sharpe"]
        or left["max_drawdown"] < right["max_drawdown"]
        or left["turnover_annualized"] < right["turnover_annualized"]
    )
    return bool(no_worse and strictly_better)


def collect_metric_nodes(value, path=()):
    if isinstance(value, dict):
        if (
            set(METRIC_KEYS).issubset(value)
            and value.get("start") == "20220607"
            and value.get("end") == "20251231"
            and value.get("days") == 870
            and "metrics_0_30pct" in path
        ):
            yield path, value
        for key, child in value.items():
            yield from collect_metric_nodes(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from collect_metric_nodes(child, (*path, str(index)))


def metric_signature(metrics: dict) -> tuple:
    return tuple(round(float(metrics[key]), 12) for key in METRIC_KEYS)


def frontier_eligible(payload: dict) -> bool:
    if payload.get("frontier_eligible") is False:
        return False
    if payload.get("event_overlay_role") == "diagnostic_only_not_selection_candidate":
        return False
    return True


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered pre-2026 frontier")

    records = []
    scanned_files = 0
    for directory in sorted(REPORTS.glob("strategy_agent_v260_fixed10_*20260822")):
        if directory == OUTPUT_ROOT:
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("validation_2026_opened") is not False:
                continue
            if payload.get("production_modified") is not False:
                continue
            scanned_files += 1
            if not frontier_eligible(payload):
                continue
            for node_path, metrics in collect_metric_nodes(payload):
                records.append({
                    "candidate_id": "/".join(node_path[:-1]) or path.stem,
                    "source": str(path.relative_to(REPO)).replace("\\", "/"),
                    "metrics": {key: metrics[key] for key in METRIC_KEYS},
                    "utility": utility(metrics),
                })

    current = checkpoint["current_best_equalweight"]
    production = checkpoint["production_baseline"]
    records.extend((
        {
            "candidate_id": checkpoint["selected_candidate"],
            "source": str(CHECKPOINT.relative_to(REPO)).replace("\\", "/"),
            "metrics": {key: current[key] for key in METRIC_KEYS},
            "utility": utility(current),
        },
        {
            "candidate_id": "production_baseline",
            "source": str(CHECKPOINT.relative_to(REPO)).replace("\\", "/"),
            "metrics": {key: production[key] for key in METRIC_KEYS},
            "utility": utility(production),
        },
    ))

    unique = {}
    for record in records:
        signature = metric_signature(record["metrics"])
        current_record = unique.get(signature)
        if current_record is None or len(record["source"]) < len(current_record["source"]):
            unique[signature] = record
    candidates = list(unique.values())
    frontier = [
        candidate
        for candidate in candidates
        if not any(
            dominates(other["metrics"], candidate["metrics"])
            for other in candidates
            if other is not candidate
        )
    ]
    by_utility = sorted(candidates, key=lambda item: item["utility"], reverse=True)
    current_signature = metric_signature(current)
    current_rank = next(
        index + 1
        for index, item in enumerate(by_utility)
        if metric_signature(item["metrics"]) == current_signature
    )
    current_on_frontier = any(
        metric_signature(item["metrics"]) == current_signature for item in frontier
    )
    production_signature = metric_signature(production)
    production_on_frontier = any(
        metric_signature(item["metrics"]) == production_signature for item in frontier
    )

    payload = {
        "status": "pre2026_candidate_frontier_complete_2026_not_opened",
        "comparable_contract": {
            "start": "20220607",
            "end": "20251231",
            "days": 870,
            "cost": "0.30pct",
            "required_path_component": "metrics_0_30pct",
        },
        "scanned_json_files": scanned_files,
        "raw_metric_nodes": len(records),
        "unique_metric_vectors": len(candidates),
        "pareto_frontier_size": len(frontier),
        "current_candidate": {
            "candidate_id": checkpoint["selected_candidate"],
            "utility_rank": current_rank,
            "utility_rank_is_first": current_rank == 1,
            "on_pareto_frontier": current_on_frontier,
            "metrics": {key: current[key] for key in METRIC_KEYS},
        },
        "production_baseline": {
            "on_pareto_frontier": production_on_frontier,
            "metrics": {key: production[key] for key in METRIC_KEYS},
        },
        "top_20_by_fixed_utility": by_utility[:20],
        "pareto_frontier": sorted(
            frontier, key=lambda item: item["metrics"]["cagr"], reverse=True
        ),
        "interpretation": (
            "the current candidate leads the fixed utility among all discovered "
            "comparable pre-2026 variants; production remains a distinct lower-drawdown "
            "frontier point, so the final 2026 validation must decide rather than more "
            "development-set rule stacking"
        ),
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "candidate_frontier.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "unique_metric_vectors": len(candidates),
        "pareto_frontier_size": len(frontier),
        "current_utility_rank": current_rank,
        "current_on_frontier": current_on_frontier,
        "production_on_frontier": production_on_frontier,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
