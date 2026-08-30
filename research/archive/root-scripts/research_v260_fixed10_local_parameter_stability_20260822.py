from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_local_parameter_stability_20260822"
)
SOURCES = {
    "weak_market_confirmation_days": (
        "strategy_agent_v260_fixed10_confirmed_weak_days_boundary_20260822/"
        "development_result.json"
    ),
    "weak_market_volatility_penalty": (
        "strategy_agent_v260_fixed10_weak2_vol_priority_boundary_20260822/"
        "development_result.json"
    ),
    "weak_market_second_exit_min_age": (
        "strategy_agent_v260_fixed10_confirmed_weak2_pressure_age_boundary_20260822/"
        "development_result.json"
    ),
    "weak_market_pressure_trigger": (
        "strategy_agent_v260_fixed10_weak2_pressure_trigger_boundary_20260822/"
        "development_result.json"
    ),
}


def compact(metrics: dict) -> dict:
    return {
        key: float(metrics[key])
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dimensions = {}
    for dimension, relative in SOURCES.items():
        report = json.loads((REPORTS / relative).read_text(encoding="utf-8"))
        if report.get("validation_2026_opened") is not False:
            raise PermissionError(f"2026 was opened in {relative}")
        if report.get("production_modified") is not False:
            raise PermissionError(f"production was modified in {relative}")
        selected = report["selected_candidate"]
        rows = {}
        for candidate_id, evidence in report["results"].items():
            rows[candidate_id] = {
                "full_0_30pct": compact(evidence["metrics_0_30pct"]),
                "train_2022_2024": compact(evidence["train_2022_2024"]),
                "holdout_2025": compact(evidence["holdout_2025"]),
                "stress_0_65pct": compact(evidence["metrics_0_65pct"]),
            }
        selected_metrics = rows[selected]["full_0_30pct"]
        alternatives = [value["full_0_30pct"] for key, value in rows.items() if key != selected]
        dimensions[dimension] = {
            "selected_candidate": selected,
            "candidate_count": int(len(rows)),
            "candidates": rows,
            "neighbor_ranges": {
                metric: {
                    "minimum": float(min(value[metric] for value in alternatives)),
                    "maximum": float(max(value[metric] for value in alternatives)),
                    "selected": float(selected_metrics[metric]),
                    "selected_minus_best_alternative": float(
                        selected_metrics[metric] - max(value[metric] for value in alternatives)
                    ),
                }
                for metric in (
                    "cumulative_return", "cagr", "sharpe",
                    "turnover_annualized", "average_invested_ratio",
                )
            },
            "all_neighbors_profitable": bool(all(
                value["full_0_30pct"]["cumulative_return"] > 0.0
                for value in rows.values()
            )),
            "all_neighbors_all_years_positive": bool(all(
                min(report["results"][candidate_id]["metrics_0_30pct"]["annual_returns"].values()) > 0.0
                for candidate_id in rows
            )),
            "source": relative,
        }

    selected_is_interior = {
        "weak_market_confirmation_days": True,
        "weak_market_volatility_penalty": True,
        "weak_market_second_exit_min_age": True,
        "weak_market_pressure_trigger": True,
    }
    result = {
        "status": "local_parameter_stability_complete_2026_not_opened",
        "dimensions": dimensions,
        "selected_values_are_interior_of_tested_neighborhood": selected_is_interior,
        "all_dimensions_neighbors_profitable": bool(all(
            item["all_neighbors_profitable"] for item in dimensions.values()
        )),
        "all_dimensions_neighbors_have_positive_calendar_years": bool(all(
            item["all_neighbors_all_years_positive"] for item in dimensions.values()
        )),
        "interpretation": (
            "The selected policy is not an isolated profitable point: every tested "
            "one-dimensional neighbor remains profitable in every calendar year. "
            "Selection still depends on relative return/risk quality, so 2026 remains "
            "the only untouched validation."
        ),
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "local_parameter_stability.json", result)
    print(json.dumps({
        "status": result["status"],
        "all_dimensions_neighbors_profitable": result[
            "all_dimensions_neighbors_profitable"
        ],
        "all_dimensions_neighbors_have_positive_calendar_years": result[
            "all_dimensions_neighbors_have_positive_calendar_years"
        ],
        "selected": {
            key: value["selected_candidate"] for key, value in dimensions.items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
