from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = (
    REPORTS
    / "strategy_agent_v260_fixed10_leave_one_year_out_parameter_stability_20260822"
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
YEARS = ("2022", "2023", "2024", "2025")


def compound(annual_returns: dict, included_years: tuple[str, ...]) -> float:
    wealth = 1.0
    for year in included_years:
        wealth *= 1.0 + float(annual_returns[year])
    return float(wealth - 1.0)


def geometric_mean(annual_returns: dict, included_years: tuple[str, ...]) -> float:
    wealth = 1.0
    for year in included_years:
        value = 1.0 + float(annual_returns[year])
        if value <= 0.0:
            return -1.0
        wealth *= value
    return float(math.pow(wealth, 1.0 / len(included_years)) - 1.0)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dimensions = {}
    for dimension, relative in SOURCES.items():
        report = json.loads((REPORTS / relative).read_text(encoding="utf-8"))
        if report.get("validation_2026_opened") is not False:
            raise PermissionError(f"2026 was opened in {relative}")
        selected = report["selected_candidate"]
        annual = {
            candidate_id: evidence["metrics_0_30pct"]["annual_returns"]
            for candidate_id, evidence in report["results"].items()
        }
        folds = {}
        selected_wins = 0
        selected_top2 = 0
        for omitted in YEARS:
            included = tuple(year for year in YEARS if year != omitted)
            rows = {
                candidate_id: {
                    "compound_return": compound(values, included),
                    "geometric_mean_return": geometric_mean(values, included),
                }
                for candidate_id, values in annual.items()
            }
            ranking = sorted(
                rows,
                key=lambda candidate_id: (
                    rows[candidate_id]["geometric_mean_return"],
                    rows[candidate_id]["compound_return"],
                    candidate_id,
                ),
                reverse=True,
            )
            selected_rank = ranking.index(selected) + 1
            selected_wins += int(selected_rank == 1)
            selected_top2 += int(selected_rank <= 2)
            folds[omitted] = {
                "included_years": list(included),
                "ranking": ranking,
                "selected_candidate_rank": int(selected_rank),
                "candidate_metrics": rows,
            }
        dimensions[dimension] = {
            "selected_candidate": selected,
            "leave_one_year_out_folds": folds,
            "selected_win_fraction": float(selected_wins / len(YEARS)),
            "selected_top2_fraction": float(selected_top2 / len(YEARS)),
            "source": relative,
        }

    result = {
        "status": "leave_one_year_out_parameter_stability_complete_2026_not_opened",
        "selection_metric": (
            "geometric mean of reported calendar-year returns across the three "
            "retained years; this is a diagnostic, not a new selection rule"
        ),
        "dimensions": dimensions,
        "all_current_values_top2_in_every_fold": bool(all(
            item["selected_top2_fraction"] == 1.0 for item in dimensions.values()
        )),
        "interpretation": (
            "A low win fraction flags a parameter whose exact value depends on a "
            "particular calendar year even if the broader mechanism remains profitable."
        ),
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(
        OUTPUT_ROOT / "leave_one_year_out_parameter_stability.json", result
    )
    print(json.dumps({
        "status": result["status"],
        "dimensions": {
            key: {
                "selected": value["selected_candidate"],
                "win_fraction": value["selected_win_fraction"],
                "top2_fraction": value["selected_top2_fraction"],
            }
            for key, value in dimensions.items()
        },
        "all_current_values_top2_in_every_fold": result[
            "all_current_values_top2_in_every_fold"
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
