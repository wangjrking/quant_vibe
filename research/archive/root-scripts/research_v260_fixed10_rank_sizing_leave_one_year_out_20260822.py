from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_entry_rank_sizing_profit_optimization_20260822/"
    "development_result.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_leave_one_year_out_20260822"
)
YEARS = ("2022", "2023", "2024", "2025")


def compounded_excluding(annual_returns: dict, excluded_year: str) -> float:
    value = 1.0
    for year in YEARS:
        if year != excluded_year:
            value *= 1.0 + float(annual_returns[year])
    return value - 1.0


def main() -> None:
    source = json.loads(REPORT.read_text(encoding="utf-8"))
    if source.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered rank sizing leave-one-year-out")
    candidates = source["candidates"]
    baseline = candidates["equalweight_entry"]["metrics_0_30pct"]["annual_returns"]
    results = {}
    for excluded in YEARS:
        baseline_return = compounded_excluding(baseline, excluded)
        values = {}
        for name, item in candidates.items():
            candidate_return = compounded_excluding(
                item["metrics_0_30pct"]["annual_returns"], excluded
            )
            values[name] = {
                "cumulative_return": candidate_return,
                "minus_equalweight": candidate_return - baseline_return,
            }
        winner = max(
            values,
            key=lambda name: values[name]["cumulative_return"],
        )
        results[excluded] = {
            "winner": winner,
            "candidates": values,
        }

    selected = source["selected_candidate"]
    selected_positive = {
        year: results[year]["candidates"][selected]["minus_equalweight"] > 0.0
        for year in YEARS
    }
    output = {
        "status": "rank_sizing_leave_one_year_out_complete_2026_not_opened",
        "selected_candidate": selected,
        "selection_rule": "profit comparison only; no new parameter or rejection gate",
        "leave_one_year_out": results,
        "selected_beats_equalweight_after_each_year_removed": selected_positive,
        "all_removed_year_cases_positive": all(selected_positive.values()),
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "leave_one_year_out.json", output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
