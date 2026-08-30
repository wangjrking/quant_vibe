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
ORIGINAL = (
    REPORTS
    / "strategy_agent_v260_fixed10_entry_rank_sizing_profit_optimization_20260822/"
    "development_result.json"
)
MIDPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_rank_sizing_midpoint_profit_check_20260823/"
    "development_result.json"
)
OUTPUT_ROOT = (
    REPORTS
    / "strategy_agent_v260_fixed10_rank_sizing_walkforward_amplitude_20260823"
)


def compound(annual_returns: dict[str, float], years: tuple[str, ...]) -> float:
    missing = [year for year in years if year not in annual_returns]
    if missing:
        raise ValueError(f"missing annual returns: {missing}")
    return float(math.prod(1.0 + float(annual_returns[year]) for year in years) - 1.0)


def walkforward_split(
    annual_by_case: dict[str, dict[str, float]],
    train_years: tuple[str, ...],
    test_years: tuple[str, ...],
) -> dict:
    train = {
        name: compound(returns, train_years)
        for name, returns in annual_by_case.items()
    }
    selected = max(sorted(train), key=lambda name: train[name])
    test = {
        name: compound(returns, test_years)
        for name, returns in annual_by_case.items()
    }
    test_winner = max(sorted(test), key=lambda name: test[name])
    return {
        "train_years": list(train_years),
        "test_years": list(test_years),
        "train_returns": train,
        "train_selected": selected,
        "test_returns": test,
        "test_winner": test_winner,
        "train_selection_is_test_winner": selected == test_winner,
        "selected_test_regret_vs_test_winner": float(
            test[test_winner] - test[selected]
        ),
    }


def main() -> None:
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    midpoint = json.loads(MIDPOINT.read_text(encoding="utf-8"))
    for report in (original, midpoint):
        if report.get("validation_2026_opened") is not False:
            raise PermissionError("walk-forward amplitude diagnostic entered 2026")
        if report.get("production_modified") is not False:
            raise PermissionError("walk-forward amplitude source modified production")

    annual_by_case = {
        "equalweight_0pct": original["candidates"]["equalweight_entry"]
        ["metrics_0_30pct"]["annual_returns"],
        "rank_tilt_5pct": midpoint["cases"]["midpoint_rank105_to_95"]
        ["metrics_0_30pct"]["annual_returns"],
        "rank_tilt_10pct": original["candidates"]["global_rank11_to_9_entry"]
        ["metrics_0_30pct"]["annual_returns"],
        "rank_tilt_20pct": original["candidates"]["rank12_to_8_control"]
        ["metrics_0_30pct"]["annual_returns"],
        "rank_tilt_30pct": original["candidates"]["rank13_to_7_control"]
        ["metrics_0_30pct"]["annual_returns"],
    }
    splits = {
        "train_2022_2023_test_2024_2025": walkforward_split(
            annual_by_case, ("2022", "2023"), ("2024", "2025")
        ),
        "train_2022_2024_test_2025": walkforward_split(
            annual_by_case, ("2022", "2023", "2024"), ("2025",)
        ),
    }
    result = {
        "status": "rank_sizing_walkforward_amplitude_complete_2026_not_opened",
        "role": "diagnostic_only_no_parameter_reselection",
        "annual_returns": annual_by_case,
        "splits": splits,
        "current_frozen_candidate": "rank_tilt_10pct",
        "all_early_sample_choices_win_following_period": all(
            item["train_selection_is_test_winner"] for item in splits.values()
        ),
        "interpretation": (
            "tilt amplitude is not temporally stable enough to claim an exact "
            "structural optimum; retain the modest frozen 10% tilt as a pre-2026 "
            "profit compromise and leave the final decision to one 2026 validation"
        ),
        "source_paths": [str(ORIGINAL), str(MIDPOINT)],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "walkforward_amplitude.json", result)
    print(json.dumps({
        "status": result["status"],
        "splits": {
            name: {
                "train_selected": item["train_selected"],
                "test_winner": item["test_winner"],
                "regret": item["selected_test_regret_vs_test_winner"],
            }
            for name, item in splits.items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
