from __future__ import annotations

import json
import math
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = (
    REPORTS
    / "strategy_agent_v260_fixed10_profit_first_holdout_consistency_20260822"
)

EXPERIMENTS = {
    "portfolio_width": (
        "strategy_agent_v260_fixed10_soft_width_profit_optimization_20260822/"
        "soft_width_results.json",
        "selected_width",
    ),
    "target_gross": (
        "strategy_agent_v260_fixed10_soft_gross_profit_optimization_20260822/"
        "soft_gross_results.json",
        "selected_gross",
    ),
    "entry_1d_blend": (
        "strategy_agent_v260_fixed10_current_entry_score_profit_optimization_20260822/"
        "development_result.json",
        "selected_entry_weight_1d",
    ),
    "entry_3d_blend": (
        "strategy_agent_v260_fixed10_current_entry_3d_profit_optimization_20260822/"
        "development_result.json",
        "selected_entry_weight_3d",
    ),
    "entry_5d_blend": (
        "strategy_agent_v260_fixed10_current_entry_5d_profit_optimization_20260822/"
        "development_result.json",
        "selected_entry_weight_5d",
    ),
    "entry_smoothing": (
        "strategy_agent_v260_fixed10_current_smoothing_profit_optimization_20260822/"
        "development_result.json",
        "selected_smoothing_window",
    ),
    "entry_latest_session_weight": (
        "strategy_agent_v260_fixed10_current_raw_alpha_profit_optimization_20260822/"
        "development_result.json",
        "selected_raw_alpha",
    ),
    "exit_smoothing": (
        "strategy_agent_v260_fixed10_current_exit_smoothing_profit_optimization_20260822/"
        "development_result.json",
        "selected_exit_smoothing_window",
    ),
    "exit_latest_session_weight": (
        "strategy_agent_v260_fixed10_current_exit_raw_alpha_profit_optimization_20260822/"
        "development_result.json",
        "selected_exit_raw_alpha",
    ),
    "exit_horizon": (
        "strategy_agent_v260_fixed10_current_exit_horizon_profit_optimization_20260822/"
        "development_result.json",
        "selected_exit_horizon",
    ),
    "partial_trim": (
        "strategy_agent_v260_fixed10_current_partial_trim_profit_optimization_20260822/"
        "development_result.json",
        "selected_candidate",
    ),
    "score_transfer": (
        "strategy_agent_v260_fixed10_current_score_transfer_profit_optimization_20260822/"
        "development_result.json",
        "selected_candidate",
    ),
}


def read_report(relative: str) -> dict:
    report = json.loads((REPORTS / relative).read_text(encoding="utf-8"))
    if report.get("validation_2026_opened") is not False:
        raise PermissionError(f"2026 entered holdout consistency: {relative}")
    if report.get("production_modified") is not False:
        raise PermissionError(f"production changed in holdout consistency: {relative}")
    return report


def compound_annual_returns(annual_returns: dict, years: tuple[str, ...]) -> float:
    missing = [year for year in years if year not in annual_returns]
    if missing:
        raise ValueError(f"missing annual returns: {missing}")
    return float(math.prod(1.0 + float(annual_returns[year]) for year in years) - 1.0)


def period_winner(candidates: dict, years: tuple[str, ...]) -> tuple[str, dict]:
    values = {
        name: compound_annual_returns(
            item["metrics_0_30pct"]["annual_returns"], years
        )
        for name, item in candidates.items()
    }
    winner = max(values, key=lambda name: (values[name], name))
    return winner, values


def normalize_selected(value, candidate_keys) -> str:
    direct = str(value)
    if direct in candidate_keys:
        return direct
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return direct
    for key in candidate_keys:
        try:
            if math.isclose(float(key), numeric, rel_tol=0.0, abs_tol=1e-12):
                return key
        except ValueError:
            continue
    return direct


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for name, (relative, selected_field) in EXPERIMENTS.items():
        report = read_report(relative)
        train_winner, train_returns = period_winner(
            report["candidates"], ("2022", "2023", "2024")
        )
        holdout_winner, holdout_returns = period_winner(
            report["candidates"], ("2025",)
        )
        selected = normalize_selected(
            report[selected_field], report["candidates"].keys()
        )
        experiments[name] = {
            "reported_full_period_selection": selected,
            "development_2022_2024_winner": train_winner,
            "holdout_2025_winner": holdout_winner,
            "development_and_holdout_agree": train_winner == holdout_winner,
            "reported_selection_matches_development": selected == train_winner,
            "reported_selection_matches_holdout": selected == holdout_winner,
            "development_returns": train_returns,
            "holdout_returns": holdout_returns,
            "source": relative,
        }

    stable = [
        name
        for name, item in experiments.items()
        if item["development_and_holdout_agree"]
        and item["reported_selection_matches_development"]
    ]
    divergent = [name for name in experiments if name not in stable]
    result = {
        "status": "profit_first_holdout_consistency_complete_2026_not_opened",
        "role": (
            "diagnostic only; profit remains primary and this report does not add "
            "a new candidate rejection gate"
        ),
        "selection_semantics": {
            "portfolio_width_equalweight_and_invested_ratio_are_soft": True,
            "development_period": ["2022", "2023", "2024"],
            "holdout_period": ["2025"],
            "metric": "baseline_0.30pct_compounded_return",
        },
        "experiments": experiments,
        "stable_experiments": stable,
        "divergent_experiments": divergent,
        "all_experiments_stable": not divergent,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    target = OUTPUT_ROOT / "profit_first_holdout_consistency.json"
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "stable": stable,
                "divergent": divergent,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
