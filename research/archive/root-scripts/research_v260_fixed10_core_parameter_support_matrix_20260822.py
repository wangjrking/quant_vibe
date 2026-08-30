from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_core_parameter_neighborhood_20260822 as neighborhood
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_min_hold4_fragility_diagnostic_20260822 as fragility
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_vol_priority_action_concentration_20260822 as concentration


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_core_parameter_support_matrix_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
DIMENSIONS = {
    "sell_score_below": {"current": 0.85, "neighbors": (0.80, 0.90)},
    "max_hold_renewal_score": {"current": 0.80, "neighbors": (0.75, 0.85)},
    "replacement_advantage": {"current": 0.05, "neighbors": (0.03, 0.07)},
    "portfolio_rebalance_interval_days": {"current": 20, "neighbors": (10, 40)},
}
BLOCK_LENGTHS = (5, 20, 60)
SEED = 260_202_608_25


def value_id(value: float | int) -> str:
    return f"{float(value):.4f}" if isinstance(value, float) else str(value)


def concise_concentration(summary: dict) -> dict:
    return {
        key: summary[key]
        for key in (
            "total_log_excess",
            "equivalent_relative_wealth_gain",
            "active_day_count",
            "positive_day_count",
            "negative_day_count",
            "top1_positive_share",
            "top5_positive_share",
            "top10_absolute_share",
            "log_excess_after_removing_top5_positive_days",
            "positive_month_fraction",
            "yearly_log_excess",
        )
    }


def classify_support(summary: dict, paired: dict) -> dict:
    support = fragility.support_summary(summary, paired)
    broad = bool(
        summary["total_log_excess"] > 0.0
        and support["positive_year_fraction"] >= 0.75
        and support["positive_after_removing_top5_days"]
        and support["return_probability_above_80pct_all_blocks"]
    )
    return {
        **support,
        "broad_support_for_current_value": broad,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered core parameter support matrix")

    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    current_daily, current_actions = neighborhood.run_case(
        context, policy, "sell_score_below", 0.85
    )
    current_metrics = neighborhood.evaluate(current_daily, current_actions)
    expected = checkpoint["current_best_equalweight"]
    if not all(
        np.isclose(current_metrics[key], expected[key], rtol=0.0, atol=1e-12)
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    ):
        raise RuntimeError("core parameter support baseline drifted")

    current_returns, _ = fragility.aligned_returns(current_daily, current_daily)
    results = {}
    for dimension, specification in DIMENSIONS.items():
        comparisons = {}
        for neighbor in specification["neighbors"]:
            neighbor_daily, neighbor_actions = neighborhood.run_case(
                context, policy, dimension, neighbor
            )
            candidate_returns, neighbor_returns = fragility.aligned_returns(
                current_daily, neighbor_daily
            )
            if not np.array_equal(current_returns, candidate_returns):
                raise RuntimeError("current return alignment drifted")
            paired = {
                str(block): bootstrap.paired_bootstrap(
                    candidate_returns,
                    neighbor_returns,
                    block,
                    SEED + len(results) * 1000 + int(round(float(neighbor) * 100)) + block,
                )
                for block in BLOCK_LENGTHS
            }
            excess = concentration.concentration_summary(
                concentration.daily_log_excess(current_daily, neighbor_daily)
            )
            comparisons[value_id(neighbor)] = {
                "neighbor_metrics": neighborhood.evaluate(
                    neighbor_daily, neighbor_actions
                ),
                "action_difference": {
                    key: value
                    for key, value in concentration.action_difference(
                        current_actions, neighbor_actions
                    ).items()
                    if key != "changed_execution_dates"
                },
                "daily_log_excess_concentration": concise_concentration(excess),
                "paired_block_bootstrap": paired,
                "support": classify_support(excess, paired),
            }
        results[dimension] = {
            "current_value": value_id(specification["current"]),
            "comparisons": comparisons,
            "current_broadly_supported_against_all_neighbors": bool(all(
                value["support"]["broad_support_for_current_value"]
                for value in comparisons.values()
            )),
        }

    result = {
        "status": "core_parameter_support_matrix_complete_2026_not_opened",
        "diagnostic_only_no_parameter_selection": True,
        "current_metrics": current_metrics,
        "dimensions": results,
        "summary": {
            dimension: {
                "broadly_supported_against_all_neighbors": value[
                    "current_broadly_supported_against_all_neighbors"
                ],
                "neighbor_support": {
                    neighbor: comparison["support"]
                    for neighbor, comparison in value["comparisons"].items()
                },
            }
            for dimension, value in results.items()
        },
        "interpretation": (
            "Parameters without broad support are treated as validation-sensitive and "
            "are not tuned further on pre-2026 data."
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "core_parameter_support_matrix.json", result)
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
