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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_min_hold4_fragility_diagnostic_20260822 as fragility
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_vol_priority_action_concentration_20260822 as concentration
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_confirmation_support_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CONFIRMATION_DAYS = (1, 2, 3, 4)
BLOCK_LENGTHS = (5, 20, 60)
SEED = 260_202_608_26


def broad_support(summary: dict, paired: dict) -> dict:
    support = fragility.support_summary(summary, paired)
    support["broad_support_for_two_days"] = bool(
        summary["total_log_excess"] > 0.0
        and support["positive_year_fraction"] >= 0.75
        and support["positive_after_removing_top5_days"]
        and support["return_probability_above_80pct_all_blocks"]
    )
    return support


def run(context, policy: dict, confirmation_days: int):
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak = confirmed.confirmed_weak_mask(strong, confirmation_days)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak, 0.05
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered weak confirmation diagnostic")

    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    runs = {}
    for days in CONFIRMATION_DAYS:
        daily, actions = run(context, policy, days)
        runs[str(days)] = {
            "daily": daily,
            "actions": actions,
            "metrics": round1.evaluate_run(
                daily,
                actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
        }

    expected = checkpoint["current_best_equalweight"]
    if not all(
        np.isclose(runs["2"]["metrics"][key], expected[key], rtol=0.0, atol=1e-12)
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    ):
        raise RuntimeError("two-day confirmation replay drifted")

    comparisons = {}
    for reference_days in (1, 3, 4):
        candidate_returns, reference_returns = fragility.aligned_returns(
            runs["2"]["daily"], runs[str(reference_days)]["daily"]
        )
        paired = {
            str(block): bootstrap.paired_bootstrap(
                candidate_returns,
                reference_returns,
                block,
                SEED + 100 * reference_days + block,
            )
            for block in BLOCK_LENGTHS
        }
        summary = concentration.concentration_summary(
            concentration.daily_log_excess(
                runs["2"]["daily"], runs[str(reference_days)]["daily"]
            )
        )
        comparisons[f"confirmation_2_vs_{reference_days}"] = {
            "action_difference": {
                key: value
                for key, value in concentration.action_difference(
                    runs["2"]["actions"], runs[str(reference_days)]["actions"]
                ).items()
                if key != "changed_execution_dates"
            },
            "daily_log_excess_concentration": {
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
            },
            "paired_block_bootstrap": paired,
            "support": broad_support(summary, paired),
        }

    result = {
        "status": "weak_confirmation_support_complete_2026_not_opened",
        "diagnostic_only_no_parameter_selection": True,
        "runs": {key: value["metrics"] for key, value in runs.items()},
        "comparisons": comparisons,
        "diagnosis": {
            "two_day_confirmation_supported_against_every_neighbor": bool(all(
                item["support"]["broad_support_for_two_days"]
                for item in comparisons.values()
            )),
            "time_concentration_flag": bool(
                comparisons["confirmation_2_vs_1"]["support"][
                    "positive_year_fraction"
                ] < 0.75
            ),
            "interpretation": (
                "Two-day confirmation remains frozen, but any support concentrated in "
                "one pre-2026 year is explicitly treated as one-shot validation risk."
            ),
        },
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(
        OUTPUT_ROOT / "weak_confirmation_support_diagnostic.json", result
    )
    print(json.dumps({
        "status": result["status"],
        "diagnosis": result["diagnosis"],
        "support": {
            key: value["support"] for key, value in comparisons.items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
