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
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_vol_priority_action_concentration_20260822 as concentration
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_market_state_branch_support_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
BLOCK_LENGTHS = (5, 20, 60)
SEED = 260_202_608_29


def uniform_schedule(shape: tuple[int, ...], trigger: int) -> np.ndarray:
    return np.full(shape, int(trigger), dtype=np.int16)


def broad_support(summary: dict, paired: dict) -> dict:
    result = fragility.support_summary(summary, paired)
    result["broad_support_for_current_branch"] = bool(
        summary["total_log_excess"] > 0.0
        and result["positive_year_fraction"] >= 0.75
        and result["positive_after_removing_top5_days"]
        and result["return_probability_above_80pct_all_blocks"]
    )
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered market-state branch diagnostic")

    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )

    schedules = {
        "current_strong4_weak5": np.where(strong, 4, 5).astype(np.int16),
        "uniform4": uniform_schedule(strong.shape, 4),
        "uniform5": uniform_schedule(strong.shape, 5),
        "uniform6": uniform_schedule(strong.shape, 6),
    }
    runs = {}
    for case_id, schedule in schedules.items():
        metrics, daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            pressure_trigger_override=schedule,
            score_sell_priority_override=sell_priority,
        )
        runs[case_id] = {
            "metrics": metrics,
            "daily": daily,
            "actions": actions,
        }

    current = runs["current_strong4_weak5"]
    expected = checkpoint["current_best_equalweight"]
    current_equivalent = bool(all(
        np.isclose(
            current["metrics"]["metrics_0_30pct"][key],
            expected[key],
            rtol=0.0,
            atol=1e-12,
        )
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    ))
    if not current_equivalent:
        raise RuntimeError("market-state branch baseline drifted")

    comparisons = {}
    for index, case_id in enumerate(("uniform4", "uniform5", "uniform6")):
        control = runs[case_id]
        current_returns, control_returns = fragility.aligned_returns(
            current["daily"], control["daily"]
        )
        paired = {
            str(block): bootstrap.paired_bootstrap(
                current_returns,
                control_returns,
                block,
                SEED + index * 1000 + block,
            )
            for block in BLOCK_LENGTHS
        }
        summary = concentration.concentration_summary(
            concentration.daily_log_excess(current["daily"], control["daily"])
        )
        comparisons[f"current_vs_{case_id}"] = {
            "control_metrics": control["metrics"],
            "action_difference": {
                key: value
                for key, value in concentration.action_difference(
                    current["actions"], control["actions"]
                ).items()
                if key != "changed_execution_dates"
            },
            "concentration": {
                key: summary[key]
                for key in (
                    "total_log_excess",
                    "equivalent_relative_wealth_gain",
                    "active_day_count",
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

    support_vs_simple_neighbors = bool(all(
        comparisons[f"current_vs_uniform{trigger}"]["support"][
            "broad_support_for_current_branch"
        ]
        for trigger in (4, 5)
    ))
    result = {
        "status": "market_state_branch_support_complete_2026_not_opened",
        "diagnostic_only_no_automatic_policy_change": True,
        "state_semantics": (
            "cross-sectional production model-score breadth, not a price-trend regime"
        ),
        "state_day_counts": {
            "strong": int(strong.sum()),
            "weak": int((~strong).sum()),
            "confirmed_weak2": int(weak2.sum()),
        },
        "runs": {case_id: value["metrics"] for case_id, value in runs.items()},
        "comparisons": comparisons,
        "diagnosis": {
            "current_branch_supported_vs_uniform4_and_uniform5": (
                support_vs_simple_neighbors
            ),
            "interpretation": (
                "Keep the regime branch only if its advantage over both adjacent "
                "uniform rules is broad across years and not concentrated in a few days."
            ),
        },
        "current_equivalence": current_equivalent,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "market_state_branch_support.json", result)
    print(json.dumps({
        "status": result["status"],
        "state_day_counts": result["state_day_counts"],
        "diagnosis": result["diagnosis"],
        "comparisons": {
            key: {
                "total_log_excess": value["concentration"]["total_log_excess"],
                "support": value["support"],
            }
            for key, value in comparisons.items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
