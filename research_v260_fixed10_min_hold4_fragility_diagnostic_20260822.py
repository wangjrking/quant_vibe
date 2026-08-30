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
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_vol_priority_action_concentration_20260822 as concentration
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_min_hold4_fragility_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
MIN_HOLD_DAYS = (3, 4, 5)
BLOCK_LENGTHS = (5, 20, 60)
SEED = 260_202_608_24


def build_priority_matrix(context) -> np.ndarray:
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    return priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )


def aligned_returns(candidate, reference) -> tuple[np.ndarray, np.ndarray]:
    left = bootstrap.frame_returns(candidate).rename(columns={"return": "candidate"})
    right = bootstrap.frame_returns(reference).rename(columns={"return": "reference"})
    joined = left.merge(right, on="date", how="outer", indicator=True, validate="one_to_one")
    if not (joined["_merge"] == "both").all():
        raise RuntimeError("minimum-hold calendars do not align")
    return (
        joined["candidate"].to_numpy(dtype=np.float64),
        joined["reference"].to_numpy(dtype=np.float64),
    )


def support_summary(concentration_result: dict, bootstraps: dict) -> dict:
    yearly = concentration_result["yearly_log_excess"]
    positive_year_fraction = float(np.mean([value > 0.0 for value in yearly.values()]))
    return {
        "positive_year_fraction": positive_year_fraction,
        "positive_after_removing_top5_days": bool(
            concentration_result["log_excess_after_removing_top5_positive_days"] > 0.0
        ),
        "return_probability_above_80pct_all_blocks": bool(all(
            value["probability_annualized_log_return_higher"] >= 0.80
            for value in bootstraps.values()
        )),
        "sharpe_probability_above_80pct_all_blocks": bool(all(
            value["probability_sharpe_higher"] >= 0.80
            for value in bootstraps.values()
        )),
        "return_ci_lower_bound_nonnegative_all_blocks": bool(all(
            value["annualized_log_return_delta"]["p025"] >= 0.0
            for value in bootstraps.values()
        )),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered minimum-hold fragility diagnostic")

    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    priority_matrix = build_priority_matrix(context)

    runs = {}
    for minimum_hold_days in MIN_HOLD_DAYS:
        daily, actions = age_guard.run_policy_at_cost(
            context,
            policy,
            extra_age,
            round1.BASELINE_COST,
            min_hold_days_override=minimum_hold_days,
            score_sell_priority_override=priority_matrix,
        )
        runs[str(minimum_hold_days)] = {
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
    current_equivalent = bool(all(
        np.isclose(runs["4"]["metrics"][key], expected[key], rtol=0.0, atol=1e-12)
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    ))
    if not current_equivalent:
        raise RuntimeError("minimum-hold 4 replay drifted from frozen checkpoint")

    comparisons = {}
    for reference_days in (3, 5):
        current_returns, reference_returns = aligned_returns(
            runs["4"]["daily"], runs[str(reference_days)]["daily"]
        )
        bootstraps = {
            str(block): bootstrap.paired_bootstrap(
                current_returns,
                reference_returns,
                block,
                SEED + 100 * reference_days + block,
            )
            for block in BLOCK_LENGTHS
        }
        daily_excess = concentration.daily_log_excess(
            runs["4"]["daily"], runs[str(reference_days)]["daily"]
        )
        excess_summary = concentration.concentration_summary(daily_excess)
        comparisons[f"min_hold_4_vs_{reference_days}"] = {
            "action_difference": concentration.action_difference(
                runs["4"]["actions"], runs[str(reference_days)]["actions"]
            ),
            "daily_log_excess_concentration": excess_summary,
            "paired_block_bootstrap": bootstraps,
            "support": support_summary(excess_summary, bootstraps),
        }

    broad_nonconcentrated_support = bool(all(
        comparison["support"]["positive_year_fraction"] >= 0.75
        and comparison["support"]["positive_after_removing_top5_days"]
        and comparison["support"]["return_probability_above_80pct_all_blocks"]
        for comparison in comparisons.values()
    ))
    result = {
        "status": "min_hold4_fragility_diagnostic_complete_2026_not_opened",
        "diagnostic_only_no_parameter_selection": True,
        "runs": {key: value["metrics"] for key, value in runs.items()},
        "comparisons": comparisons,
        "diagnosis": {
            "minimum_hold_4_is_sharp_local_optimum": bool(
                runs["4"]["metrics"]["cagr"] > runs["3"]["metrics"]["cagr"]
                and runs["4"]["metrics"]["cagr"] > runs["5"]["metrics"]["cagr"]
            ),
            "broad_nonconcentrated_support": broad_nonconcentrated_support,
            "interpretation": (
                "A sharp discrete optimum is retained only as a validation risk flag. "
                "This diagnostic does not replace or retune the frozen candidate."
            ),
        },
        "current_equivalence": current_equivalent,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "min_hold4_fragility_diagnostic.json", result)
    print(json.dumps({
        "status": result["status"],
        "diagnosis": result["diagnosis"],
        "comparisons": {
            key: {
                "total_log_excess": value["daily_log_excess_concentration"][
                    "total_log_excess"
                ],
                "support": value["support"],
            }
            for key, value in comparisons.items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
