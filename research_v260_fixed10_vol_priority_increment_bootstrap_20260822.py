from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_vol_priority_increment_bootstrap_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
BLOCK_LENGTHS = (5, 20, 60)
SEED = 260_202_608_23


def run(context, policy: dict, penalty: float):
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, penalty
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        score_sell_priority_override=priority_matrix,
    )


def aligned(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    left = bootstrap.frame_returns(candidate).rename(columns={"return": "candidate"})
    right = bootstrap.frame_returns(reference).rename(columns={"return": "reference"})
    result = left.merge(right, on="date", validate="one_to_one")
    if len(result) != len(left) or len(result) != len(right):
        raise RuntimeError("volatility-priority calendars do not align")
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered volatility-priority bootstrap")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    candidate_daily, candidate_actions = run(context, policy, 0.05)
    reference_daily, reference_actions = run(context, policy, 0.00)
    candidate_metrics = round1.evaluate_run(
        candidate_daily, candidate_actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    reference_metrics = round1.evaluate_run(
        reference_daily, reference_actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    equivalent = all(
        np.isclose(candidate_metrics[key], expected[key], rtol=0.0, atol=1e-12)
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    )
    if not equivalent:
        raise RuntimeError("volatility-priority candidate drifted")

    frame = aligned(candidate_daily, reference_daily)
    candidate = frame["candidate"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    comparisons = {
        str(block): bootstrap.paired_bootstrap(
            candidate, reference, block, SEED + block
        )
        for block in BLOCK_LENGTHS
    }
    support = {
        "return_probability_at_least_80pct_all_blocks": all(
            item["probability_annualized_log_return_higher"] >= 0.80
            for item in comparisons.values()
        ),
        "sharpe_probability_at_least_80pct_all_blocks": all(
            item["probability_sharpe_higher"] >= 0.80
            for item in comparisons.values()
        ),
        "return_ci_lower_bound_nonnegative_all_blocks": all(
            item["annualized_log_return_delta"]["p025"] >= 0.0
            for item in comparisons.values()
        ),
    }
    result = {
        "status": "vol_priority_increment_bootstrap_complete_2026_not_opened",
        "candidate": "confirmed_weak2_volatility_penalty_0.05",
        "reference": "confirmed_weak2_no_volatility_priority",
        "candidate_metrics": candidate_metrics,
        "reference_metrics": reference_metrics,
        "candidate_minus_reference": {
            key: float(candidate_metrics[key] - reference_metrics[key])
            for key in (
                "cumulative_return", "cagr", "sharpe", "max_drawdown",
                "turnover_annualized", "average_invested_ratio",
            )
        },
        "paired_block_bootstrap": comparisons,
        "strong_incremental_support": support,
        "decision": (
            "retain_as_frozen_candidate_component_pending_2026_validation"
            if all(support.values())
            else "increment_not_statistically_decisive_keep_flagged_as_complexity_risk"
        ),
        "candidate_equivalence": equivalent,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "vol_priority_increment_bootstrap.json", result)
    print(json.dumps({
        "status": result["status"],
        "candidate_minus_reference": result["candidate_minus_reference"],
        "strong_incremental_support": support,
        "decision": result["decision"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
