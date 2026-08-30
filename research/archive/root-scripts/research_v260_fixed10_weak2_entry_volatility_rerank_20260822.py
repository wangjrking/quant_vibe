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
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak2_entry_volatility_rerank_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PENALTIES = (0.0, 0.025, 0.05, 0.075)
POOL_SIZE = 30
MAX_MDD_TOLERANCE = 0.001


def entry_order(
    production_order: np.ndarray,
    score: np.ndarray,
    volatility_rank: np.ndarray,
    confirmed_weak: np.ndarray,
    penalty: float,
) -> np.ndarray:
    alternative = np.asarray(score, dtype=np.float64) - float(penalty) * np.nan_to_num(
        np.asarray(volatility_rank, dtype=np.float64), nan=1.0
    )
    return defensive.bounded_score_order(
        production_order,
        alternative,
        confirmed_weak,
        POOL_SIZE,
    )


def run_at_cost(context, policy, order, priority_matrix, extra_age, cost):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
    )
    strong = regime.strong_market_mask(context.score, context.protocol)
    return round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        order,
        policy,
        round1.DEVELOPMENT_END,
        slip=float(cost),
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=cadence.rebalance_schedule(
            len(context.arrays["dates"]),
            int(policy["portfolio_rebalance_interval_days"]),
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=context.empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
        score_sell_pressure_extra_min_age_override=extra_age,
        score_sell_priority_override=priority_matrix,
        exit_score_override=context.score,
    )


def evaluate(context, policy, order, priority_matrix, extra_age):
    daily, actions = run_at_cost(
        context, policy, order, priority_matrix, extra_age, round1.BASELINE_COST
    )
    stress_daily, stress_actions = run_at_cost(
        context, policy, order, priority_matrix, extra_age, round1.STRESS_COST
    )
    return {
        "metrics_0_30pct": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        ),
        "train_2022_2024": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        ),
        "holdout_2025": round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        ),
        "metrics_0_65pct": round1.evaluate_run(
            stress_daily,
            stress_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ),
    }, daily, actions


def selection_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    current = candidate["metrics_0_30pct"]
    reference = baseline["metrics_0_30pct"]
    return {
        "cumulative_return_improved": current["cumulative_return"]
        > reference["cumulative_return"],
        "sharpe_improved": current["sharpe"] > reference["sharpe"],
        "drawdown_within_10bp_tolerance": current["max_drawdown"]
        <= reference["max_drawdown"] + MAX_MDD_TOLERANCE,
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "train_return_not_worse": candidate["train_2022_2024"]["cumulative_return"]
        >= baseline["train_2022_2024"]["cumulative_return"],
        "holdout_return_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "all_years_positive": min(current["annual_returns"].values()) > 0.0,
        "full_10_positions": current["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for entry rerank")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )

    results, cache = {}, {}
    for penalty in PENALTIES:
        candidate_id = f"weak2_entry_vol_penalty_{penalty:.3f}"
        order = entry_order(
            context.order,
            context.score,
            volatility_rank,
            confirmed_weak,
            penalty,
        )
        results[candidate_id], daily, actions = evaluate(
            context, policy, order, priority_matrix, extra_age
        )
        cache[candidate_id] = (daily, actions, order)

    baseline_id = "weak2_entry_vol_penalty_0.000"
    baseline = results[baseline_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
            )
        )
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("entry rerank baseline drifted")
    gates = {
        candidate_id: selection_gates(result, baseline)
        for candidate_id, result in results.items()
        if candidate_id != baseline_id
    }
    eligible = [
        candidate_id
        for candidate_id, candidate_gates in gates.items()
        if all(candidate_gates.values())
    ]
    selected = max(
        [baseline_id, *eligible],
        key=lambda candidate_id: (
            results[candidate_id]["metrics_0_30pct"]["sharpe"],
            results[candidate_id]["metrics_0_30pct"]["cagr"],
            -results[candidate_id]["metrics_0_30pct"]["max_drawdown"],
        ),
    )
    repeat, repeat_daily, repeat_actions = evaluate(
        context, policy, cache[selected][2], priority_matrix, extra_age
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
        "metrics": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("entry rerank replay failed")

    result = {
        "status": "weak2_entry_volatility_rerank_complete_2026_not_opened",
        "candidate_budget": [
            f"weak2_entry_vol_penalty_{value:.3f}" for value in PENALTIES
        ],
        "only_change": (
            "after two consecutive weak sessions, reorder only the production top-30 "
            "entry pool by score minus penalty times trailing-20d volatility rank; "
            "exit score and eligibility remain production-score based"
        ),
        "results": results,
        "selection_gates_against_production_entry_order": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
