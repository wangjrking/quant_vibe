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
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_membership_order_attribution_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def transformed(order: np.ndarray, mode: str) -> np.ndarray:
    result = np.asarray(order, dtype=np.int64).copy()
    if mode == "baseline":
        return result
    if mode == "swap_rank1_rank2":
        result[:, [0, 1]] = result[:, [1, 0]]
    elif mode == "rotate_top10_same_membership":
        result[:, :10] = np.roll(result[:, :10], shift=1, axis=1)
    elif mode == "reverse_top10_same_membership":
        result[:, :10] = result[:, :10][:, ::-1]
    elif mode == "swap_rank10_rank11_membership_boundary":
        result[:, [9, 10]] = result[:, [10, 9]]
    else:
        raise ValueError(f"unknown order transformation: {mode}")
    return result


def run(context, policy: dict, order: np.ndarray, priority_matrix: np.ndarray):
    strong = regime.strong_market_mask(context.score, context.protocol)
    case = copy.copy(context)
    case.order = order
    return age_guard.run_policy_at_cost(
        case,
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
        raise PermissionError("2026 data entered rank attribution")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )

    modes = (
        "baseline",
        "swap_rank1_rank2",
        "rotate_top10_same_membership",
        "reverse_top10_same_membership",
        "swap_rank10_rank11_membership_boundary",
    )
    results = {}
    cache = {}
    for mode in modes:
        daily, actions = run(
            context, policy, transformed(context.order, mode), priority_matrix
        )
        metrics = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        results[mode] = metrics
        cache[mode] = (daily, actions)

    baseline = results["baseline"]
    expected = checkpoint["current_best_equalweight"]
    equivalent = all(
        np.isclose(baseline[key], expected[key], rtol=0.0, atol=1e-12)
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    )
    if not equivalent:
        raise RuntimeError("rank attribution baseline drifted")

    deltas = {
        mode: {
            key: float(metrics[key] - baseline[key])
            for key in (
                "cumulative_return", "cagr", "sharpe", "max_drawdown",
                "turnover_annualized", "average_invested_ratio",
            )
        }
        for mode, metrics in results.items()
        if mode != "baseline"
    }
    repeat_daily, repeat_actions = run(
        context,
        policy,
        transformed(context.order, "swap_rank10_rank11_membership_boundary"),
        priority_matrix,
    )
    deterministic = {
        "daily": round1.frame_hash(repeat_daily)
        == round1.frame_hash(cache["swap_rank10_rank11_membership_boundary"][0]),
        "actions": round1.frame_hash(repeat_actions)
        == round1.frame_hash(cache["swap_rank10_rank11_membership_boundary"][1]),
    }
    if not all(deterministic.values()):
        raise RuntimeError("rank attribution replay failed")

    result = {
        "status": "rank_membership_order_attribution_complete_2026_not_opened",
        "method": (
            "deterministic order perturbations only; score values, exit priorities, "
            "eligibility and all execution rules remain unchanged"
        ),
        "results": results,
        "deltas_vs_baseline": deltas,
        "baseline_equivalent": equivalent,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "rank_membership_order_attribution.json", result)
    print(json.dumps({
        "status": result["status"],
        "deltas_vs_baseline": deltas,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
