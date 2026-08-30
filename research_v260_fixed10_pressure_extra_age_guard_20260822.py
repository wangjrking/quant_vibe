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

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_extra_age_guard_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_regime_pressure_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("extra_sell_min_age4", "extra_sell_min_age8")


def run_policy_at_cost(
    context,
    policy: dict,
    extra_min_age,
    cost: float,
    simulator=None,
    entry_block_mask_override=None,
    maintenance_buy_block_mask_override=None,
    selection_mask_override=None,
    min_hold_days_override=None,
    sell_score_below_override=None,
    replacement_advantage_override=None,
    max_hold_renewal_score_override=None,
    pressure_trigger_override=None,
    pressure_limit_override=None,
    score_sell_priority_override=None,
    exit_score_override=None,
    score_exit_confirmation_mask_override=None,
    max_entry_open_gap_override=None,
    min_entry_open_gap_override=None,
    defer_sell_on_limit_up_override=None,
    portfolio_rebalance_active_override=None,
    portfolio_rebalance_overweight_deviation_override=None,
    portfolio_rebalance_underweight_deviation_override=None,
    position_observer=None,
    end_date=None,
    target_positions_override: int = 10,
    target_gross_override: float = 1.0,
):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
    )
    if maintenance_buy_block_mask_override is not None:
        maintenance_block = np.asarray(
            maintenance_buy_block_mask_override, dtype=np.bool_
        )
    strong = regime.strong_market_mask(context.score, context.protocol)
    common = dict(
        simulator=runtime.simulate if simulator is None else simulator,
        portfolio_rebalance_active_override=(
            cadence.rebalance_schedule(
                len(context.arrays["dates"]),
                int(policy["portfolio_rebalance_interval_days"]),
            )
            if portfolio_rebalance_active_override is None
            else portfolio_rebalance_active_override
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        portfolio_rebalance_overweight_deviation_override=(
            portfolio_rebalance_overweight_deviation_override
        ),
        portfolio_rebalance_underweight_deviation_override=(
            portfolio_rebalance_underweight_deviation_override
        ),
        entry_block_mask_override=(
            context.empty_block
            if entry_block_mask_override is None
            else entry_block_mask_override
        ),
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=(
            trigger.pressure_trigger_schedule(strong)
            if pressure_trigger_override is None
            else pressure_trigger_override
        ),
        score_sell_pressure_limit_override=(
            policy["score_sell_pressure_limit"]
            if pressure_limit_override is None
            else pressure_limit_override
        ),
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
        score_sell_pressure_extra_min_age_override=extra_min_age,
        selection_mask_override=selection_mask_override,
        min_hold_days_override=min_hold_days_override,
        sell_score_below_override=sell_score_below_override,
        replacement_advantage_override=replacement_advantage_override,
        max_hold_renewal_score_override=max_hold_renewal_score_override,
        score_sell_priority_override=score_sell_priority_override,
        exit_score_override=exit_score_override,
        score_exit_confirmation_mask_override=score_exit_confirmation_mask_override,
        max_entry_open_gap_override=max_entry_open_gap_override,
        min_entry_open_gap_override=min_entry_open_gap_override,
        defer_sell_on_limit_up_override=defer_sell_on_limit_up_override,
        position_observer=position_observer,
    )
    effective_end_date = (
        round1.DEVELOPMENT_END if end_date is None else str(end_date)
    )
    return round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        effective_end_date,
        slip=float(cost),
        record_actions=True,
        target_positions_override=target_positions_override,
        target_gross_override=target_gross_override,
        **common,
    )


def run_policy(
    context,
    policy: dict,
    extra_min_age,
    simulator=None,
    entry_block_mask_override=None,
    maintenance_buy_block_mask_override=None,
    selection_mask_override=None,
    min_hold_days_override=None,
    sell_score_below_override=None,
    pressure_trigger_override=None,
    score_sell_priority_override=None,
    exit_score_override=None,
    score_exit_confirmation_mask_override=None,
    max_entry_open_gap_override=None,
    min_entry_open_gap_override=None,
    defer_sell_on_limit_up_override=None,
):
    daily, actions = run_policy_at_cost(
        context,
        policy,
        extra_min_age,
        round1.BASELINE_COST,
        simulator=simulator,
        entry_block_mask_override=entry_block_mask_override,
        maintenance_buy_block_mask_override=maintenance_buy_block_mask_override,
        selection_mask_override=selection_mask_override,
        min_hold_days_override=min_hold_days_override,
        sell_score_below_override=sell_score_below_override,
        pressure_trigger_override=pressure_trigger_override,
        score_sell_priority_override=score_sell_priority_override,
        exit_score_override=exit_score_override,
        score_exit_confirmation_mask_override=score_exit_confirmation_mask_override,
        max_entry_open_gap_override=max_entry_open_gap_override,
        min_entry_open_gap_override=min_entry_open_gap_override,
        defer_sell_on_limit_up_override=defer_sell_on_limit_up_override,
    )
    stress_daily, stress_actions = run_policy_at_cost(
        context,
        policy,
        extra_min_age,
        round1.STRESS_COST,
        simulator=simulator,
        entry_block_mask_override=entry_block_mask_override,
        maintenance_buy_block_mask_override=maintenance_buy_block_mask_override,
        selection_mask_override=selection_mask_override,
        min_hold_days_override=min_hold_days_override,
        sell_score_below_override=sell_score_below_override,
        pressure_trigger_override=pressure_trigger_override,
        score_sell_priority_override=score_sell_priority_override,
        exit_score_override=exit_score_override,
        score_exit_confirmation_mask_override=score_exit_confirmation_mask_override,
        max_entry_open_gap_override=max_entry_open_gap_override,
        min_entry_open_gap_override=min_entry_open_gap_override,
        defer_sell_on_limit_up_override=defer_sell_on_limit_up_override,
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
        "cagr_improved": current["cagr"] > reference["cagr"],
        "sharpe_improved": current["sharpe"] > reference["sharpe"],
        "drawdown_not_worse": current["max_drawdown"]
        <= reference["max_drawdown"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "all_years_positive": all(
            value > 0 for value in current["annual_returns"].values()
        ),
        "full_10_positions": current["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is not available for age-guard research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    overrides = {"extra_sell_min_age4": None, "extra_sell_min_age8": 8}
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        results[candidate_id], daily, actions = run_policy(
            context, policy, overrides[candidate_id]
        )
        cache[candidate_id] = (daily, actions)

    baseline_id, candidate_id = CANDIDATES
    baseline, candidate = results[baseline_id], results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
            )
        )
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("pressure extra-age baseline drifted")
    gates = selection_gates(candidate, baseline)
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat, repeat_daily, repeat_actions = run_policy(
        context, policy, overrides[selected]
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1]) == round1.frame_hash(repeat_actions),
        "metrics": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure extra-age replay failed")

    selected_policy = copy.deepcopy(policy)
    selected_policy["score_sell_pressure_extra_min_age"] = 4 if selected == baseline_id else 8
    result = {
        "status": "pressure_extra_age_guard_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "the second same-day pressure exit must have been held at least eight "
            "sessions; the ordinary first score exit retains the four-day minimum"
        ),
        "results": results,
        "selection_gates": gates,
        "selected_candidate": selected,
        "selected_policy": selected_policy,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "source_manifests": context.manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
