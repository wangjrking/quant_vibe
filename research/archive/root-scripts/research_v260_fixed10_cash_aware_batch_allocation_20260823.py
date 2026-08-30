from __future__ import annotations

import copy
import functools
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_cash_aware_batch_allocation_20260823"
)
BUY_COMMISSION = 0.0003


def allocate_cash_aware_lots(
    *,
    candidates: np.ndarray,
    target_pcts: np.ndarray,
    raw_opens: np.ndarray,
    equity_before: float,
    cash_available: float,
    slippage_ratio: float,
) -> np.ndarray:
    del candidates
    arrays = (target_pcts, raw_opens)
    if len({value.shape for value in arrays}) != 1 or target_pcts.ndim != 1:
        raise ValueError("batch allocation inputs must be aligned vectors")
    if (
        not np.isfinite(target_pcts).all()
        or not np.isfinite(raw_opens).all()
        or (target_pcts < 0.0).any()
        or (raw_opens <= 0.0).any()
        or not np.isfinite(equity_before)
        or not np.isfinite(cash_available)
        or equity_before <= 0.0
        or cash_available < 0.0
        or not 0.0 <= slippage_ratio < 1.0
    ):
        raise ValueError("batch allocation inputs are invalid")
    lot_costs = raw_opens * (1.0 + slippage_ratio) * (1.0 + BUY_COMMISSION) * 100.0
    desired_lots = np.floor(equity_before * target_pcts / lot_costs).astype(np.int64)
    desired_spend = float(np.dot(desired_lots, lot_costs))
    if desired_spend <= cash_available + 1e-9:
        return desired_lots.astype(np.float64) * 100.0

    scale = max(cash_available / desired_spend, 0.0)
    lots = np.floor(desired_lots.astype(np.float64) * scale).astype(np.int64)
    remaining = float(cash_available - np.dot(lots, lot_costs))
    while True:
        eligible = np.flatnonzero((lots < desired_lots) & (lot_costs <= remaining + 1e-9))
        if not len(eligible):
            break
        shortfall = np.divide(
            desired_lots[eligible] - lots[eligible],
            np.maximum(desired_lots[eligible], 1),
            dtype=np.float64,
        )
        best = int(eligible[int(np.argmax(shortfall))])
        lots[best] += 1
        remaining -= float(lot_costs[best])
    return lots.astype(np.float64) * 100.0


def run_batch(context, policy: dict, cost: float):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers = sizing.entry_rank_multipliers(context.order)
    simulator = functools.partial(
        sizing.runtime.simulate,
        candidate_target_multiplier_override=multipliers,
        candidate_quantity_allocator_override=allocate_cash_aware_lots,
    )
    return sizing.age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        simulator=simulator,
        score_sell_priority_override=sell_priority,
        target_positions_override=sizing.TARGET_POSITIONS,
        target_gross_override=1.0,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-aware batch allocation research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    cases = {}
    frames = {}
    for name, runner in (
        (
            "current_sequential_allocation",
            lambda cost: sizing.run_case(
                context,
                policy,
                cost,
                sizing.TOP_WEIGHT_MULTIPLIER,
                sizing.BOTTOM_WEIGHT_MULTIPLIER,
            ),
        ),
        ("cash_aware_batch_allocation", lambda cost: run_batch(context, policy, cost)),
    ):
        daily, actions = runner(sizing.BASELINE_COST)
        stress_daily, stress_actions = runner(sizing.STRESS_COST)
        cases[name] = {
            "metrics_0_30pct": sizing.evaluate(daily, actions),
            "metrics_0_65pct": sizing.evaluate(stress_daily, stress_actions),
            "buy_target_diagnostics": sizing.realized_buy_target_diagnostics(actions),
        }
        frames[name] = (daily, actions)

    repeat_daily, repeat_actions = run_batch(context, policy, sizing.BASELINE_COST)
    candidate_daily, candidate_actions = frames["cash_aware_batch_allocation"]
    deterministic = {
        "daily": round1.frame_hash(candidate_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(candidate_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("cash-aware batch allocation replay is not deterministic")

    current = cases["current_sequential_allocation"]
    candidate = cases["cash_aware_batch_allocation"]
    delta = {
        cost: sizing.checkpoint_tools.metric_delta(
            candidate[f"metrics_{cost}"], current[f"metrics_{cost}"]
        )
        for cost in ("0_30pct", "0_65pct")
    }
    profit_upgrade = bool(
        delta["0_30pct"]["cumulative_return"] > 0.0
        and delta["0_65pct"]["cumulative_return"] > 0.0
    )
    result = {
        "status": "cash_aware_batch_allocation_complete_2026_not_opened",
        "single_new_rule": (
            "allocate available cash across the same new-entry batch pro rata, then "
            "assign affordable 100-share lots to the largest remaining target gap"
        ),
        "parameter_search_count": 0,
        "stock_selection_and_exit_rules_unchanged": True,
        "portfolio_width_equalweight_and_investment_are_soft": True,
        "cases": cases,
        "candidate_minus_current": delta,
        "profit_upgrade_at_both_costs": profit_upgrade,
        "selection_decision": (
            "eligible_for_further_robustness"
            if profit_upgrade
            else "reject_keep_current_frozen_candidate"
        ),
        "deterministic_replay": deterministic,
        "development_boundary": [
            sizing.research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ],
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps({
        "status": result["status"],
        "baseline_cumulative_delta": delta["0_30pct"]["cumulative_return"],
        "stress_cumulative_delta": delta["0_65pct"]["cumulative_return"],
        "invested_ratio_delta": delta["0_30pct"]["average_invested_ratio"],
        "selection_decision": result["selection_decision"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
