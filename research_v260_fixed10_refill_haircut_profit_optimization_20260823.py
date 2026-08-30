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
    "strategy_agent_v260_fixed10_refill_haircut_profit_optimization_20260823"
)


def refill_haircut_multipliers(
    order: np.ndarray,
    positions: int = sizing.TARGET_POSITIONS,
    top_multiplier: float = sizing.TOP_WEIGHT_MULTIPLIER,
    bottom_multiplier: float = sizing.BOTTOM_WEIGHT_MULTIPLIER,
) -> np.ndarray:
    if order.ndim != 2:
        raise ValueError("order must be a two-dimensional ranking matrix")
    if positions <= 1 or positions > order.shape[1]:
        raise ValueError("positions must be between two and the stock count")
    if not 0.0 < bottom_multiplier <= top_multiplier:
        raise ValueError("multipliers must be positive and ordered")
    result = np.full(order.shape, bottom_multiplier, dtype=np.float64)
    schedule = np.linspace(
        top_multiplier,
        bottom_multiplier,
        positions,
        dtype=np.float64,
    )
    if not np.isclose(schedule.sum(), positions, rtol=0.0, atol=1e-12):
        raise RuntimeError("top10 schedule does not preserve reference gross")
    rows = np.arange(order.shape[0])[:, None]
    result[rows, order[:, :positions]] = schedule[None, :]
    return result


def run_refill_haircut(context, policy: dict, cost: float):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers = refill_haircut_multipliers(context.order)
    simulator = functools.partial(
        sizing.runtime.simulate,
        candidate_target_multiplier_override=multipliers,
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
        raise PermissionError("2026 entered refill-haircut development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    cases = {}
    frames = {}
    for name, runner in (
        (
            "current_refill_full_weight",
            lambda cost: sizing.run_case(
                context,
                policy,
                cost,
                sizing.TOP_WEIGHT_MULTIPLIER,
                sizing.BOTTOM_WEIGHT_MULTIPLIER,
            ),
        ),
        (
            "refill_outside_top10_at_9pct",
            lambda cost: run_refill_haircut(context, policy, cost),
        ),
    ):
        daily, actions = runner(sizing.BASELINE_COST)
        stress_daily, stress_actions = runner(sizing.STRESS_COST)
        cases[name] = {
            "metrics_0_30pct": sizing.evaluate(daily, actions),
            "metrics_0_65pct": sizing.evaluate(stress_daily, stress_actions),
            "buy_target_diagnostics": sizing.realized_buy_target_diagnostics(actions),
        }
        frames[name] = (daily, actions)

    repeat_daily, repeat_actions = run_refill_haircut(
        context, policy, sizing.BASELINE_COST
    )
    candidate_daily, candidate_actions = frames["refill_outside_top10_at_9pct"]
    deterministic = {
        "daily": round1.frame_hash(candidate_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(candidate_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("refill-haircut replay is not deterministic")

    current = cases["current_refill_full_weight"]
    candidate = cases["refill_outside_top10_at_9pct"]
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
        "status": "refill_haircut_profit_optimization_complete_2026_not_opened",
        "single_new_rule": (
            "when execution refill reaches a name outside the signal-date global "
            "top10, use the existing 0.90 multiplier instead of forcing 1.00"
        ),
        "parameter_search_count": 0,
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
        "selection_decision": result["selection_decision"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
