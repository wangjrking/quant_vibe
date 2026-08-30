from __future__ import annotations

import copy
import functools
import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
from research_v260_runtime import fixed10_residual_cash_sweep_v112 as sweep_runtime


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_residual_cash_sweep_20260823"
)


def run_with_runtime(
    context,
    policy: dict,
    cost: float,
    *,
    sweep: bool,
    rank_sizing: bool = True,
    sweep_max_orders: int | None = None,
    sweep_trigger: str = "any_trade",
    sweep_recipient: str = "most_underweight",
):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    simulator = functools.partial(
        sweep_runtime.simulate,
        residual_cash_sweep_override=sweep,
        residual_cash_sweep_max_orders_override=sweep_max_orders,
        residual_cash_sweep_trigger_override=sweep_trigger,
        residual_cash_sweep_recipient_override=sweep_recipient,
    )
    if rank_sizing:
        simulator = functools.partial(
            simulator,
            candidate_target_multiplier_override=sizing.entry_rank_multipliers(
                context.order
            ),
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
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered residual-cash sweep research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    cases = {}
    frames = {}
    for cost_name, cost in (
        ("0_30pct", sizing.BASELINE_COST),
        ("0_65pct", sizing.STRESS_COST),
    ):
        current_daily, current_actions = sizing.run_case(
            context,
            policy,
            cost,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        copy_daily, copy_actions = run_with_runtime(
            context, policy, cost, sweep=False
        )
        sweep_daily, sweep_actions = run_with_runtime(
            context, policy, cost, sweep=True
        )
        if not current_daily.equals(copy_daily) or not current_actions.equals(copy_actions):
            raise RuntimeError("cash-sweep runtime changed default candidate behavior")
        cases[cost_name] = {
            "current_no_sweep": sizing.evaluate(current_daily, current_actions),
            "residual_cash_sweep": sizing.evaluate(sweep_daily, sweep_actions),
        }
        frames[cost_name] = {
            "current_daily": current_daily,
            "current_actions": current_actions,
            "sweep_daily": sweep_daily,
            "sweep_actions": sweep_actions,
        }

    replay_daily, replay_actions = run_with_runtime(
        context, policy, sizing.BASELINE_COST, sweep=True
    )
    deterministic = {
        "daily": replay_daily.equals(frames["0_30pct"]["sweep_daily"]),
        "actions": replay_actions.equals(frames["0_30pct"]["sweep_actions"]),
    }
    if not all(deterministic.values()):
        raise RuntimeError("residual-cash sweep replay drifted")

    deltas = {
        cost_name: sizing.checkpoint_tools.metric_delta(
            values["residual_cash_sweep"], values["current_no_sweep"]
        )
        for cost_name, values in cases.items()
    }
    selected = (
        "residual_cash_sweep"
        if deltas["0_30pct"]["cumulative_return"] > 0.0
        else "current_no_sweep"
    )
    current_actions = frames["0_30pct"]["current_actions"]
    sweep_actions = frames["0_30pct"]["sweep_actions"]
    result = {
        "status": "rank_sizing_residual_cash_sweep_complete_2026_not_opened",
        "single_new_rule": (
            "after a normal trade, spend affordable board-lot residual cash on "
            "the most underweight existing holding that remains buyable"
        ),
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "equalweight_and_full_investment_are_soft_directions": True,
        "parameter_search_count": 0,
        "cases": cases,
        "sweep_minus_current": deltas,
        "selected_arm": selected,
        "selected_changes_frozen_candidate": False,
        "followup_required_before_candidate_change": selected == "residual_cash_sweep",
        "action_diagnostics_0_30pct": {
            "current_action_count": int(len(current_actions)),
            "sweep_action_count": int(len(sweep_actions)),
            "additional_topup_actions": int(len(sweep_actions) - len(current_actions)),
            "new_names_from_sweep": 0,
            "sales_from_sweep": 0,
        },
        "default_runtime_exact_replay": True,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_arm": selected,
                "baseline_return_delta": deltas["0_30pct"]["cumulative_return"],
                "stress_return_delta": deltas["0_65pct"]["cumulative_return"],
                "invested_ratio_delta": deltas["0_30pct"][
                    "average_invested_ratio"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
