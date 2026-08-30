from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_rank_sizing_residual_cash_sweep_20260823 as sweep


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_cash_sweep_order_simplification_20260823"
)


def run_arm(context, policy: dict, cost: float, max_orders: int | None):
    return sweep.run_with_runtime(
        context,
        policy,
        cost,
        sweep=True,
        rank_sizing=True,
        sweep_max_orders=max_orders,
    )


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-sweep order simplification")
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
        unlimited_daily, unlimited_actions = run_arm(
            context, policy, cost, None
        )
        one_daily, one_actions = run_arm(context, policy, cost, 1)
        cases[cost_name] = {
            "unlimited_affordable_topups": sizing.evaluate(
                unlimited_daily, unlimited_actions
            ),
            "one_topup_order_per_trade_day": sizing.evaluate(
                one_daily, one_actions
            ),
        }
        frames[cost_name] = {
            "unlimited_daily": unlimited_daily,
            "unlimited_actions": unlimited_actions,
            "one_daily": one_daily,
            "one_actions": one_actions,
        }

    replay_daily, replay_actions = run_arm(
        context, policy, sizing.BASELINE_COST, 1
    )
    deterministic = {
        "daily": replay_daily.equals(frames["0_30pct"]["one_daily"]),
        "actions": replay_actions.equals(frames["0_30pct"]["one_actions"]),
    }
    if not all(deterministic.values()):
        raise RuntimeError("one-order cash-sweep replay drifted")
    deltas = {
        cost_name: sizing.checkpoint_tools.metric_delta(
            values["one_topup_order_per_trade_day"],
            values["unlimited_affordable_topups"],
        )
        for cost_name, values in cases.items()
    }
    selected = max(
        cases["0_30pct"],
        key=lambda name: (
            float(cases["0_30pct"][name]["cumulative_return"]),
            name,
        ),
    )
    result = {
        "status": "cash_sweep_order_simplification_complete_2026_not_opened",
        "only_change": (
            "limit the residual-cash sweep to the single most-underweight buyable "
            "holding per trade day instead of filling every affordable shortfall"
        ),
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "parameter_search_count": 0,
        "cases": cases,
        "one_order_minus_unlimited": deltas,
        "selected_arm": selected,
        "selected_changes_frozen_candidate": False,
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
                "baseline_turnover_delta": deltas["0_30pct"][
                    "turnover_annualized"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
