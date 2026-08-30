from __future__ import annotations

import copy
import functools
import inspect
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
    "strategy_agent_v260_fixed10_rank_sizing_maintenance_profit_optimization_20260823"
)


def run_maintained_case(context, policy: dict, cost: float):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers = sizing.entry_rank_multipliers(
        context.order,
        sizing.TARGET_POSITIONS,
        sizing.TOP_WEIGHT_MULTIPLIER,
        sizing.BOTTOM_WEIGHT_MULTIPLIER,
    )
    simulator = functools.partial(
        sizing.runtime.simulate,
        candidate_target_multiplier_override=multipliers,
        maintenance_target_multiplier_override=multipliers,
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


def select_by_pre2026_net_return(metrics: dict[str, dict]) -> str:
    if set(metrics) != {"new_entry_only", "new_entry_and_maintenance"}:
        raise ValueError("maintenance comparison must contain exactly two arms")
    return max(
        metrics,
        key=lambda name: (float(metrics[name]["cumulative_return"]), name),
    )


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered maintenance sizing optimization")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
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
        maintained_daily, maintained_actions = run_maintained_case(
            context, policy, cost
        )
        cases[cost_name] = {
            "new_entry_only": sizing.evaluate(current_daily, current_actions),
            "new_entry_and_maintenance": sizing.evaluate(
                maintained_daily, maintained_actions
            ),
        }
        frames[cost_name] = {
            "current_daily": current_daily,
            "current_actions": current_actions,
            "maintained_daily": maintained_daily,
            "maintained_actions": maintained_actions,
        }

    selected = select_by_pre2026_net_return(cases["0_30pct"])
    replay_daily, replay_actions = run_maintained_case(
        context, policy, sizing.BASELINE_COST
    )
    deterministic = {
        "daily": replay_daily.equals(frames["0_30pct"]["maintained_daily"]),
        "actions": replay_actions.equals(
            frames["0_30pct"]["maintained_actions"]
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("maintained rank sizing replay drifted")
    current_replay_equivalent = all(
        np.isclose(
            cases["0_30pct"]["new_entry_only"][key],
            checkpoint["metrics_0_30pct"][key],
            rtol=0.0,
            atol=1e-12,
        )
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    )
    if not current_replay_equivalent:
        raise RuntimeError("current new-entry-only replay drifted")

    result = {
        "status": "rank_sizing_maintenance_profit_comparison_complete_2026_not_opened",
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "single_new_rule_tested": (
            "reuse the frozen 1.10-to-0.90 global-rank multipliers during the "
            "existing 20-session maintenance rebalance"
        ),
        "arms": cases,
        "selected_arm": selected,
        "selected_changes_frozen_candidate": selected == "new_entry_and_maintenance",
        "baseline_delta_maintained_minus_entry_only": (
            sizing.checkpoint_tools.metric_delta(
                cases["0_30pct"]["new_entry_and_maintenance"],
                cases["0_30pct"]["new_entry_only"],
            )
        ),
        "stress_delta_maintained_minus_entry_only": (
            sizing.checkpoint_tools.metric_delta(
                cases["0_65pct"]["new_entry_and_maintenance"],
                cases["0_65pct"]["new_entry_only"],
            )
        ),
        "runtime_hook": {
            "parameter": "maintenance_target_multiplier_override",
            "default": inspect.signature(sizing.runtime.simulate)
            .parameters["maintenance_target_multiplier_override"]
            .default,
            "default_preserves_existing_behavior": True,
        },
        "current_replay_equivalent": current_replay_equivalent,
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
                "baseline_delta": result[
                    "baseline_delta_maintained_minus_entry_only"
                ],
                "stress_delta": result[
                    "stress_delta_maintained_minus_entry_only"
                ],
                "validation_2026_opened": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
