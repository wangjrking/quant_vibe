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

import research_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822 as checkpoint_tools
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
    "strategy_agent_v260_fixed10_rank_sizing_pressure_trigger_simplification_20260823"
)


def uniform4_policy(policy: dict) -> dict:
    result = copy.deepcopy(policy)
    if result.get("score_sell_pressure_trigger") != 4:
        raise ValueError("current strong-market pressure trigger is not four")
    result["score_sell_pressure_trigger_rule"] = "uniform4"
    return result


def run_arm(context, policy: dict, cost: float, uniform_trigger: bool):
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
    )
    return sizing.age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        simulator=simulator,
        pressure_trigger_override=(4 if uniform_trigger else None),
        score_sell_priority_override=sell_priority,
        target_positions_override=sizing.TARGET_POSITIONS,
        target_gross_override=1.0,
    )


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered pressure-trigger simplification")
    current_policy = copy.deepcopy(checkpoint["selected_policy"])
    simplified_policy = uniform4_policy(current_policy)
    changed = sorted(
        key
        for key in set(current_policy) | set(simplified_policy)
        if current_policy.get(key) != simplified_policy.get(key)
    )
    if changed != ["score_sell_pressure_trigger_rule"]:
        raise RuntimeError("pressure-trigger simplification changed unexpected keys")
    context = sizing.width_tools.harness.load_context(current_policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    arms = {}
    frames = {}
    for name, policy, uniform_trigger in (
        ("current_strong4_weak5", current_policy, False),
        ("uniform4", simplified_policy, True),
    ):
        daily, actions = run_arm(
            context, policy, sizing.BASELINE_COST, uniform_trigger
        )
        stress_daily, stress_actions = run_arm(
            context, policy, sizing.STRESS_COST, uniform_trigger
        )
        arms[name] = {
            "metrics_0_30pct": sizing.evaluate(daily, actions),
            "metrics_0_65pct": sizing.evaluate(stress_daily, stress_actions),
        }
        frames[name] = (daily, actions)

    current = arms["current_strong4_weak5"]
    expected = checkpoint["metrics_0_30pct"]
    for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown"):
        if not np.isclose(
            current["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("current rank-sizing baseline drifted")
    selected = max(
        arms,
        key=lambda name: arms[name]["metrics_0_30pct"]["cumulative_return"],
    )
    selected_policy = current_policy if selected == "current_strong4_weak5" else simplified_policy
    repeat_daily, repeat_actions = run_arm(
        context,
        selected_policy,
        sizing.BASELINE_COST,
        selected == "uniform4",
    )
    deterministic = {
        "daily": round1.frame_hash(frames[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(frames[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure-trigger simplification replay failed")

    challenger = arms["uniform4"]
    result = {
        "status": "rank_sizing_pressure_trigger_simplification_complete_2026_not_opened",
        "only_change": (
            "replace the strong4/weak5 pressure trigger branch with uniform4; "
            "sell threshold, sell limit and all other rules remain unchanged"
        ),
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "equalweight_and_full_investment_are_soft_directions": True,
        "changed_policy_keys": changed,
        "arms": arms,
        "selected_arm": selected,
        "challenger_minus_current": {
            "0_30pct": checkpoint_tools.metric_delta(
                challenger["metrics_0_30pct"], current["metrics_0_30pct"]
            ),
            "0_65pct": checkpoint_tools.metric_delta(
                challenger["metrics_0_65pct"], current["metrics_0_65pct"]
            ),
        },
        "selected_policy": selected_policy,
        "selected_changes_frozen_candidate": selected == "uniform4",
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
                "baseline_delta": result["challenger_minus_current"]["0_30pct"][
                    "cumulative_return"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
