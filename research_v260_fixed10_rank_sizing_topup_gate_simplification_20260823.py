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
    "strategy_agent_v260_fixed10_rank_sizing_topup_gate_simplification_20260823"
)


def without_maintenance_topup_score_gate(policy: dict) -> dict:
    result = copy.deepcopy(policy)
    if result.get("maintenance_topup_requires_score") is None:
        raise ValueError("maintenance top-up score gate is already absent")
    result["maintenance_topup_requires_score"] = None
    return result


def changed_policy_keys(left: dict, right: dict) -> list[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def run_arm(context, policy: dict, cost: float):
    return sizing.run_case(context, policy, cost, 1.10, 0.90)


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered maintenance top-up simplification")
    current_policy = copy.deepcopy(checkpoint["selected_policy"])
    simplified_policy = without_maintenance_topup_score_gate(current_policy)
    changed_keys = changed_policy_keys(current_policy, simplified_policy)
    if changed_keys != ["maintenance_topup_requires_score"]:
        raise RuntimeError("top-up simplification changed unexpected policy keys")
    context = sizing.width_tools.harness.load_context(current_policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    arms = {}
    frames = {}
    for name, policy in (
        ("current_rank_sizing", current_policy),
        ("remove_maintenance_topup_score_gate", simplified_policy),
    ):
        daily, actions = run_arm(context, policy, sizing.BASELINE_COST)
        stress_daily, stress_actions = run_arm(context, policy, sizing.STRESS_COST)
        arms[name] = {
            "metrics_0_30pct": sizing.evaluate(daily, actions),
            "metrics_0_65pct": sizing.evaluate(stress_daily, stress_actions),
        }
        frames[name] = (daily, actions)

    current = arms["current_rank_sizing"]
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
    repeat_daily, repeat_actions = run_arm(
        context,
        current_policy if selected == "current_rank_sizing" else simplified_policy,
        sizing.BASELINE_COST,
    )
    deterministic = {
        "daily": round1.frame_hash(frames[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(frames[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("maintenance top-up simplification replay failed")

    challenger = arms["remove_maintenance_topup_score_gate"]
    result = {
        "status": "rank_sizing_topup_gate_simplification_complete_2026_not_opened",
        "only_change": "remove maintenance_topup_requires_score from the current rank-sizing candidate",
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "equalweight_and_full_investment_are_soft_directions": True,
        "changed_policy_keys": changed_keys,
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
        "selected_policy": (
            current_policy if selected == "current_rank_sizing" else simplified_policy
        ),
        "selected_changes_frozen_candidate": selected != "current_rank_sizing",
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
