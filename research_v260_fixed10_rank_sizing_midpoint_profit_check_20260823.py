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


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_midpoint_profit_check_20260823"
)
MIDPOINT_TOP = 1.05
MIDPOINT_BOTTOM = 0.95


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered rank-sizing midpoint research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    designs = {
        "midpoint_rank105_to_95": (MIDPOINT_TOP, MIDPOINT_BOTTOM),
        "current_rank110_to_90": (
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        ),
    }
    cases = {}
    frames = {}
    for name, (top, bottom) in designs.items():
        daily, actions = sizing.run_case(
            context, policy, sizing.BASELINE_COST, top, bottom
        )
        stress_daily, stress_actions = sizing.run_case(
            context, policy, sizing.STRESS_COST, top, bottom
        )
        cases[name] = {
            "top_multiplier": top,
            "bottom_multiplier": bottom,
            "metrics_0_30pct": sizing.evaluate(daily, actions),
            "metrics_0_65pct": sizing.evaluate(stress_daily, stress_actions),
        }
        frames[name] = (daily, actions)

    repeat_daily, repeat_actions = sizing.run_case(
        context,
        policy,
        sizing.BASELINE_COST,
        MIDPOINT_TOP,
        MIDPOINT_BOTTOM,
    )
    midpoint_daily, midpoint_actions = frames["midpoint_rank105_to_95"]
    deterministic = {
        "daily": round1.frame_hash(midpoint_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(midpoint_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("rank-sizing midpoint replay is not deterministic")

    midpoint = cases["midpoint_rank105_to_95"]
    current = cases["current_rank110_to_90"]
    delta = {
        cost: sizing.checkpoint_tools.metric_delta(
            midpoint[f"metrics_{cost}"], current[f"metrics_{cost}"]
        )
        for cost in ("0_30pct", "0_65pct")
    }
    midpoint_wins_both = bool(
        delta["0_30pct"]["cumulative_return"] > 0.0
        and delta["0_65pct"]["cumulative_return"] > 0.0
    )
    result = {
        "status": "rank_sizing_midpoint_profit_check_complete_2026_not_opened",
        "single_new_probe": "linear new-entry rank sizing 1.05..0.95",
        "parameter_search_count": 1,
        "equalweight_is_soft_reference": True,
        "cases": cases,
        "midpoint_minus_current": delta,
        "midpoint_wins_at_both_costs": midpoint_wins_both,
        "selection_decision": (
            "replace_with_midpoint_pending_robustness"
            if midpoint_wins_both
            else "retain_current_rank110_to_90"
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
