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
    "strategy_agent_v260_fixed10_sweep_rank_interaction_20260823"
)


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered sweep/rank interaction research")
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
        equal_daily, equal_actions = sweep.run_with_runtime(
            context, policy, cost, sweep=True, rank_sizing=False
        )
        rank_daily, rank_actions = sweep.run_with_runtime(
            context, policy, cost, sweep=True, rank_sizing=True
        )
        cases[cost_name] = {
            "exact_equalweight_with_sweep": sizing.evaluate(
                equal_daily, equal_actions
            ),
            "rank11_to_9_with_sweep": sizing.evaluate(rank_daily, rank_actions),
        }
        frames[cost_name] = {
            "equal_daily": equal_daily,
            "equal_actions": equal_actions,
            "rank_daily": rank_daily,
            "rank_actions": rank_actions,
        }

    replay_daily, replay_actions = sweep.run_with_runtime(
        context,
        policy,
        sizing.BASELINE_COST,
        sweep=True,
        rank_sizing=False,
    )
    deterministic = {
        "daily": replay_daily.equals(frames["0_30pct"]["equal_daily"]),
        "actions": replay_actions.equals(frames["0_30pct"]["equal_actions"]),
    }
    if not all(deterministic.values()):
        raise RuntimeError("equalweight cash-sweep replay drifted")

    deltas = {
        cost_name: sizing.checkpoint_tools.metric_delta(
            values["rank11_to_9_with_sweep"],
            values["exact_equalweight_with_sweep"],
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
        "status": "sweep_rank_interaction_complete_2026_not_opened",
        "question": (
            "after residual-cash sweeping, does the frozen 1.10-to-0.90 new-entry "
            "rank tilt still add net profit over exact 0.10 entry targets"
        ),
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "parameter_search_count": 0,
        "cases": cases,
        "rank_minus_exact_equalweight": deltas,
        "selected_arm": selected,
        "selected_changes_frozen_candidate": False,
        "followup_required_before_candidate_change": True,
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
                "rank_minus_equal_baseline_return": deltas["0_30pct"][
                    "cumulative_return"
                ],
                "rank_minus_equal_stress_return": deltas["0_65pct"][
                    "cumulative_return"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
