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
    "strategy_agent_v260_fixed10_rank_sizing_reverse_placebo_20260823"
)


def linear_rank_multipliers(
    order: np.ndarray,
    positions: int,
    first_multiplier: float,
    last_multiplier: float,
) -> np.ndarray:
    if order.ndim != 2:
        raise ValueError("order must be two-dimensional")
    if positions <= 1 or positions > order.shape[1]:
        raise ValueError("positions must fit the ranking matrix")
    if min(first_multiplier, last_multiplier) <= 0.0:
        raise ValueError("rank multipliers must remain positive")
    schedule = np.linspace(
        first_multiplier, last_multiplier, positions, dtype=np.float64
    )
    if not np.isclose(schedule.sum(), float(positions), rtol=0.0, atol=1e-12):
        raise ValueError("placebo schedule must preserve reference gross")
    result = np.ones(order.shape, dtype=np.float64)
    rows = np.arange(order.shape[0])[:, None]
    result[rows, order[:, :positions]] = schedule[None, :]
    return result


def run_placebo(context, policy: dict, cost: float, first: float, last: float):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers = linear_rank_multipliers(
        context.order, sizing.TARGET_POSITIONS, first, last
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
        score_sell_priority_override=sell_priority,
        target_positions_override=sizing.TARGET_POSITIONS,
        target_gross_override=1.0,
    )


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered reverse rank sizing placebo")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    results = {}
    deterministic = {}
    for cost_name, cost in (
        ("0_30pct", sizing.BASELINE_COST),
        ("0_65pct", sizing.STRESS_COST),
    ):
        equal_daily, equal_actions = sizing.run_case(
            context, policy, cost, None, None
        )
        forward_daily, forward_actions = sizing.run_case(
            context,
            policy,
            cost,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        reverse_daily, reverse_actions = run_placebo(
            context, policy, cost, 0.90, 1.10
        )
        equal = sizing.evaluate(equal_daily, equal_actions)
        forward = sizing.evaluate(forward_daily, forward_actions)
        reverse = sizing.evaluate(reverse_daily, reverse_actions)
        results[cost_name] = {
            "equalweight": equal,
            "forward_high_score_overweight": forward,
            "reverse_low_score_overweight_placebo": reverse,
            "forward_minus_equalweight": sizing.checkpoint_tools.metric_delta(
                forward, equal
            ),
            "reverse_minus_equalweight": sizing.checkpoint_tools.metric_delta(
                reverse, equal
            ),
            "forward_minus_reverse": sizing.checkpoint_tools.metric_delta(
                forward, reverse
            ),
        }
        replay_daily, replay_actions = run_placebo(
            context, policy, cost, 0.90, 1.10
        )
        deterministic[cost_name] = {
            "daily": replay_daily.equals(reverse_daily),
            "actions": replay_actions.equals(reverse_actions),
        }
    if not all(
        all(item.values()) for item in deterministic.values()
    ):
        raise RuntimeError("reverse placebo replay drifted")

    result = {
        "status": "rank_sizing_reverse_placebo_complete_2026_not_opened",
        "role": "directional_causality_diagnostic_not_selectable",
        "design": {
            "forward": "global score top10 receives 1.10 down to 0.90",
            "reverse_placebo": "global score top10 receives 0.90 up to 1.10",
            "application": "new_entries_only",
            "parameter_search": False,
        },
        "results": results,
        "forward_beats_reverse_at_both_costs": all(
            results[cost]["forward_minus_reverse"]["cumulative_return"] > 0.0
            for cost in results
        ),
        "reverse_beats_equalweight_at_both_costs": all(
            results[cost]["reverse_minus_equalweight"]["cumulative_return"] > 0.0
            for cost in results
        ),
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "reverse_placebo.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "forward_beats_reverse_at_both_costs": result[
                    "forward_beats_reverse_at_both_costs"
                ],
                "reverse_beats_equalweight_at_both_costs": result[
                    "reverse_beats_equalweight_at_both_costs"
                ],
                "results": {
                    cost: {
                        "forward_minus_equalweight": item[
                            "forward_minus_equalweight"
                        ]["cumulative_return"],
                        "reverse_minus_equalweight": item[
                            "reverse_minus_equalweight"
                        ]["cumulative_return"],
                        "forward_minus_reverse": item[
                            "forward_minus_reverse"
                        ]["cumulative_return"],
                    }
                    for cost, item in results.items()
                },
                "validation_2026_opened": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
