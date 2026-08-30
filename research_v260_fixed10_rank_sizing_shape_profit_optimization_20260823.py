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
import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as linear_tools
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_soft_width_profit_optimization_20260822 as width_tools
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_shape_profit_optimization_20260823"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
BASELINE_COST = 0.0030
STRESS_COST = 0.0065
TARGET_POSITIONS = 10


def step_rank_multipliers(
    order: np.ndarray,
    positions: int = TARGET_POSITIONS,
    top_multiplier: float = 1.10,
    bottom_multiplier: float = 0.90,
) -> np.ndarray:
    if order.ndim != 2:
        raise ValueError("order must be a two-dimensional ranking matrix")
    if positions <= 1 or positions > order.shape[1]:
        raise ValueError("positions must be between two and the stock count")
    if bottom_multiplier <= 0.0 or top_multiplier < bottom_multiplier:
        raise ValueError("rank sizing bounds must be positive and descending")
    if not np.isclose(
        top_multiplier + bottom_multiplier, 2.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("step multipliers must be symmetric around one")

    half = positions // 2
    ranked = np.concatenate(
        [
            np.full(half, top_multiplier, dtype=np.float64),
            np.ones(positions - 2 * half, dtype=np.float64),
            np.full(half, bottom_multiplier, dtype=np.float64),
        ]
    )
    if not np.isclose(ranked.sum(), float(positions), rtol=0.0, atol=1e-12):
        raise RuntimeError("step rank sizing does not preserve target gross")
    result = np.ones(order.shape, dtype=np.float64)
    rows = np.arange(order.shape[0])[:, None]
    result[rows, order[:, :positions]] = ranked[None, :]
    return result


def run_with_multipliers(context, policy: dict, cost: float, multipliers):
    extra_age, sell_priority = width_tools.build_overrides(context, policy)
    simulator = runtime.simulate
    if multipliers is not None:
        simulator = functools.partial(
            runtime.simulate,
            candidate_target_multiplier_override=multipliers,
        )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        simulator=simulator,
        score_sell_priority_override=sell_priority,
        target_positions_override=TARGET_POSITIONS,
        target_gross_override=1.0,
    )


def evaluate(daily, actions) -> dict:
    return round1.evaluate_run(
        daily,
        actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
        target_positions=TARGET_POSITIONS,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered rank sizing shape development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    designs = {
        "equalweight_entry": None,
        "linear_rank11_to_9_entry": linear_tools.entry_rank_multipliers(
            context.order, TARGET_POSITIONS, 1.10, 0.90
        ),
        "step_top5_11_bottom5_9_entry": step_rank_multipliers(
            context.order, TARGET_POSITIONS, 1.10, 0.90
        ),
    }
    results = {}
    frames = {}
    for name, multipliers in designs.items():
        daily, actions = run_with_multipliers(
            context, policy, BASELINE_COST, multipliers
        )
        stress_daily, stress_actions = run_with_multipliers(
            context, policy, STRESS_COST, multipliers
        )
        results[name] = {
            "metrics_0_30pct": evaluate(daily, actions),
            "metrics_0_65pct": evaluate(stress_daily, stress_actions),
        }
        frames[name] = (daily, actions)

    selected = max(
        results,
        key=lambda name: results[name]["metrics_0_30pct"]["cumulative_return"],
    )
    repeat_daily, repeat_actions = run_with_multipliers(
        context, policy, BASELINE_COST, designs[selected]
    )
    deterministic = {
        "daily": round1.frame_hash(frames[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(frames[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("rank sizing shape replay is not deterministic")

    current = results["linear_rank11_to_9_entry"]
    challenger = results["step_top5_11_bottom5_9_entry"]
    result = {
        "status": "rank_sizing_shape_profit_comparison_complete_2026_not_opened",
        "only_change": (
            "new-entry rank sizing shape changes from linear 1.10..0.90 to "
            "top-five 1.10 and bottom-five 0.90; holdings, exits, costs and "
            "gross direction remain unchanged"
        ),
        "selection_rule": "maximize_pre2026_net_cumulative_return_at_0_30pct_cost",
        "equalweight_and_full_investment_are_soft_directions": True,
        "parameter_search": False,
        "results": results,
        "selected_arm": selected,
        "challenger_minus_current": {
            "0_30pct": checkpoint_tools.metric_delta(
                challenger["metrics_0_30pct"], current["metrics_0_30pct"]
            ),
            "0_65pct": checkpoint_tools.metric_delta(
                challenger["metrics_0_65pct"], current["metrics_0_65pct"]
            ),
        },
        "selected_changes_frozen_candidate": selected
        != "linear_rank11_to_9_entry",
        "deterministic_replay": deterministic,
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_arm": selected,
                "challenger_baseline_delta": result["challenger_minus_current"][
                    "0_30pct"
                ]["cumulative_return"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
