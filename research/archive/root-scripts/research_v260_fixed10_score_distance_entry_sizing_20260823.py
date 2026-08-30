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
    "strategy_agent_v260_fixed10_score_distance_entry_sizing_20260823"
)


def score_distance_multipliers(
    score: np.ndarray,
    order: np.ndarray,
    positions: int = sizing.TARGET_POSITIONS,
    maximum_tilt: float = 0.10,
) -> tuple[np.ndarray, dict]:
    if score.ndim != 2 or order.shape != score.shape:
        raise ValueError("score and order must be aligned two-dimensional matrices")
    if positions <= 1 or positions > score.shape[1]:
        raise ValueError("positions must be between two and the stock count")
    if not 0.0 <= maximum_tilt < 1.0:
        raise ValueError("maximum tilt must be in [0, 1)")

    result = np.ones(score.shape, dtype=np.float64)
    active_days = 0
    tied_or_invalid_days = 0
    realized_ranges = []
    for day in range(score.shape[0]):
        indices = order[day, :positions]
        values = score[day, indices].astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            tied_or_invalid_days += 1
            continue
        deviations = values - float(values.mean())
        scale = float(np.max(np.abs(deviations)))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            tied_or_invalid_days += 1
            continue
        multipliers = 1.0 + maximum_tilt * deviations / scale
        if not np.isclose(multipliers.sum(), positions, rtol=0.0, atol=1e-12):
            raise RuntimeError("score-distance multipliers do not preserve reference gross")
        if multipliers.min() < 1.0 - maximum_tilt - 1e-12:
            raise RuntimeError("score-distance lower bound drifted")
        if multipliers.max() > 1.0 + maximum_tilt + 1e-12:
            raise RuntimeError("score-distance upper bound drifted")
        result[day, indices] = multipliers
        active_days += 1
        realized_ranges.append(float(multipliers.max() - multipliers.min()))

    diagnostics = {
        "days": int(score.shape[0]),
        "active_days": int(active_days),
        "tied_or_invalid_days": int(tied_or_invalid_days),
        "mean_realized_multiplier_range": (
            float(np.mean(realized_ranges)) if realized_ranges else 0.0
        ),
        "minimum_multiplier": float(result.min()),
        "maximum_multiplier": float(result.max()),
    }
    return result, diagnostics


def run_score_distance(context, policy: dict, cost: float):
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers, diagnostics = score_distance_multipliers(
        context.score,
        context.order,
    )
    simulator = functools.partial(
        sizing.runtime.simulate,
        candidate_target_multiplier_override=multipliers,
    )
    daily, actions = sizing.age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        simulator=simulator,
        score_sell_priority_override=sell_priority,
        target_positions_override=sizing.TARGET_POSITIONS,
        target_gross_override=1.0,
    )
    return daily, actions, diagnostics


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered score-distance entry sizing research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    current_daily, current_actions = sizing.run_case(
        context,
        policy,
        sizing.BASELINE_COST,
        sizing.TOP_WEIGHT_MULTIPLIER,
        sizing.BOTTOM_WEIGHT_MULTIPLIER,
    )
    current_stress_daily, current_stress_actions = sizing.run_case(
        context,
        policy,
        sizing.STRESS_COST,
        sizing.TOP_WEIGHT_MULTIPLIER,
        sizing.BOTTOM_WEIGHT_MULTIPLIER,
    )
    distance_daily, distance_actions, diagnostics = run_score_distance(
        context, policy, sizing.BASELINE_COST
    )
    distance_stress_daily, distance_stress_actions, stress_diagnostics = (
        run_score_distance(context, policy, sizing.STRESS_COST)
    )
    repeat_daily, repeat_actions, repeat_diagnostics = run_score_distance(
        context, policy, sizing.BASELINE_COST
    )

    current = {
        "metrics_0_30pct": sizing.evaluate(current_daily, current_actions),
        "metrics_0_65pct": sizing.evaluate(
            current_stress_daily, current_stress_actions
        ),
    }
    distance = {
        "metrics_0_30pct": sizing.evaluate(distance_daily, distance_actions),
        "metrics_0_65pct": sizing.evaluate(
            distance_stress_daily, distance_stress_actions
        ),
    }
    delta = {
        cost: sizing.checkpoint_tools.metric_delta(
            distance[f"metrics_{cost}"], current[f"metrics_{cost}"]
        )
        for cost in ("0_30pct", "0_65pct")
    }
    deterministic = {
        "daily": round1.frame_hash(distance_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(distance_actions)
        == round1.frame_hash(repeat_actions),
        "diagnostics": diagnostics == repeat_diagnostics,
    }
    if not all(deterministic.values()):
        raise RuntimeError("score-distance entry sizing replay is not deterministic")

    profit_upgrade = bool(
        delta["0_30pct"]["cumulative_return"] > 0.0
        and delta["0_65pct"]["cumulative_return"] > 0.0
    )
    result = {
        "status": "score_distance_entry_sizing_complete_2026_not_opened",
        "research_question": (
            "Does using actual top10 score distances for new-entry sizing improve "
            "cost-adjusted profit over the frozen rank-only sizing?"
        ),
        "only_change": (
            "new-entry multipliers use demeaned top10 score distances, bounded "
            "within 0.90..1.10 and gross-neutral before execution"
        ),
        "parameter_search_count": 0,
        "portfolio_width_equalweight_and_investment_are_soft": True,
        "current_frozen_candidate": current,
        "score_distance_candidate": distance,
        "score_distance_minus_current": delta,
        "multiplier_diagnostics": diagnostics,
        "stress_multiplier_diagnostics": stress_diagnostics,
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
