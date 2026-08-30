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
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822"
)
COST_LEVELS = (0.0030, 0.0040, 0.0065)
START_OFFSETS = (0, 5, 20, 60)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source = json.loads(sizing.CHECKPOINT.read_text(encoding="utf-8"))
    if source.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered rank sizing checkpoint")
    policy = copy.deepcopy(source["selected_policy"])
    policy["entry_rank_sizing"] = {
        "application": "new_entries_only",
        "ranking": "global_existing_frozen_score_order_before_execution_refill",
        "positions": sizing.TARGET_POSITIONS,
        "top_multiplier": sizing.TOP_WEIGHT_MULTIPLIER,
        "bottom_multiplier": sizing.BOTTOM_WEIGHT_MULTIPLIER,
        "schedule": "linear_descending",
        "reference_top10_multiplier_sum": 10.0,
        "refill_outside_global_top10_multiplier": 1.0,
        "actual_entry_batch_gross_neutral": False,
        "equalweight_is_soft_reference": True,
    }
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 checkpoint boundary drift")

    cost_metrics, frames = {}, {}
    for cost in COST_LEVELS:
        daily, actions = sizing.run_case(
            context,
            policy,
            cost,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        key = f"{cost:.4f}"
        cost_metrics[key] = sizing.evaluate(daily, actions)
        frames[key] = (daily, actions)

    daily, actions = frames["0.0030"]
    start_offset_metrics = {
        str(offset): round1.evaluate_run(
            daily,
            actions,
            research_base.FIRST_BUY
            if offset == 0
            else robustness.window_start_date(context.arrays, offset),
            round1.DEVELOPMENT_END,
            target_positions=sizing.TARGET_POSITIONS,
        )
        for offset in START_OFFSETS
    }
    repeat_daily, repeat_actions = sizing.run_case(
        context,
        policy,
        0.0030,
        sizing.TOP_WEIGHT_MULTIPLIER,
        sizing.BOTTOM_WEIGHT_MULTIPLIER,
    )
    deterministic = {
        "daily": round1.frame_hash(daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("rank sizing checkpoint replay failed")

    equalweight = source["current_best_equalweight"]
    candidate = cost_metrics["0.0030"]
    safety_checks = {
        "pre2026_boundary": context.access["logical_max_date"]
        == round1.DEVELOPMENT_END,
        "no_bj": context.access["bj_stock_count"] == 0,
        "finite_metrics": all(
            isinstance(candidate[key], (int, float))
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
        "deterministic": all(deterministic.values()),
        "reference_top10_multiplier_sum_is_10": abs(
            sizing.entry_rank_multipliers(
                context.order[:1], sizing.TARGET_POSITIONS
            )[0, context.order[0, : sizing.TARGET_POSITIONS]].sum()
            - sizing.TARGET_POSITIONS
        )
        < 1e-12,
    }
    if not all(safety_checks.values()):
        raise RuntimeError("rank sizing checkpoint safety check failed")

    result = {
        "status": "rank_sizing_profit_candidate_frozen_pre2026_validation_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "selected_candidate": "fixed10_global_rank11_to_9_entry",
        "selected_policy": policy,
        "selection_reason": (
            "global score top10 new-entry sizing has higher pre-2026 net cumulative "
            "return than equalweight at both 0.30% "
            "and 0.65% costs; position count, equalweight and full investment are "
            "diagnostics rather than rejection gates"
        ),
        "metrics_0_30pct": candidate,
        "cost_metrics": cost_metrics,
        "start_offset_metrics": start_offset_metrics,
        "candidate_minus_equalweight": sizing.checkpoint_tools.metric_delta(
            candidate, equalweight
        ),
        "safety_checks": safety_checks,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "pre2026_checkpoint.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate": result["selected_candidate"],
                "cumulative_return": candidate["cumulative_return"],
                "cagr": candidate["cagr"],
                "sharpe": candidate["sharpe"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
