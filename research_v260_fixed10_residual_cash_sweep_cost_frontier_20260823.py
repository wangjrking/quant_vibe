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
    "strategy_agent_v260_fixed10_residual_cash_sweep_cost_frontier_20260823"
)
COSTS = (0.0030, 0.0065, 0.0100, 0.0150)


def summarize_frontier(results: dict[str, dict]) -> dict:
    if not results:
        raise ValueError("cost frontier cannot be empty")
    ordered = sorted(results, key=float)
    positive = [
        key
        for key in ordered
        if results[key]["sweep_minus_current"]["cumulative_return"] > 0.0
    ]
    return {
        "tested_cost_count": len(ordered),
        "all_tested_costs_positive": len(positive) == len(ordered),
        "highest_tested_positive_cost": float(positive[-1]) if positive else None,
        "first_tested_nonpositive_cost": next(
            (
                float(key)
                for key in ordered
                if results[key]["sweep_minus_current"]["cumulative_return"] <= 0.0
            ),
            None,
        ),
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-sweep cost frontier")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    results = {}
    for cost in COSTS:
        current_daily, current_actions = sizing.run_case(
            context,
            policy,
            cost,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        sweep_daily, sweep_actions = sweep.run_with_runtime(
            context, policy, cost, sweep=True, rank_sizing=True
        )
        current = sizing.evaluate(current_daily, current_actions)
        candidate = sizing.evaluate(sweep_daily, sweep_actions)
        results[f"{cost:.4f}"] = {
            "current": current,
            "residual_cash_sweep": candidate,
            "sweep_minus_current": sizing.checkpoint_tools.metric_delta(
                candidate, current
            ),
        }
    summary = summarize_frontier(results)
    result = {
        "status": "residual_cash_sweep_cost_frontier_complete_2026_not_opened",
        "role": "cost_sensitivity_not_parameter_selection",
        "costs": list(COSTS),
        "results": results,
        "summary": summary,
        "interpretation": (
            "the cost frontier discloses how much extra execution friction the "
            "cash-sweep profit edge can absorb; it does not select a cost assumption"
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "cost_frontier.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": summary,
                "cumulative_return_deltas": {
                    key: value["sweep_minus_current"]["cumulative_return"]
                    for key, value in results.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
