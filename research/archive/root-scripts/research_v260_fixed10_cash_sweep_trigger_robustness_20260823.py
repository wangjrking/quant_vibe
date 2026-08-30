from __future__ import annotations

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
import research_v260_fixed10_residual_cash_sweep_capital_scale_20260823 as scale


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_cash_sweep_trigger_robustness_20260823"
)
COSTS = {"0_30pct": 0.003, "0_40pct": 0.004, "0_65pct": 0.0065}


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-sweep trigger robustness")
    policy = checkpoint["selected_policy"]
    base_context = sizing.width_tools.harness.load_context(policy)
    results = {}
    for initial_cash in scale.CAPITALS:
        context = scale.context_with_initial_cash(base_context, initial_cash)
        capital_result = {}
        for cost_name, cost in COSTS.items():
            arms = {}
            for trigger in ("any_trade", "buy_trade"):
                daily, actions = sweep.run_with_runtime(
                    context,
                    policy,
                    cost,
                    sweep=True,
                    sweep_trigger=trigger,
                )
                arms[trigger] = sizing.evaluate(daily, actions)
            capital_result[cost_name] = {
                **arms,
                "buy_trade_minus_any_trade": sizing.checkpoint_tools.metric_delta(
                    arms["buy_trade"], arms["any_trade"]
                ),
            }
        results[str(int(initial_cash))] = capital_result

    summary = {
        cost_name: {
            capital: values[cost_name]["buy_trade_minus_any_trade"][
                "cumulative_return"
            ]
            for capital, values in results.items()
        }
        for cost_name in COSTS
    }
    result = {
        "status": "cash_sweep_trigger_robustness_complete_2026_not_opened",
        "single_mechanism": "buy-day-only trigger versus any-trade trigger",
        "capital_scales": list(scale.CAPITALS),
        "cost_scenarios": COSTS,
        "results": results,
        "cumulative_return_deltas": summary,
        "interpretation": (
            "The baseline-cost result is the profit selection input. Capital and "
            "higher-cost results disclose implementation sensitivity and are not "
            "additional rejection gates."
        ),
        "data_access": base_context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "robustness.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cumulative_return_deltas": summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
