from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


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
    "strategy_agent_v260_fixed10_residual_cash_sweep_capital_scale_20260823"
)
CAPITALS = (350_000.0, 700_000.0, 1_400_000.0, 7_000_000.0)


def context_with_initial_cash(context, initial_cash: float):
    if not np.isfinite(initial_cash) or initial_cash <= 0.0:
        raise ValueError("initial cash must be finite and positive")
    protocol = copy.deepcopy(context.protocol)
    protocol["execution"]["initial_cash"] = float(initial_cash)
    return replace(context, protocol=protocol)


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-sweep capital-scale research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    base_context = sizing.width_tools.harness.load_context(policy)
    results = {}
    for initial_cash in CAPITALS:
        context = context_with_initial_cash(base_context, initial_cash)
        capital_result = {}
        for cost_name, cost in (
            ("0_30pct", sizing.BASELINE_COST),
            ("0_65pct", sizing.STRESS_COST),
        ):
            current_daily, current_actions = sizing.run_case(
                context,
                policy,
                cost,
                sizing.TOP_WEIGHT_MULTIPLIER,
                sizing.BOTTOM_WEIGHT_MULTIPLIER,
            )
            sweep_daily, sweep_actions = sweep.run_with_runtime(
                context, policy, cost, sweep=True
            )
            current_metrics = sizing.evaluate(current_daily, current_actions)
            sweep_metrics = sizing.evaluate(sweep_daily, sweep_actions)
            capital_result[cost_name] = {
                "current": current_metrics,
                "residual_cash_sweep": sweep_metrics,
                "sweep_minus_current": sizing.checkpoint_tools.metric_delta(
                    sweep_metrics, current_metrics
                ),
            }
        results[str(int(initial_cash))] = capital_result

    baseline_return_positive = {
        capital: bool(
            value["0_30pct"]["sweep_minus_current"]["cumulative_return"] > 0.0
        )
        for capital, value in results.items()
    }
    stress_return_positive = {
        capital: bool(
            value["0_65pct"]["sweep_minus_current"]["cumulative_return"] > 0.0
        )
        for capital, value in results.items()
    }
    result = {
        "status": "residual_cash_sweep_capital_scale_complete_2026_not_opened",
        "single_variable": "initial_cash_only",
        "capital_scales": list(CAPITALS),
        "results": results,
        "summary": {
            "baseline_return_positive": baseline_return_positive,
            "stress_return_positive": stress_return_positive,
            "positive_at_all_capitals_and_costs": bool(
                all(baseline_return_positive.values())
                and all(stress_return_positive.values())
            ),
        },
        "interpretation": (
            "capital scale is robustness evidence only; it is not a new hard gate "
            "and no capital value is selected or tuned"
        ),
        "data_access": base_context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "capital_scale.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "positive_at_all_capitals_and_costs": result["summary"][
                    "positive_at_all_capitals_and_costs"
                ],
                "baseline_return_delta": {
                    capital: value["0_30pct"]["sweep_minus_current"][
                        "cumulative_return"
                    ]
                    for capital, value in results.items()
                },
                "stress_return_delta": {
                    capital: value["0_65pct"]["sweep_minus_current"][
                        "cumulative_return"
                    ]
                    for capital, value in results.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
