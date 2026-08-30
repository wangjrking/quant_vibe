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

import research_v260_fixed10_current_core_parameter_neighborhood_20260822 as neighborhood
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_capital_scale_robustness_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CAPITALS = (350_000.0, 700_000.0, 1_400_000.0)


def context_with_initial_cash(context, initial_cash: float):
    value = float(initial_cash)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("initial cash must be finite and positive")
    protocol = copy.deepcopy(context.protocol)
    protocol["execution"]["initial_cash"] = value
    return replace(context, protocol=protocol)


def metric_spread(results: dict, key: str) -> dict:
    values = np.asarray(
        [result["metrics"][key] for result in results.values()], dtype=np.float64
    )
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered capital-scale robustness")

    policy = copy.deepcopy(checkpoint["selected_policy"])
    base_context = harness.load_context(policy)
    results = {}
    for initial_cash in CAPITALS:
        context = context_with_initial_cash(base_context, initial_cash)
        daily, actions = neighborhood.run_case(
            context, policy, "sell_score_below", 0.85
        )
        metrics = neighborhood.evaluate(daily, actions)
        results[str(int(initial_cash))] = {
            "initial_cash": initial_cash,
            "metrics": metrics,
            "daily_hash": round1.frame_hash(daily),
            "action_hash": round1.frame_hash(actions),
        }

    reference = results["700000"]["metrics"]
    expected = checkpoint["current_best_equalweight"]
    reference_equivalent = bool(all(
        np.isclose(reference[key], expected[key], rtol=0.0, atol=1e-12)
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    ))
    if not reference_equivalent:
        raise RuntimeError("700k capital replay drifted")

    all_exactly10 = bool(all(
        value["metrics"]["full_10_position_ratio"] == 1.0
        for value in results.values()
    ))
    all_years_positive = bool(all(
        min(value["metrics"]["annual_returns"].values()) > 0.0
        for value in results.values()
    ))
    result = {
        "status": "capital_scale_robustness_complete_2026_not_opened",
        "single_variable": "initial_cash_only",
        "results": results,
        "summary": {
            "all_capitals_exactly10": all_exactly10,
            "all_capitals_all_years_positive": all_years_positive,
            "cagr": metric_spread(results, "cagr"),
            "sharpe": metric_spread(results, "sharpe"),
            "max_drawdown": metric_spread(results, "max_drawdown"),
            "average_invested_ratio": metric_spread(
                results, "average_invested_ratio"
            ),
            "capital_scale_robust": bool(
                all_exactly10
                and all_years_positive
                and metric_spread(results, "cagr")["range"] <= 0.02
                and metric_spread(results, "sharpe")["range"] <= 0.05
            ),
        },
        "reference_700k_equivalence": reference_equivalent,
        "data_access": base_context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "capital_scale_robustness.json", result)
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
