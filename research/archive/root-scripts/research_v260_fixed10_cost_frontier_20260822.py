from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_cost_frontier_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
COST_LEVELS = (0.0, 0.0010, 0.0030, 0.0040, 0.0065, 0.0100, 0.0150)


def current_priority(context):
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    return strong, priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )


def run_fixed10(context, policy: dict, strong, priority_matrix, cost: float):
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        cost,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def run_production(context, cost: float):
    protocol = copy.deepcopy(context.protocol)
    protocol["execution"]["fixed_slippage_ratio"] = float(cost)
    return research_base.run_shell(
        context.harness,
        context.arrays,
        protocol,
        context.score,
        context.order,
        context.definition,
        "production_shell",
        round1.DEVELOPMENT_END,
        actions=True,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered cost-frontier diagnostic")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong, priority_matrix = current_priority(context)
    results = {}
    current_cache = {}
    production_cache = {}
    for cost in COST_LEVELS:
        key = f"{cost:.4f}"
        current_daily, current_actions = run_fixed10(
            context, policy, strong, priority_matrix, cost
        )
        production_daily, production_actions = run_production(context, cost)
        current_cache[key] = (current_daily, current_actions)
        production_cache[key] = (production_daily, production_actions)
        current_metrics = round1.evaluate_run(
            current_daily,
            current_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        )
        production_metrics = round1.evaluate_run(
            production_daily,
            production_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        )
        results[key] = {
            "fixed10": current_metrics,
            "production": production_metrics,
            "fixed10_minus_production": {
                metric: float(current_metrics[metric] - production_metrics[metric])
                for metric in (
                    "cumulative_return", "cagr", "sharpe", "max_drawdown",
                    "turnover_annualized", "average_invested_ratio",
                )
            },
        }

    expected_current = checkpoint["current_best_equalweight"]
    expected_production = checkpoint["production_baseline"]
    baseline_equivalence = {
        "fixed10": {
            key: bool(np.isclose(
                results["0.0030"]["fixed10"][key], expected_current[key],
                rtol=0.0, atol=1e-12,
            ))
            for key in (
                "cumulative_return", "cagr", "sharpe", "max_drawdown",
                "turnover_annualized", "average_invested_ratio",
            )
        },
        "production": {
            key: bool(np.isclose(
                results["0.0030"]["production"][key], expected_production[key],
                rtol=0.0, atol=1e-12,
            ))
            for key in (
                "cumulative_return", "cagr", "sharpe", "max_drawdown",
                "turnover_annualized", "average_invested_ratio",
            )
        },
    }
    if not all(baseline_equivalence["fixed10"].values()) or not all(
        baseline_equivalence["production"].values()
    ):
        raise RuntimeError("cost-frontier baseline drifted")
    repeat_current = run_fixed10(context, policy, strong, priority_matrix, 0.0030)
    repeat_production = run_production(context, 0.0030)
    deterministic = {
        "fixed10_daily": round1.frame_hash(current_cache["0.0030"][0])
        == round1.frame_hash(repeat_current[0]),
        "fixed10_actions": round1.frame_hash(current_cache["0.0030"][1])
        == round1.frame_hash(repeat_current[1]),
        "production_daily": round1.frame_hash(production_cache["0.0030"][0])
        == round1.frame_hash(repeat_production[0]),
        "production_actions": round1.frame_hash(production_cache["0.0030"][1])
        == round1.frame_hash(repeat_production[1]),
    }
    if not all(deterministic.values()):
        raise RuntimeError("cost-frontier replay failed")
    fixed10_positive_costs = [
        float(key)
        for key, item in results.items()
        if item["fixed10"]["cumulative_return"] > 0.0
    ]
    fixed10_outperformance_costs = [
        float(key)
        for key, item in results.items()
        if item["fixed10_minus_production"]["cumulative_return"] > 0.0
    ]
    result = {
        "status": "cost_frontier_complete_2026_not_opened",
        "method": (
            "replay current fixed10 and production at identical deterministic "
            "one-way slippage assumptions; no rule selection"
        ),
        "results": results,
        "highest_tested_cost_with_positive_fixed10_return": max(fixed10_positive_costs),
        "highest_tested_equal_cost_with_fixed10_return_above_production": max(
            fixed10_outperformance_costs
        ),
        "baseline_equivalence": baseline_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "cost_frontier.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
