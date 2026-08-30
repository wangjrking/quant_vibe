from __future__ import annotations

import copy
import json
import math
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
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_core_parameter_neighborhood_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
YEARS = ("2022", "2023", "2024", "2025")
DIMENSIONS = {
    "sell_score_below": (0.80, 0.85, 0.90),
    "minimum_hold_days": (3, 4, 5),
    "max_hold_renewal_score": (0.75, 0.80, 0.85),
    "replacement_advantage": (0.03, 0.05, 0.07),
    "portfolio_rebalance_interval_days": (10, 20, 40),
    "maintenance_topup_score": (0.75, 0.80, 0.85),
}
CURRENT = {
    "sell_score_below": 0.85,
    "minimum_hold_days": 4,
    "max_hold_renewal_score": 0.80,
    "replacement_advantage": 0.05,
    "portfolio_rebalance_interval_days": 20,
    "maintenance_topup_score": 0.80,
}


def priority_matrix(context, strong: np.ndarray) -> np.ndarray:
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    return priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )


def run_case(context, policy: dict, dimension: str, value: float):
    strong = regime.strong_market_mask(context.score, context.protocol)
    kwargs = {
        "pressure_trigger_override": trigger.pressure_trigger_schedule(strong),
        "score_sell_priority_override": priority_matrix(context, strong),
    }
    if dimension == "sell_score_below":
        kwargs["sell_score_below_override"] = float(value)
    elif dimension == "minimum_hold_days":
        kwargs["min_hold_days_override"] = int(value)
    elif dimension == "max_hold_renewal_score":
        kwargs["max_hold_renewal_score_override"] = float(value)
    elif dimension == "replacement_advantage":
        kwargs["replacement_advantage_override"] = float(value)
    elif dimension == "portfolio_rebalance_interval_days":
        kwargs["portfolio_rebalance_active_override"] = cadence.rebalance_schedule(
            len(context.arrays["dates"]), int(value)
        )
    elif dimension == "maintenance_topup_score":
        kwargs["maintenance_buy_block_mask_override"] = (
            ~np.isfinite(context.score) | (context.score < float(value))
        )
    else:
        raise ValueError(f"unknown dimension: {dimension}")
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        **kwargs,
    )


def evaluate(daily, actions) -> dict:
    return round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )


def geometric_mean(annual_returns: dict, excluded: str) -> float:
    retained = [year for year in YEARS if year != excluded]
    wealth = math.prod(1.0 + float(annual_returns[year]) for year in retained)
    return float(math.pow(wealth, 1.0 / len(retained)) - 1.0)


def value_id(value: float) -> str:
    return f"{float(value):.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered current core-parameter neighborhood")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    results = {}
    hashes = {}
    for dimension, values in DIMENSIONS.items():
        dimension_results = {}
        for value in values:
            daily, actions = run_case(context, policy, dimension, value)
            key = value_id(value)
            dimension_results[key] = evaluate(daily, actions)
            hashes[f"{dimension}:{key}"] = {
                "daily": round1.frame_hash(daily),
                "actions": round1.frame_hash(actions),
            }
        results[dimension] = dimension_results

    equivalence = {}
    expected = checkpoint["current_best_equalweight"]
    for dimension, current in CURRENT.items():
        metrics = results[dimension][value_id(current)]
        equivalence[dimension] = all(
            np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12)
            for key in (
                "cumulative_return", "cagr", "sharpe", "max_drawdown",
                "turnover_annualized", "average_invested_ratio",
            )
        )
    if not all(equivalence.values()):
        raise RuntimeError("current core-parameter neighborhood baseline drifted")

    leave_one_year_out = {}
    for dimension, values in DIMENSIONS.items():
        rows = results[dimension]
        current_id = value_id(CURRENT[dimension])
        folds = {}
        for excluded in YEARS:
            ranking = sorted(
                rows,
                key=lambda key: (
                    geometric_mean(rows[key]["annual_returns"], excluded), key
                ),
                reverse=True,
            )
            folds[excluded] = {
                "ranking": ranking,
                "current_rank": int(ranking.index(current_id) + 1),
                "geometric_mean_return": {
                    key: geometric_mean(rows[key]["annual_returns"], excluded)
                    for key in ranking
                },
            }
        leave_one_year_out[dimension] = {
            "current_value": current_id,
            "folds": folds,
            "win_fraction": float(np.mean([
                fold["current_rank"] == 1 for fold in folds.values()
            ])),
            "top2_fraction": float(np.mean([
                fold["current_rank"] <= 2 for fold in folds.values()
            ])),
        }

    repeat_daily, repeat_actions = run_case(
        context, policy, "sell_score_below", CURRENT["sell_score_below"]
    )
    deterministic = {
        "daily": round1.frame_hash(repeat_daily)
        == hashes["sell_score_below:0.8500"]["daily"],
        "actions": round1.frame_hash(repeat_actions)
        == hashes["sell_score_below:0.8500"]["actions"],
    }
    if not all(deterministic.values()):
        raise RuntimeError("current core-parameter neighborhood replay failed")

    result = {
        "status": "current_core_parameter_neighborhood_complete_2026_not_opened",
        "method": "one-dimensional neighbors replayed on the current final candidate",
        "current_values": CURRENT,
        "results": results,
        "leave_one_year_out": leave_one_year_out,
        "all_current_values_top2_in_every_fold": bool(all(
            item["top2_fraction"] == 1.0 for item in leave_one_year_out.values()
        )),
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "current_core_parameter_neighborhood.json", result)
    print(json.dumps({
        "status": result["status"],
        "leave_one_year_out": {
            key: {
                "current": value["current_value"],
                "win_fraction": value["win_fraction"],
                "top2_fraction": value["top2_fraction"],
            }
            for key, value in leave_one_year_out.items()
        },
        "all_current_values_top2_in_every_fold": result[
            "all_current_values_top2_in_every_fold"
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
