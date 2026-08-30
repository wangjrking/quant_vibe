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
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_soft_width_profit_optimization_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
WIDTHS = (5, 7, 8, 9, 10, 11, 12, 15, 20)
BASELINE_COST = 0.0030
STRESS_COST = 0.0065
MAX_DRAWDOWN = 0.35


def build_overrides(context, policy: dict):
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    return extra_age, sell_priority


def run_width(context, policy: dict, width: int, cost: float):
    extra_age, sell_priority = build_overrides(context, policy)
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        score_sell_priority_override=sell_priority,
        target_positions_override=width,
    )


def metrics(daily, actions, width: int) -> dict:
    return round1.evaluate_run(
        daily,
        actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
        target_positions=width,
    )


def profit_eligible(baseline: dict, stress: dict) -> bool:
    return bool(
        baseline["cumulative_return"] > 0.0
        and stress["cumulative_return"] > 0.0
        and baseline["sharpe"] > 0.0
        and baseline["max_drawdown"] <= MAX_DRAWDOWN
    )


def choose_width(results: dict[str, dict]) -> str:
    eligible = [
        key
        for key, value in results.items()
        if value["profit_eligible"]
    ]
    if not eligible:
        raise RuntimeError("no width satisfies the basic profit and risk floors")
    return max(
        eligible,
        key=lambda key: (
            results[key]["metrics_0_30pct"]["cumulative_return"],
            results[key]["metrics_0_30pct"]["sharpe"],
            -results[key]["metrics_0_30pct"]["max_drawdown"],
        ),
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered soft-width optimization")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    results: dict[str, dict] = {}
    baseline_frames = {}
    for width in WIDTHS:
        daily, actions = run_width(context, policy, width, BASELINE_COST)
        stress_daily, stress_actions = run_width(
            context, policy, width, STRESS_COST
        )
        baseline = metrics(daily, actions, width)
        stress = metrics(stress_daily, stress_actions, width)
        results[str(width)] = {
            "target_positions": width,
            "target_weight": 1.0 / width,
            "metrics_0_30pct": baseline,
            "metrics_0_65pct": stress,
            "profit_eligible": profit_eligible(baseline, stress),
        }
        baseline_frames[str(width)] = (daily, actions)

    selected = choose_width(results)
    selected_width = int(selected)
    repeat_daily, repeat_actions = run_width(
        context, policy, selected_width, BASELINE_COST
    )
    reference_daily, reference_actions = baseline_frames[selected]
    deterministic = {
        "daily": round1.frame_hash(reference_daily)
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(reference_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("selected width replay is not deterministic")

    current = results["10"]["metrics_0_30pct"]
    winner = results[selected]["metrics_0_30pct"]
    result = {
        "status": "soft_equalweight_width_profit_optimization_complete_2026_not_opened",
        "selection_rule": "maximize_pre2026_cumulative_return_subject_to_basic_risk_floors",
        "soft_diagnostics_only": [
            "exactly_10_positions",
            "exact_10_percent_target_weight",
            "minimum_invested_ratio",
        ],
        "hard_floors": {
            "baseline_return_positive": True,
            "stress_0_65_return_positive": True,
            "sharpe_positive": True,
            "max_drawdown_at_most": MAX_DRAWDOWN,
            "deterministic_replay": True,
        },
        "candidates": results,
        "selected_width": selected_width,
        "selected_weight": 1.0 / selected_width,
        "selected_minus_fixed10": round1.numeric_delta(winner, current),
        "deterministic_replay": deterministic,
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "soft_width_results.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
