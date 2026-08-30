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

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_extra_age_regime_decomposition_20260822 as decomposition
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age_checkpoint_20260822"
)
CURRENT_CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_extra_age_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
ORIGINAL_CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
COST_LEVELS = (0.0030, 0.0040, 0.0065)
START_OFFSETS = (0, 5, 20, 60)


def metric_delta(candidate: dict, reference: dict) -> dict:
    return {
        key: float(candidate[key] - reference[key])
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    current = json.loads(CURRENT_CHECKPOINT.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_CHECKPOINT.read_text(encoding="utf-8"))
    if current["validation_2026_opened"] or original["validation_2026_opened"]:
        raise PermissionError("2026 data entered weak-market age robustness")
    policy = copy.deepcopy(current["selected_policy"])
    policy["score_sell_pressure_extra_min_age_rule"] = "4_strong_market_8_weak_market"
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    schedule = decomposition.age_schedule(strong, 4, 8)

    cost_results, cache = {}, {}
    for cost in COST_LEVELS:
        daily, actions = age_guard.run_policy_at_cost(context, policy, schedule, cost)
        key = f"{cost:.4f}"
        cost_results[key] = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        cache[key] = (daily, actions)

    baseline_daily, baseline_actions = cache["0.0030"]
    start_offset_metrics = {
        str(offset): round1.evaluate_run(
            baseline_daily,
            baseline_actions,
            research_base.FIRST_BUY if offset == 0 else robustness.window_start_date(context.arrays, offset),
            round1.DEVELOPMENT_END,
        )
        for offset in START_OFFSETS
    }
    repeat_daily, repeat_actions = age_guard.run_policy_at_cost(
        context, policy, schedule, 0.0030
    )
    deterministic = {
        "daily": round1.frame_hash(baseline_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(baseline_actions) == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("weak-market age replay failed")

    candidate = cost_results["0.0030"]
    prior = current["current_best_equalweight"]
    current_costs = current["robustness"]["cost_metrics"]
    gates = {
        "cumulative_return_improved": candidate["cumulative_return"] > prior["cumulative_return"],
        "cagr_improved": candidate["cagr"] > prior["cagr"],
        "sharpe_improved": candidate["sharpe"] > prior["sharpe"],
        "drawdown_not_worse": candidate["max_drawdown"] <= prior["max_drawdown"],
        "all_cost_levels_not_worse": all(
            cost_results[key]["cumulative_return"] >= current_costs[key]["cumulative_return"]
            for key in cost_results
        ),
        "all_years_positive": min(candidate["annual_returns"].values()) > 0.0,
        "all_start_offsets_profitable": min(
            item["cumulative_return"] for item in start_offset_metrics.values()
        ) > 0.0,
        "full_10_positions": candidate["full_10_position_ratio"] == 1.0,
    }
    if not all(gates.values()):
        raise RuntimeError("weak-market age candidate did not pass robustness gates")
    result = {
        "status": "weak_market_pressure_age_candidate_accepted_pre2026_validation_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selected_policy": policy,
        "strong_market_days": int(np.sum(strong)),
        "weak_market_days": int(np.sum(~strong)),
        "cost_metrics": cost_results,
        "start_offset_metrics": start_offset_metrics,
        "minimum_start_offset_cagr": float(min(item["cagr"] for item in start_offset_metrics.values())),
        "candidate_minus_previous_best": metric_delta(candidate, prior),
        "candidate_minus_original_equalweight": metric_delta(candidate, original["current_best_equalweight"]),
        "candidate_minus_production": metric_delta(candidate, original["production_baseline"]),
        "acceptance_gates": gates,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "source_manifests": context.manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    checkpoint = {
        "status": "pre2026_development_checkpoint_validation_unopened",
        "source_strategy": context.rules["strategy_id"],
        "selected_candidate": "weak_market_pressure_extra_sell_min_age8",
        "selected_policy": policy,
        "current_best_equalweight": candidate,
        "previous_best_equalweight": prior,
        "production_baseline": original["production_baseline"],
        "robustness": {
            "cost_metrics": cost_results,
            "start_offset_metrics": start_offset_metrics,
            "acceptance_gates": gates,
            "deterministic_replay": deterministic,
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "robustness_closure.json", result)
    round1.atomic_json(OUTPUT_ROOT / "pre2026_checkpoint.json", checkpoint)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
