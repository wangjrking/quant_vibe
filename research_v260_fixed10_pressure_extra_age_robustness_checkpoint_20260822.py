from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_extra_age_checkpoint_20260822"
)
CURRENT_CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_regime_pressure_checkpoint_20260822/"
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
        "cumulative_return": float(candidate["cumulative_return"] - reference["cumulative_return"]),
        "cagr": float(candidate["cagr"] - reference["cagr"]),
        "sharpe": float(candidate["sharpe"] - reference["sharpe"]),
        "max_drawdown": float(candidate["max_drawdown"] - reference["max_drawdown"]),
        "turnover_annualized": float(candidate["turnover_annualized"] - reference["turnover_annualized"]),
        "average_invested_ratio": float(candidate["average_invested_ratio"] - reference["average_invested_ratio"]),
    }


def acceptance_gates(candidate, current, cost_results, current_costs, start_results):
    return {
        "cumulative_return_improved": candidate["cumulative_return"] > current["cumulative_return"],
        "cagr_improved": candidate["cagr"] > current["cagr"],
        "sharpe_improved": candidate["sharpe"] > current["sharpe"],
        "drawdown_improved": candidate["max_drawdown"] < current["max_drawdown"],
        "all_cost_levels_not_worse": all(
            cost_results[key]["cumulative_return"] >= current_costs[key]["cumulative_return"]
            for key in cost_results
        ),
        "all_years_positive": min(candidate["annual_returns"].values()) > 0.0,
        "all_start_offsets_profitable": min(
            item["cumulative_return"] for item in start_results.values()
        ) > 0.0,
        "full_10_positions": candidate["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    current = json.loads(CURRENT_CHECKPOINT.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_CHECKPOINT.read_text(encoding="utf-8"))
    if current["validation_2026_opened"] or original["validation_2026_opened"]:
        raise PermissionError("2026 data entered extra-age robustness")
    policy = copy.deepcopy(current["selected_policy"])
    policy["score_sell_pressure_extra_min_age"] = 8
    context = harness.load_context(policy)

    cost_results, cache = {}, {}
    for cost in COST_LEVELS:
        daily, actions = age_guard.run_policy_at_cost(context, policy, 8, cost)
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
        context, policy, 8, 0.0030
    )
    deterministic = {
        "daily": round1.frame_hash(baseline_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(baseline_actions) == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("extra-age robustness replay failed")

    candidate = cost_results["0.0030"]
    prior = current["current_best_equalweight"]
    gates = acceptance_gates(
        candidate,
        prior,
        cost_results,
        current["robustness"]["cost_metrics"],
        start_offset_metrics,
    )
    if not all(gates.values()):
        raise RuntimeError("extra-age candidate did not pass robustness gates")
    result = {
        "status": "pressure_extra_age_candidate_accepted_pre2026_validation_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selected_policy": policy,
        "cost_levels": list(COST_LEVELS),
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
        "selected_candidate": "pressure_extra_sell_min_age8",
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
