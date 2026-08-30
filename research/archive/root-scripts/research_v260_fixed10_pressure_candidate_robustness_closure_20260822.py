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
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_candidate_robustness_closure_20260822"
)
PRESSURE_CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
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


def run_at_cost(context, policy, cost):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
    )
    return round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=float(cost),
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=cadence.rebalance_schedule(
            len(context.arrays["dates"]),
            int(policy["portfolio_rebalance_interval_days"]),
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=context.empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=policy["score_sell_pressure_trigger"],
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
    )


def metric_delta(candidate, reference):
    return {
        "cumulative_return": float(
            candidate["cumulative_return"] - reference["cumulative_return"]
        ),
        "cagr": float(candidate["cagr"] - reference["cagr"]),
        "sharpe": float(candidate["sharpe"] - reference["sharpe"]),
        "max_drawdown": float(
            candidate["max_drawdown"] - reference["max_drawdown"]
        ),
        "turnover_annualized": float(
            candidate["turnover_annualized"] - reference["turnover_annualized"]
        ),
        "average_invested_ratio": float(
            candidate["average_invested_ratio"]
            - reference["average_invested_ratio"]
        ),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pressure = json.loads(PRESSURE_CHECKPOINT.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_CHECKPOINT.read_text(encoding="utf-8"))
    if pressure["validation_2026_opened"] or original["validation_2026_opened"]:
        raise PermissionError("2026 data entered pressure robustness closure")
    policy = copy.deepcopy(pressure["selected_policy"])
    context = harness.load_context(policy)
    cost_results, cache = {}, {}
    for cost in COST_LEVELS:
        daily, actions = run_at_cost(context, policy, cost)
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
            (
                research_base.FIRST_BUY
                if offset == 0
                else robustness.window_start_date(context.arrays, offset)
            ),
            round1.DEVELOPMENT_END,
        )
        for offset in START_OFFSETS
    }
    repeat_daily, repeat_actions = run_at_cost(context, policy, 0.0030)
    deterministic = {
        "daily": round1.frame_hash(baseline_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(baseline_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure robustness closure replay failed")
    candidate = cost_results["0.0030"]
    result = {
        "status": "pressure_candidate_robustness_closed_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selected_policy": policy,
        "cost_levels": list(COST_LEVELS),
        "cost_metrics": cost_results,
        "start_offset_metrics": start_offset_metrics,
        "minimum_start_offset_cagr": float(
            min(item["cagr"] for item in start_offset_metrics.values())
        ),
        "minimum_cost_level_cumulative_return": float(
            min(item["cumulative_return"] for item in cost_results.values())
        ),
        "candidate_minus_original_equalweight": metric_delta(
            candidate, original["current_best_equalweight"]
        ),
        "candidate_minus_production": metric_delta(
            candidate, original["production_baseline"]
        ),
        "all_calendar_years_positive": bool(
            min(candidate["annual_returns"].values()) > 0.0
        ),
        "all_cost_levels_profitable": bool(
            min(item["cumulative_return"] for item in cost_results.values()) > 0.0
        ),
        "all_start_offsets_profitable": bool(
            min(item["cumulative_return"] for item in start_offset_metrics.values())
            > 0.0
        ),
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "source_manifests": context.manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "robustness_closure.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
