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
import research_v260_fixed10_hold_robustness_20260822 as robustness
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
    "strategy_agent_v260_fixed10_strong_market_renewal_threshold_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
COST_LEVELS = (0.0030, 0.0040, 0.0065)


def run(context, policy: dict, strong, priority_matrix, renewal_schedule, cost: float):
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        cost,
        max_hold_renewal_score_override=renewal_schedule,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered renewal-threshold development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    thresholds = {
        "current_080_all_markets": 0.80,
        "control_strong0775_weak080": 0.775,
        "candidate_strong075_weak080": 0.75,
        "control_strong070_weak080": 0.70,
    }
    schedules = {
        candidate_id: np.where(strong, strong_threshold, 0.80).astype(np.float64)
        for candidate_id, strong_threshold in thresholds.items()
    }
    results, cache = {}, {}
    for candidate_id, schedule in schedules.items():
        cost_metrics = {}
        for cost in COST_LEVELS:
            daily, actions = run(
                context, policy, strong, priority_matrix, schedule, cost
            )
            cost_metrics[f"{cost:.4f}"] = round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            )
            if cost == 0.0030:
                cache[candidate_id] = (daily, actions)
        daily, actions = cache[candidate_id]
        results[candidate_id] = {
            "strong_market_renewal_score": thresholds[candidate_id],
            "weak_market_renewal_score": 0.80,
            "cost_metrics": cost_metrics,
            "train_2022_2024": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025": round1.evaluate_run(
                daily, actions, "20250102", round1.DEVELOPMENT_END
            ),
            "start_offset_metrics": {
                str(offset): round1.evaluate_run(
                    daily,
                    actions,
                    research_base.FIRST_BUY
                    if offset == 0
                    else robustness.window_start_date(context.arrays, offset),
                    round1.DEVELOPMENT_END,
                )
                for offset in (0, 5, 20, 60)
            },
        }

    baseline_id = "current_080_all_markets"
    candidate_id = "candidate_strong075_weak080"
    baseline, candidate = results[baseline_id], results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    equivalent = {
        key: bool(np.isclose(
            baseline["cost_metrics"]["0.0030"][key], expected[key],
            rtol=0.0, atol=1e-12,
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalent.values()):
        raise RuntimeError("renewal-threshold baseline drifted")
    current = candidate["cost_metrics"]["0.0030"]
    reference = baseline["cost_metrics"]["0.0030"]
    gates = {
        "training_cagr_improved": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "training_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "full_period_cumulative_not_worse": current["cumulative_return"]
        >= reference["cumulative_return"],
        "full_period_sharpe_not_worse": current["sharpe"] >= reference["sharpe"],
        "drawdown_not_worse": current["max_drawdown"] <= reference["max_drawdown"],
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["cost_metrics"]["0.0065"]["cumulative_return"]
        >= baseline["cost_metrics"]["0.0065"]["cumulative_return"],
        "turnover_not_worse": current["turnover_annualized"]
        <= reference["turnover_annualized"],
        "both_neighbor_controls_improve_training_sharpe": all(
            results[control]["train_2022_2024"]["sharpe"]
            > baseline["train_2022_2024"]["sharpe"]
            for control in (
                "control_strong0775_weak080",
                "control_strong070_weak080",
            )
        ),
        "all_start_offsets_profitable": min(
            item["cumulative_return"]
            for item in candidate["start_offset_metrics"].values()
        ) > 0.0,
        "all_years_positive": min(current["annual_returns"].values()) > 0.0,
        "exactly10_full_period": current["full_10_position_ratio"] == 1.0,
    }
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat_daily, repeat_actions = run(
        context, policy, strong, priority_matrix, schedules[selected], 0.0030
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("renewal-threshold replay failed")
    result = {
        "status": "candidate_accepted_pre2026_validation_not_opened"
        if selected == candidate_id
        else "candidate_rejected_pre2026_validation_not_opened",
        "only_change": (
            "lower max-hold renewal score from 0.80 to 0.75 only on strong-market "
            "dates; weak-market threshold and every other rule remain unchanged"
        ),
        "results": results,
        "candidate_gates": gates,
        "selected_candidate": selected,
        "selected_policy": {
            **policy,
            "max_hold_renewal_score_rule": None
            if selected == baseline_id
            else "0.75_strong_market_0.80_weak_market",
        },
        "checkpoint_equivalence": equivalent,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
