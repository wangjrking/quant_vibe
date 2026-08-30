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
    "strategy_agent_v260_fixed10_downside_volatility_sell_priority_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
COST_LEVELS = (0.0030, 0.0040, 0.0065)


def trailing_downside_semideviation(close_qfq: np.ndarray, window: int) -> np.ndarray:
    close = np.asarray(close_qfq, dtype=np.float64)
    if close.ndim != 2 or window < 2:
        raise ValueError("close matrix and rolling window are invalid")
    returns = np.full(close.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(close[1:])
        & np.isfinite(close[:-1])
        & (close[1:] > 0)
        & (close[:-1] > 0)
    )
    adjacent = np.full_like(close[1:], np.nan)
    adjacent[valid] = np.log(close[1:][valid] / close[:-1][valid])
    returns[1:] = adjacent
    finite = np.isfinite(returns)
    downside_sq = np.where(finite, np.minimum(returns, 0.0) ** 2, 0.0)
    cumulative_sq = np.vstack(
        [np.zeros((1, close.shape[1])), np.cumsum(downside_sq, axis=0)]
    )
    counts = np.vstack(
        [np.zeros((1, close.shape[1]), dtype=np.int32), np.cumsum(finite, axis=0)]
    )
    result = np.full(close.shape, np.nan, dtype=np.float64)
    for end in range(window, close.shape[0]):
        start = end - window + 1
        count = counts[end + 1] - counts[start]
        total_sq = cumulative_sq[end + 1] - cumulative_sq[start]
        complete = count == window
        result[end, complete] = np.sqrt(total_sq[complete] / window)
    return result


def run(context, policy: dict, priority_matrix, cost: float):
    strong = regime.strong_market_mask(context.score, context.protocol)
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        cost,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered downside-risk development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak2 = confirmed.confirmed_weak_mask(strong, 2)
    total_vol = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    downside_vol = trailing_downside_semideviation(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    total_rank = percentile.cross_sectional_percent_rank(total_vol)
    downside_rank = percentile.cross_sectional_percent_rank(downside_vol)
    priorities = {
        "current_total_volatility_005": priority.sell_priority_matrix(
            context.score, total_rank, ~confirmed_weak2, 0.05
        ),
        "control_downside_semideviation_0025": priority.sell_priority_matrix(
            context.score, downside_rank, ~confirmed_weak2, 0.025
        ),
        "candidate_downside_semideviation_005": priority.sell_priority_matrix(
            context.score, downside_rank, ~confirmed_weak2, 0.05
        ),
        "control_downside_semideviation_0075": priority.sell_priority_matrix(
            context.score, downside_rank, ~confirmed_weak2, 0.075
        ),
    }
    results, cache = {}, {}
    for candidate_id, priority_matrix in priorities.items():
        cost_metrics = {}
        for cost in COST_LEVELS:
            daily, actions = run(context, policy, priority_matrix, cost)
            cost_metrics[f"{cost:.4f}"] = round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            )
            if cost == 0.0030:
                cache[candidate_id] = (daily, actions)
        daily, actions = cache[candidate_id]
        results[candidate_id] = {
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

    baseline_id = "current_total_volatility_005"
    candidate_id = "candidate_downside_semideviation_005"
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
        raise RuntimeError("downside-risk baseline drifted")
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
        "neighbor_controls_training_sharpe_positive_delta": all(
            results[control]["train_2022_2024"]["sharpe"]
            > baseline["train_2022_2024"]["sharpe"]
            for control in (
                "control_downside_semideviation_0025",
                "control_downside_semideviation_0075",
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
        context, policy, priorities[selected], 0.0030
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("downside-risk replay failed")
    result = {
        "status": "candidate_accepted_pre2026_validation_not_opened"
        if selected == candidate_id
        else "candidate_rejected_pre2026_validation_not_opened",
        "only_change": (
            "on confirmed-weak dates, replace total-volatility percentile in the "
            "score-sell priority with trailing-20-session downside semideviation percentile"
        ),
        "results": results,
        "candidate_gates": gates,
        "selected_candidate": selected,
        "selected_policy": {
            **policy,
            "confirmed_weak_sell_priority": (
                policy["confirmed_weak_sell_priority"]
                if selected == baseline_id
                else {
                    "confirmation_days": 2,
                    "formula": (
                        "exit_score - 0.05 * "
                        "trailing_20d_downside_semideviation_percentile"
                    ),
                    "changes_eligibility": False,
                    "changes_exit_count": False,
                }
            ),
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
