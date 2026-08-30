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
import research_v260_fixed10_decision_staleness_robustness_20260822 as staleness
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_exit_timing_robustness_tradeoff_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def consecutive_below_mask(score: np.ndarray, threshold: float, days: int) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("score must be date by stock")
    count = int(days)
    if count < 1:
        raise ValueError("days must be positive")
    below = np.isfinite(values) & (values < float(threshold))
    result = below.copy()
    for lag in range(1, count):
        prior = np.zeros_like(below)
        prior[lag:] = below[:-lag]
        result &= prior
    return result


def run_case(context, policy: dict, confirmation_days: int, exit_lag: int):
    exit_score = staleness.lag_rows(context.score, exit_lag)
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
    confirmation = (
        None
        if int(confirmation_days) == 1
        else consecutive_below_mask(
            exit_score, float(policy["sell_score_below"]), confirmation_days
        )
    )
    return age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
        exit_score_override=exit_score,
        score_exit_confirmation_mask_override=confirmation,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered exit-timing robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)

    cases = {
        "current_1d": (1, 0),
        "confirm_2d": (2, 0),
        "current_1d_exit_score_lag1": (1, 1),
        "confirm_2d_exit_score_lag1": (2, 1),
    }
    results = {}
    cache = {}
    for case_id, (confirmation_days, exit_lag) in cases.items():
        result, daily, actions = run_case(
            context, policy, confirmation_days, exit_lag
        )
        results[case_id] = result
        cache[case_id] = (daily, actions)

    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(
            np.isclose(
                results["current_1d"]["metrics_0_30pct"][key],
                expected[key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("exit-timing baseline drifted")

    metrics = {
        case_id: result["metrics_0_30pct"] for case_id, result in results.items()
    }
    base = metrics["current_1d"]
    confirm = metrics["confirm_2d"]
    base_lag = metrics["current_1d_exit_score_lag1"]
    confirm_lag = metrics["confirm_2d_exit_score_lag1"]
    comparison = {
        "confirm_2d_delta_vs_current": {
            key: float(confirm[key] - base[key])
            for key in (
                "cumulative_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "turnover_annualized",
            )
        },
        "one_day_exit_staleness_loss": {
            "current_1d": float(
                base_lag["cumulative_return"] - base["cumulative_return"]
            ),
            "confirm_2d": float(
                confirm_lag["cumulative_return"] - confirm["cumulative_return"]
            ),
        },
        "confirmation_reduces_absolute_staleness_loss": abs(
            confirm_lag["cumulative_return"] - confirm["cumulative_return"]
        )
        < abs(base_lag["cumulative_return"] - base["cumulative_return"]),
    }
    selection = (
        "retain_current_1d"
        if confirm["cumulative_return"] < base["cumulative_return"]
        or confirm["sharpe"] < base["sharpe"]
        or results["confirm_2d"]["metrics_0_65pct"]["cumulative_return"]
        < results["current_1d"]["metrics_0_65pct"]["cumulative_return"]
        else "promote_confirm_2d"
    )

    repeat, repeat_daily, repeat_actions = run_case(context, policy, 1, 0)
    deterministic = {
        "daily": round1.frame_hash(cache["current_1d"][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache["current_1d"][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("exit-timing replay failed")

    payload = {
        "status": "exit_timing_tradeoff_complete_2026_not_opened",
        "method": (
            "compare the current one-session score exit with a logical two-consecutive-"
            "session confirmation, both with and without one-session exit-score staleness"
        ),
        "results": results,
        "comparison": comparison,
        "selection": selection,
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "exit_timing_tradeoff.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": selection,
                "comparison": comparison,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
