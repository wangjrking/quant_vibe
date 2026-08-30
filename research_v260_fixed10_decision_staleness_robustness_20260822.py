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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
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
    "strategy_agent_v260_fixed10_decision_staleness_robustness_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def lag_rows(values: np.ndarray, sessions: int) -> np.ndarray:
    lag = int(sessions)
    if lag < 0 or lag >= len(values):
        raise ValueError("sessions must be within the available calendar")
    result = np.empty_like(values)
    if lag == 0:
        result[:] = values
    else:
        result[:lag] = values[0]
        result[lag:] = values[:-lag]
    return result


def run_case(
    base_context,
    policy: dict,
    score_sessions: int,
    order_sessions: int | None = None,
    exit_score_sessions: int | None = None,
):
    order_lag = int(score_sessions if order_sessions is None else order_sessions)
    score = lag_rows(base_context.score, score_sessions)
    order = lag_rows(base_context.order, order_lag)
    context = replace(base_context, score=score, order=order)
    strong = regime.strong_market_mask(score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        score, volatility_rank, ~confirmed_weak, 0.05
    )
    exit_score = (
        None
        if exit_score_sessions is None
        else lag_rows(base_context.score, exit_score_sessions)
    )
    return age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
        exit_score_override=exit_score,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered decision-staleness robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)

    results = {}
    cache = {}
    cases = {
        "current": (0, 0, None),
        "entry_order_lag1": (0, 1, None),
        "exit_score_lag1": (0, 0, 1),
        "supporting_score_surface_lag1_exit_current": (1, 0, 0),
        "score_surface_lag1": (1, 0, None),
        "all_decisions_lag1": (1, 1, None),
        "all_decisions_lag2": (2, 2, None),
    }
    for case_id, (score_lag, order_lag, exit_score_lag) in cases.items():
        result, daily, actions = run_case(
            context,
            policy,
            score_lag,
            order_lag,
            exit_score_lag,
        )
        results[case_id] = result
        cache[case_id] = (daily, actions)

    expected = checkpoint["current_best_equalweight"]
    baseline_equivalence = {
        key: bool(
            np.isclose(
                results["current"]["metrics_0_30pct"][key],
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
    if not all(baseline_equivalence.values()):
        raise RuntimeError("staleness baseline drifted")

    summary = {}
    baseline = results["current"]["metrics_0_30pct"]
    production = checkpoint["production_baseline"]
    for sessions, result in results.items():
        metrics = result["metrics_0_30pct"]
        summary[sessions] = {
            "metrics": metrics,
            "all_calendar_years_positive": all(
                value > 0.0 for value in metrics["annual_returns"].values()
            ),
            "full_10_positions": metrics["full_10_position_ratio"] == 1.0,
            "cumulative_return_delta_vs_current": float(
                metrics["cumulative_return"] - baseline["cumulative_return"]
            ),
            "cumulative_return_delta_vs_production": float(
                metrics["cumulative_return"] - production["cumulative_return"]
            ),
            "stress_cumulative_return": result["metrics_0_65pct"][
                "cumulative_return"
            ],
        }

    repeat, repeat_daily, repeat_actions = run_case(context, policy, 0, 0, None)
    deterministic = {
        "daily": round1.frame_hash(cache["current"][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache["current"][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("decision-staleness baseline replay failed")

    payload = {
        "status": "decision_staleness_robustness_complete_2026_not_opened",
        "method": (
            "lag the entire score and ranking decision surface by one or two official "
            "sessions; delayed variants are robustness probes and are not selectable"
        ),
        "summary": summary,
        "robustness_decision": {
            "one_session_delay_all_years_positive": summary[
                "all_decisions_lag1"
            ][
                "all_calendar_years_positive"
            ],
            "two_session_delay_all_years_positive": summary[
                "all_decisions_lag2"
            ][
                "all_calendar_years_positive"
            ],
            "one_session_delay_beats_production_cumulative": summary[
                "all_decisions_lag1"
            ]["cumulative_return_delta_vs_production"]
            > 0.0,
            "two_session_delay_beats_production_cumulative": summary[
                "all_decisions_lag2"
            ]["cumulative_return_delta_vs_production"]
            > 0.0,
            "one_session_attribution": {
                "entry_order_only_cumulative_delta": summary[
                    "entry_order_lag1"
                ]["cumulative_return_delta_vs_current"],
                "score_surface_only_cumulative_delta": summary[
                    "score_surface_lag1"
                ]["cumulative_return_delta_vs_current"],
                "exit_score_only_cumulative_delta": summary[
                    "exit_score_lag1"
                ]["cumulative_return_delta_vs_current"],
                "supporting_score_surface_only_cumulative_delta": summary[
                    "supporting_score_surface_lag1_exit_current"
                ]["cumulative_return_delta_vs_current"],
            },
        },
        "baseline_equivalence": baseline_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "decision_staleness_robustness.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "robustness_decision": payload["robustness_decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
