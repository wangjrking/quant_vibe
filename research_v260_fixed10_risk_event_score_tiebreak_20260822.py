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

import research_v260_fixed10_abnormal_announcement_proxy_20260822 as proxy
import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_risk_event_score_tiebreak_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CASES = (
    "risk_event_off",
    "severe_score_tiebreak",
    "alert_score_tiebreak",
    "severe_or_alert_score_tiebreak",
)
COSTS = (0.0030, 0.0065)


def score_tiebreak_order(
    score: np.ndarray,
    base_order: np.ndarray,
    event_mask: np.ndarray,
    margin: float,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    order = np.asarray(base_order)
    events = np.asarray(event_mask, dtype=np.bool_)
    if values.shape != order.shape or values.shape != events.shape:
        raise ValueError("score, order and event mask must align")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("event tiebreak margin must be finite and non-negative")
    if order.ndim != 2:
        raise ValueError("event tiebreak requires two-dimensional daily rankings")

    result = np.empty_like(order)
    for row_index in range(order.shape[0]):
        original = [int(index) for index in order[row_index]]
        adjusted = values[row_index] - float(margin) * events[row_index]
        result[row_index] = np.asarray(
            sorted(
                original,
                key=lambda index: (
                    -round(float(adjusted[index]), 12)
                    if np.isfinite(adjusted[index])
                    else np.inf,
                    bool(events[row_index, index]),
                ),
            ),
            dtype=order.dtype,
        )
    return result


def current_overrides(context) -> tuple[np.ndarray, np.ndarray]:
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score,
        volatility_rank,
        ~weak2,
        penalty=0.05,
    )
    return extra_age, sell_priority


def evaluate_case(context, policy, extra_age, sell_priority, order) -> tuple[dict, object, object]:
    case_context = replace(context, order=np.asarray(order).copy())
    cost_metrics = {}
    baseline_daily = baseline_actions = None
    for cost in COSTS:
        daily, actions = age_guard.run_policy_at_cost(
            case_context,
            policy,
            extra_age,
            cost,
            score_sell_priority_override=sell_priority,
            exit_score_override=context.score,
        )
        cost_metrics[f"{cost:.4f}"] = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        if cost == round1.BASELINE_COST:
            baseline_daily, baseline_actions = daily, actions
    if baseline_daily is None or baseline_actions is None:
        raise RuntimeError("baseline cost was not evaluated")
    return (
        {
            "metrics_0_30pct": cost_metrics["0.0030"],
            "metrics_0_65pct": cost_metrics["0.0065"],
            "train_2022_2024": round1.evaluate_run(
                baseline_daily, baseline_actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025": round1.evaluate_run(
                baseline_daily, baseline_actions, "20250102", round1.DEVELOPMENT_END
            ),
            "start_offset_metrics": {
                str(offset): round1.evaluate_run(
                    baseline_daily,
                    baseline_actions,
                    research_base.FIRST_BUY
                    if offset == 0
                    else robustness.window_start_date(context.arrays, offset),
                    round1.DEVELOPMENT_END,
                )
                for offset in (0, 5, 20, 60)
            },
        },
        baseline_daily,
        baseline_actions,
    )


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for event-rule development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    if context.access["logical_max_date"] >= "20260101":
        raise PermissionError("event-rule development crossed into 2026")

    features = proxy.load_features()
    ordinary, ordinary_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features, 1, 0, 0
    )
    severe, severe_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features, 0, 1, 0
    )
    alert, alert_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features, 0, 0, 1
    )
    empty = np.zeros(context.score.shape, dtype=np.bool_)
    masks = {
        "risk_event_off": empty,
        "severe_score_tiebreak": severe,
        "alert_score_tiebreak": alert,
        "severe_or_alert_score_tiebreak": severe | alert,
    }
    margin = float(policy["replacement_advantage"])
    orders = {
        case_id: score_tiebreak_order(
            context.score, context.order, mask, margin
        )
        for case_id, mask in masks.items()
    }
    if not np.array_equal(orders["risk_event_off"], context.order):
        raise RuntimeError("empty event tiebreak changed the frozen ranking")

    extra_age, sell_priority = current_overrides(context)
    results, cache = {}, {}
    for case_id in CASES:
        results[case_id], daily, actions = evaluate_case(
            context, policy, extra_age, sell_priority, orders[case_id]
        )
        cache[case_id] = (daily, actions)

    baseline_id = "risk_event_off"
    candidate_id = "severe_score_tiebreak"
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                results[baseline_id]["metrics_0_30pct"][key],
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
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("event tiebreak baseline drifted from checkpoint")

    action_audit = {}
    baseline_actions = cache[baseline_id][1]
    for case_id in CASES[1:]:
        actions = cache[case_id][1]
        baseline_keys = set(
            zip(
                baseline_actions["buy_date"].astype(str),
                baseline_actions["action"].astype(str),
                baseline_actions["stock_code"].astype(str),
            )
        )
        case_keys = set(
            zip(
                actions["buy_date"].astype(str),
                actions["action"].astype(str),
                actions["stock_code"].astype(str),
            )
        )
        action_audit[case_id] = {
            "ranking_cells_changed": int(
                np.count_nonzero(orders[case_id] != context.order)
            ),
            "baseline_only_action_keys": int(len(baseline_keys - case_keys)),
            "case_only_action_keys": int(len(case_keys - baseline_keys)),
            "metric_delta_vs_off": metric_delta(
                results[case_id]["metrics_0_30pct"],
                results[baseline_id]["metrics_0_30pct"],
            ),
        }

    candidate = results[candidate_id]
    baseline = results[baseline_id]
    gates = {
        "train_cagr_not_worse": candidate["train_2022_2024"]["cagr"]
        >= baseline["train_2022_2024"]["cagr"],
        "train_sharpe_not_worse": candidate["train_2022_2024"]["sharpe"]
        >= baseline["train_2022_2024"]["sharpe"],
        "holdout_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "drawdown_not_worse": candidate["metrics_0_30pct"]["max_drawdown"]
        <= baseline["metrics_0_30pct"]["max_drawdown"],
        "all_offsets_profitable": min(
            item["cumulative_return"]
            for item in candidate["start_offset_metrics"].values()
        ) > 0.0,
        "exactly10": candidate["metrics_0_30pct"]["full_10_position_ratio"] == 1.0,
    }

    repeat, repeat_daily, repeat_actions = evaluate_case(
        context, policy, extra_age, sell_priority, orders[candidate_id]
    )
    deterministic = {
        "daily": round1.frame_hash(cache[candidate_id][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[candidate_id][1])
        == round1.frame_hash(repeat_actions),
        "metrics": repeat["metrics_0_30pct"] == candidate["metrics_0_30pct"],
    }
    if not all(deterministic.values()):
        raise RuntimeError("event tiebreak deterministic replay failed")

    result = {
        "status": "event_score_tiebreak_diagnostic_complete_2026_not_opened",
        "authoritative_candidate_changed": False,
        "reason": (
            "the true Tushare event series is concentrated in 2026; pre-2026 "
            "announcement proxies may test mechanics but cannot select a live rule"
        ),
        "recommended_predeclared_diagnostic": {
            "ordinary_abnormal_volatility": "observe only",
            "severe_abnormal_volatility": (
                "new-entry score tiebreak only; subtract the existing 0.05 replacement "
                "margin for ranking, prefer the clean stock on an adjusted-score tie, "
                "and never change eligibility"
            ),
            "exchange_focus_security": (
                "observe only; the proxy tiebreak reduced development return and Sharpe"
            ),
            "existing_holdings": "no forced exit and no event-driven sell priority",
            "maintenance_topups": "unchanged frozen score-quality rule",
            "new_fitted_parameters": 0,
        },
        "rejected_rule_families": {
            "ordinary_entry_rule": "too broad and previously reduced return quality",
            "exchange_focus_entry_tiebreak": (
                "reduced pre-2026 proxy return, Sharpe and drawdown quality"
            ),
            "forced_event_exit": "event status can describe strong winners and is not a sell signal",
            "event_sell_priority": "previous tests reduced return, Sharpe and drawdown quality",
        },
        "proxy_component_audit": {
            "ordinary_1_session": ordinary_audit,
            "severe_1_session": severe_audit,
            "alert_1_session": alert_audit,
            "ordinary_used_for_trading": False,
        },
        "results": results,
        "action_audit": action_audit,
        "candidate_gates": gates,
        "diagnostic_candidate": candidate_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
