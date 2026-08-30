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

import research_v260_fixed10_abnormal_announcement_proxy_20260822 as proxy
import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_severe_event_temporal_robustness_20260822 as intervention
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_risk_event_minimal_policy_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CASES = (
    "risk_event_off",
    "severe_next_session_only",
    "severe_and_alert_next_session_only",
    "all_events_next_session_control",
)


def compose_minimal_masks(
    ordinary: np.ndarray,
    severe: np.ndarray,
    alert: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value, dtype=np.bool_) for value in (ordinary, severe, alert)]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("risk-event masks do not align")
    ordinary_mask, severe_mask, alert_mask = arrays
    empty = np.zeros(severe_mask.shape, dtype=np.bool_)
    return {
        "risk_event_off": empty,
        "severe_next_session_only": severe_mask.copy(),
        "severe_and_alert_next_session_only": severe_mask | alert_mask,
        "all_events_next_session_control": ordinary_mask | severe_mask | alert_mask,
    }


def current_policy_overrides(context) -> tuple[np.ndarray, np.ndarray]:
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    score_priority = priority.sell_priority_matrix(
        context.score,
        volatility_rank,
        ~weak2,
        penalty=0.05,
    )
    return extra_age, score_priority


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
        )
    }


def compose_maintenance_block(
    score: np.ndarray,
    quality_threshold: float | None,
    event_block: np.ndarray,
) -> np.ndarray:
    events = np.asarray(event_block, dtype=np.bool_)
    values = np.asarray(score, dtype=np.float64)
    if events.shape != values.shape:
        raise ValueError("maintenance event mask does not align with score")
    if quality_threshold is None:
        quality_block = np.zeros(values.shape, dtype=np.bool_)
    else:
        if not np.isfinite(quality_threshold):
            raise ValueError("maintenance quality threshold must be finite")
        quality_block = ~np.isfinite(values) | (values < float(quality_threshold))
    return quality_block | events


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
        context.arrays["dates"],
        context.arrays["stocks"],
        features,
        ordinary_window=1,
        severe_window=0,
        warning_window=0,
    )
    severe, severe_audit = proxy.announcement_block_matrix(
        context.arrays["dates"],
        context.arrays["stocks"],
        features,
        ordinary_window=0,
        severe_window=1,
        warning_window=0,
    )
    alert, alert_audit = proxy.announcement_block_matrix(
        context.arrays["dates"],
        context.arrays["stocks"],
        features,
        ordinary_window=0,
        severe_window=0,
        warning_window=1,
    )
    masks = compose_minimal_masks(ordinary, severe, alert)
    extra_age, score_priority = current_policy_overrides(context)

    results: dict[str, dict] = {}
    cache = {}
    maintenance_threshold = policy.get("maintenance_topup_requires_score")
    for case_id in CASES:
        maintenance_block = compose_maintenance_block(
            context.score,
            maintenance_threshold,
            masks[case_id],
        )
        item, daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            entry_block_mask_override=masks[case_id],
            maintenance_buy_block_mask_override=maintenance_block,
            score_sell_priority_override=score_priority,
        )
        results[case_id] = item
        cache[case_id] = (daily, actions)

    baseline_id = "risk_event_off"
    baseline = results[baseline_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key],
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
        raise RuntimeError("minimal event-rule baseline drifted from checkpoint")

    intervention_audit = {}
    for case_id in CASES[1:]:
        blocked = intervention.blocked_baseline_buys(
            cache[baseline_id][1],
            context.arrays["dates"],
            context.arrays["stocks"],
            masks[case_id],
        )
        intervention_audit[case_id] = {
            "blocked_baseline_buy_count": int(len(blocked)),
            "distinct_signal_date_count": int(
                0 if blocked.empty else blocked["signal_date"].nunique()
            ),
            "distinct_stock_count": int(
                0 if blocked.empty else blocked["stock_code"].nunique()
            ),
            "affected_years": []
            if blocked.empty
            else sorted({str(value)[:4] for value in blocked["signal_date"]}),
            "metric_delta_vs_off": metric_delta(
                results[case_id]["metrics_0_30pct"],
                baseline["metrics_0_30pct"],
            ),
        }

    # The true Tushare event series starts in 2026, so proxy performance cannot
    # select a trading rule. The only admissible conclusion is a predeclared,
    # diagnostic-only policy for the later one-shot validation.
    selected = baseline_id
    selected_maintenance_block = compose_maintenance_block(
        context.score,
        maintenance_threshold,
        masks[selected],
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        entry_block_mask_override=masks[selected],
        maintenance_buy_block_mask_override=selected_maintenance_block,
        score_sell_priority_override=score_priority,
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
        "metrics": all(
            baseline["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("minimal event-rule deterministic replay failed")

    result = {
        "status": "minimal_event_policy_diagnostic_only_2026_not_opened",
        "frontier_eligible": False,
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [
            str(context.arrays["dates"][0]),
            str(context.arrays["dates"][-1]),
        ],
        "authoritative_strategy_decision": {
            "selected_candidate": selected,
            "checkpoint_changed": False,
            "reason": (
                "true Tushare risk-event rows are unavailable before 2026; "
                "proxy rows may test mechanics but cannot select policy"
            ),
        },
        "minimal_diagnostic_policy": {
            "ordinary_abnormal_volatility": "record only; no trading action",
            "severe_abnormal_volatility": (
                "block the next new entry and maintenance buy only; no fixed multi-day lock"
            ),
            "exchange_focus_security": (
                "block new entry and maintenance buy only while the official alert is active"
            ),
            "existing_holdings": (
                "never force exit and never change sell priority; use the frozen score-decay exit"
            ),
            "event_effect_on_candidate_selection": "diagnostic only",
            "fitted_numeric_parameters": 0,
        },
        "rejected_rule_families": {
            "forced_event_exit": "causally too strong and contradicted by prior tests",
            "event_first_sell_priority": "reduced return, Sharpe and drawdown quality",
            "ordinary_event_entry_block": "too broad for an informational event",
            "fixed_five_session_lock": "unsupported by true pre-2026 event data",
        },
        "proxy_component_audit": {
            "ordinary_1_session": ordinary_audit,
            "severe_1_session": severe_audit,
            "alert_1_session_proxy": alert_audit,
        },
        "proxy_results": results,
        "direct_interventions": intervention_audit,
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
