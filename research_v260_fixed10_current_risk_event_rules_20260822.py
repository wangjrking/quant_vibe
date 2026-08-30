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
    "strategy_agent_v260_fixed10_current_risk_event_rules_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CASES = (
    "risk_event_off",
    "severe_pause5",
    "severe_pause5_alert_quality",
    "severe_pause5_ordinary_alert_quality",
)
QUALITY_THRESHOLD = 0.85


def compose_entry_blocks(
    score: np.ndarray,
    ordinary: np.ndarray,
    severe: np.ndarray,
    alert: np.ndarray,
    quality_threshold: float = QUALITY_THRESHOLD,
) -> dict[str, np.ndarray]:
    values = np.asarray(score, dtype=np.float64)
    components = [
        np.asarray(item, dtype=np.bool_)
        for item in (ordinary, severe, alert)
    ]
    if values.ndim != 2 or any(item.shape != values.shape for item in components):
        raise ValueError("risk-event entry inputs do not align")
    if not np.isfinite(quality_threshold):
        raise ValueError("quality threshold must be finite")
    ordinary_mask, severe_mask, alert_mask = components
    below_quality = ~np.isfinite(values) | (values < float(quality_threshold))
    empty = np.zeros(values.shape, dtype=np.bool_)
    return {
        "risk_event_off": empty,
        "severe_pause5": severe_mask.copy(),
        "severe_pause5_alert_quality": (
            severe_mask | (alert_mask & below_quality)
        ),
        "severe_pause5_ordinary_alert_quality": (
            severe_mask | ((ordinary_mask | alert_mask) & below_quality)
        ),
    }


def current_policy_overrides(context) -> tuple[np.ndarray, np.ndarray]:
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 3)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    score_priority = priority.sell_priority_matrix(
        context.score,
        volatility_rank,
        ~confirmed_weak,
        penalty=0.05,
    )
    return extra_age, score_priority


def selection_gates(
    candidate: dict,
    baseline: dict,
    direct: dict,
) -> dict[str, bool]:
    return {
        "train_cagr_improved": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "train_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "holdout_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "drawdown_not_worse": candidate["metrics_0_30pct"]["max_drawdown"]
        <= baseline["metrics_0_30pct"]["max_drawdown"],
        "full_10_positions": candidate["metrics_0_30pct"]["full_10_position_ratio"]
        == 1.0,
        "evidence_spans_three_dates": direct["distinct_signal_date_count"] >= 3,
        "evidence_spans_two_years": len(direct["affected_years"]) >= 2,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for event-rule tuning")
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
        severe_window=5,
        warning_window=0,
    )
    alert, alert_audit = proxy.announcement_block_matrix(
        context.arrays["dates"],
        context.arrays["stocks"],
        features,
        ordinary_window=0,
        severe_window=0,
        warning_window=5,
    )
    entry_blocks = compose_entry_blocks(
        context.score, ordinary, severe, alert
    )
    extra_age, score_priority = current_policy_overrides(context)

    results, cache = {}, {}
    for candidate_id in CASES:
        results[candidate_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            entry_block_mask_override=entry_blocks[candidate_id],
            score_sell_priority_override=score_priority,
        )
        cache[candidate_id] = (daily, actions)

    baseline_id = CASES[0]
    baseline = results[baseline_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(
            baseline["metrics_0_30pct"][key],
            expected[key],
            rtol=0.0,
            atol=1e-12,
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("event-rule baseline drifted from the current checkpoint")

    direct_interventions, gates = {}, {}
    for candidate_id in CASES[1:]:
        blocked = intervention.blocked_baseline_buys(
            cache[baseline_id][1],
            context.arrays["dates"],
            context.arrays["stocks"],
            entry_blocks[candidate_id],
        )
        direct = {
            "blocked_baseline_buy_count": int(len(blocked)),
            "distinct_signal_date_count": int(
                0 if blocked.empty else blocked["signal_date"].nunique()
            ),
            "distinct_stock_count": int(
                0 if blocked.empty else blocked["stock_code"].nunique()
            ),
            "affected_years": (
                [] if blocked.empty else sorted(
                    {str(value)[:4] for value in blocked["signal_date"]}
                )
            ),
        }
        direct_interventions[candidate_id] = direct
        gates[candidate_id] = selection_gates(
            results[candidate_id], baseline, direct
        )

    eligible = [
        candidate_id
        for candidate_id in CASES[1:]
        if all(gates[candidate_id].values())
    ]
    selected = max(
        [baseline_id, *eligible],
        key=lambda candidate_id: (
            results[candidate_id]["metrics_0_30pct"]["sharpe"],
            results[candidate_id]["metrics_0_30pct"]["cagr"],
            -results[candidate_id]["metrics_0_30pct"]["max_drawdown"],
        ),
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        entry_block_mask_override=entry_blocks[selected],
        score_sell_priority_override=score_priority,
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
        "metrics": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("event-rule deterministic replay failed")

    result = {
        "status": "current_risk_event_rules_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [
            str(context.arrays["dates"][0]), str(context.arrays["dates"][-1])
        ],
        "candidate_budget": list(CASES),
        "rule_semantics": {
            "ordinary_abnormal": (
                "do not force exit; optional one-session entry quality confirmation "
                "uses the existing 0.85 score threshold"
            ),
            "severe_abnormal": "pause new entries for five official sessions",
            "exchange_focus": (
                "do not force exit; during the proxy warning window require the "
                "existing 0.85 entry-quality threshold"
            ),
            "held_positions": (
                "event status never changes sell eligibility or exit count; "
                "the frozen score-decay exit remains authoritative"
            ),
            "fitted_numeric_parameters": 0,
        },
        "proxy_component_audit": {
            "ordinary_1d": ordinary_audit,
            "severe_5d": severe_audit,
            "exchange_focus_5d": alert_audit,
        },
        "results": results,
        "direct_interventions": direct_interventions,
        "selection_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "interpretation": (
            "Pre-2026 announcement classifications are used only as a mechanism proxy. "
            "The actual Tushare interfaces begin mainly in 2026 and remain unopened "
            "for final validation during this development pass."
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
