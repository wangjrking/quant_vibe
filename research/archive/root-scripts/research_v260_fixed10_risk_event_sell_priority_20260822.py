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
import research_v260_fixed10_current_risk_event_rules_20260822 as entry_rules
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_severe_event_temporal_robustness_20260822 as intervention


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_risk_event_sell_priority_20260822"
)
CHECKPOINT_PATH = entry_rules.CHECKPOINT_PATH
CASES = (
    "score_decay_priority_only",
    "severe_event_first",
    "severe_or_alert_first",
    "all_risk_events_first",
)


def event_first_priority(
    base_priority: np.ndarray,
    event_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(base_priority, dtype=np.float64)
    events = np.asarray(event_mask, dtype=np.bool_)
    if values.ndim != 2 or events.shape != values.shape:
        raise ValueError("event-priority inputs do not align")
    within_tier_rank = percentile.cross_sectional_percent_rank(values)
    safe_rank = np.nan_to_num(within_tier_rank, nan=1.0).astype(np.float64)
    return (~events).astype(np.float64) * 2.0 + safe_rank


def selection_gates(
    candidate: dict,
    baseline: dict,
    action_diagnostic: dict,
) -> dict[str, bool]:
    changed_dates = action_diagnostic["changed_signal_dates"]
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
        "changes_span_three_dates": len(changed_dates) >= 3,
        "changes_span_two_years": len({date[:4] for date in changed_dates}) >= 2,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for event priority")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    if context.access["logical_max_date"] >= "20260101":
        raise PermissionError("event-priority development crossed into 2026")

    features = proxy.load_features()
    ordinary, ordinary_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features,
        ordinary_window=1, severe_window=0, warning_window=0,
    )
    severe, severe_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features,
        ordinary_window=0, severe_window=5, warning_window=0,
    )
    alert, alert_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features,
        ordinary_window=0, severe_window=0, warning_window=5,
    )
    extra_age, base_priority = entry_rules.current_policy_overrides(context)
    priorities = {
        "score_decay_priority_only": base_priority,
        "severe_event_first": event_first_priority(base_priority, severe),
        "severe_or_alert_first": event_first_priority(
            base_priority, severe | alert
        ),
        "all_risk_events_first": event_first_priority(
            base_priority, ordinary | severe | alert
        ),
    }

    results, cache = {}, {}
    for candidate_id in CASES:
        results[candidate_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=priorities[candidate_id],
        )
        cache[candidate_id] = (daily, actions)

    baseline_id = CASES[0]
    baseline = results[baseline_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(
            baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("event sell-priority baseline drifted")

    action_diagnostics, gates = {}, {}
    for candidate_id in CASES[1:]:
        diagnostic = intervention.action_set_diagnostics(
            cache[baseline_id][1], cache[candidate_id][1]
        )
        action_diagnostics[candidate_id] = diagnostic
        gates[candidate_id] = selection_gates(
            results[candidate_id], baseline, diagnostic
        )

    eligible = [
        candidate_id for candidate_id in CASES[1:]
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
        score_sell_priority_override=priorities[selected],
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
        raise RuntimeError("event sell-priority replay failed")

    result = {
        "status": "risk_event_sell_priority_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "candidate_budget": list(CASES),
        "only_change": (
            "among positions already eligible for the frozen score-decay exit, "
            "event-tier names sell first; events never create a new exit"
        ),
        "tier_semantics": (
            "event names receive tier 0 and other names tier 1; the existing "
            "score/volatility priority remains the deterministic within-tier order"
        ),
        "proxy_component_audit": {
            "ordinary_1d": ordinary_audit,
            "severe_5d": severe_audit,
            "exchange_focus_5d": alert_audit,
        },
        "results": results,
        "action_diagnostics": action_diagnostics,
        "selection_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
