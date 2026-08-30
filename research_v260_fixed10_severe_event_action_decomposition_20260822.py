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
import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_weak_market_size_tail_guard_20260822 as scorecard
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_severe_event_action_decomposition_20260822"
)
CASES = {
    "severe_gate_off": (False, False),
    "severe_5d_entry_only": (True, False),
    "severe_5d_topup_only": (False, True),
    "severe_5d_entry_and_topup": (True, True),
}


def compose_masks(
    severe_block: np.ndarray,
    quality_block: np.ndarray,
    block_entry: bool,
    block_topup: bool,
) -> tuple[np.ndarray, np.ndarray]:
    severe = np.asarray(severe_block, dtype=np.bool_)
    quality = np.asarray(quality_block, dtype=np.bool_)
    if severe.shape != quality.shape:
        raise ValueError("severe and quality masks do not align")
    entry = severe.copy() if block_entry else np.zeros_like(severe)
    topup = quality | severe if block_topup else quality.copy()
    return entry, topup


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/"
            "pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(round1.DEVELOPMENT_END)
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered severe-event decomposition")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(len(arrays["dates"]), policy["portfolio_rebalance_interval_days"])
    quality_block = quality.maintenance_quality_block(score, True)
    severe_block, severe_audit = proxy.announcement_block_matrix(
        arrays["dates"], arrays["stocks"], proxy.load_features(),
        ordinary_window=0, severe_window=5, warning_window=0,
    )
    results, cache = {}, {}
    for case_id, (block_entry, block_topup) in CASES.items():
        entry_mask, topup_mask = compose_masks(
            severe_block, quality_block, block_entry, block_topup
        )
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy["portfolio_rebalance_min_weight_deviation"],
            entry_block_mask_override=entry_mask,
            maintenance_buy_block_mask_override=topup_mask,
        )
        daily, actions = round1.run_fixed10(
            harness, arrays, protocol, definition, score, order, policy,
            round1.DEVELOPMENT_END, slip=round1.BASELINE_COST, record_actions=True, **common,
        )
        stress_daily, stress_actions = round1.run_fixed10(
            harness, arrays, protocol, definition, score, order, policy,
            round1.DEVELOPMENT_END, slip=round1.STRESS_COST, record_actions=True, **common,
        )
        metrics = round1.evaluate_run(daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END)
        results[case_id] = {
            "block_new_entry": block_entry,
            "block_maintenance_topup": block_topup,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily, stress_actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            ),
            "utility": scorecard.utility(metrics),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
        }
        cache[case_id] = (daily, actions)

    baseline_id = "severe_gate_off"
    baseline = results[baseline_id]
    expected = checkpoint["current_best_equalweight"]
    equivalent = {
        key: bool(np.isclose(baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12))
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown", "turnover_annualized", "average_invested_ratio")
    }
    if not all(equivalent.values()):
        raise RuntimeError("severe-event decomposition baseline drifted")
    raw_winner = max(results, key=lambda key: results[key]["utility"])
    winner = results[raw_winner]
    accepted = bool(
        raw_winner != baseline_id
        and winner["metrics_0_30pct"]["cagr"] > baseline["metrics_0_30pct"]["cagr"]
        and winner["metrics_0_30pct"]["sharpe"] > baseline["metrics_0_30pct"]["sharpe"]
        and winner["metrics_0_30pct"]["max_drawdown"] <= baseline["metrics_0_30pct"]["max_drawdown"]
        and winner["metrics_0_65pct"]["cumulative_return"] > baseline["metrics_0_65pct"]["cumulative_return"]
    )
    selected = raw_winner if accepted else baseline_id
    selected_daily, selected_actions = cache[selected]
    entry_mask, topup_mask = compose_masks(
        severe_block, quality_block, *CASES[selected]
    )
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness, arrays, protocol, definition, score, order, policy,
        round1.DEVELOPMENT_END, slip=round1.BASELINE_COST, record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy["portfolio_rebalance_min_weight_deviation"],
        entry_block_mask_override=entry_mask,
        maintenance_buy_block_mask_override=topup_mask,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions) == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("severe-event decomposition replay failed")
    result = {
        "status": "action_scope_confirmed_pre2026_2026_not_opened" if accepted else "action_scope_rejected_pre2026_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "severe_event_proxy_audit": severe_audit,
        "results": results,
        "raw_winner": raw_winner,
        "accepted": accepted,
        "selected_candidate": selected,
        "checkpoint_equivalence": equivalent,
        "deterministic_replay": deterministic,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
