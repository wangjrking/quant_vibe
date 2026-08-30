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

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_size_tail_guard_20260822 as coarse
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_weak_market_size_guard_confirmation_20260822"
)
CASES = {
    "production_size_guard_off": {"percentile": 0.0, "selectable": False},
    "control_bottom_05pct": {"percentile": 0.05, "selectable": False},
    "candidate_bottom_10pct": {"percentile": 0.10, "selectable": True},
    "control_bottom_15pct": {"percentile": 0.15, "selectable": False},
}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered size-guard confirmation")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    weak_market = ~regime.strong_market_mask(score, protocol)
    production_mask = harness.v174.selection_mask(
        arrays, definition["max_rank_deterioration"]
    )
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    results = {}
    run_cache = {}
    for case_id, case in CASES.items():
        selection_mask = coarse.size_tail_selection_mask(
            production_mask,
            arrays["total_mv"],
            arrays["signal_clean"],
            weak_market,
            case["percentile"],
        )
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=np.zeros(score.shape, dtype=np.bool_),
            maintenance_buy_block_mask_override=maintenance_block,
            selection_mask_override=selection_mask,
        )
        daily, actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            policy,
            round1.DEVELOPMENT_END,
            slip=round1.BASELINE_COST,
            record_actions=True,
            **common,
        )
        stress_daily, stress_actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            policy,
            round1.DEVELOPMENT_END,
            slip=round1.STRESS_COST,
            record_actions=True,
            **common,
        )
        metrics = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        results[case_id] = {
            "selectable": bool(case["selectable"]),
            "minimum_size_percentile": float(case["percentile"]),
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "utility": coarse.utility(metrics),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "removed_stock_dates": int(
                production_mask.sum() - selection_mask.sum()
            ),
        }
        run_cache[case_id] = (daily, actions)

    baseline_id = "production_size_guard_off"
    candidate_id = "candidate_bottom_10pct"
    baseline = results[baseline_id]
    candidate = results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
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
        raise RuntimeError("size-guard confirmation baseline drifted")
    utility_deltas = {
        case_id: float(item["utility"] - baseline["utility"])
        for case_id, item in results.items()
        if case_id != baseline_id
    }
    annual_deltas = {
        year: float(
            candidate["metrics_0_30pct"]["annual_returns"][year]
            - baseline["metrics_0_30pct"]["annual_returns"][year]
        )
        for year in baseline["metrics_0_30pct"]["annual_returns"]
    }
    local_robust = bool(
        utility_deltas["control_bottom_05pct"] > 0
        and utility_deltas["control_bottom_15pct"] > 0
    )
    candidate_passed = bool(
        local_robust
        and candidate["utility"] > baseline["utility"]
        and candidate["metrics_0_30pct"]["cagr"]
        > baseline["metrics_0_30pct"]["cagr"]
        and candidate["metrics_0_30pct"]["sharpe"]
        > baseline["metrics_0_30pct"]["sharpe"]
        and candidate["metrics_0_30pct"]["max_drawdown"]
        < baseline["metrics_0_30pct"]["max_drawdown"]
        and candidate["metrics_0_65pct"]["cumulative_return"]
        > baseline["metrics_0_65pct"]["cumulative_return"]
        and all(value > 0 for value in candidate["metrics_0_30pct"]["annual_returns"].values())
    )
    selected = candidate_id if candidate_passed else baseline_id
    selected_daily, selected_actions = run_cache[selected]
    repeat_mask = coarse.size_tail_selection_mask(
        production_mask,
        arrays["total_mv"],
        arrays["signal_clean"],
        weak_market,
        CASES[selected]["percentile"],
    )
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=np.zeros(score.shape, dtype=np.bool_),
        maintenance_buy_block_mask_override=maintenance_block,
        selection_mask_override=repeat_mask,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("size-guard confirmation replay failed")
    selected_policy = copy.deepcopy(policy)
    selected_policy["weak_market_new_entry_minimum_size_percentile"] = CASES[
        selected
    ]["percentile"]
    result = {
        "status": "candidate_confirmed_pre2026_2026_not_opened" if candidate_passed else "candidate_rejected_pre2026_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selectable_candidate": candidate_id,
        "nonselectable_controls": ["control_bottom_05pct", "control_bottom_15pct"],
        "results": results,
        "utility_deltas_vs_baseline": utility_deltas,
        "candidate_annual_return_deltas": annual_deltas,
        "local_neighborhood_robust": local_robust,
        "candidate_passed": candidate_passed,
        "selected_candidate": selected,
        "selected_policy": selected_policy,
        "checkpoint_equivalence": checkpoint_equivalence,
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
