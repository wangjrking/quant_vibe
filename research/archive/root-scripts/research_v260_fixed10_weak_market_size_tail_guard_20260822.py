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
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_weak_market_size_tail_guard_20260822"
)
TAIL_PERCENTILES = {
    "weak_market_size_guard_off": 0.0,
    "weak_market_exclude_bottom_10pct_size": 0.10,
    "weak_market_exclude_bottom_20pct_size": 0.20,
    "weak_market_exclude_bottom_30pct_size": 0.30,
}


def size_tail_selection_mask(
    production_mask: np.ndarray,
    total_mv: np.ndarray,
    signal_clean: np.ndarray,
    weak_market: np.ndarray,
    minimum_percentile: float,
) -> np.ndarray:
    result = np.asarray(production_mask, dtype=np.bool_).copy()
    values = np.asarray(total_mv, dtype=np.float64)
    clean = np.asarray(signal_clean, dtype=np.bool_)
    weak = np.asarray(weak_market, dtype=np.bool_)
    if result.shape != values.shape or clean.shape != result.shape:
        raise ValueError("size-tail matrices do not align")
    if weak.shape != (result.shape[0],):
        raise ValueError("weak-market vector does not align")
    percentile = float(minimum_percentile)
    if not 0.0 <= percentile < 1.0:
        raise ValueError("minimum size percentile is outside [0, 1)")
    if percentile == 0.0:
        return result
    for day in np.flatnonzero(weak):
        universe = result[day] & clean[day] & np.isfinite(values[day])
        reference = values[day, universe]
        if reference.size < 10:
            continue
        threshold = float(np.quantile(reference, percentile, method="linear"))
        result[day] &= ~np.isfinite(values[day]) | (values[day] >= threshold)
    return result


def utility(metrics: dict) -> float:
    return float(
        metrics["cagr"]
        + 0.20 * metrics["sharpe"]
        - 0.50 * metrics["max_drawdown"]
        - 0.001 * metrics["turnover_annualized"]
    )


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
        raise PermissionError("2026 data entered weak-market size development")
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
    for candidate_id, percentile in TAIL_PERCENTILES.items():
        selection_mask = size_tail_selection_mask(
            production_mask,
            arrays["total_mv"],
            arrays["signal_clean"],
            weak_market,
            percentile,
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
        results[candidate_id] = {
            "policy": {
                **policy,
                "weak_market_new_entry_minimum_size_percentile": percentile,
            },
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "utility": utility(metrics),
            "positive_annual_returns": bool(
                all(value > 0 for value in metrics["annual_returns"].values())
            ),
            "window_start_metrics": {
                str(offset): round1.evaluate_run(
                    daily,
                    actions,
                    robustness.window_start_date(arrays, offset),
                    round1.DEVELOPMENT_END,
                )
                for offset in (5, 20, 60)
            },
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "weak_market_days": int(weak_market.sum()),
            "selection_mask_removed_stock_dates": int(
                production_mask.sum() - selection_mask.sum()
            ),
        }
        run_cache[candidate_id] = (daily, actions)

    baseline_id = "weak_market_size_guard_off"
    baseline = results[baseline_id]["metrics_0_30pct"]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(baseline[key], expected[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("weak-market size baseline drifted from checkpoint")
    raw_winner = max(
        results,
        key=lambda key: (
            results[key]["positive_annual_returns"],
            results[key]["utility"],
        ),
    )
    baseline_utility = results[baseline_id]["utility"]
    improvements = {
        key: float(results[key]["utility"] - baseline_utility)
        for key in results
        if key != baseline_id
    }
    robust = bool(
        raw_winner != baseline_id
        and results[raw_winner]["positive_annual_returns"]
        and sum(value > 0 for value in improvements.values()) >= 2
        and results[raw_winner]["metrics_0_65pct"]["cumulative_return"]
        > results[baseline_id]["metrics_0_65pct"]["cumulative_return"]
    )
    selected = raw_winner if robust else baseline_id
    selected_daily, selected_actions = run_cache[selected]
    percentile = TAIL_PERCENTILES[selected]
    repeat_mask = size_tail_selection_mask(
        production_mask,
        arrays["total_mv"],
        arrays["signal_clean"],
        weak_market,
        percentile,
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
        raise RuntimeError("weak-market size replay failed")
    result = {
        "status": "development_complete_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "only_rule_changed": (
            "new-entry exclusion of the smallest total-market-cap tail only on "
            "production-defined weak-market days"
        ),
        "candidate_budget": TAIL_PERCENTILES,
        "results": results,
        "raw_winner": raw_winner,
        "neighborhood_utility_improvements": improvements,
        "neighborhood_robust": robust,
        "selected_candidate": selected,
        "selected_policy": results[selected]["policy"],
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
