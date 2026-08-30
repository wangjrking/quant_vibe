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
import research_v260_fixed10_entry_beta_guard_20260822 as beta_guard
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_entry_beta_tail_guard_20260822"
)
SELECTABLE_PERCENTILES = {
    "production_entry": None,
    "exclude_highest_beta_1pct": 0.99,
    "exclude_highest_beta_2_5pct": 0.975,
}
ROBUSTNESS_PERCENTILES = {
    "control_exclude_highest_beta_0_5pct": 0.995,
    "control_exclude_highest_beta_2pct": 0.98,
    "control_exclude_highest_beta_5pct": 0.95,
}
NEIGHBORS = {
    "exclude_highest_beta_1pct": (
        "control_exclude_highest_beta_0_5pct",
        "control_exclude_highest_beta_2pct",
    ),
    "exclude_highest_beta_2_5pct": (
        "control_exclude_highest_beta_2pct",
        "control_exclude_highest_beta_5pct",
    ),
}
MIN_CAGR_IMPROVEMENT = 0.005
MIN_SHARPE_IMPROVEMENT = 0.01
MIN_DRAWDOWN_IMPROVEMENT = 0.005


def cross_sectional_percent_rank(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("beta matrix must be two-dimensional")
    result = np.full(matrix.shape, np.nan, dtype=np.float32)
    for day in range(len(matrix)):
        finite_index = np.flatnonzero(np.isfinite(matrix[day]))
        if not len(finite_index):
            continue
        order = np.argsort(matrix[day, finite_index], kind="stable")
        denominator = max(len(finite_index) - 1, 1)
        ranks = np.empty(len(finite_index), dtype=np.float32)
        ranks[order] = np.arange(len(finite_index), dtype=np.float32) / denominator
        result[day, finite_index] = ranks
    return result


def entry_selection_mask(
    candidate_id: str,
    production_mask: np.ndarray,
    beta_percentile: np.ndarray,
) -> np.ndarray:
    all_percentiles = {**SELECTABLE_PERCENTILES, **ROBUSTNESS_PERCENTILES}
    if candidate_id not in all_percentiles:
        raise ValueError(f"unknown candidate: {candidate_id}")
    result = np.asarray(production_mask, dtype=np.bool_).copy()
    ceiling = all_percentiles[candidate_id]
    if ceiling is not None:
        values = np.asarray(beta_percentile, dtype=np.float64)
        result &= ~np.isfinite(values) | (values <= float(ceiling))
    return result


def materially_better(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["cagr"] - baseline["cagr"] >= MIN_CAGR_IMPROVEMENT
        or candidate["sharpe"] - baseline["sharpe"] >= MIN_SHARPE_IMPROVEMENT
        or baseline["max_drawdown"] - candidate["max_drawdown"]
        >= MIN_DRAWDOWN_IMPROVEMENT
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
        raise PermissionError("2026 data entered beta-tail development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    production_mask = harness.v174.selection_mask(
        arrays, definition["max_rank_deterioration"]
    )
    beta = beta_guard.trailing_market_beta(
        arrays["close_qfq"],
        beta_guard.BETA_LOOKBACK,
        beta_guard.BETA_MIN_OBSERVATIONS,
    )
    beta_percentile = cross_sectional_percent_rank(beta)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    entry_block = np.zeros(score.shape, dtype=np.bool_)
    maintenance_block = quality.maintenance_quality_block(
        score, policy.get("maintenance_topup_requires_score") is not None
    )
    results = {}
    run_cache = {}
    mask_cache = {}
    all_percentiles = {**SELECTABLE_PERCENTILES, **ROBUSTNESS_PERCENTILES}

    for candidate_id, ceiling in all_percentiles.items():
        selection_mask = entry_selection_mask(
            candidate_id, production_mask, beta_percentile
        )
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
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
        baseline_metrics = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        stress_metrics = round1.evaluate_run(
            stress_daily,
            stress_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        )
        windows = {
            str(offset): round1.evaluate_run(
                daily,
                actions,
                robustness.window_start_date(arrays, offset),
                round1.DEVELOPMENT_END,
            )
            for offset in (5, 20, 60)
        }
        results[candidate_id] = {
            "policy": {
                **policy,
                "new_entry_market_beta_percentile_max": ceiling,
                "market_beta_lookback": beta_guard.BETA_LOOKBACK,
                "market_beta_min_observations": beta_guard.BETA_MIN_OBSERVATIONS,
            },
            "metrics_0_30pct": baseline_metrics,
            "metrics_0_65pct": stress_metrics,
            "pre2026_selection_utility": round1.selection_utility(
                baseline_metrics, stress_metrics
            ),
            "annual_returns": baseline_metrics["annual_returns"],
            "window_start_metrics": windows,
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "selection_mask_true_count": int(selection_mask.sum()),
        }
        run_cache[candidate_id] = (daily, actions)
        mask_cache[candidate_id] = selection_mask

    baseline_id = "production_entry"
    current = results[baseline_id]["metrics_0_30pct"]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(current[key], expected[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("beta-tail baseline drifted from checkpoint")

    ranked_id = max(
        SELECTABLE_PERCENTILES,
        key=lambda key: results[key]["pre2026_selection_utility"],
    )
    baseline_utility = results[baseline_id]["pre2026_selection_utility"]
    neighbors = NEIGHBORS.get(ranked_id, ())
    neighborhood_robust = bool(
        neighbors
        and all(
            results[control_id]["pre2026_selection_utility"] >= baseline_utility
            for control_id in neighbors
        )
    )
    material_improvement = bool(
        ranked_id != baseline_id
        and materially_better(
            results[ranked_id]["metrics_0_30pct"],
            results[baseline_id]["metrics_0_30pct"],
        )
        and neighborhood_robust
    )
    selected_id = ranked_id if material_improvement else baseline_id
    selected_daily, selected_actions = run_cache[selected_id]
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
        entry_block_mask_override=entry_block,
        maintenance_buy_block_mask_override=maintenance_block,
        selection_mask_override=mask_cache[selected_id],
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("beta-tail deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "tuning_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "validation_boundary": ["20260101", "closed_until_final_one_shot"],
        "candidate_budget": list(SELECTABLE_PERCENTILES),
        "predeclared_robustness_controls_not_selectable": list(
            ROBUSTNESS_PERCENTILES
        ),
        "only_change": (
            "exclude only the highest trailing-beta cross-sectional tail from new "
            "entries; preserve production score, exits, exactly ten equal weights, and refill"
        ),
        "results": results,
        "selected_candidate_id": selected_id,
        "selected_reason": (
            "material pre-2026 improvement with neighboring percentile robustness"
            if material_improvement
            else "candidate did not clear material and neighboring robustness gates"
        ),
        "practical_improvement_gate": {
            "minimum_cagr_improvement": MIN_CAGR_IMPROVEMENT,
            "minimum_sharpe_improvement": MIN_SHARPE_IMPROVEMENT,
            "minimum_drawdown_improvement": MIN_DRAWDOWN_IMPROVEMENT,
            "raw_ranked_candidate_id": ranked_id,
            "neighbor_controls": list(neighbors),
            "neighborhood_robust": neighborhood_robust,
            "passed": material_improvement,
        },
        "selected_policy": copy.deepcopy(results[selected_id]["policy"]),
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "selected": selected_id,
                "ranked": ranked_id,
                "gate": result["practical_improvement_gate"],
                "results": {
                    key: {
                        "metrics": value["metrics_0_30pct"],
                        "utility": value["pre2026_selection_utility"],
                    }
                    for key, value in results.items()
                },
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
