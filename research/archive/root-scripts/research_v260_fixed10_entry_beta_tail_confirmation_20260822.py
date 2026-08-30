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
import research_v260_fixed10_entry_beta_tail_guard_20260822 as beta_tail
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_entry_beta_tail_confirmation_20260822"
)
SELECTABLE_PERCENTILES = {
    "production_entry": None,
    "exclude_highest_beta_0_5pct": 0.995,
}
ROBUSTNESS_PERCENTILES = {
    "control_exclude_highest_beta_0_25pct": 0.9975,
    "control_exclude_highest_beta_0_75pct": 0.9925,
}
MIN_CAGR_IMPROVEMENT = 0.005
MIN_SHARPE_IMPROVEMENT = 0.01
MIN_DRAWDOWN_IMPROVEMENT = 0.005


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
        raise PermissionError("2026 data entered beta-tail confirmation")
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
    beta_percentile = beta_tail.cross_sectional_percent_rank(beta)
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
        metrics = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        stress = round1.evaluate_run(
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
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "pre2026_selection_utility": round1.selection_utility(metrics, stress),
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
        raise RuntimeError("beta-tail confirmation baseline drifted from checkpoint")

    candidate_id = "exclude_highest_beta_0_5pct"
    baseline_utility = results[baseline_id]["pre2026_selection_utility"]
    candidate_utility = results[candidate_id]["pre2026_selection_utility"]
    neighborhood_robust = all(
        results[control_id]["pre2026_selection_utility"] >= baseline_utility
        for control_id in ROBUSTNESS_PERCENTILES
    )
    annual_positive = all(
        value > 0
        for value in results[candidate_id]["metrics_0_30pct"][
            "annual_returns"
        ].values()
    )
    passed = bool(
        candidate_utility > baseline_utility
        and materially_better(
            results[candidate_id]["metrics_0_30pct"],
            results[baseline_id]["metrics_0_30pct"],
        )
        and neighborhood_robust
        and annual_positive
    )
    selected_id = candidate_id if passed else baseline_id
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
        raise RuntimeError("beta-tail confirmation deterministic replay failed")

    result = {
        "status": "pre2026_candidate_selected_2026_not_opened" if passed else "rejected",
        "source_strategy": rules["strategy_id"],
        "tuning_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "validation_boundary": ["20260101", "closed_until_final_one_shot"],
        "candidate_budget": list(SELECTABLE_PERCENTILES),
        "predeclared_robustness_controls_not_selectable": list(
            ROBUSTNESS_PERCENTILES
        ),
        "only_change": (
            "exclude the highest trailing-beta 0.5 percent tail from new entries; "
            "all holding, exit, equal-weight, refill, and execution rules stay unchanged"
        ),
        "results": results,
        "selected_candidate_id": selected_id,
        "selected_reason": (
            "candidate improved pre-2026 utility and both neighboring controls"
            if passed
            else "candidate failed material, annual, or neighboring robustness"
        ),
        "confirmation_gate": {
            "candidate_utility_above_baseline": candidate_utility > baseline_utility,
            "material_improvement": materially_better(
                results[candidate_id]["metrics_0_30pct"],
                results[baseline_id]["metrics_0_30pct"],
            ),
            "annual_returns_all_positive": annual_positive,
            "neighborhood_robust": neighborhood_robust,
            "passed": passed,
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
                "gate": result["confirmation_gate"],
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
