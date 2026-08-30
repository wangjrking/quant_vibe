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
import research_v260_fixed10_score_5d10d_current_rules_20260822 as score_tools
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_score_5d_minor_blend_20260822"
)
SELECTABLE_WEIGHTS_5D = {
    "score_10d": 0.0,
    "score_5d10_10d90": 0.10,
    "score_5d20_10d80": 0.20,
}
ROBUSTNESS_WEIGHTS_5D = {"control_score_5d30_10d70": 0.30}
NEIGHBORS = {
    "score_5d10_10d90": ("score_10d", "score_5d20_10d80"),
    "score_5d20_10d80": (
        "score_5d10_10d90",
        "control_score_5d30_10d70",
    ),
}
MIN_CAGR_IMPROVEMENT = 0.005
MIN_SHARPE_IMPROVEMENT = 0.01
MIN_DRAWDOWN_IMPROVEMENT = 0.005


def candidate_score(
    candidate_id: str,
    arrays: dict[str, np.ndarray],
    baseline_score: np.ndarray,
    baseline_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    all_weights = {**SELECTABLE_WEIGHTS_5D, **ROBUSTNESS_WEIGHTS_5D}
    if candidate_id not in all_weights:
        raise ValueError(f"unknown candidate: {candidate_id}")
    weight_5d = all_weights[candidate_id]
    if weight_5d == 0:
        return baseline_score, baseline_order
    raw = (
        float(weight_5d) * arrays["rank_5d"]
        + (1.0 - float(weight_5d)) * arrays["rank_10d"]
    )
    return score_tools.smooth_score(
        raw, score_tools.SMOOTHING_WINDOW, score_tools.RAW_ALPHA
    )


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
        raise PermissionError("2026 data entered minor score blend development")
    definition = harness.production_definition(protocol)
    baseline_score, baseline_order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    entry_block = np.zeros(baseline_score.shape, dtype=np.bool_)
    results = {}
    run_cache = {}
    score_cache = {}
    all_weights = {**SELECTABLE_WEIGHTS_5D, **ROBUSTNESS_WEIGHTS_5D}

    for candidate_id, weight_5d in all_weights.items():
        score, order = candidate_score(
            candidate_id, arrays, baseline_score, baseline_order
        )
        maintenance_block = quality.maintenance_quality_block(
            score, policy.get("maintenance_topup_requires_score") is not None
        )
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
            maintenance_buy_block_mask_override=maintenance_block,
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
            "score_formula": {
                "rank_5d": weight_5d,
                "rank_10d": 1.0 - weight_5d,
            },
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "pre2026_selection_utility": round1.selection_utility(metrics, stress),
            "window_start_metrics": windows,
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
        }
        run_cache[candidate_id] = (daily, actions)
        score_cache[candidate_id] = (score, order, maintenance_block)

    baseline_id = "score_10d"
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
        raise RuntimeError("minor score blend baseline drifted from checkpoint")

    ranked_id = max(
        SELECTABLE_WEIGHTS_5D,
        key=lambda key: results[key]["pre2026_selection_utility"],
    )
    baseline_utility = results[baseline_id]["pre2026_selection_utility"]
    neighbors = NEIGHBORS.get(ranked_id, ())
    neighborhood_robust = bool(
        neighbors
        and all(
            results[neighbor]["pre2026_selection_utility"] >= baseline_utility
            for neighbor in neighbors
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
    score, order, maintenance_block = score_cache[selected_id]
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
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("minor score blend deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "tuning_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "validation_boundary": ["20260101", "closed_until_final_one_shot"],
        "candidate_budget": list(SELECTABLE_WEIGHTS_5D),
        "predeclared_robustness_controls_not_selectable": list(
            ROBUSTNESS_WEIGHTS_5D
        ),
        "only_change": (
            "add only a small five-day component to the production ten-day score; "
            "all holding, equal-weight, exit, refill, and execution rules stay unchanged"
        ),
        "results": results,
        "selected_candidate_id": selected_id,
        "selected_reason": (
            "material pre-2026 improvement with neighboring weight robustness"
            if material_improvement
            else "candidate did not clear material and neighboring robustness gates"
        ),
        "practical_improvement_gate": {
            "raw_ranked_candidate_id": ranked_id,
            "neighbor_controls": list(neighbors),
            "neighborhood_robust": neighborhood_robust,
            "passed": material_improvement,
        },
        "selected_policy": {
            **copy.deepcopy(policy),
            "score_formula": results[selected_id]["score_formula"],
        },
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
