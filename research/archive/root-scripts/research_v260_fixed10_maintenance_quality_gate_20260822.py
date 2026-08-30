from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_renewal_maintenance_20260822 as renewal
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_maintenance_quality_gate_20260822"
)
CANDIDATES = ("current_topup", "topup_requires_score_080")
MAINTENANCE_SCORE_THRESHOLD = 0.80


def maintenance_quality_block(score: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.zeros(score.shape, dtype=np.bool_)
    return ~np.isfinite(score) | (score < MAINTENANCE_SCORE_THRESHOLD)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    if policy["max_hold_renewal_score"] != MAINTENANCE_SCORE_THRESHOLD:
        raise ValueError("maintenance quality threshold must reuse the renewal threshold")

    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered maintenance-quality development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    entry_block = np.zeros(score.shape, dtype=np.bool_)

    results = {}
    run_cache = {}
    for candidate_id in CANDIDATES:
        maintenance_block = maintenance_quality_block(
            score, candidate_id == "topup_requires_score_080"
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
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
            maintenance_buy_block_mask_override=maintenance_block,
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
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
            maintenance_buy_block_mask_override=maintenance_block,
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
        train = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        )
        holdout = round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
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
                "maintenance_topup_requires_score": (
                    None
                    if candidate_id == "current_topup"
                    else MAINTENANCE_SCORE_THRESHOLD
                ),
            },
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout,
            "window_start_metrics": windows,
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
            "blocked_stock_dates": int(maintenance_block.sum()),
        }
        run_cache[candidate_id] = (daily, actions)

    current = results["current_topup"]["metrics_0_30pct"]
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
        raise RuntimeError("maintenance-quality current path drifted from checkpoint")

    walk_forward = robustness.walk_forward_selections(results)
    selected_id, selected_reason = renewal.robust_selection(results, walk_forward)
    selected_daily, selected_actions = run_cache[selected_id]
    selected_block = maintenance_quality_block(
        score, selected_id == "topup_requires_score_080"
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
        entry_block_mask_override=entry_block,
        maintenance_buy_block_mask_override=selected_block,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("maintenance-quality deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATES),
        "only_change": "whether a periodic maintenance buy requires the existing 0.80 renewal score",
        "results": results,
        "walk_forward_selection_evidence": walk_forward,
        "selected_candidate_id": selected_id,
        "selected_reason": selected_reason,
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
                "selected_reason": selected_reason,
                "results": results,
                "walk_forward": walk_forward,
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
