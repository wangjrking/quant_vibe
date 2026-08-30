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
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_renewal_maintenance_20260822 as renewal
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_asymmetric_rebalance_v111 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_rebalance_40d_refinement_20260822"
)
CANDIDATE_IDS = (
    "portfolio_rebalance_20d",
    "portfolio_rebalance_40d",
    "portfolio_rebalance_20d_strong_market_only",
    "portfolio_rebalance_5d_strong_market_only",
    "portfolio_topup_5d_never_trim",
    "portfolio_topup_5d_never_trim_score085",
)
MIN_CAGR_IMPROVEMENT = 0.005
MIN_SHARPE_IMPROVEMENT = 0.01
MIN_DRAWDOWN_IMPROVEMENT = 0.005


def candidate_id(interval: int) -> str:
    return f"portfolio_rebalance_{int(interval)}d"


def candidate_schedule(candidate: str, score: np.ndarray, protocol: dict) -> np.ndarray:
    if candidate == "portfolio_rebalance_20d":
        return cadence.rebalance_schedule(len(score), 20)
    if candidate == "portfolio_rebalance_40d":
        return cadence.rebalance_schedule(len(score), 40)
    if candidate == "portfolio_rebalance_20d_strong_market_only":
        return cadence.rebalance_schedule(len(score), 20) & regime.strong_market_mask(
            score, protocol
        )
    if candidate == "portfolio_rebalance_5d_strong_market_only":
        return cadence.rebalance_schedule(len(score), 5) & regime.strong_market_mask(
            score, protocol
        )
    if candidate in {
        "portfolio_topup_5d_never_trim",
        "portfolio_topup_5d_never_trim_score085",
    }:
        return cadence.rebalance_schedule(len(score), 5)
    raise ValueError(f"unknown candidate: {candidate}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    if policy["portfolio_rebalance_interval_days"] != 20:
        raise ValueError("current checkpoint must anchor the 20-session cadence")

    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered 40-day rebalance development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    entry_block = np.zeros(score.shape, dtype=np.bool_)

    results = {}
    run_cache = {}
    for current_id in CANDIDATE_IDS:
        active = candidate_schedule(current_id, score, protocol)
        topup_threshold = (
            0.85
            if current_id == "portfolio_topup_5d_never_trim_score085"
            else 0.80
        )
        maintenance_block = ~np.isfinite(score) | (score < topup_threshold)
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            portfolio_rebalance_overweight_deviation_override=(
                10.0
                if current_id
                in {
                    "portfolio_topup_5d_never_trim",
                    "portfolio_topup_5d_never_trim_score085",
                }
                else 0.01
            ),
            portfolio_rebalance_underweight_deviation_override=0.01,
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
        train = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        )
        holdout = round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
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
        results[current_id] = {
            "policy": {
                **policy,
                "portfolio_rebalance_interval_days": (
                    40
                    if current_id == "portfolio_rebalance_40d"
                    else 5
                    if current_id
                    in {
                        "portfolio_rebalance_5d_strong_market_only",
                        "portfolio_topup_5d_never_trim",
                        "portfolio_topup_5d_never_trim_score085",
                    }
                    else 20
                ),
                "portfolio_rebalance_market_gate": (
                    "strong_market_only"
                    if current_id
                    in {
                        "portfolio_rebalance_20d_strong_market_only",
                        "portfolio_rebalance_5d_strong_market_only",
                    }
                    else None
                ),
                "portfolio_rebalance_overweight_deviation": (
                    10.0
                    if current_id
                    in {
                        "portfolio_topup_5d_never_trim",
                        "portfolio_topup_5d_never_trim_score085",
                    }
                    else 0.01
                ),
                "portfolio_rebalance_underweight_deviation": 0.01,
                "maintenance_topup_requires_score": topup_threshold,
            },
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout,
            "window_start_metrics": windows,
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
        }
        run_cache[current_id] = (daily, actions)

    baseline_id = "portfolio_rebalance_20d"
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
        raise RuntimeError("40-day rebalance baseline drifted from checkpoint")

    ranked_id = max(results, key=lambda key: renewal.selection_key(results[key]))
    baseline_train = results[baseline_id]["train_2022_2024"]
    ranked_train = results[ranked_id]["train_2022_2024"]
    material_improvement = bool(
        ranked_id != baseline_id
        and (
            ranked_train["cagr"] - baseline_train["cagr"]
            >= MIN_CAGR_IMPROVEMENT
            or ranked_train["sharpe"] - baseline_train["sharpe"]
            >= MIN_SHARPE_IMPROVEMENT
            or baseline_train["max_drawdown"] - ranked_train["max_drawdown"]
            >= MIN_DRAWDOWN_IMPROVEMENT
        )
    )
    selected_id = ranked_id if material_improvement else baseline_id
    selected_daily, selected_actions = run_cache[selected_id]
    selected_topup_threshold = results[selected_id]["policy"][
        "maintenance_topup_requires_score"
    ]
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
        portfolio_rebalance_active_override=candidate_schedule(
            selected_id, score, protocol
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        portfolio_rebalance_overweight_deviation_override=results[selected_id][
            "policy"
        ]["portfolio_rebalance_overweight_deviation"],
        portfolio_rebalance_underweight_deviation_override=0.01,
        entry_block_mask_override=entry_block,
        maintenance_buy_block_mask_override=(
            ~np.isfinite(score) | (score < selected_topup_threshold)
        ),
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("40-day rebalance deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATE_IDS),
        "only_change": (
            "periodic equal-weight maintenance cadence, including a strong-market-only "
            "gate that leaves holdings, scores, exits, and gross target unchanged"
        ),
        "results": results,
        "walk_forward_selection_evidence": robustness.walk_forward_selections(results),
        "selected_candidate_id": selected_id,
        "selected_reason": (
            "material aggregate 2022-2024 improvement"
            if material_improvement
            else "alternative improvement was below the predeclared practical threshold"
        ),
        "practical_improvement_gate": {
            "minimum_cagr_improvement": MIN_CAGR_IMPROVEMENT,
            "minimum_sharpe_improvement": MIN_SHARPE_IMPROVEMENT,
            "minimum_drawdown_improvement": MIN_DRAWDOWN_IMPROVEMENT,
            "raw_ranked_candidate_id": ranked_id,
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
                "results": results,
                "walk_forward": result["walk_forward_selection_evidence"],
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
