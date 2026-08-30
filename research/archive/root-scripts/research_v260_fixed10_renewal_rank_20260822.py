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
import research_v260_fixed10_portfolio_rebalance_band_20260822 as rebalance_band
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_rebalance_band_v109 as rebalance_band_v109
from research_v260_runtime import fixed10_renewal_rank_v109 as renewal_rank_v109


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_renewal_rank_20260822"
)
MAX_HOLD_RENEWAL_RANKS = (None, 10, 20, 30)
PORTFOLIO_REBALANCE_INTERVAL_DAYS = 20
PORTFOLIO_REBALANCE_BAND = 0.01
SHARPE_PRACTICAL_EQUIVALENCE = 0.01


def candidate_id(rank: int | None) -> str:
    return "renewal_score_080" if rank is None else f"renewal_rank_top{rank:02d}"


def candidate_policy(rank: int | None) -> dict:
    current = rebalance_band.candidate_policy(PORTFOLIO_REBALANCE_BAND)
    current["max_hold_renewal_rank"] = rank
    return current


def selection_key(item: dict) -> tuple:
    train = item["train_2022_2024"]
    return (
        train["sharpe"],
        train["cagr"],
        -train["max_drawdown"],
        -train["turnover_annualized"],
    )


def robust_selection(results: dict, selection_pool: dict, walk_forward: list[dict]) -> tuple[str, str]:
    aggregate_best = max(
        selection_pool, key=lambda key: selection_key(selection_pool[key])
    )
    fold_winners = {item["selected_policy_id"] for item in walk_forward}
    if len(fold_winners) == 1:
        stable_id = next(iter(fold_winners))
        if stable_id in selection_pool:
            best_sharpe = selection_pool[aggregate_best]["train_2022_2024"]["sharpe"]
            stable_sharpe = selection_pool[stable_id]["train_2022_2024"]["sharpe"]
            if best_sharpe - stable_sharpe <= SHARPE_PRACTICAL_EQUIVALENCE:
                return stable_id, "walk_forward_consistent_within_sharpe_equivalence_band"
    return aggregate_best, "aggregate_training_selection"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered renewal-rank development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)

    results = {}
    run_cache = {}
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), PORTFOLIO_REBALANCE_INTERVAL_DAYS
    )
    for rank in MAX_HOLD_RENEWAL_RANKS:
        current_id = candidate_id(rank)
        current = candidate_policy(rank)
        daily, actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            current,
            round1.DEVELOPMENT_END,
            slip=round1.BASELINE_COST,
            record_actions=True,
            simulator=renewal_rank_v109.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=PORTFOLIO_REBALANCE_BAND,
            max_hold_renewal_rank_override=rank,
        )
        stress_daily, stress_actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            current,
            round1.DEVELOPMENT_END,
            slip=round1.STRESS_COST,
            record_actions=True,
            simulator=renewal_rank_v109.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=PORTFOLIO_REBALANCE_BAND,
            max_hold_renewal_rank_override=rank,
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
        holdout_2025 = round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        )
        window_metrics = {
            str(offset): round1.evaluate_run(
                daily,
                actions,
                robustness.window_start_date(arrays, offset),
                round1.DEVELOPMENT_END,
            )
            for offset in (5, 20, 60)
        }
        annual_values = np.asarray(list(metrics["annual_returns"].values()), dtype=float)
        results[current_id] = {
            "policy": current,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout_2025,
            "window_start_metrics": window_metrics,
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "min_annual_return": float(np.min(annual_values)),
            "median_annual_return": float(np.median(annual_values)),
            "min_window_cagr": float(
                min(item["cagr"] for item in window_metrics.values())
            ),
        }
        run_cache[current_id] = (daily, actions)

    base_policy = rebalance_band.candidate_policy(PORTFOLIO_REBALANCE_BAND)
    base_daily, base_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        base_policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=rebalance_band_v109.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=PORTFOLIO_REBALANCE_BAND,
    )
    disabled_daily, disabled_actions = run_cache[candidate_id(None)]
    disabled_equivalence = {
        "daily": round1.frame_hash(base_daily) == round1.frame_hash(disabled_daily),
        "actions": round1.frame_hash(base_actions) == round1.frame_hash(disabled_actions),
    }
    if not all(disabled_equivalence.values()):
        raise RuntimeError("renewal-rank runtime changed the disabled path")

    eligible = {
        key: item
        for key, item in results.items()
        if min(item["train_2022_2024"]["annual_returns"].values()) >= 0.0
        and item["min_window_cagr"] > 0.0
        and item["metrics_0_65pct"]["cumulative_return"] > 0.0
        and item["metrics_0_30pct"]["full_10_position_ratio"] == 1.0
    }
    selection_pool = eligible or results
    walk_forward = robustness.walk_forward_selections(results)
    selected_id, selection_reason = robust_selection(
        results, selection_pool, walk_forward
    )
    selected_rank = results[selected_id]["policy"]["max_hold_renewal_rank"]
    selected_daily, selected_actions = run_cache[selected_id]
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        results[selected_id]["policy"],
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=renewal_rank_v109.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=PORTFOLIO_REBALANCE_BAND,
        max_hold_renewal_rank_override=selected_rank,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("renewal-rank deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "score_rule_changed": False,
        "portfolio": {"positions": 10, "target_weight_each": 0.10, "gross_target": 1.0},
        "candidate_budget": [
            None if value is None else int(value)
            for value in MAX_HOLD_RENEWAL_RANKS
        ],
        "selection_rule": {
            "training_boundary": "2022-2024",
            "sharpe_practical_equivalence": SHARPE_PRACTICAL_EQUIVALENCE,
            "preference_within_band": "walk_forward_consistency_then_lower_complexity",
            "selected_reason": selection_reason,
        },
        "results": results,
        "walk_forward_selection_evidence": walk_forward,
        "eligible_candidates": list(eligible),
        "selected_candidate_id": selected_id,
        "selected_policy": copy.deepcopy(results[selected_id]["policy"]),
        "disabled_path_equivalent_to_current_candidate": disabled_equivalence,
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
                "status": result["status"],
                "selected": selected_id,
                "selected_result": results[selected_id],
                "default_equivalence": disabled_equivalence,
                "walk_forward": result["walk_forward_selection_evidence"],
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
