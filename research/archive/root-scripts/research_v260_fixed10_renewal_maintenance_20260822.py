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
import research_v260_fixed10_portfolio_rebalance_band_20260822 as rebalance_band
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_rebalance_band_v109 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_renewal_maintenance_20260822"
)
CANDIDATES = {
    "renewal_reset_only": "no_score_exit",
    "renewal_reset_and_rebalance": "no_score_exit_rebalance",
}
PORTFOLIO_REBALANCE_INTERVAL_DAYS = 20
PORTFOLIO_REBALANCE_BAND = 0.01
SHARPE_PRACTICAL_EQUIVALENCE = 0.01


def candidate_policy(candidate_id: str) -> dict:
    if candidate_id not in CANDIDATES:
        raise KeyError(candidate_id)
    policy = rebalance_band.candidate_policy(PORTFOLIO_REBALANCE_BAND)
    policy["renewal_policy"] = CANDIDATES[candidate_id]
    return policy


def selection_key(item: dict) -> tuple:
    train = item["train_2022_2024"]
    return (
        train["sharpe"],
        train["cagr"],
        -train["max_drawdown"],
        -train["turnover_annualized"],
    )


def robust_selection(results: dict, walk_forward: list[dict]) -> tuple[str, str]:
    aggregate_best = max(results, key=lambda key: selection_key(results[key]))
    fold_winners = {item["selected_policy_id"] for item in walk_forward}
    if len(fold_winners) == 1:
        stable = next(iter(fold_winners))
        best_sharpe = results[aggregate_best]["train_2022_2024"]["sharpe"]
        stable_sharpe = results[stable]["train_2022_2024"]["sharpe"]
        if best_sharpe - stable_sharpe <= SHARPE_PRACTICAL_EQUIVALENCE:
            return stable, "walk_forward_consistent_within_sharpe_equivalence_band"
    return aggregate_best, "aggregate_training_selection"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered renewal-maintenance development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), PORTFOLIO_REBALANCE_INTERVAL_DAYS
    )

    results = {}
    run_cache = {}
    for candidate_id in CANDIDATES:
        policy = candidate_policy(candidate_id)
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
            portfolio_rebalance_min_weight_deviation_override=PORTFOLIO_REBALANCE_BAND,
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
            portfolio_rebalance_min_weight_deviation_override=PORTFOLIO_REBALANCE_BAND,
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
            "policy": policy,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout,
            "window_start_metrics": windows,
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
        }
        run_cache[candidate_id] = (daily, actions)

    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    current_metrics = results["renewal_reset_and_rebalance"]["metrics_0_30pct"]
    checkpoint_metrics = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(current_metrics[key], checkpoint_metrics[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("current renewal-maintenance candidate drifted from checkpoint")

    walk_forward = robustness.walk_forward_selections(results)
    selected_id, selected_reason = robust_selection(results, walk_forward)
    selected_daily, selected_actions = run_cache[selected_id]
    selected_policy = results[selected_id]["policy"]
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        selected_policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=PORTFOLIO_REBALANCE_BAND,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("renewal-maintenance deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATES),
        "only_change": "whether max-hold renewal also forces a quantity rebalance",
        "selection_rule": {
            "training_boundary": "2022-2024",
            "sharpe_practical_equivalence": SHARPE_PRACTICAL_EQUIVALENCE,
            "selected_reason": selected_reason,
        },
        "results": results,
        "walk_forward_selection_evidence": walk_forward,
        "selected_candidate_id": selected_id,
        "selected_policy": copy.deepcopy(selected_policy),
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
                "checkpoint_equivalence": checkpoint_equivalence,
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
