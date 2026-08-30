from __future__ import annotations

import copy
import hashlib
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
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_price_trailing_current_20260822"
)
THRESHOLDS = {
    "trailing_off": None,
    "trailing_drawdown_010": 0.10,
    "trailing_drawdown_015": 0.15,
    "trailing_drawdown_020": 0.20,
}


def candidate_policy(base_policy: dict, threshold: float | None) -> dict:
    policy = copy.deepcopy(base_policy)
    policy["price_peak_drawdown_exit"] = threshold
    return policy


def policy_difference(left: dict, right: dict) -> set[str]:
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}


def utility(metrics: dict) -> float:
    return float(
        metrics["cagr"]
        + 0.20 * metrics["sharpe"]
        - 0.50 * metrics["max_drawdown"]
        - 0.001 * metrics["turnover_annualized"]
    )


def frame_hash(frame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    base_policy = checkpoint["selected_policy"]
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered current trailing-exit development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), base_policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    common = dict(
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=base_policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=np.zeros(score.shape, dtype=np.bool_),
        maintenance_buy_block_mask_override=maintenance_block,
    )
    results = {}
    deterministic = {}
    for candidate_id, threshold in THRESHOLDS.items():
        policy = candidate_policy(base_policy, threshold)
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
        results[candidate_id] = {
            "policy": policy,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
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
        }
        deterministic[candidate_id] = {
            "daily_hash": frame_hash(daily),
            "actions_hash": frame_hash(actions),
        }

    baseline = results["trailing_off"]["metrics_0_30pct"]
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
        raise RuntimeError("trailing-exit baseline drifted from checkpoint")
    ranked = sorted(
        THRESHOLDS,
        key=lambda candidate_id: (
            results[candidate_id]["positive_annual_returns"],
            results[candidate_id]["utility"],
            results[candidate_id]["metrics_0_30pct"]["cagr"],
        ),
        reverse=True,
    )
    raw_winner = ranked[0]
    baseline_utility = results["trailing_off"]["utility"]
    threshold_candidates = [key for key in THRESHOLDS if key != "trailing_off"]
    neighborhood_improvements = {
        key: float(results[key]["utility"] - baseline_utility)
        for key in threshold_candidates
    }
    robust_candidate = bool(
        raw_winner != "trailing_off"
        and results[raw_winner]["positive_annual_returns"]
        and sum(value > 0 for value in neighborhood_improvements.values()) >= 2
        and results[raw_winner]["metrics_0_65pct"]["cumulative_return"]
        > results["trailing_off"]["metrics_0_65pct"]["cumulative_return"]
    )
    selected = raw_winner if robust_candidate else "trailing_off"
    result = {
        "status": "development_complete_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "only_rule_changed": "held-position peak-to-current qfq drawdown exit",
        "thresholds": THRESHOLDS,
        "results": results,
        "raw_winner": raw_winner,
        "neighborhood_utility_improvements": neighborhood_improvements,
        "neighborhood_robust": robust_candidate,
        "selected_candidate": selected,
        "selected_policy": results[selected]["policy"],
        "checkpoint_equivalence": checkpoint_equivalence,
        "policy_difference_from_checkpoint": sorted(
            policy_difference(base_policy, results[selected]["policy"])
        ),
        "deterministic_hashes": deterministic,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
