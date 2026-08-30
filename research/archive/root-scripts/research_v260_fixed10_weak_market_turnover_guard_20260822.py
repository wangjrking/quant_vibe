from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_daily_exit_after_renewal_gate_20260822 as final_round
import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_renewal_quality_v109 as renewal_quality_v109


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_weak_market_turnover_guard_20260822"
)
GUARD_CANDIDATES = (
    {
        "candidate_id": "weak_turnover_guard_off",
        "weak_sell_score_below": 0.85,
        "weak_renewal_score": 0.80,
    },
    {
        "candidate_id": "weak_suppress_score_exit",
        "weak_sell_score_below": 0.00,
        "weak_renewal_score": 0.80,
    },
    {
        "candidate_id": "weak_suppress_score_and_renewal_exit",
        "weak_sell_score_below": 0.00,
        "weak_renewal_score": 0.00,
    },
)
STRONG_SELL_SCORE_BELOW = 0.85
STRONG_RENEWAL_SCORE = 0.80
TARGET_CRASH_WINDOW = ("20231205", "20240208")


def candidate_policy(candidate: dict) -> dict:
    current = final_round.candidate_policy(1)
    current["weak_market_turnover_guard"] = candidate["candidate_id"]
    current["weak_market_sell_score_below"] = float(
        candidate["weak_sell_score_below"]
    )
    current["weak_market_renewal_score"] = float(candidate["weak_renewal_score"])
    current["market_state_source"] = "production_breadth_state"
    return current


def schedules(score: np.ndarray, protocol: dict, candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    strong = regime.strong_market_mask(score, protocol)
    sell_threshold = np.where(
        strong,
        STRONG_SELL_SCORE_BELOW,
        float(candidate["weak_sell_score_below"]),
    ).astype(np.float64)
    renewal_threshold = np.where(
        strong,
        STRONG_RENEWAL_SCORE,
        float(candidate["weak_renewal_score"]),
    ).astype(np.float64)
    return sell_threshold, renewal_threshold


def selection_key(item: dict) -> tuple:
    train = item["train_2022_2024"]
    return (
        train["sharpe"],
        train["cagr"],
        -train["max_drawdown"],
        -train["turnover_annualized"],
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered weak-market turnover-guard development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    strong = regime.strong_market_mask(score, protocol)

    results = {}
    run_cache = {}
    for candidate in GUARD_CANDIDATES:
        current_id = candidate["candidate_id"]
        current = candidate_policy(candidate)
        sell_schedule, renewal_schedule = schedules(score, protocol, candidate)
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
            simulator=renewal_quality_v109.simulate,
            sell_score_below_override=sell_schedule,
            max_hold_renewal_score_override=renewal_schedule,
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
            simulator=renewal_quality_v109.simulate,
            sell_score_below_override=sell_schedule,
            max_hold_renewal_score_override=renewal_schedule,
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
            "target_crash_window_actions": attribution.actions_in_window(
                actions, *TARGET_CRASH_WINDOW
            ),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "weak_market_days": int(np.sum(~strong)),
            "strong_market_days": int(np.sum(strong)),
            "min_annual_return": float(np.min(annual_values)),
            "median_annual_return": float(np.median(annual_values)),
            "min_window_cagr": float(
                min(item["cagr"] for item in window_metrics.values())
            ),
        }
        run_cache[current_id] = (daily, actions)

    base_candidate = GUARD_CANDIDATES[0]
    base_policy = final_round.candidate_policy(1)
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
        simulator=renewal_quality_v109.simulate,
    )
    scheduled_daily, scheduled_actions = run_cache[base_candidate["candidate_id"]]
    disabled_equivalence = {
        "daily": round1.frame_hash(base_daily) == round1.frame_hash(scheduled_daily),
        "actions": round1.frame_hash(base_actions)
        == round1.frame_hash(scheduled_actions),
    }
    if not all(disabled_equivalence.values()):
        raise RuntimeError("weak-market guard changed the disabled path")

    eligible = {
        key: item
        for key, item in results.items()
        if min(item["train_2022_2024"]["annual_returns"].values()) >= 0.0
        and item["min_window_cagr"] > 0.0
        and item["metrics_0_65pct"]["cumulative_return"] > 0.0
        and item["metrics_0_30pct"]["full_10_position_ratio"] == 1.0
    }
    selection_pool = eligible or results
    selected_id = max(selection_pool, key=lambda key: selection_key(selection_pool[key]))
    selected_candidate = next(
        item for item in GUARD_CANDIDATES if item["candidate_id"] == selected_id
    )
    selected_policy = results[selected_id]["policy"]
    selected_daily, selected_actions = run_cache[selected_id]
    selected_sell, selected_renewal = schedules(score, protocol, selected_candidate)
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
        simulator=renewal_quality_v109.simulate,
        sell_score_below_override=selected_sell,
        max_hold_renewal_score_override=selected_renewal,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("weak-market turnover-guard replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "score_rule_changed": False,
        "portfolio": {"positions": 10, "target_weight_each": 0.10, "gross_target": 1.0},
        "candidate_budget": [item["candidate_id"] for item in GUARD_CANDIDATES],
        "results": results,
        "walk_forward_selection_evidence": robustness.walk_forward_selections(results),
        "eligible_candidates": list(eligible),
        "selected_candidate_id": selected_id,
        "selected_policy": copy.deepcopy(selected_policy),
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
