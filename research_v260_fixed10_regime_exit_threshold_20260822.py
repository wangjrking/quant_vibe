from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_daily_exit_budget_20260822 as daily_exit
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_safe_v109 as safe_v109


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_regime_exit_threshold_20260822"
)
WEAK_MARKET_EXIT_THRESHOLDS = (0.80, 0.85, 0.90)
STRONG_MARKET_EXIT_THRESHOLD = 0.85


def candidate_id(weak_threshold: float) -> str:
    return f"weak_exit_{weak_threshold:.2f}_strong_exit_{STRONG_MARKET_EXIT_THRESHOLD:.2f}"


def candidate_policy(weak_threshold: float) -> dict:
    current = daily_exit.candidate_policy(1)
    current["weak_market_sell_score_below"] = float(weak_threshold)
    current["strong_market_sell_score_below"] = STRONG_MARKET_EXIT_THRESHOLD
    current["market_state_source"] = "production_breadth_state"
    return current


def strong_market_mask(score: np.ndarray, protocol: dict) -> np.ndarray:
    breadth = np.sum(
        np.isfinite(score)
        & (score >= float(protocol["breadth"]["score_threshold"])),
        axis=1,
    )
    state = protocol["breadth_state"]
    above = breadth >= int(state["breadth_count_threshold"])
    return above if state["high_when"] == "above" else ~above


def threshold_schedule(score: np.ndarray, protocol: dict, weak_threshold: float) -> np.ndarray:
    high_state = strong_market_mask(score, protocol)
    return np.where(
        high_state,
        STRONG_MARKET_EXIT_THRESHOLD,
        float(weak_threshold),
    ).astype(np.float64)


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
        raise PermissionError("2026 data entered regime-exit development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    high_state = strong_market_mask(score, protocol)

    results = {}
    run_cache = {}
    for weak_threshold in WEAK_MARKET_EXIT_THRESHOLDS:
        current_id = candidate_id(weak_threshold)
        current = candidate_policy(weak_threshold)
        schedule = threshold_schedule(score, protocol, weak_threshold)
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
            simulator=safe_v109.simulate,
            sell_score_below_override=schedule,
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
            simulator=safe_v109.simulate,
            sell_score_below_override=schedule,
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
            "weak_market_days": int(np.sum(~high_state)),
            "strong_market_days": int(np.sum(high_state)),
            "min_annual_return": float(np.min(annual_values)),
            "median_annual_return": float(np.median(annual_values)),
            "min_window_cagr": float(
                min(item["cagr"] for item in window_metrics.values())
            ),
        }
        run_cache[current_id] = (daily, actions)

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
    selected_daily, selected_actions = run_cache[selected_id]
    selected_policy = results[selected_id]["policy"]
    selected_schedule = threshold_schedule(
        score,
        protocol,
        selected_policy["weak_market_sell_score_below"],
    )
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
        simulator=safe_v109.simulate,
        sell_score_below_override=selected_schedule,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("regime-exit deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "score_rule_changed": False,
        "market_timing_added": False,
        "portfolio": {"positions": 10, "target_weight_each": 0.10, "gross_target": 1.0},
        "candidate_budget": [float(value) for value in WEAK_MARKET_EXIT_THRESHOLDS],
        "selection_rule": (
            "all 2022-2024 calendar returns positive, then training Sharpe, CAGR, "
            "drawdown and turnover; 2025 is confirmation only"
        ),
        "results": results,
        "walk_forward_selection_evidence": robustness.walk_forward_selections(results),
        "eligible_candidates": list(eligible),
        "selected_candidate_id": selected_id,
        "selected_policy": copy.deepcopy(selected_policy),
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
                "walk_forward": result["walk_forward_selection_evidence"],
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
