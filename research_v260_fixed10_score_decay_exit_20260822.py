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
from research_v260_runtime import fixed10_score_decay_v109 as score_decay_v109


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_score_decay_exit_20260822"
)
SCORE_PEAK_DROP_THRESHOLDS = (None, 0.10, 0.15, 0.20)


def candidate_id(threshold: float | None) -> str:
    return "score_peak_drop_off" if threshold is None else f"score_peak_drop_{threshold:.2f}"


def candidate_policy(threshold: float | None) -> dict:
    current = daily_exit.candidate_policy(1)
    current["score_peak_drop_exit"] = threshold
    return current


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
        raise PermissionError("2026 data entered score-decay development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)

    results = {}
    run_cache = {}
    for threshold in SCORE_PEAK_DROP_THRESHOLDS:
        current_id = candidate_id(threshold)
        current = candidate_policy(threshold)
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
            simulator=score_decay_v109.simulate,
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
            simulator=score_decay_v109.simulate,
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
            "min_annual_return": float(np.min(annual_values)),
            "median_annual_return": float(np.median(annual_values)),
            "min_window_cagr": float(
                min(item["cagr"] for item in window_metrics.values())
            ),
        }
        run_cache[current_id] = (daily, actions)

    current_policy = candidate_policy(None)
    safe_daily, safe_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        current_policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=safe_v109.simulate,
    )
    decay_off_daily, decay_off_actions = run_cache[candidate_id(None)]
    default_equivalence = {
        "daily": round1.frame_hash(safe_daily)
        == round1.frame_hash(decay_off_daily),
        "actions": round1.frame_hash(safe_actions)
        == round1.frame_hash(decay_off_actions),
    }
    if not all(default_equivalence.values()):
        raise RuntimeError("score-decay runtime changed the disabled path")

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
        simulator=score_decay_v109.simulate,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("score-decay deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "score_rule_changed": False,
        "portfolio": {"positions": 10, "target_weight_each": 0.10, "gross_target": 1.0},
        "candidate_budget": [
            None if value is None else float(value)
            for value in SCORE_PEAK_DROP_THRESHOLDS
        ],
        "selection_rule": (
            "all 2022-2024 calendar returns positive, then training Sharpe, CAGR, "
            "drawdown and turnover; 2025 is confirmation only"
        ),
        "results": results,
        "walk_forward_selection_evidence": robustness.walk_forward_selections(results),
        "eligible_candidates": list(eligible),
        "selected_candidate_id": selected_id,
        "selected_policy": copy.deepcopy(results[selected_id]["policy"]),
        "disabled_path_equivalent_to_previous_runtime": default_equivalence,
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
                "default_equivalence": default_equivalence,
                "walk_forward": result["walk_forward_selection_evidence"],
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
