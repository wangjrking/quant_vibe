from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_equalweight_maintenance_20260822 as round3
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as round2
import research_v260_fixed10_renewal_policy_20260822 as round10
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_safe_v109 as safe_v109


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_exit_confirmation_20260822"
)
CONFIRMATION_DAYS = (1, 2, 3)


def candidate_policy(confirmation_days: int) -> dict:
    current = round10.candidate_policy("no_score_exit_rebalance")
    current["sell_confirmation_days"] = int(confirmation_days)
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
        raise PermissionError("2026 data entered exit-confirmation development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)

    results = {}
    run_cache = {}
    for confirmation_days in CONFIRMATION_DAYS:
        candidate_id = f"confirmation_{confirmation_days}d"
        current = candidate_policy(confirmation_days)
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
                round2.window_start_date(arrays, offset),
                round1.DEVELOPMENT_END,
            )
            for offset in (5, 20, 60)
        }
        annual_values = np.asarray(list(metrics["annual_returns"].values()), dtype=float)
        results[candidate_id] = {
            "policy": current,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout_2025,
            "window_start_metrics": window_metrics,
            "action_diagnostics": round3.maintenance_action_metrics(actions),
            "min_annual_return": float(np.min(annual_values)),
            "median_annual_return": float(np.median(annual_values)),
            "min_window_cagr": float(
                min(item["cagr"] for item in window_metrics.values())
            ),
        }
        run_cache[candidate_id] = (daily, actions)

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
        simulator=safe_v109.simulate,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("exit-confirmation deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "score_rule_changed": False,
        "portfolio": {"positions": 10, "target_weight_each": 0.10, "gross_target": 1.0},
        "candidate_budget": [int(value) for value in CONFIRMATION_DAYS],
        "selection_rule": (
            "all 2022-2024 calendar returns positive, then training Sharpe, CAGR, "
            "drawdown and turnover; 2025 is confirmation only"
        ),
        "results": results,
        "eligible_candidates": list(eligible),
        "selected_candidate_id": selected_id,
        "selected_policy": copy.deepcopy(results[selected_id]["policy"]),
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
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
