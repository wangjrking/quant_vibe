from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_hold_robustness_20260822"
)

POLICIES = {
    "daily1_max20_confirm1": (20, 1),
    "daily1_max20_confirm2": (20, 2),
    "daily1_max25_confirm1": (25, 1),
    "daily1_max25_confirm2": (25, 2),
    "daily1_max30_confirm1": (30, 1),
    "daily1_max30_confirm2": (30, 2),
}


def policy(max_hold: int, confirmation: int) -> dict:
    return {
        "min_hold_mode": "production_dynamic",
        "max_hold_days": int(max_hold),
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
        "sell_confirmation_days": int(confirmation),
        "renewal_policy": "no_score_exit",
        "age_bands": None,
        "max_daily_score_sells": 1,
    }


def window_start_date(arrays: dict[str, np.ndarray], offset: int) -> str:
    dates = arrays["dates"].astype(str)
    first_buy_index = int(np.searchsorted(dates, research_base.FIRST_BUY))
    target = min(first_buy_index + int(offset), len(dates) - 1)
    return str(dates[target])


def robust_key(item: dict) -> tuple:
    return (
        item["min_annual_return"],
        item["min_window_cagr"],
        item["median_annual_return"],
        item["metrics_0_30pct"]["sharpe"],
        -item["metrics_0_30pct"]["max_drawdown"],
    )


def walk_forward_selections(results: dict) -> list[dict]:
    folds = ((["2022", "2023"], "2024"), (["2022", "2023", "2024"], "2025"))
    evidence = []
    for train_years, test_year in folds:
        def train_key(policy_id: str) -> tuple:
            annual = results[policy_id]["metrics_0_30pct"]["annual_returns"]
            values = np.asarray([annual[year] for year in train_years], dtype=float)
            return float(np.min(values)), float(np.median(values)), policy_id

        selected = max(results, key=train_key)
        annual = results[selected]["metrics_0_30pct"]["annual_returns"]
        evidence.append(
            {
                "train_years": train_years,
                "selected_policy_id": selected,
                "forward_test_year": test_year,
                "forward_test_return": float(annual[test_year]),
            }
        )
    return evidence


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered robustness development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)

    results = {}
    for policy_id, (max_hold, confirmation) in POLICIES.items():
        current = policy(max_hold, confirmation)
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
        window_start_metrics = {}
        for offset in (5, 20, 60):
            window_start = window_start_date(arrays, offset)
            window_start_metrics[str(offset)] = round1.evaluate_run(
                daily,
                actions,
                window_start,
                round1.DEVELOPMENT_END,
            )
        annual_values = np.asarray(list(metrics["annual_returns"].values()), dtype=float)
        results[policy_id] = {
            "policy": current,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "window_start_metrics": window_start_metrics,
            "min_annual_return": float(np.min(annual_values)),
            "median_annual_return": float(np.median(annual_values)),
            "min_window_cagr": float(
                min(item["cagr"] for item in window_start_metrics.values())
            ),
        }

    eligible = {
        key: item
        for key, item in results.items()
        if item["min_annual_return"] >= 0.0
        and item["min_window_cagr"] > 0.0
        and item["metrics_0_65pct"]["cumulative_return"] > 0.0
        and item["metrics_0_30pct"]["full_10_position_ratio"] == 1.0
    }
    selection_pool = eligible or results
    selected_id = max(selection_pool, key=lambda key: robust_key(selection_pool[key]))
    replay_daily, replay_actions = round1.run_fixed10(
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
    )
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
    )
    deterministic_replay = {
        "daily": round1.frame_hash(replay_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(replay_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic_replay.values()):
        raise RuntimeError("selected fixed10 policy deterministic replay failed")
    walk_forward = walk_forward_selections(results)
    frozen = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_rule": (
            "max minimum calendar-year return, then minimum later-window CAGR, "
            "median annual return, Sharpe and drawdown"
        ),
        "score_rule_changed": False,
        "portfolio": {"positions": 10, "target_weight_each": 0.10, "gross_target": 1.0},
        "candidate_budget": list(POLICIES),
        "results": results,
        "eligible_candidates": list(eligible),
        "selected_policy_id": selected_id,
        "selected_policy": copy.deepcopy(results[selected_id]["policy"]),
        "deterministic_replay": deterministic_replay,
        "walk_forward_selection_evidence": walk_forward,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", frozen)
    print(
        json.dumps(
            {
                "status": frozen["status"],
                "selected": selected_id,
                "selected_result": results[selected_id],
                "eligible": list(eligible),
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
