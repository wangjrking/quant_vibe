from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_coordinate_recheck_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
MARGINS = (0.03, 0.05, 0.07)
MAX_HOLDS = (20, 25, 30)


def training_key(item: dict) -> tuple:
    train = item["train_2022_2024"]
    return (
        train["sharpe"],
        train["cagr"],
        -train["max_drawdown"],
        -train["turnover_annualized"],
    )


def confirmation_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    return {
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "full_10_positions": candidate["metrics_0_30pct"]["full_10_position_ratio"]
        == 1.0,
        "all_calendar_years_positive": min(
            candidate["metrics_0_30pct"]["annual_returns"].values()
        )
        > 0.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered pressure coordinate recheck")
    base_policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered pressure coordinate recheck")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), base_policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    empty_block = np.zeros(score.shape, dtype=np.bool_)

    def run(policy: dict) -> tuple[dict, object, object]:
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=empty_block,
            maintenance_buy_block_mask_override=maintenance_block,
            score_sell_pressure_trigger_override=policy[
                "score_sell_pressure_trigger"
            ],
            score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
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
        result = {
            "policy": copy.deepcopy(policy),
            "metrics_0_30pct": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            ),
            "train_2022_2024": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025": round1.evaluate_run(
                daily, actions, "20250102", round1.DEVELOPMENT_END
            ),
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
        }
        return result, daily, actions

    margin_results = {}
    for margin in MARGINS:
        policy = copy.deepcopy(base_policy)
        policy["replacement_advantage"] = float(margin)
        margin_results[f"margin_{int(round(margin * 100)):02d}"] = run(policy)[0]
    margin_baseline_id = "margin_05"
    margin_ranked_id = max(margin_results, key=lambda key: training_key(margin_results[key]))
    margin_gates = confirmation_gates(
        margin_results[margin_ranked_id], margin_results[margin_baseline_id]
    )
    selected_margin_id = (
        margin_ranked_id if all(margin_gates.values()) else margin_baseline_id
    )
    selected_policy = copy.deepcopy(margin_results[selected_margin_id]["policy"])

    max_hold_results = {}
    final_cache = {}
    for max_hold in MAX_HOLDS:
        policy = copy.deepcopy(selected_policy)
        policy["max_hold_days"] = int(max_hold)
        item, daily, actions = run(policy)
        candidate_id = f"max_hold_{max_hold}"
        max_hold_results[candidate_id] = item
        final_cache[candidate_id] = (daily, actions)
    max_hold_baseline_id = "max_hold_25"
    max_hold_ranked_id = max(
        max_hold_results, key=lambda key: training_key(max_hold_results[key])
    )
    max_hold_gates = confirmation_gates(
        max_hold_results[max_hold_ranked_id], max_hold_results[max_hold_baseline_id]
    )
    selected_max_hold_id = (
        max_hold_ranked_id if all(max_hold_gates.values()) else max_hold_baseline_id
    )
    selected_result = max_hold_results[selected_max_hold_id]
    selected_daily, selected_actions = final_cache[selected_max_hold_id]
    repeat_result, repeat_daily, repeat_actions = run(selected_result["policy"])
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure coordinate replay failed")

    original_metrics = checkpoint["current_best_equalweight"]
    selected_metrics = selected_result["metrics_0_30pct"]
    checkpoint_equivalence = {
        "baseline_margin_and_max_hold_retained": (
            selected_margin_id == margin_baseline_id
            and selected_max_hold_id == max_hold_baseline_id
        ),
        "repeat_metrics_equal": all(
            np.isclose(
                selected_metrics[key],
                repeat_result["metrics_0_30pct"][key],
                rtol=0.0,
                atol=1e-12,
            )
            for key in ("cagr", "sharpe", "max_drawdown")
        ),
    }
    result = {
        "status": "sequential_coordinate_recheck_complete_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "method": (
            "coarse sequential coordinates only: replacement margin, then max hold; "
            "no cross-product grid"
        ),
        "margin_budget": list(MARGINS),
        "margin_results": margin_results,
        "margin_training_ranked": margin_ranked_id,
        "margin_confirmation_gates": margin_gates,
        "selected_margin": selected_margin_id,
        "max_hold_budget": list(MAX_HOLDS),
        "max_hold_results": max_hold_results,
        "max_hold_training_ranked": max_hold_ranked_id,
        "max_hold_confirmation_gates": max_hold_gates,
        "selected_max_hold": selected_max_hold_id,
        "selected_policy": selected_result["policy"],
        "changed_from_checkpoint": any(
            not np.isclose(selected_metrics[key], original_metrics[key])
            for key in ("cagr", "sharpe", "max_drawdown")
        ),
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
