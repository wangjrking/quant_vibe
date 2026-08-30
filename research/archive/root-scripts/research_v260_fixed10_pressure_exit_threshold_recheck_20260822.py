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

import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_exit_threshold_recheck_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
THRESHOLDS = (0.80, 0.85, 0.90)
BASELINE_ID = "sell_below_085"


def candidate_policy(base_policy: dict, threshold: float) -> dict:
    policy = copy.deepcopy(base_policy)
    policy["sell_score_below"] = float(threshold)
    return policy


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
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered pressure threshold recheck")
    base_policy = checkpoint["selected_policy"]
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered pressure threshold recheck")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), base_policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    empty_block = np.zeros(score.shape, dtype=np.bool_)
    common = dict(
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=base_policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=base_policy[
            "score_sell_pressure_trigger"
        ],
        score_sell_pressure_limit_override=base_policy["score_sell_pressure_limit"],
    )
    results, cache = {}, {}
    for threshold in THRESHOLDS:
        candidate_id = f"sell_below_{int(round(threshold * 100)):03d}"
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
        results[candidate_id] = {
            "policy": policy,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "train_2022_2024": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025_not_used_for_selection": round1.evaluate_run(
                daily, actions, "20250102", round1.DEVELOPMENT_END
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
        }
        cache[candidate_id] = (daily, actions)

    baseline = results[BASELINE_ID]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key],
                checkpoint["current_best_equalweight"][key],
                rtol=0.0,
                atol=1e-12,
            )
        )
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
        raise RuntimeError("pressure threshold baseline drifted")

    ranked_id = max(results, key=lambda key: selection_key(results[key]))
    ranked = results[ranked_id]
    practical_gates = {
        "training_selected": ranked_id != BASELINE_ID,
        "forward_2025_not_worse": ranked[
            "holdout_2025_not_used_for_selection"
        ]["cumulative_return"]
        >= baseline["holdout_2025_not_used_for_selection"]["cumulative_return"],
        "stress_not_worse": ranked["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "full_10_positions": ranked["metrics_0_30pct"]["full_10_position_ratio"]
        == 1.0,
        "all_calendar_years_positive": min(
            ranked["metrics_0_30pct"]["annual_returns"].values()
        )
        > 0.0,
    }
    accepted = bool(all(practical_gates.values()))
    selected_id = ranked_id if accepted else BASELINE_ID
    selected_daily, selected_actions = cache[selected_id]
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
        **common,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure threshold replay failed")

    result = {
        "status": "development_recheck_complete_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "only_change": "score exit threshold under the selected pressure-exit rule",
        "candidate_budget": list(THRESHOLDS),
        "results": results,
        "training_ranked_candidate": ranked_id,
        "practical_gates": practical_gates,
        "accepted_change": accepted,
        "selected_candidate_id": selected_id,
        "selected_policy": results[selected_id]["policy"],
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
