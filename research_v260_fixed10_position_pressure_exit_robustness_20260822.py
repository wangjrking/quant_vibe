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

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_robustness_20260822"
)
CASES = {
    "baseline_one_exit": None,
    "control_trigger4_limit2": 4,
    "candidate_trigger5_limit2": 5,
    "control_trigger6_limit2": 6,
}


def summarize_pressure(records: list[dict], trigger: int | None) -> dict:
    if trigger is None:
        triggered = []
    else:
        triggered = [
            row
            for row in records
            if int(row["score_sell_candidate_count"]) >= int(trigger)
        ]
    return {
        "observed_days": len(records),
        "triggered_days": len(triggered),
        "triggered_years": sorted({str(row["signal_date"])[:4] for row in triggered}),
        "maximum_simultaneous_score_sell_candidates": max(
            (int(row["score_sell_candidate_count"]) for row in records), default=0
        ),
        "records": triggered,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/"
            "pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered pressure-exit robustness")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    empty_block = np.zeros(score.shape, dtype=np.bool_)
    results = {}
    for case_id, trigger in CASES.items():
        pressure_records: list[dict] = []
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=empty_block,
            maintenance_buy_block_mask_override=maintenance_block,
            score_sell_pressure_trigger_override=trigger,
            score_sell_pressure_limit_override=(2 if trigger is not None else None),
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
            score_sell_pressure_observer=pressure_records.append,
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
        results[case_id] = {
            "trigger": trigger,
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
            "pressure_diagnostics": summarize_pressure(pressure_records, trigger),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
        }

    baseline = results["baseline_one_exit"]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
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
        raise RuntimeError("pressure robustness baseline drifted")

    neighbors = ["control_trigger4_limit2", "candidate_trigger5_limit2", "control_trigger6_limit2"]
    neighborhood_gates = {}
    for case_id in neighbors:
        item = results[case_id]
        neighborhood_gates[case_id] = {
            "train_cagr_higher": item["train_2022_2024"]["cagr"]
            > baseline["train_2022_2024"]["cagr"],
            "train_sharpe_higher": item["train_2022_2024"]["sharpe"]
            > baseline["train_2022_2024"]["sharpe"],
            "train_drawdown_lower": item["train_2022_2024"]["max_drawdown"]
            < baseline["train_2022_2024"]["max_drawdown"],
            "holdout_return_higher": item["holdout_2025"]["cumulative_return"]
            > baseline["holdout_2025"]["cumulative_return"],
            "stress_cumulative_higher": item["metrics_0_65pct"]["cumulative_return"]
            > baseline["metrics_0_65pct"]["cumulative_return"],
        }
    candidate_pressure = results["candidate_trigger5_limit2"]["pressure_diagnostics"]
    acceptance_gates = {
        "all_neighbors_improve_train_cagr_sharpe_drawdown_holdout_and_stress": all(
            all(gates.values()) for gates in neighborhood_gates.values()
        ),
        "candidate_triggered_at_least_four_days": candidate_pressure["triggered_days"] >= 4,
        "candidate_trigger_spans_at_least_two_years": len(candidate_pressure["triggered_years"]) >= 2,
    }
    accepted = bool(all(acceptance_gates.values()))
    result = {
        "status": (
            "neighborhood_robust_candidate_supported_pre2026_2026_not_opened"
            if accepted
            else "candidate_rejected_after_neighborhood_check_pre2026_2026_not_opened"
        ),
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "rule": (
            "normally allow one score exit; allow two when at least five held "
            "positions simultaneously satisfy the unchanged score-exit contract"
        ),
        "results": results,
        "neighborhood_gates": neighborhood_gates,
        "acceptance_gates": acceptance_gates,
        "accepted": accepted,
        "selected_candidate": (
            "candidate_trigger5_limit2" if accepted else "baseline_one_exit"
        ),
        "checkpoint_equivalence": checkpoint_equivalence,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
