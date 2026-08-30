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
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822"
)
ROBUSTNESS_REPORT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_robustness_20260822/"
    "development_result.json"
)
BASELINE_ID = "baseline_one_exit"
SELECTED_ID = "control_trigger4_limit2"
SUPPORT_ID = "candidate_trigger5_limit2"
BOUNDARY_ID = "control_trigger6_limit2"


def improves_core_contract(candidate: dict, baseline: dict) -> dict[str, bool]:
    return {
        "train_cagr_higher": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "train_sharpe_higher": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "train_drawdown_lower": candidate["train_2022_2024"]["max_drawdown"]
        < baseline["train_2022_2024"]["max_drawdown"],
        "forward_2025_return_higher": candidate["holdout_2025"]["cumulative_return"]
        > baseline["holdout_2025"]["cumulative_return"],
        "stress_cumulative_higher": candidate["metrics_0_65pct"]["cumulative_return"]
        > baseline["metrics_0_65pct"]["cumulative_return"],
    }


def positive_annual_delta_count(candidate: dict, baseline: dict) -> int:
    candidate_annual = candidate["metrics_0_30pct"]["annual_returns"]
    baseline_annual = baseline["metrics_0_30pct"]["annual_returns"]
    return sum(
        float(candidate_annual[year]) > float(baseline_annual[year])
        for year in sorted(baseline_annual)
    )


def coarse_mechanism_gates(results: dict) -> dict[str, bool]:
    baseline = results[BASELINE_ID]
    selected = results[SELECTED_ID]
    support = results[SUPPORT_ID]
    selected_pressure = selected["pressure_diagnostics"]
    support_pressure = support["pressure_diagnostics"]
    return {
        "selected_improves_core_contract": all(
            improves_core_contract(selected, baseline).values()
        ),
        "neighbor_support_improves_core_contract": all(
            improves_core_contract(support, baseline).values()
        ),
        "selected_improves_at_least_three_calendar_years": (
            positive_annual_delta_count(selected, baseline) >= 3
        ),
        "neighbor_improves_at_least_three_calendar_years": (
            positive_annual_delta_count(support, baseline) >= 3
        ),
        "selected_has_at_least_twelve_direct_triggers": (
            int(selected_pressure["triggered_days"]) >= 12
        ),
        "selected_spans_at_least_three_years": (
            len(selected_pressure["triggered_years"]) >= 3
        ),
        "neighbor_has_direct_evidence": int(support_pressure["triggered_days"]) >= 4,
    }


def run_selected_replay(policy: dict) -> tuple:
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered pressure-exit checkpoint")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    empty_block = np.zeros(score.shape, dtype=np.bool_)
    common = dict(
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=4,
        score_sell_pressure_limit_override=2,
    )
    first_daily, first_actions = round1.run_fixed10(
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
    second_daily, second_actions = round1.run_fixed10(
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
    deterministic = {
        "daily": round1.frame_hash(first_daily) == round1.frame_hash(second_daily),
        "actions": round1.frame_hash(first_actions)
        == round1.frame_hash(second_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("pressure-exit checkpoint replay failed")
    metrics = round1.evaluate_run(
        first_daily, first_actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    stress = round1.evaluate_run(
        stress_daily,
        stress_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    return metrics, stress, deterministic, rules, manifests, access


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    robustness = json.loads(ROBUSTNESS_REPORT.read_text(encoding="utf-8"))
    if robustness["validation_2026_opened"]:
        raise PermissionError("robustness report opened 2026")
    results = robustness["results"]
    gates = coarse_mechanism_gates(results)
    if not all(gates.values()):
        raise RuntimeError("coarse position-pressure mechanism gates failed")

    previous_checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/"
            "strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/"
            "pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(previous_checkpoint["selected_policy"])
    policy["score_sell_pressure_trigger"] = 4
    policy["score_sell_pressure_limit"] = 2
    policy["score_sell_pressure_confirmation_days"] = 1
    metrics, stress, deterministic, rules, manifests, access = run_selected_replay(policy)

    expected = results[SELECTED_ID]
    replay_equivalence = {
        key: bool(
            np.isclose(
                metrics[key], expected["metrics_0_30pct"][key], rtol=0.0, atol=1e-12
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
    replay_equivalence["stress_cumulative_return"] = bool(
        np.isclose(
            stress["cumulative_return"],
            expected["metrics_0_65pct"]["cumulative_return"],
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not all(replay_equivalence.values()):
        raise RuntimeError("pressure-exit checkpoint drifted from robustness evidence")

    baseline = results[BASELINE_ID]["metrics_0_30pct"]
    delta = {
        "cumulative_return_percentage_points": 100.0
        * (metrics["cumulative_return"] - baseline["cumulative_return"]),
        "cagr_percentage_points": 100.0 * (metrics["cagr"] - baseline["cagr"]),
        "sharpe": metrics["sharpe"] - baseline["sharpe"],
        "max_drawdown_percentage_points": 100.0
        * (metrics["max_drawdown"] - baseline["max_drawdown"]),
        "turnover_annualized": metrics["turnover_annualized"]
        - baseline["turnover_annualized"],
    }
    checkpoint = {
        "status": "pre2026_exploratory_checkpoint_pressure_exit_supported",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "simple_rule": (
            "keep one score exit per day; when at least four held positions "
            "simultaneously satisfy the unchanged score-exit rule on the current "
            "signal day, allow at most two"
        ),
        "selection_reason": (
            "trigger 4 and neighboring trigger 5 both improve the core contract; "
            "trigger 4 has 25 direct events spanning all four calendar years"
        ),
        "boundary_control": {
            "trigger": 6,
            "interpretation": (
                "sparse mechanism boundary, not an all-or-nothing veto on triggers 4-5"
            ),
        },
        "selected_policy": policy,
        "current_best_equalweight": metrics,
        "stress_0_65pct": stress,
        "delta_vs_previous_checkpoint": delta,
        "coarse_mechanism_gates": gates,
        "direct_trigger_count": expected["pressure_diagnostics"]["triggered_days"],
        "direct_trigger_years": expected["pressure_diagnostics"]["triggered_years"],
        "deterministic_replay": deterministic,
        "replay_equivalence": replay_equivalence,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "pre2026_checkpoint.json", checkpoint)
    print(json.dumps(checkpoint, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
