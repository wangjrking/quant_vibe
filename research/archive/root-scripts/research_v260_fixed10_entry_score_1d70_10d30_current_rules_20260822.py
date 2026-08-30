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
import research_v260_fixed10_score_5d10d_current_rules_20260822 as score_tools
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_entry_score_1d70_10d30_current_rules_20260822"
)
CASES = {
    "entry_score_pure10d": 0.0,
    "control_entry_1d10_10d90": 0.10,
    "control_entry_1d30_10d70": 0.30,
    "candidate_entry_1d70_10d30": 0.70,
}
SELECTABLE = "candidate_entry_1d70_10d30"


def entry_score(
    arrays: dict[str, np.ndarray],
    weight_1d: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= float(weight_1d) <= 1.0:
        raise ValueError("1d entry weight must be within [0, 1]")
    raw = (
        float(weight_1d) * np.asarray(arrays["rank_1d"], dtype=np.float64)
        + (1.0 - float(weight_1d))
        * np.asarray(arrays["rank_10d"], dtype=np.float64)
    )
    return score_tools.smooth_score(
        raw, score_tools.SMOOTHING_WINDOW, score_tools.RAW_ALPHA
    )


def candidate_passes(
    baseline_train: dict,
    candidate_train: dict,
    baseline_holdout: dict,
    candidate_holdout: dict,
    baseline_stress: dict,
    candidate_stress: dict,
) -> dict:
    gates = {
        "train_cagr_higher": candidate_train["cagr"] > baseline_train["cagr"],
        "train_sharpe_higher": candidate_train["sharpe"] > baseline_train["sharpe"],
        "train_drawdown_not_worse": (
            candidate_train["max_drawdown"] <= baseline_train["max_drawdown"]
        ),
        "holdout_2025_return_higher": (
            candidate_holdout["cumulative_return"]
            > baseline_holdout["cumulative_return"]
        ),
        "holdout_2025_sharpe_higher": (
            candidate_holdout["sharpe"] > baseline_holdout["sharpe"]
        ),
        "stress_cumulative_higher": (
            candidate_stress["cumulative_return"]
            > baseline_stress["cumulative_return"]
        ),
    }
    return {"gates": gates, "passed": bool(all(gates.values()))}


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
        raise PermissionError("2026 data entered 1d/10d entry-score development")
    definition = harness.production_definition(protocol)
    exit_score, exit_order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    entry_block = np.zeros(exit_score.shape, dtype=np.bool_)
    maintenance_block = quality.maintenance_quality_block(exit_score, True)
    results, cache = {}, {}
    for case_id, weight_1d in CASES.items():
        score, order = entry_score(arrays, weight_1d)
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
            maintenance_buy_block_mask_override=maintenance_block,
            exit_score_override=exit_score,
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
        results[case_id] = {
            "entry_score_formula": {
                "rank_1d": weight_1d,
                "rank_10d": 1.0 - weight_1d,
                "smoothing_window": 7,
                "current_raw_alpha": 0.10,
            },
            "exit_score_formula": {"rank_10d": 1.0},
            "metrics_0_30pct": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            ),
            "train_2022_2024": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025_not_used_for_selection": round1.evaluate_run(
                daily, actions, "20250102", round1.DEVELOPMENT_END
            ),
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
        }
        cache[case_id] = (daily, actions, score, order)

    baseline_id = "entry_score_pure10d"
    baseline = results[baseline_id]
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
        raise RuntimeError("1d/10d entry-score baseline drifted")

    candidate = results[SELECTABLE]
    decision = candidate_passes(
        baseline["train_2022_2024"],
        candidate["train_2022_2024"],
        baseline["holdout_2025_not_used_for_selection"],
        candidate["holdout_2025_not_used_for_selection"],
        baseline["metrics_0_65pct"],
        candidate["metrics_0_65pct"],
    )
    selected = SELECTABLE if decision["passed"] else baseline_id
    selected_daily, selected_actions, score, order = cache[selected]
    repeat_daily, repeat_actions = round1.run_fixed10(
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
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=entry_block,
        maintenance_buy_block_mask_override=maintenance_block,
        exit_score_override=exit_score,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("1d/10d entry-score replay failed")
    result = {
        "status": (
            "candidate_supported_pre2026_2026_not_opened"
            if decision["passed"]
            else "candidate_rejected_pre2026_2026_not_opened"
        ),
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "candidate_budget": [SELECTABLE],
        "predeclared_controls_not_selectable": [
            "control_entry_1d10_10d90",
            "control_entry_1d30_10d70",
        ],
        "only_change": (
            "use 70% one-day and 30% ten-day rank for new-entry ordering only; "
            "existing-position exits, renewal, top-ups, and all execution rules remain "
            "on the frozen pure-ten-day policy"
        ),
        "results": results,
        "candidate_decision": decision,
        "selected_candidate": selected,
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
