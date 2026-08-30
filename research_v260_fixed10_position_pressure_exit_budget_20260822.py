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
    "strategy_agent_v260_fixed10_position_pressure_exit_budget_20260822"
)
CASES = {
    "one_score_exit_per_day": (None, None),
    "two_exits_when_at_least_three_weak": (3, 2),
    "two_exits_when_at_least_five_weak": (5, 2),
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
        raise PermissionError("2026 data entered position-pressure exit development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    empty_block = np.zeros(score.shape, dtype=np.bool_)
    results, cache = {}, {}
    for case_id, (trigger, elevated_limit) in CASES.items():
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=empty_block,
            maintenance_buy_block_mask_override=maintenance_block,
            score_sell_pressure_trigger_override=trigger,
            score_sell_pressure_limit_override=elevated_limit,
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
            "pressure_trigger_weak_positions": trigger,
            "temporary_score_exit_limit": elevated_limit,
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
        cache[case_id] = (daily, actions)

    baseline_id = "one_score_exit_per_day"
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
        raise RuntimeError("position-pressure exit baseline drifted")
    ranked = max(
        results,
        key=lambda key: round1.selection_utility(
            results[key]["train_2022_2024"], results[key]["metrics_0_65pct"]
        ),
    )
    candidate = results[ranked]
    accepted = bool(
        ranked != baseline_id
        and candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"]
        and candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"]
        and candidate["train_2022_2024"]["max_drawdown"]
        < baseline["train_2022_2024"]["max_drawdown"]
        and candidate["holdout_2025_not_used_for_selection"]["cumulative_return"]
        >= baseline["holdout_2025_not_used_for_selection"]["cumulative_return"]
        and candidate["metrics_0_65pct"]["cumulative_return"]
        > baseline["metrics_0_65pct"]["cumulative_return"]
    )
    selected = ranked if accepted else baseline_id
    result = {
        "status": (
            "candidate_supported_pre2026_2026_not_opened"
            if accepted
            else "candidate_rejected_pre2026_2026_not_opened"
        ),
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "only_change": (
            "keep one score exit per day normally; allow at most two only when "
            "multiple held positions simultaneously satisfy the existing score-exit rule"
        ),
        "candidate_budget": list(CASES),
        "results": results,
        "raw_ranked_candidate": ranked,
        "accepted": accepted,
        "selected_candidate": selected,
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
