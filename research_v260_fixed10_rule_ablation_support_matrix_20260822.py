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

import research_v260_fixed10_current_rule_ablation_20260822 as ablation
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_min_hold4_fragility_diagnostic_20260822 as fragility
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_vol_priority_action_concentration_20260822 as concentration


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rule_ablation_support_matrix_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
BLOCK_LENGTHS = (5, 20, 60)
SEED = 260_202_608_27


def support(summary: dict, paired: dict) -> dict:
    result = fragility.support_summary(summary, paired)
    result["broad_support_for_retaining_rule"] = bool(
        summary["total_log_excess"] > 0.0
        and result["positive_year_fraction"] >= 0.75
        and result["positive_after_removing_top5_days"]
        and result["return_probability_above_80pct_all_blocks"]
    )
    return result


def simplification_eligible(metrics: dict, production: dict, evidence: dict) -> bool:
    baseline = metrics["metrics_0_30pct"]
    stress = metrics["metrics_0_65pct"]
    return bool(
        not evidence["broad_support_for_retaining_rule"]
        and baseline["cumulative_return"] > production["cumulative_return"]
        and baseline["cagr"] > production["cagr"]
        and min(baseline["annual_returns"].values()) > 0.0
        and stress["cumulative_return"] > 0.0
        and baseline["full_10_position_ratio"] == 1.0
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered rule ablation support matrix")

    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    cases = ablation.build_cases(context, policy)
    runs = {}
    for case_id, (case_policy, case_age, pressure_trigger, sell_priority) in cases.items():
        metrics, daily, actions = age_guard.run_policy(
            context,
            case_policy,
            case_age,
            pressure_trigger_override=pressure_trigger,
            score_sell_priority_override=sell_priority,
        )
        runs[case_id] = {"metrics": metrics, "daily": daily, "actions": actions}

    current = runs["full_current"]
    expected = checkpoint["current_best_equalweight"]
    if not all(
        np.isclose(
            current["metrics"]["metrics_0_30pct"][key],
            expected[key],
            rtol=0.0,
            atol=1e-12,
        )
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    ):
        raise RuntimeError("rule ablation support baseline drifted")

    comparisons = {}
    for index, (case_id, removed) in enumerate(runs.items()):
        if case_id == "full_current":
            continue
        current_returns, removed_returns = fragility.aligned_returns(
            current["daily"], removed["daily"]
        )
        paired = {
            str(block): bootstrap.paired_bootstrap(
                current_returns,
                removed_returns,
                block,
                SEED + index * 1000 + block,
            )
            for block in BLOCK_LENGTHS
        }
        summary = concentration.concentration_summary(
            concentration.daily_log_excess(current["daily"], removed["daily"])
        )
        evidence = support(summary, paired)
        comparisons[case_id] = {
            "removed_metrics": removed["metrics"],
            "action_difference": {
                key: value
                for key, value in concentration.action_difference(
                    current["actions"], removed["actions"]
                ).items()
                if key != "changed_execution_dates"
            },
            "concentration": {
                key: summary[key]
                for key in (
                    "total_log_excess",
                    "equivalent_relative_wealth_gain",
                    "active_day_count",
                    "top5_positive_share",
                    "top10_absolute_share",
                    "log_excess_after_removing_top5_positive_days",
                    "positive_month_fraction",
                    "yearly_log_excess",
                )
            },
            "paired_block_bootstrap": paired,
            "support": evidence,
            "simplification_eligible": simplification_eligible(
                removed["metrics"], checkpoint["production_baseline"], evidence
            ),
        }

    simplification_candidates = [
        case_id
        for case_id, value in comparisons.items()
        if value["simplification_eligible"]
    ]
    result = {
        "status": "rule_ablation_support_matrix_complete_2026_not_opened",
        "diagnostic_only_no_automatic_policy_change": True,
        "comparisons": comparisons,
        "summary": {
            case_id: {
                "broad_support_for_retaining_rule": value["support"][
                    "broad_support_for_retaining_rule"
                ],
                "positive_year_fraction": value["support"][
                    "positive_year_fraction"
                ],
                "simplification_eligible": value["simplification_eligible"],
            }
            for case_id, value in comparisons.items()
        },
        "simplification_candidates": simplification_candidates,
        "interpretation": (
            "A rule is only a simplification candidate when its incremental evidence is "
            "not broad and the ablated strategy still beats production on pre-2026 return."
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "rule_ablation_support_matrix.json", result)
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
        "simplification_candidates": simplification_candidates,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
