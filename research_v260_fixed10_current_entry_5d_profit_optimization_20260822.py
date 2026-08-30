from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_score_5d10d_current_rules_20260822 as score_tools
import research_v260_fixed10_soft_width_profit_optimization_20260822 as width
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_entry_5d_profit_optimization_20260822"
)
CHECKPOINT = width.CHECKPOINT
ENTRY_5D_WEIGHTS = (0.0, 0.1, 0.3, 0.5)


def entry_order(arrays: dict[str, np.ndarray], weight_5d: float) -> np.ndarray:
    raw = (
        float(weight_5d) * np.asarray(arrays["rank_5d"], dtype=np.float64)
        + (1.0 - float(weight_5d))
        * np.asarray(arrays["rank_10d"], dtype=np.float64)
    )
    return score_tools.smooth_score(raw, 7, 0.1)[1]


def run_case(context, policy: dict, weight_5d: float, cost: float):
    case_context = replace(
        context, order=entry_order(context.arrays, weight_5d)
    )
    extra_age, sell_priority = width.build_overrides(context, policy)
    return age_guard.run_policy_at_cost(
        case_context,
        policy,
        extra_age,
        cost,
        score_sell_priority_override=sell_priority,
        exit_score_override=context.score,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered 5d entry-score optimization")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)

    results = {}
    frames = {}
    for weight_5d in ENTRY_5D_WEIGHTS:
        daily, actions = run_case(
            context, policy, weight_5d, width.BASELINE_COST
        )
        stress_daily, stress_actions = run_case(
            context, policy, weight_5d, width.STRESS_COST
        )
        baseline = width.metrics(daily, actions, 10)
        stress = width.metrics(stress_daily, stress_actions, 10)
        key = f"{weight_5d:.1f}"
        results[key] = {
            "entry_weight_5d": weight_5d,
            "entry_weight_10d": 1.0 - weight_5d,
            "exit_score": "unchanged_pure_10d",
            "metrics_0_30pct": baseline,
            "metrics_0_65pct": stress,
            "profit_eligible": width.profit_eligible(baseline, stress),
        }
        frames[key] = (daily, actions)

    selected = width.choose_width(results)
    repeat_daily, repeat_actions = run_case(
        context, policy, float(selected), width.BASELINE_COST
    )
    daily, actions = frames[selected]
    deterministic = {
        "daily": round1.frame_hash(daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("selected 5d entry-score replay is not deterministic")

    result = {
        "status": "current_entry_5d_profit_optimization_complete_2026_not_opened",
        "only_change": "five-day versus ten-day weight used for new-entry ordering",
        "selection_rule": "maximize_pre2026_cumulative_return_subject_to_basic_risk_floors",
        "candidates": results,
        "selected_entry_weight_5d": float(selected),
        "selected_entry_weight_10d": 1.0 - float(selected),
        "selected_minus_current_pure10d": round1.numeric_delta(
            results[selected]["metrics_0_30pct"],
            results["0.0"]["metrics_0_30pct"],
        ),
        "deterministic_replay": deterministic,
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
