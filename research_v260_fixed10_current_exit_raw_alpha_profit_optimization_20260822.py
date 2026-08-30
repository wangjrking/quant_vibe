from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_soft_width_profit_optimization_20260822 as width
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_exit_raw_alpha_profit_optimization_20260822"
)
CHECKPOINT = width.CHECKPOINT
EXIT_RAW_ALPHAS = (0.0, 0.1, 0.2)


def exit_score(context, alpha: float):
    score, _order = context.harness.v95.score_pair(
        context.arrays, 0.0, 7, float(alpha)
    )
    return score


def run_case(context, policy: dict, alpha: float, cost: float):
    extra_age, sell_priority = width.build_overrides(context, policy)
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        score_sell_priority_override=sell_priority,
        exit_score_override=exit_score(context, alpha),
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered exit-alpha optimization")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)

    results = {}
    frames = {}
    for alpha in EXIT_RAW_ALPHAS:
        daily, actions = run_case(
            context, policy, alpha, width.BASELINE_COST
        )
        stress_daily, stress_actions = run_case(
            context, policy, alpha, width.STRESS_COST
        )
        baseline = width.metrics(daily, actions, 10)
        stress = width.metrics(stress_daily, stress_actions, 10)
        key = f"{alpha:.1f}"
        results[key] = {
            "entry_raw_alpha": 0.1,
            "exit_raw_alpha": alpha,
            "smoothing_window": 7,
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
        raise RuntimeError("selected exit-alpha replay is not deterministic")

    result = {
        "status": "current_exit_raw_alpha_profit_optimization_complete_2026_not_opened",
        "only_change": "latest-session alpha used by the seven-session exit score",
        "entry_score": "unchanged_pure_10d_window7_alpha010",
        "selection_rule": "maximize_pre2026_cumulative_return_subject_to_basic_risk_floors",
        "candidates": results,
        "selected_exit_raw_alpha": float(selected),
        "selected_minus_current_alpha010_exit": round1.numeric_delta(
            results[selected]["metrics_0_30pct"],
            results["0.1"]["metrics_0_30pct"],
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
