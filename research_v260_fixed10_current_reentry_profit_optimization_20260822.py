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
    "strategy_agent_v260_fixed10_current_reentry_profit_optimization_20260822"
)
CHECKPOINT = width.CHECKPOINT
COOLDOWN_DAYS = (0, 1, 3)


def run_case(context, base_policy: dict, cooldown: int, cost: float):
    policy = copy.deepcopy(base_policy)
    policy["reentry_cooldown_days"] = int(cooldown)
    extra_age, sell_priority = width.build_overrides(context, policy)
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        score_sell_priority_override=sell_priority,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered reentry optimization")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)

    results = {}
    frames = {}
    for cooldown in COOLDOWN_DAYS:
        daily, actions = run_case(
            context, policy, cooldown, width.BASELINE_COST
        )
        stress_daily, stress_actions = run_case(
            context, policy, cooldown, width.STRESS_COST
        )
        baseline = width.metrics(daily, actions, 10)
        stress = width.metrics(stress_daily, stress_actions, 10)
        key = str(cooldown)
        results[key] = {
            "reentry_cooldown_days": cooldown,
            "metrics_0_30pct": baseline,
            "metrics_0_65pct": stress,
            "profit_eligible": width.profit_eligible(baseline, stress),
        }
        frames[key] = (daily, actions)

    selected = width.choose_width(results)
    repeat_daily, repeat_actions = run_case(
        context, policy, int(selected), width.BASELINE_COST
    )
    selected_daily, selected_actions = frames[selected]
    deterministic = {
        "daily": round1.frame_hash(selected_daily)
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("selected reentry cooldown replay is not deterministic")

    result = {
        "status": "current_reentry_profit_optimization_complete_2026_not_opened",
        "only_change": "days before a sold stock may be bought again",
        "selection_rule": "maximize_pre2026_cumulative_return_subject_to_basic_risk_floors",
        "candidates": results,
        "selected_cooldown_days": int(selected),
        "selected_minus_current": round1.numeric_delta(
            results[selected]["metrics_0_30pct"],
            results["0"]["metrics_0_30pct"],
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
