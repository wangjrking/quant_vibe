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
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_score_5d10d_current_rules_20260822 as score_tools
import research_v260_fixed10_soft_width_profit_optimization_20260822 as width
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_exit_horizon_profit_optimization_20260822"
)
CHECKPOINT = width.CHECKPOINT
EXIT_HORIZONS = ("pure_10d", "1d_10pct", "3d_10pct", "5d_10pct")


def raw_exit_score(arrays: dict[str, np.ndarray], horizon: str) -> np.ndarray:
    rank_10d = np.asarray(arrays["rank_10d"], dtype=np.float64)
    if horizon == "pure_10d":
        return rank_10d
    rank_key = {
        "1d_10pct": "rank_1d",
        "3d_10pct": "rank_3d",
        "5d_10pct": "rank_5d",
    }.get(horizon)
    if rank_key is None:
        raise ValueError(f"unknown exit horizon: {horizon}")
    return 0.1 * np.asarray(arrays[rank_key], dtype=np.float64) + 0.9 * rank_10d


def exit_score(context, horizon: str) -> np.ndarray:
    return score_tools.smooth_score(
        raw_exit_score(context.arrays, horizon), 7, 0.1
    )[0]


def run_case(context, policy: dict, horizon: str, cost: float):
    extra_age, sell_priority = width.build_overrides(context, policy)
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        score_sell_priority_override=sell_priority,
        exit_score_override=exit_score(context, horizon),
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered exit-horizon optimization")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)

    results = {}
    frames = {}
    for horizon in EXIT_HORIZONS:
        daily, actions = run_case(
            context, policy, horizon, width.BASELINE_COST
        )
        stress_daily, stress_actions = run_case(
            context, policy, horizon, width.STRESS_COST
        )
        baseline = width.metrics(daily, actions, 10)
        stress = width.metrics(stress_daily, stress_actions, 10)
        results[horizon] = {
            "entry_score": "unchanged_pure_10d",
            "exit_horizon": horizon,
            "short_horizon_weight": 0.0 if horizon == "pure_10d" else 0.1,
            "metrics_0_30pct": baseline,
            "metrics_0_65pct": stress,
            "profit_eligible": width.profit_eligible(baseline, stress),
        }
        frames[horizon] = (daily, actions)

    selected = width.choose_width(results)
    repeat_daily, repeat_actions = run_case(
        context, policy, selected, width.BASELINE_COST
    )
    daily, actions = frames[selected]
    deterministic = {
        "daily": round1.frame_hash(daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("selected exit-horizon replay is not deterministic")

    result = {
        "status": "current_exit_horizon_profit_optimization_complete_2026_not_opened",
        "only_change": "ten-percent short-horizon component used by the exit score",
        "selection_rule": "maximize_pre2026_cumulative_return_subject_to_basic_risk_floors",
        "candidates": results,
        "selected_exit_horizon": selected,
        "selected_minus_current_pure10d_exit": round1.numeric_delta(
            results[selected]["metrics_0_30pct"],
            results["pure_10d"]["metrics_0_30pct"],
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
