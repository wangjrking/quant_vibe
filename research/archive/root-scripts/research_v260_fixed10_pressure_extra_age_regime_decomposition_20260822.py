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
import research_v260_fixed10_regime_exit_threshold_20260822 as regime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_extra_age_regime_decomposition_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_extra_age_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("age8_all", "age8_strong_age4_weak", "age4_strong_age8_weak")


def age_schedule(strong_market_mask, strong_age: int, weak_age: int) -> np.ndarray:
    strong = np.asarray(strong_market_mask, dtype=np.bool_)
    return np.where(strong, int(strong_age), int(weak_age)).astype(np.int16)


def dominates(candidate: dict, baseline: dict) -> dict[str, bool]:
    current = candidate["metrics_0_30pct"]
    reference = baseline["metrics_0_30pct"]
    return {
        "return_improved": current["cumulative_return"] > reference["cumulative_return"],
        "sharpe_improved": current["sharpe"] > reference["sharpe"],
        "drawdown_not_worse": current["max_drawdown"] <= reference["max_drawdown"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is not available for age decomposition")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    schedules = {
        "age8_all": 8,
        "age8_strong_age4_weak": age_schedule(strong, 8, 4),
        "age4_strong_age8_weak": age_schedule(strong, 4, 8),
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        results[candidate_id], daily, actions = age_guard.run_policy(
            context, policy, schedules[candidate_id]
        )
        cache[candidate_id] = (daily, actions)

    baseline_id = "age8_all"
    gates = {
        key: dominates(value, results[baseline_id])
        for key, value in results.items()
        if key != baseline_id
    }
    eligible = [key for key, value in gates.items() if all(value.values())]
    selected = (
        max(eligible, key=lambda key: results[key]["metrics_0_30pct"]["sharpe"])
        if eligible
        else baseline_id
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context, policy, schedules[selected]
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1]) == round1.frame_hash(repeat_actions),
        "metrics": results[selected]["metrics_0_30pct"]["cumulative_return"]
        == repeat["metrics_0_30pct"]["cumulative_return"],
    }
    if not all(deterministic.values()):
        raise RuntimeError("age regime decomposition replay failed")
    expected = checkpoint["current_best_equalweight"]
    if results[baseline_id]["metrics_0_30pct"]["cumulative_return"] != expected["cumulative_return"]:
        raise RuntimeError("age regime decomposition checkpoint drifted")
    result = {
        "status": "pressure_extra_age_regime_decomposition_complete_2026_not_opened",
        "candidate_budget": list(CANDIDATES),
        "strong_market_days": int(np.sum(strong)),
        "weak_market_days": int(np.sum(~strong)),
        "results": results,
        "dominance_gates_vs_age8_all": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
