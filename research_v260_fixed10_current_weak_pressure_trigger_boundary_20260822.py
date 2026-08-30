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

import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_weak_pressure_trigger_boundary_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age10_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
WEAK_TRIGGERS = (4, 5, 6)


def trigger_schedule(strong_market: np.ndarray, weak_trigger: int) -> np.ndarray:
    strong = np.asarray(strong_market, dtype=np.bool_)
    return np.where(strong, 4, int(weak_trigger)).astype(np.int64)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for trigger boundaries")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_min_age = age_boundary.pressure_age_schedule(strong, 10)
    results, cache = {}, {}
    for weak_trigger in WEAK_TRIGGERS:
        candidate_id = f"strong4_weak{weak_trigger}"
        schedule = trigger_schedule(strong, weak_trigger)
        results[candidate_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_min_age,
            pressure_trigger_override=schedule,
        )
        cache[candidate_id] = (daily, actions)

    baseline_id = "strong4_weak5"
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                results[baseline_id]["metrics_0_30pct"][key],
                expected[key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("weak pressure trigger baseline drifted")
    baseline = results[baseline_id]
    gates = {
        candidate_id: age_guard.selection_gates(item, baseline)
        for candidate_id, item in results.items()
        if candidate_id != baseline_id
    }
    eligible = [key for key, value in gates.items() if all(value.values())]
    selected = max(
        [baseline_id, *eligible],
        key=lambda candidate_id: (
            results[candidate_id]["metrics_0_30pct"]["sharpe"],
            results[candidate_id]["metrics_0_30pct"]["cagr"],
            -results[candidate_id]["metrics_0_30pct"]["max_drawdown"],
        ),
    )
    selected_trigger = int(selected.removeprefix("strong4_weak"))
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_min_age,
        pressure_trigger_override=trigger_schedule(strong, selected_trigger),
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1]) == round1.frame_hash(repeat_actions),
        "metrics": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("weak pressure trigger replay failed")
    result = {
        "status": "current_weak_pressure_trigger_boundary_complete_2026_not_opened",
        "candidate_budget": [f"strong4_weak{value}" for value in WEAK_TRIGGERS],
        "only_change": "weak-market count of simultaneous score-exit candidates required for the second exit",
        "results": results,
        "selection_gates_against_weak5": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
