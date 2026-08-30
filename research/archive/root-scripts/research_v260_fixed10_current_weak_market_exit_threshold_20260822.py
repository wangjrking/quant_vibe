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
    "strategy_agent_v260_fixed10_current_weak_market_exit_threshold_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = ("current_exit_085", "weak_exit_080")


def exit_threshold_schedule(strong_market: np.ndarray, weak_threshold: float) -> np.ndarray:
    strong = np.asarray(strong_market, dtype=np.bool_)
    return np.where(strong, 0.85, float(weak_threshold)).astype(np.float64)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for exit research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_min_age = np.where(strong, 4, 8).astype(np.int64)
    schedules = {
        "current_exit_085": None,
        "weak_exit_080": exit_threshold_schedule(strong, 0.80),
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        results[candidate_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_min_age,
            sell_score_below_override=schedules[candidate_id],
        )
        cache[candidate_id] = (daily, actions)

    baseline_id, candidate_id = CANDIDATES
    baseline, candidate = results[baseline_id], results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
            )
        )
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("weak-market exit baseline drifted")
    gates = age_guard.selection_gates(candidate, baseline)
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_min_age,
        sell_score_below_override=schedules[selected],
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
        raise RuntimeError("weak-market exit replay failed")
    result = {
        "status": "current_weak_market_exit_threshold_complete_2026_not_opened",
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "weak-market score exits require score below 0.80 instead of 0.85; "
            "strong-market threshold remains 0.85"
        ),
        "weak_market_days": int((~strong).sum()),
        "results": results,
        "selection_gates": gates,
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
