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
import research_v260_fixed10_pressure_defensive_sleeve_20260822 as sleeve
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age10_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = (
    "production_order",
    "confirmed_weak_9alpha_1defensive",
    "confirmed_weak_8alpha_2defensive",
)
CONFIRMATION_DAYS = 3


def confirmed_weak_mask(strong_market: np.ndarray, days: int = CONFIRMATION_DAYS) -> np.ndarray:
    strong = np.asarray(strong_market, dtype=np.bool_)
    required = max(int(days), 1)
    result = np.zeros(strong.shape, dtype=np.bool_)
    streak = 0
    for index, is_strong in enumerate(strong):
        streak = 0 if bool(is_strong) else streak + 1
        result[index] = streak >= required
    return result


def dominates(candidate: dict, baseline: dict) -> dict[str, bool]:
    return age_guard.selection_gates(candidate, baseline)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for sleeve research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed_weak_mask(strong)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    defensive_order = sleeve.defensive_sleeve_order(
        context.order, volatility, confirmed_weak
    )
    one_slot_order = sleeve.defensive_sleeve_order(
        context.order,
        volatility,
        confirmed_weak,
        alpha_slots=9,
        defensive_slots=1,
    )
    orders = {
        "production_order": context.order,
        "confirmed_weak_9alpha_1defensive": one_slot_order,
        "confirmed_weak_8alpha_2defensive": defensive_order,
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        case_context = copy.copy(context)
        case_context.order = orders[candidate_id]
        results[candidate_id], daily, actions = age_guard.run_policy(
            case_context, policy, extra_age
        )
        cache[candidate_id] = (daily, actions)

    baseline_id = CANDIDATES[0]
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
        raise RuntimeError("confirmed defensive sleeve baseline drifted")
    gates = {
        candidate_id: dominates(results[candidate_id], results[baseline_id])
        for candidate_id in CANDIDATES[1:]
    }
    eligible = [
        candidate_id
        for candidate_id, candidate_gates in gates.items()
        if all(candidate_gates.values())
    ]
    selected = max(
        [baseline_id, *eligible],
        key=lambda candidate_id: (
            results[candidate_id]["metrics_0_30pct"]["sharpe"],
            results[candidate_id]["metrics_0_30pct"]["cagr"],
        ),
    )
    repeat_context = copy.copy(context)
    repeat_context.order = orders[selected]
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        repeat_context, policy, extra_age
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
        raise RuntimeError("confirmed defensive sleeve replay failed")
    result = {
        "status": "confirmed_severe_weak_defensive_sleeve_complete_2026_not_opened",
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "after three consecutive weak-market sessions, keep eight production-ranked "
            "entry slots and fill two from the lowest trailing-volatility names in top30"
        ),
        "weak_market_days": int((~strong).sum()),
        "confirmed_weak_days": int(confirmed_weak.sum()),
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
