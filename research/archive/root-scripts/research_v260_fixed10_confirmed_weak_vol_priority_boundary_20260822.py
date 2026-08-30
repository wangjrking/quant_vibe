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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak_vol_priority_boundary_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age10_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PENALTIES = (0.00, 0.05, 0.10, 0.15)


def candidate_id(penalty: float) -> str:
    return f"confirmed_weak_vol_penalty_{penalty:.2f}"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for priority boundaries")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 3)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    results, cache = {}, {}
    for penalty in PENALTIES:
        current_id = candidate_id(penalty)
        matrix = None if penalty == 0.0 else priority.sell_priority_matrix(
            context.score,
            volatility_rank,
            ~confirmed_weak,
            penalty,
        )
        results[current_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=matrix,
        )
        cache[current_id] = (daily, actions)

    baseline_id = candidate_id(0.0)
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(
            results[baseline_id]["metrics_0_30pct"][key], expected[key],
            rtol=0.0, atol=1e-12,
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("confirmed weak priority baseline drifted")
    gates = {
        current_id: age_guard.selection_gates(item, results[baseline_id])
        for current_id, item in results.items()
        if current_id != baseline_id
    }
    neighboring_robustness = {
        "penalty_005_passed": all(gates[candidate_id(0.05)].values()),
        "penalty_010_passed": all(gates[candidate_id(0.10)].values()),
        "penalty_015_passed": all(gates[candidate_id(0.15)].values()),
    }
    eligible = [key for key, value in gates.items() if all(value.values())]
    selected = max(
        [baseline_id, *eligible],
        key=lambda current_id: (
            results[current_id]["metrics_0_30pct"]["sharpe"],
            results[current_id]["metrics_0_30pct"]["cagr"],
            -results[current_id]["metrics_0_30pct"]["max_drawdown"],
        ),
    )
    selected_penalty = float(selected.rsplit("_", 1)[1])
    selected_matrix = None if selected_penalty == 0.0 else priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, selected_penalty
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=selected_matrix,
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
        raise RuntimeError("confirmed weak priority boundary replay failed")
    result = {
        "status": "confirmed_weak_vol_priority_boundary_complete_2026_not_opened",
        "candidate_budget": [candidate_id(value) for value in PENALTIES],
        "only_change": "volatility-rank penalty used only to order eligible exits after three consecutive weak days",
        "confirmed_weak_days": int(confirmed_weak.sum()),
        "results": results,
        "selection_gates_against_no_penalty": gates,
        "neighboring_robustness": neighboring_robustness,
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
