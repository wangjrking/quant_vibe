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
    "strategy_agent_v260_fixed10_confirmed_weak_days_boundary_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CONFIRMATION_DAYS = (1, 2, 3, 4)
PENALTY = 0.05
MAX_MDD_TOLERANCE = 0.001


def practical_selection_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    current = candidate["metrics_0_30pct"]
    reference = baseline["metrics_0_30pct"]
    return {
        "cumulative_return_improved": current["cumulative_return"]
        > reference["cumulative_return"],
        "cagr_improved": current["cagr"] > reference["cagr"],
        "sharpe_improved": current["sharpe"] > reference["sharpe"],
        "drawdown_within_10bp_tolerance": current["max_drawdown"]
        <= reference["max_drawdown"] + MAX_MDD_TOLERANCE,
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "all_years_positive": all(value > 0 for value in current["annual_returns"].values()),
        "full_10_positions": current["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for weak-day boundary")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)

    results, cache, weak_day_counts = {}, {}, {}
    for days in CONFIRMATION_DAYS:
        candidate_id = f"confirmed_weak_{days}d"
        weak = confirmed.confirmed_weak_mask(strong, days)
        weak_day_counts[candidate_id] = int(weak.sum())
        matrix = priority.sell_priority_matrix(
            context.score, volatility_rank, ~weak, penalty=PENALTY
        )
        results[candidate_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=matrix,
        )
        cache[candidate_id] = (daily, actions, matrix)

    baseline_id = "confirmed_weak_3d"
    baseline = results[baseline_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(
            baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("confirmed weak-day baseline drifted")
    gates = {
        candidate_id: practical_selection_gates(result, baseline)
        for candidate_id, result in results.items()
        if candidate_id != baseline_id
    }
    eligible = [
        candidate_id for candidate_id, candidate_gates in gates.items()
        if all(candidate_gates.values())
    ]
    selected = max(
        [baseline_id, *eligible],
        key=lambda candidate_id: (
            results[candidate_id]["metrics_0_30pct"]["sharpe"],
            results[candidate_id]["metrics_0_30pct"]["cagr"],
            -results[candidate_id]["metrics_0_30pct"]["max_drawdown"],
        ),
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=cache[selected][2],
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
        "metrics": all(
            results[selected]["metrics_0_30pct"][key]
            == repeat["metrics_0_30pct"][key]
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("confirmed weak-day boundary replay failed")

    result = {
        "status": "confirmed_weak_days_boundary_complete_2026_not_opened",
        "candidate_budget": [f"confirmed_weak_{days}d" for days in CONFIRMATION_DAYS],
        "only_change": (
            "number of consecutive weak-market sessions required before the frozen "
            "0.05 volatility sell-priority overlay activates"
        ),
        "selection_materiality": {
            "max_drawdown_tolerance": MAX_MDD_TOLERANCE,
            "interpretation": (
                "a drawdown difference within 10 basis points is treated as immaterial, "
                "but return, Sharpe, and stress performance must all improve"
            ),
        },
        "weak_day_counts": weak_day_counts,
        "results": results,
        "selection_gates_against_3d": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
