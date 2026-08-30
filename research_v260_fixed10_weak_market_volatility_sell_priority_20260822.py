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
import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_volatility_sell_priority_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age10_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CANDIDATES = (
    "score_only_priority",
    "weak_score_minus_010_vol_rank",
    "confirmed_weak_score_minus_010_vol_rank",
)
VOLATILITY_PENALTY = 0.10


def sell_priority_matrix(
    score: np.ndarray,
    volatility_rank: np.ndarray,
    strong_market: np.ndarray,
    penalty: float = VOLATILITY_PENALTY,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    vol_rank = np.asarray(volatility_rank, dtype=np.float64)
    strong = np.asarray(strong_market, dtype=np.bool_)
    if values.shape != vol_rank.shape or strong.shape != (values.shape[0],):
        raise ValueError("weak-market sell-priority inputs do not align")
    result = values.copy()
    weak_rows = np.flatnonzero(~strong)
    result[weak_rows] = values[weak_rows] - float(penalty) * np.nan_to_num(
        vol_rank[weak_rows], nan=0.0
    )
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for priority research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priorities = {
        "score_only_priority": None,
        "weak_score_minus_010_vol_rank": sell_priority_matrix(
            context.score, volatility_rank, strong
        ),
        "confirmed_weak_score_minus_010_vol_rank": sell_priority_matrix(
            context.score,
            volatility_rank,
            ~confirmed.confirmed_weak_mask(strong, 3),
        ),
    }
    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        results[candidate_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=priorities[candidate_id],
        )
        cache[candidate_id] = (daily, actions)

    baseline_id = CANDIDATES[0]
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
        raise RuntimeError("volatility sell-priority baseline drifted")
    gates = {
        candidate_id: age_guard.selection_gates(
            results[candidate_id], results[baseline_id]
        )
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
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=priorities[selected],
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
        raise RuntimeError("volatility sell-priority replay failed")
    result = {
        "status": "weak_market_volatility_sell_priority_complete_2026_not_opened",
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "on weak-market days order already-eligible score exits by "
            "score minus 0.10 times trailing-20d volatility percentile"
        ),
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
