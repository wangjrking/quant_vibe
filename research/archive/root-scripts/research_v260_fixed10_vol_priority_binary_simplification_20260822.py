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
    "strategy_agent_v260_fixed10_vol_priority_binary_simplification_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
BINARY_PENALTY = 0.05
CANDIDATES = ("continuous_005", "binary_top_half", "binary_top_quartile")


def binary_priority_matrix(
    score: np.ndarray,
    volatility_rank: np.ndarray,
    confirmed_weak: np.ndarray,
    threshold: float,
    penalty: float = BINARY_PENALTY,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    rank = np.asarray(volatility_rank, dtype=np.float64)
    weak = np.asarray(confirmed_weak, dtype=np.bool_)
    if values.shape != rank.shape or weak.shape != (values.shape[0],):
        raise ValueError("binary volatility-priority inputs do not align")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("volatility threshold must be between zero and one")
    result = values.copy()
    rows = np.flatnonzero(weak)
    flagged = np.isfinite(rank[rows]) & (rank[rows] >= float(threshold))
    result[rows] -= float(penalty) * flagged.astype(np.float64)
    return result


def simplification_gates(candidate: dict, reference: dict) -> dict[str, bool]:
    current = candidate["metrics_0_30pct"]
    baseline = reference["metrics_0_30pct"]
    return {
        "cagr_not_worse": current["cagr"] >= baseline["cagr"],
        "sharpe_not_worse": current["sharpe"] >= baseline["sharpe"],
        "drawdown_not_worse": current["max_drawdown"] <= baseline["max_drawdown"],
        "train_return_not_worse": candidate["train_2022_2024"]["cumulative_return"]
        >= reference["train_2022_2024"]["cumulative_return"],
        "holdout_return_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= reference["holdout_2025"]["cumulative_return"],
        "stress_return_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= reference["metrics_0_65pct"]["cumulative_return"],
        "all_years_positive": min(current["annual_returns"].values()) > 0.0,
        "exactly10": current["full_10_position_ratio"] == 1.0,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered binary priority simplification")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    matrices = {
        "continuous_005": priority.sell_priority_matrix(
            context.score, volatility_rank, ~weak2, 0.05
        ),
        "binary_top_half": binary_priority_matrix(
            context.score, volatility_rank, weak2, 0.50
        ),
        "binary_top_quartile": binary_priority_matrix(
            context.score, volatility_rank, weak2, 0.75
        ),
    }

    results, cache = {}, {}
    for candidate_id in CANDIDATES:
        results[candidate_id], daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=matrices[candidate_id],
        )
        cache[candidate_id] = (daily, actions)

    reference_id = CANDIDATES[0]
    reference = results[reference_id]
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(np.isclose(
            reference["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("continuous 0.05 reference drifted")
    gates = {
        candidate_id: simplification_gates(results[candidate_id], reference)
        for candidate_id in CANDIDATES[1:]
    }
    eligible = [candidate_id for candidate_id, values in gates.items() if all(values.values())]
    selected = (
        max(
            eligible,
            key=lambda candidate_id: (
                results[candidate_id]["metrics_0_30pct"]["sharpe"],
                results[candidate_id]["metrics_0_30pct"]["cagr"],
            ),
        )
        if eligible
        else reference_id
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=matrices[selected],
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
        raise RuntimeError("binary priority simplification replay failed")

    result = {
        "status": "vol_priority_binary_simplification_complete_2026_not_opened",
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "replace the continuous weak-market volatility percentile term with a "
            "single high-volatility bucket indicator"
        ),
        "results": results,
        "simplification_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != reference_id,
        "checkpoint_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "decision": (
            "adopt_simpler_binary_priority"
            if selected != reference_id
            else "retain_continuous_005_binary_rules_fail_robustness_gates"
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps({
        "status": result["status"],
        "selected_candidate": selected,
        "simplification_gates": gates,
        "decision": result["decision"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
