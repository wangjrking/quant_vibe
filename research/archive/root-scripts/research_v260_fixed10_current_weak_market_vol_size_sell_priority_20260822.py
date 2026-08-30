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
import research_v260_fixed10_current_weak_market_size_guard_recheck_20260822 as size_recheck
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as vol_priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_weak_market_vol_size_sell_priority_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
SIZE_PENALTIES = (0.0, 0.0125, 0.025, 0.05)


def vol_size_priority_matrix(
    current_priority: np.ndarray,
    size_percentile: np.ndarray,
    confirmed_weak: np.ndarray,
    size_penalty: float,
) -> np.ndarray:
    priority = np.asarray(current_priority, dtype=np.float64)
    size_rank = np.asarray(size_percentile, dtype=np.float64)
    weak = np.asarray(confirmed_weak, dtype=np.bool_)
    if priority.shape != size_rank.shape or weak.shape != (priority.shape[0],):
        raise ValueError("vol-size sell-priority inputs do not align")
    result = priority.copy()
    weak_rows = np.flatnonzero(weak)
    result[weak_rows] += float(size_penalty) * np.nan_to_num(
        size_rank[weak_rows], nan=0.5
    )
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered vol-size sell-priority research")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility_rank = percentile.cross_sectional_percent_rank(
        defensive.trailing_log_volatility(
            context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
        )
    )
    current_priority = vol_priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    size_rank = percentile.cross_sectional_percent_rank(context.arrays["total_mv"])

    results, cache = {}, {}
    for penalty in SIZE_PENALTIES:
        case_id = f"size_penalty_{penalty:.4f}"
        priority = vol_size_priority_matrix(
            current_priority, size_rank, confirmed_weak, penalty
        )
        result, daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            score_sell_priority_override=priority,
        )
        results[case_id] = result
        cache[case_id] = (daily, actions)

    baseline_id = "size_penalty_0.0000"
    candidate_id = "size_penalty_0.0250"
    baseline, candidate = results[baseline_id], results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(np.isclose(
            baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("vol-size priority baseline drifted")

    deltas = {
        case_id: {
            section: {
                key: float(result[section][key] - baseline[section][key])
                for key in (
                    "cumulative_return", "cagr", "sharpe", "max_drawdown",
                    "turnover_annualized",
                )
            }
            for section in (
                "metrics_0_30pct", "train_2022_2024", "holdout_2025",
                "metrics_0_65pct",
            )
        }
        for case_id, result in results.items()
        if case_id != baseline_id
    }
    neighbor_ids = ("size_penalty_0.0125", "size_penalty_0.0500")
    neighbor_confirmation = sum(
        deltas[case_id]["train_2022_2024"]["cagr"] > 0.0
        and deltas[case_id]["holdout_2025"]["cumulative_return"] >= 0.0
        for case_id in neighbor_ids
    )
    gates = {
        "train_utility_improved": size_recheck.utility(
            candidate["train_2022_2024"]
        ) > size_recheck.utility(baseline["train_2022_2024"]),
        "holdout_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "drawdown_not_worse": candidate["metrics_0_30pct"]["max_drawdown"]
        <= baseline["metrics_0_30pct"]["max_drawdown"],
        "both_neighbor_directions_confirm": neighbor_confirmation == 2,
        "all_years_positive": all(
            value > 0.0
            for value in candidate["metrics_0_30pct"]["annual_returns"].values()
        ),
        "full_10_positions": candidate["metrics_0_30pct"][
            "full_10_position_ratio"
        ] == 1.0,
    }
    selected = candidate_id if all(gates.values()) else baseline_id
    selected_penalty = float(selected.removeprefix("size_penalty_"))
    repeat_priority = vol_size_priority_matrix(
        current_priority, size_rank, confirmed_weak, selected_penalty
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=repeat_priority,
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
        raise RuntimeError("vol-size priority deterministic replay failed")

    payload = {
        "status": "current_vol_size_sell_priority_complete_2026_not_opened",
        "only_change": (
            "on confirmed-weak days, among already-eligible exits only, add a small "
            "positive size-percentile term so smaller and more volatile holdings sell first"
        ),
        "size_penalties": list(SIZE_PENALTIES),
        "results": results,
        "deltas_vs_current": deltas,
        "selection_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != baseline_id,
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate": selected,
        "selection_gates": gates,
        "candidate_deltas": deltas[candidate_id],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
