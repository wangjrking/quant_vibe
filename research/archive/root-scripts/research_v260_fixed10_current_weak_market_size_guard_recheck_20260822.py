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
import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_size_tail_guard_20260822 as size_guard
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_weak_market_size_guard_recheck_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PERCENTILES = (0.00, 0.05, 0.10, 0.15)


def utility(metrics: dict) -> float:
    return float(
        metrics["cagr"]
        + 0.25 * metrics["sharpe"]
        - 0.50 * metrics["max_drawdown"]
        - 0.0025 * metrics["turnover_annualized"]
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered current size-guard recheck")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    production_selection = context.harness.v174.selection_mask(
        context.arrays, context.definition["max_rank_deterioration"]
    )

    results = {}
    cache = {}
    for cutoff in PERCENTILES:
        case_id = f"bottom_{int(round(cutoff * 100)):02d}pct"
        selection = size_guard.size_tail_selection_mask(
            production_selection,
            context.arrays["total_mv"],
            context.arrays["signal_clean"],
            confirmed_weak,
            cutoff,
        )
        result, daily, actions = age_guard.run_policy(
            context,
            policy,
            extra_age,
            selection_mask_override=selection,
            score_sell_priority_override=sell_priority,
        )
        results[case_id] = {
            "minimum_size_percentile": cutoff,
            "metrics_0_30pct": result["metrics_0_30pct"],
            "train_2022_2024": result["train_2022_2024"],
            "holdout_2025": result["holdout_2025"],
            "metrics_0_65pct": result["metrics_0_65pct"],
            "utility_0_30pct": utility(result["metrics_0_30pct"]),
            "train_utility": utility(result["train_2022_2024"]),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "removed_stock_dates": int(
                production_selection.sum() - selection.sum()
            ),
        }
        cache[case_id] = (daily, actions)

    baseline = results["bottom_00pct"]
    candidate = results["bottom_10pct"]
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key],
                expected[key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("current size-guard baseline drifted")

    deltas = {
        case_id: {
            "train_utility": float(item["train_utility"] - baseline["train_utility"]),
            "full_utility": float(
                item["utility_0_30pct"] - baseline["utility_0_30pct"]
            ),
            "holdout_2025_return": float(
                item["holdout_2025"]["cumulative_return"]
                - baseline["holdout_2025"]["cumulative_return"]
            ),
            "stress_return": float(
                item["metrics_0_65pct"]["cumulative_return"]
                - baseline["metrics_0_65pct"]["cumulative_return"]
            ),
        }
        for case_id, item in results.items()
        if case_id != "bottom_00pct"
    }
    breadth = sum(
        deltas[case_id]["train_utility"] > 0.0
        and deltas[case_id]["holdout_2025_return"] >= 0.0
        for case_id in ("bottom_05pct", "bottom_10pct", "bottom_15pct")
    )
    gates = {
        "candidate_train_utility_improved": deltas["bottom_10pct"][
            "train_utility"
        ]
        > 0.0,
        "candidate_holdout_not_worse": deltas["bottom_10pct"][
            "holdout_2025_return"
        ]
        >= 0.0,
        "candidate_stress_not_worse": deltas["bottom_10pct"]["stress_return"]
        >= 0.0,
        "candidate_drawdown_not_worse": candidate["metrics_0_30pct"][
            "max_drawdown"
        ]
        <= baseline["metrics_0_30pct"]["max_drawdown"],
        "at_least_two_neighbor_points_confirm": breadth >= 2,
        "all_years_positive": all(
            value > 0.0
            for value in candidate["metrics_0_30pct"]["annual_returns"].values()
        ),
        "full_10_positions": candidate["metrics_0_30pct"][
            "full_10_position_ratio"
        ]
        == 1.0,
    }
    selected = "bottom_10pct" if all(gates.values()) else "bottom_00pct"

    selected_cutoff = results[selected]["minimum_size_percentile"]
    repeat_selection = size_guard.size_tail_selection_mask(
        production_selection,
        context.arrays["total_mv"],
        context.arrays["signal_clean"],
        confirmed_weak,
        selected_cutoff,
    )
    repeat, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        selection_mask_override=repeat_selection,
        score_sell_priority_override=sell_priority,
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("current size-guard replay failed")

    payload = {
        "status": "current_weak_market_size_guard_recheck_complete_2026_not_opened",
        "rule": (
            "after two consecutive weak-market sessions, new entries must not be in "
            "the bottom cross-sectional total-market-value percentile; held positions "
            "and exits are unchanged"
        ),
        "results": results,
        "deltas_vs_baseline": deltas,
        "selection_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != "bottom_00pct",
        "baseline_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_candidate": selected,
                "selection_gates": gates,
                "deltas_vs_baseline": deltas,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
