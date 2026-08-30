from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
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
    "strategy_agent_v260_fixed10_score_precision_robustness_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
DECIMALS = (6, 5, 4)


def stable_descending_order(score: np.ndarray, stocks: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    labels = np.asarray(stocks, dtype=str)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("score/stocks shape mismatch")
    result = np.empty(values.shape, dtype=np.int32)
    for row_index, row in enumerate(values):
        safe = np.where(np.isfinite(row), row, -np.inf)
        result[row_index] = np.lexsort((labels, -safe)).astype(np.int32)
    return result


def topn_overlap(left: np.ndarray, right: np.ndarray, n: int) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("orders must share a date by stock shape")
    count = int(n)
    if count < 1 or count > left.shape[1]:
        raise ValueError("invalid top-n")
    overlaps = [
        len(set(map(int, a[:count])) & set(map(int, b[:count]))) / count
        for a, b in zip(left, right)
    ]
    return float(np.mean(overlaps))


def run_case(base_context, policy: dict, decimals: int | None):
    if decimals is None:
        context = base_context
    else:
        score = np.round(base_context.score.astype(np.float64), int(decimals))
        order = stable_descending_order(score, base_context.arrays["stocks"])
        context = replace(base_context, score=score, order=order)
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
    return context, age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered score-precision robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    base_context = harness.load_context(policy)

    contexts = {}
    results = {}
    cache = {}
    for case_id, decimals in [("full_precision", None)] + [
        (f"round_{value}dp", value) for value in DECIMALS
    ]:
        context, (result, daily, actions) = run_case(
            base_context, policy, decimals
        )
        contexts[case_id] = context
        results[case_id] = result
        cache[case_id] = (daily, actions)

    expected = checkpoint["current_best_equalweight"]
    baseline_equivalence = {
        key: bool(
            np.isclose(
                results["full_precision"]["metrics_0_30pct"][key],
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
    if not all(baseline_equivalence.values()):
        raise RuntimeError("score-precision baseline drifted")

    baseline_metrics = results["full_precision"]["metrics_0_30pct"]
    summary = {}
    for decimals in DECIMALS:
        case_id = f"round_{decimals}dp"
        context = contexts[case_id]
        metrics = results[case_id]["metrics_0_30pct"]
        summary[case_id] = {
            "top10_mean_overlap": topn_overlap(
                base_context.order, context.order, 10
            ),
            "metrics": metrics,
            "delta_vs_full_precision": {
                key: float(metrics[key] - baseline_metrics[key])
                for key in (
                    "cumulative_return",
                    "cagr",
                    "sharpe",
                    "max_drawdown",
                    "turnover_annualized",
                )
            },
            "all_calendar_years_positive": all(
                value > 0.0 for value in metrics["annual_returns"].values()
            ),
            "stress_cumulative_return": results[case_id]["metrics_0_65pct"][
                "cumulative_return"
            ],
            "full_10_positions": metrics["full_10_position_ratio"] == 1.0,
        }

    repeat_context, (repeat, repeat_daily, repeat_actions) = run_case(
        base_context, policy, 5
    )
    deterministic = {
        "order": np.array_equal(
            contexts["round_5dp"].order, repeat_context.order
        ),
        "daily": round1.frame_hash(cache["round_5dp"][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache["round_5dp"][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("score-precision replay failed")

    payload = {
        "status": "score_precision_robustness_complete_2026_not_opened",
        "method": (
            "round the complete decision score, rebuild a deterministic stock-code "
            "tie-broken ranking, and replay the unchanged strategy"
        ),
        "summary": summary,
        "operational_decision": (
            "preserve at least the coarsest tested precision that remains economically "
            "equivalent; never rely on database-dependent tie ordering"
        ),
        "baseline_equivalence": baseline_equivalence,
        "deterministic_replay": deterministic,
        "data_access": base_context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "score_precision_robustness.json", payload)
    print(
        json.dumps(
            {"status": payload["status"], "summary": summary},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
