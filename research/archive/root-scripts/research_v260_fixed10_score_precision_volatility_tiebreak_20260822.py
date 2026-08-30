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
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_score_precision_volatility_tiebreak_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
SCORE_DECIMALS = 4
CASES = (
    "full_precision",
    "round4_stock_code_control",
    "round4_low_volatility_tiebreak",
)


def rounded_tiebreak_order(
    score: np.ndarray,
    volatility: np.ndarray,
    stocks: np.ndarray,
    decimals: int,
    use_volatility: bool,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    risk = np.asarray(volatility, dtype=np.float64)
    labels = np.asarray(stocks, dtype=str)
    if values.ndim != 2 or risk.shape != values.shape:
        raise ValueError("score/volatility matrices do not align")
    if values.shape[1] != len(labels):
        raise ValueError("stock labels do not align")
    if int(decimals) < 0:
        raise ValueError("score precision cannot be negative")
    rounded = np.round(values, int(decimals))
    result = np.empty(values.shape, dtype=np.int32)
    for row_index in range(values.shape[0]):
        score_key = np.where(np.isfinite(rounded[row_index]), rounded[row_index], -np.inf)
        if use_volatility:
            risk_key = np.where(np.isfinite(risk[row_index]), risk[row_index], np.inf)
        else:
            risk_key = np.zeros(values.shape[1], dtype=np.float64)
        result[row_index] = np.lexsort(
            (labels, risk_key, -score_key)
        ).astype(np.int32)
    return result


def order_overlap(left: np.ndarray, right: np.ndarray, n: int) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("order matrices do not align")
    count = int(n)
    if count < 1 or count > a.shape[1]:
        raise ValueError("invalid overlap width")
    return float(np.mean([
        len(set(map(int, x[:count])) & set(map(int, y[:count]))) / count
        for x, y in zip(a, b)
    ]))


def run_case(context, policy: dict, order: np.ndarray):
    case = copy.copy(context)
    case.order = np.asarray(order)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    return age_guard.run_policy(
        case,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
        exit_score_override=context.score,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered score tiebreak development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    if context.access["logical_max_date"] >= "20260101":
        raise PermissionError("score tiebreak development crossed into 2026")

    trailing_volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    orders = {
        "full_precision": context.order,
        "round4_stock_code_control": rounded_tiebreak_order(
            context.score,
            trailing_volatility,
            context.arrays["stocks"],
            SCORE_DECIMALS,
            use_volatility=False,
        ),
        "round4_low_volatility_tiebreak": rounded_tiebreak_order(
            context.score,
            trailing_volatility,
            context.arrays["stocks"],
            SCORE_DECIMALS,
            use_volatility=True,
        ),
    }
    results = {}
    cache = {}
    for case_id in CASES:
        item, daily, actions = run_case(context, policy, orders[case_id])
        results[case_id] = item
        cache[case_id] = (daily, actions)

    baseline_id = "full_precision"
    candidate_id = "round4_low_volatility_tiebreak"
    baseline = results[baseline_id]
    candidate = results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(
            baseline["metrics_0_30pct"][key],
            expected[key],
            rtol=0.0,
            atol=1e-12,
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("score tiebreak baseline drifted from checkpoint")

    current = candidate["metrics_0_30pct"]
    reference = baseline["metrics_0_30pct"]
    gates = {
        "training_cagr_improved": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "training_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "full_period_cumulative_not_worse": current["cumulative_return"]
        >= reference["cumulative_return"],
        "full_period_sharpe_not_worse": current["sharpe"] >= reference["sharpe"],
        "drawdown_not_worse": current["max_drawdown"] <= reference["max_drawdown"],
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "turnover_not_worse": current["turnover_annualized"]
        <= reference["turnover_annualized"],
        "all_years_positive": min(current["annual_returns"].values()) > 0.0,
        "exactly10_full_period": current["full_10_position_ratio"] == 1.0,
    }
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat, repeat_daily, repeat_actions = run_case(
        context, policy, orders[selected]
    )
    deterministic = {
        "order": np.array_equal(
            orders[selected],
            rounded_tiebreak_order(
                context.score,
                trailing_volatility,
                context.arrays["stocks"],
                SCORE_DECIMALS,
                use_volatility=True,
            ) if selected == candidate_id else context.order,
        ),
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
        raise RuntimeError("score tiebreak deterministic replay failed")

    payload = {
        "status": "candidate_accepted_pre2026_validation_not_opened"
        if selected == candidate_id
        else "candidate_rejected_pre2026_validation_not_opened",
        "only_change": (
            "new-entry ranking rounds the frozen score to four decimals and, only "
            "inside an exact rounded-score tie, prefers lower trailing-20-session "
            "volatility; all exits retain the original full-precision score"
        ),
        "candidate_budget": {
            "selectable": [candidate_id],
            "nonselectable_attribution_control": ["round4_stock_code_control"],
            "post_hoc_additions": 0,
        },
        "results": results,
        "order_diagnostics": {
            case_id: {
                "top10_mean_overlap_vs_full_precision": order_overlap(
                    context.order, orders[case_id], 10
                ),
                "top30_mean_overlap_vs_full_precision": order_overlap(
                    context.order, orders[case_id], 30
                ),
            }
            for case_id in CASES[1:]
        },
        "candidate_gates": gates,
        "selected_candidate": selected,
        "checkpoint_changed": selected != baseline_id,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", payload)
    print(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
