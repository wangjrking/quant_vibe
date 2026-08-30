from __future__ import annotations

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
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_replacement_failure_segmentation_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
HORIZONS = (5, 10, 20)


def run_with_observers(context, policy: dict):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
    )
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_min_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    positions: list[dict] = []
    pressure: list[dict] = []
    daily, actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=cadence.rebalance_schedule(
            len(context.arrays["dates"]),
            int(policy["portfolio_rebalance_interval_days"]),
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=context.empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
        score_sell_pressure_extra_min_age_override=extra_min_age,
        score_sell_priority_override=priority_matrix,
        score_sell_pressure_observer=pressure.append,
        position_observer=positions.append,
    )
    return daily, actions, positions, pressure, volatility_rank, strong, confirmed_weak


def forward_return(close: np.ndarray, start: int, idx: int, horizon: int) -> float:
    end = start + horizon
    if end >= close.shape[0]:
        return np.nan
    first, last = float(close[start, idx]), float(close[end, idx])
    if not np.isfinite(first) or not np.isfinite(last) or first <= 0.0:
        return np.nan
    return last / first - 1.0


def summarize(rows: list[dict]) -> dict:
    result = {"transition_days": int(len(rows))}
    for horizon in HORIZONS:
        key = f"replacement_excess_{horizon}d"
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[f"observed_{horizon}d"] = int(len(values))
        result[f"mean_excess_{horizon}d"] = (
            float(np.mean(values)) if len(values) else None
        )
        result[f"median_excess_{horizon}d"] = (
            float(np.median(values)) if len(values) else None
        )
        result[f"win_rate_{horizon}d"] = (
            float(np.mean(values > 0.0)) if len(values) else None
        )
    return result


def grouped(rows: list[dict], key: str) -> dict:
    return {
        str(value): summarize([row for row in rows if row[key] == value])
        for value in sorted({row[key] for row in rows}, key=str)
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered replacement failure segmentation")
    context = harness.load_context(checkpoint["selected_policy"])
    (
        daily,
        actions,
        observations,
        pressure_records,
        volatility_rank,
        strong,
        confirmed_weak,
    ) = run_with_observers(context, checkpoint["selected_policy"])
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("replacement failure segmentation baseline drifted")

    stocks = np.asarray(context.arrays["stocks"], dtype=str)
    dates = np.asarray(context.arrays["dates"], dtype=str)
    stock_index = {code: idx for idx, code in enumerate(stocks)}
    date_index = {date: idx for idx, date in enumerate(dates)}
    close = np.asarray(context.arrays["close_qfq"], dtype=np.float64)
    pressure_by_buy_date = {
        str(record["buy_date"]): record for record in pressure_records
    }
    rows = []
    previous: set[str] = set()
    for item in observations:
        current = set(str(code) for code in item["positions_after_trades"])
        exited = sorted(previous - current)
        entered = sorted(current - previous)
        previous = current
        if not exited or not entered:
            continue
        signal_date = str(item["signal_date"])
        buy_date = str(item["buy_date"])
        signal_idx = date_index[signal_date]
        buy_idx = date_index[buy_date]
        pressure = pressure_by_buy_date[buy_date]
        entry_volatility = np.asarray(
            [volatility_rank[signal_idx, stock_index[code]] for code in entered],
            dtype=np.float64,
        )
        finite_volatility = entry_volatility[np.isfinite(entry_volatility)]
        mean_entry_volatility = (
            float(np.mean(finite_volatility)) if len(finite_volatility) else np.nan
        )
        row = {
            "signal_date": signal_date,
            "buy_date": buy_date,
            "year": buy_date[:4],
            "entered_count": int(len(entered)),
            "exited_count": int(len(exited)),
            "market_state": (
                "confirmed_weak"
                if bool(confirmed_weak[signal_idx])
                else "strong_or_first_weak"
            ),
            "pressure_mode": (
                "pressure_limit2"
                if int(pressure["resolved_limit"]) > int(pressure["base_limit"])
                else "ordinary_limit1"
            ),
            "candidate_count_band": (
                "at_least_5"
                if int(pressure["score_sell_candidate_count"]) >= 5
                else "below_5"
            ),
            "entry_volatility_band": (
                "high_ge_075"
                if mean_entry_volatility >= 0.75
                else "middle_040_075"
                if mean_entry_volatility >= 0.40
                else "low_below_040"
            ),
            "mean_entry_volatility_percentile": mean_entry_volatility,
            "production_strong_market": bool(strong[signal_idx]),
        }
        for horizon in HORIZONS:
            sold = np.asarray(
                [
                    forward_return(close, buy_idx, stock_index[code], horizon)
                    for code in exited
                ],
                dtype=np.float64,
            )
            bought = np.asarray(
                [
                    forward_return(close, buy_idx, stock_index[code], horizon)
                    for code in entered
                ],
                dtype=np.float64,
            )
            sold = sold[np.isfinite(sold)]
            bought = bought[np.isfinite(bought)]
            row[f"replacement_excess_{horizon}d"] = (
                float(np.mean(bought) - np.mean(sold))
                if len(sold) and len(bought)
                else np.nan
            )
        rows.append(row)

    result = {
        "status": "replacement_failure_segmentation_complete_2026_not_opened",
        "method": (
            "segment actual position replacement days by the unchanged pressure state, "
            "confirmed weak state and entry volatility; diagnostics do not alter rules"
        ),
        "overall": summarize(rows),
        "by_year": grouped(rows, "year"),
        "by_pressure_mode": grouped(rows, "pressure_mode"),
        "by_market_state": grouped(rows, "market_state"),
        "by_candidate_count_band": grouped(rows, "candidate_count_band"),
        "by_entry_volatility_band": grouped(rows, "entry_volatility_band"),
        "baseline_equivalence": equivalence,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "segmentation.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
