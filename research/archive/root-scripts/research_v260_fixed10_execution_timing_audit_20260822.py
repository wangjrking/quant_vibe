from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


REPORTS = REPO / "quant/data_file/reports"
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = REPORTS / "strategy_agent_v260_fixed10_execution_timing_audit_20260822"


def audit_actions(actions: pd.DataFrame, arrays: dict) -> dict:
    dates = [str(value) for value in arrays["dates"]]
    stocks = [str(value) for value in arrays["stocks"]]
    date_index = {value: index for index, value in enumerate(dates)}
    stock_index = {value: index for index, value in enumerate(stocks)}
    next_session = dict(zip(dates[:-1], dates[1:]))
    nonadjacent = 0
    missing_key = 0
    price_mismatch = 0
    nonfinite_price = 0
    for row in actions.itertuples(index=False):
        signal_date = str(row.signal_date)
        buy_date = str(row.buy_date)
        stock_code = str(row.stock_code)
        if next_session.get(signal_date) != buy_date:
            nonadjacent += 1
        d_idx = date_index.get(signal_date)
        s_idx = stock_index.get(stock_code)
        if d_idx is None or s_idx is None or d_idx >= len(dates) - 1:
            missing_key += 1
            continue
        expected = float(arrays["buy_open"][d_idx, s_idx])
        actual = float(row.execution_open_raw)
        if not np.isfinite(expected) or not np.isfinite(actual):
            nonfinite_price += 1
        elif not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
            price_mismatch += 1
    action_types_per_key = (
        actions.groupby(["buy_date", "stock_code"])["action"].nunique()
        if len(actions)
        else pd.Series(dtype=np.int64)
    )
    same_stock_same_day_buy_sell = int((action_types_per_key > 1).sum())
    return {
        "action_count": int(len(actions)),
        "nonadjacent_signal_execution_count": int(nonadjacent),
        "missing_execution_key_count": int(missing_key),
        "execution_open_raw_mismatch_count": int(price_mismatch),
        "nonfinite_execution_price_count": int(nonfinite_price),
        "same_stock_same_day_buy_sell_count": same_stock_same_day_buy_sell,
        "all_gates_passed": not any(
            (nonadjacent, missing_key, price_mismatch, nonfinite_price,
             same_stock_same_day_buy_sell)
        ),
    }


def audit_portfolio(daily: pd.DataFrame, actions: pd.DataFrame) -> dict:
    positions = daily["positions"].to_numpy(dtype=np.int64)
    invested = daily["invested_ratio"].to_numpy(dtype=np.float64)
    equity = daily["equity"].to_numpy(dtype=np.float64)
    returns = daily["return"].to_numpy(dtype=np.float64)
    buy_targets = actions.loc[actions["action"] == "BUY", "target_pct"].to_numpy(
        dtype=np.float64
    )
    sell_targets = actions.loc[actions["action"] == "SELL", "target_pct"].to_numpy(
        dtype=np.float64
    )
    exactly10 = bool(len(positions) and np.all(positions == 10))
    buy_targets_equal = bool(
        len(buy_targets) and np.allclose(buy_targets, 0.10, rtol=0.0, atol=1e-12)
    )
    sell_targets_valid = bool(
        len(sell_targets)
        and np.all(
            np.isclose(sell_targets, 0.0, rtol=0.0, atol=1e-12)
            | np.isclose(sell_targets, 0.10, rtol=0.0, atol=1e-12)
        )
    )
    invested_valid = bool(
        np.isfinite(invested).all()
        and np.all(invested >= -1e-12)
        and np.all(invested <= 1.0 + 1e-12)
    )
    equity_valid = bool(np.isfinite(equity).all() and np.all(equity > 0.0))
    returns_valid = bool(np.isfinite(returns).all())
    hard_integrity_passed = bool(
        invested_valid and equity_valid and returns_valid
    )
    soft_direction_met = bool(
        exactly10
        and buy_targets_equal
        and sell_targets_valid
        and (float(invested.mean()) >= 0.95 if len(invested) else False)
    )
    return {
        "days": int(len(daily)),
        "positions_min": int(positions.min()) if len(positions) else 0,
        "positions_max": int(positions.max()) if len(positions) else 0,
        "exactly10_every_day": exactly10,
        "buy_target_weight": 0.10,
        "all_buy_targets_equal_10pct": buy_targets_equal,
        "full_exit_sell_count": int(np.isclose(sell_targets, 0.0).sum()),
        "equalweight_trim_sell_count": int(np.isclose(sell_targets, 0.10).sum()),
        "all_sell_targets_are_exit0_or_equalweight10pct": sell_targets_valid,
        "average_invested_ratio": float(invested.mean()) if len(invested) else 0.0,
        "minimum_invested_ratio": float(invested.min()) if len(invested) else 0.0,
        "invested_ratio_finite_and_bounded": invested_valid,
        "equity_positive_and_finite": equity_valid,
        "returns_finite": returns_valid,
        "portfolio_direction_is_soft": True,
        "soft_direction_met": soft_direction_met,
        "hard_integrity_passed": hard_integrity_passed,
        "all_gates_passed": hard_integrity_passed,
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("execution timing audit cannot consume 2026")
    policy = checkpoint["selected_policy"]
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(context.arrays["close_qfq"], 20)
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    daily, actions = age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        0.003,
        score_sell_priority_override=sell_priority,
    )
    audit = audit_actions(actions, context.arrays)
    portfolio_audit = audit_portfolio(daily, actions)
    if not audit["all_gates_passed"] or not portfolio_audit["hard_integrity_passed"]:
        raise RuntimeError("execution timing, raw-open, or portfolio integrity audit failed")
    result = {
        "status": "pre2026_execution_timing_audit_passed",
        "candidate_id": checkpoint["selected_candidate"],
        "audit": audit,
        "portfolio_audit": portfolio_audit,
        "logical_max_date": context.access["logical_max_date"],
        "semantics": {
            "ranking_uses_signal_date_score": True,
            "execution_uses_next_official_session": True,
            "execution_price": "raw_unadjusted_open",
            "same_day_round_trip_prohibited": True,
            "positions_equalweight_and_investment_are_soft_diagnostics": True,
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "execution_timing_audit.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
