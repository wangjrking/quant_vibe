from __future__ import annotations

import numpy as np
import pandas as pd

from research_v260_fixed10_execution_timing_audit_20260822 import (
    audit_actions,
    audit_portfolio,
)


def arrays():
    return {
        "dates": np.array(["20240102", "20240103", "20240104"]),
        "stocks": np.array(["000001.SZ", "600000.SH"]),
        "buy_open": np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]),
    }


def test_action_must_use_next_session_raw_open() -> None:
    actions = pd.DataFrame([
        {
            "signal_date": "20240102",
            "buy_date": "20240103",
            "action": "BUY",
            "stock_code": "000001.SZ",
            "execution_open_raw": 10.0,
        }
    ])
    assert audit_actions(actions, arrays())["all_gates_passed"] is True


def test_nonadjacent_or_wrong_open_fails() -> None:
    actions = pd.DataFrame([
        {
            "signal_date": "20240102",
            "buy_date": "20240104",
            "action": "BUY",
            "stock_code": "000001.SZ",
            "execution_open_raw": 99.0,
        }
    ])
    audit = audit_actions(actions, arrays())
    assert audit["nonadjacent_signal_execution_count"] == 1
    assert audit["execution_open_raw_mismatch_count"] == 1
    assert audit["all_gates_passed"] is False


def test_same_stock_same_day_buy_and_sell_fails() -> None:
    actions = pd.DataFrame([
        {
            "signal_date": "20240102",
            "buy_date": "20240103",
            "action": action,
            "stock_code": "000001.SZ",
            "execution_open_raw": 10.0,
        }
        for action in ("BUY", "SELL")
    ])
    audit = audit_actions(actions, arrays())
    assert audit["same_stock_same_day_buy_sell_count"] == 1
    assert audit["all_gates_passed"] is False


def test_portfolio_reports_exact10_equal_targets_and_valid_equity() -> None:
    daily = pd.DataFrame({
        "positions": [10, 10],
        "invested_ratio": [0.96, 0.98],
        "equity": [1_000_000.0, 1_010_000.0],
        "return": [0.0, 0.01],
    })
    actions = pd.DataFrame({
        "action": ["BUY", "SELL"],
        "target_pct": [0.10, 0.0],
    })
    audit = audit_portfolio(daily, actions)
    assert audit["soft_direction_met"] is True
    assert audit["hard_integrity_passed"] is True
    assert audit["all_gates_passed"] is True


def test_equalweight_trim_sell_may_keep_10pct_target() -> None:
    daily = pd.DataFrame({
        "positions": [10],
        "invested_ratio": [0.97],
        "equity": [1_000_000.0],
        "return": [0.0],
    })
    actions = pd.DataFrame({
        "action": ["BUY", "SELL"],
        "target_pct": [0.10, 0.10],
    })
    audit = audit_portfolio(daily, actions)
    assert audit["equalweight_trim_sell_count"] == 1
    assert audit["all_gates_passed"] is True


def test_position_target_and_investment_drift_are_soft() -> None:
    daily = pd.DataFrame({
        "positions": [10, 9],
        "invested_ratio": [0.80, 0.82],
        "equity": [1_000_000.0, 1_010_000.0],
        "return": [0.0, 0.01],
    })
    actions = pd.DataFrame({"action": ["BUY"], "target_pct": [0.09]})
    audit = audit_portfolio(daily, actions)
    assert audit["exactly10_every_day"] is False
    assert audit["all_buy_targets_equal_10pct"] is False
    assert audit["soft_direction_met"] is False
    assert audit["hard_integrity_passed"] is True
    assert audit["all_gates_passed"] is True


def test_invalid_equity_remains_a_hard_integrity_failure() -> None:
    daily = pd.DataFrame({
        "positions": [9],
        "invested_ratio": [0.80],
        "equity": [float("nan")],
        "return": [0.0],
    })
    actions = pd.DataFrame({"action": ["BUY"], "target_pct": [0.09]})
    audit = audit_portfolio(daily, actions)
    assert audit["soft_direction_met"] is False
    assert audit["hard_integrity_passed"] is False
    assert audit["all_gates_passed"] is False
