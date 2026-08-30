from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research_v260_fixed10_vol_priority_action_concentration_20260822 as target


def actions(rows):
    return pd.DataFrame(
        rows,
        columns=["signal_date", "buy_date", "action", "stock_code"],
    )


def test_action_difference_counts_exact_key_changes():
    candidate = actions([
        ("20240102", "20240103", "SELL", "000001.SZ"),
        ("20240102", "20240103", "BUY", "000002.SZ"),
    ])
    reference = actions([
        ("20240102", "20240103", "SELL", "000003.SZ"),
        ("20240102", "20240103", "BUY", "000002.SZ"),
    ])
    result = target.action_difference(candidate, reference)
    assert result["symmetric_difference_count"] == 2
    assert result["changed_execution_date_count"] == 1


def test_action_difference_rejects_duplicate_keys():
    frame = actions([
        ("20240102", "20240103", "SELL", "000001.SZ"),
        ("20240102", "20240103", "SELL", "000001.SZ"),
    ])
    with pytest.raises(ValueError, match="duplicate action keys"):
        target.action_difference(frame, actions([]))


def test_daily_log_excess_and_concentration_are_deterministic():
    candidate = pd.DataFrame({
        "date": ["20240102", "20240103", "20240104"],
        "return": [0.02, -0.01, 0.03],
    })
    reference = pd.DataFrame({
        "date": ["20240104", "20240102", "20240103"],
        "return": [0.01, 0.00, 0.00],
    })
    excess = target.daily_log_excess(candidate, reference)
    assert excess["date"].tolist() == ["20240102", "20240103", "20240104"]
    summary = target.concentration_summary(excess)
    expected = (
        np.log1p(0.02) - np.log1p(0.0)
        + np.log1p(-0.01) - np.log1p(0.0)
        + np.log1p(0.03) - np.log1p(0.01)
    )
    assert summary["total_log_excess"] == pytest.approx(expected)
    assert summary["positive_day_count"] == 2
    assert summary["negative_day_count"] == 1


def test_daily_log_excess_rejects_calendar_mismatch():
    candidate = pd.DataFrame({"date": ["20240102"], "return": [0.0]})
    reference = pd.DataFrame({"date": ["20240103"], "return": [0.0]})
    with pytest.raises(ValueError, match="calendars do not align"):
        target.daily_log_excess(candidate, reference)
