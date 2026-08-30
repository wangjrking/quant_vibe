from __future__ import annotations

import pandas as pd
import pytest

from research_v260_fixed10_residual_cash_sweep_mechanism_attribution_20260823 import (
    mechanism_attribution,
)


def _daily(returns, invested, turnover):
    return pd.DataFrame(
        {
            "date": ["20240102", "20240103", "20240104"],
            "return": returns,
            "invested_ratio": invested,
            "turnover": turnover,
        }
    )


def _actions(rows):
    return pd.DataFrame(
        rows,
        columns=["buy_date", "action", "stock_code"],
    )


def test_mechanism_attribution_identifies_extra_buys_and_exposure():
    current_daily = _daily([0.0, 0.01, -0.01], [0.95, 0.95, 0.96], [0.1, 0.0, 0.0])
    sweep_daily = _daily([0.0, 0.02, -0.005], [0.99, 0.99, 0.99], [0.12, 0.0, 0.0])
    current_actions = _actions([("20240102", "BUY", "000001.SZ")])
    sweep_actions = _actions(
        [
            ("20240102", "BUY", "000001.SZ"),
            ("20240102", "BUY", "000002.SZ"),
        ]
    )

    result = mechanism_attribution(
        current_daily, current_actions, sweep_daily, sweep_actions
    )

    assert result["additional_buy_action_keys"] == 1
    assert result["additional_buy_action_days"] == 1
    assert result["positive_invested_delta_days"] == 3
    assert result["all_days"]["compound_excess"] > 0.0


def test_mechanism_attribution_rejects_duplicate_action_keys():
    daily = _daily([0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0])
    duplicate = _actions(
        [
            ("20240102", "BUY", "000001.SZ"),
            ("20240102", "BUY", "000001.SZ"),
        ]
    )

    with pytest.raises(ValueError, match="action keys must be unique"):
        mechanism_attribution(daily, duplicate, daily, duplicate)
