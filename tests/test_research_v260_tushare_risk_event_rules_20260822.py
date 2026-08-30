from __future__ import annotations

import numpy as np
import pandas as pd

from quant.main.research_v260_tushare_risk_event_rules_20260822 import (
    RULES,
    _feature_matrix,
    _monthly_return_deltas,
)


def test_rule_budget_is_small_and_semantically_distinct() -> None:
    assert set(RULES) == {
        "event_start_only",
        "all_event_momentum_confirmed",
        "focused_momentum_confirmed",
    }
    assert RULES["event_start_only"]["ordinary_shock"] == "diagnostic_only"
    assert RULES["focused_momentum_confirmed"]["ordinary_shock"] == "diagnostic_only"


def test_feature_matrix_is_exact_key_only() -> None:
    dates = np.array(["20260820", "20260821"])
    stocks = np.array(["000001.SZ", "000002.SZ"])
    frame = pd.DataFrame(
        {
            "signal_date": ["20260820", "20260822", "20260821"],
            "stock_code": ["000001.SZ", "000001.SZ", "999999.SZ"],
            "shock_count_1d": [1, 1, 1],
        }
    )
    actual = _feature_matrix(dates, stocks, frame, "shock_count_1d")
    expected = np.array([[True, False], [False, False]])
    np.testing.assert_array_equal(actual, expected)


def test_feature_matrix_rejects_missing_contract_field() -> None:
    frame = pd.DataFrame({"signal_date": [], "stock_code": []})
    try:
        _feature_matrix(np.array([]), np.array([]), frame, "shock_count_1d")
    except ValueError as exc:
        assert "shock_count_1d" in str(exc)
    else:
        raise AssertionError("missing feature field must fail closed")


def test_monthly_return_deltas_use_compounded_returns() -> None:
    baseline = pd.DataFrame(
        {"date": ["20260102", "20260105"], "return": [0.10, -0.10]}
    )
    candidate = pd.DataFrame(
        {"date": ["20260102", "20260105"], "return": [0.10, 0.0]}
    )
    actual = _monthly_return_deltas(baseline, candidate, "20260101", "20260131")
    assert len(actual) == 1
    assert np.isclose(actual[0]["baseline_return"], -0.01)
    assert np.isclose(actual[0]["candidate_return"], 0.10)
    assert np.isclose(actual[0]["delta"], 0.11)
