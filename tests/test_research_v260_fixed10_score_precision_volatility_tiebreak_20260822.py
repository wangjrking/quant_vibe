from __future__ import annotations

import numpy as np
import pytest

from research_v260_fixed10_score_precision_volatility_tiebreak_20260822 import (
    order_overlap,
    rounded_tiebreak_order,
)


def test_low_volatility_breaks_only_rounded_score_ties() -> None:
    score = np.array([[0.90004, 0.90003, 0.89994]], dtype=float)
    volatility = np.array([[0.30, 0.10, 0.01]], dtype=float)
    stocks = np.array(["000001.SZ", "000002.SZ", "000003.SZ"])

    actual = rounded_tiebreak_order(score, volatility, stocks, 4, True)

    assert actual.tolist() == [[1, 0, 2]]


def test_stock_code_is_final_deterministic_tiebreak() -> None:
    score = np.array([[0.9, 0.9]], dtype=float)
    volatility = np.array([[0.1, 0.1]], dtype=float)
    stocks = np.array(["000002.SZ", "000001.SZ"])

    actual = rounded_tiebreak_order(score, volatility, stocks, 4, True)

    assert actual.tolist() == [[1, 0]]


def test_nonfinite_score_is_always_last() -> None:
    score = np.array([[np.nan, 0.1]], dtype=float)
    volatility = np.array([[0.0, 0.5]], dtype=float)
    stocks = np.array(["000001.SZ", "000002.SZ"])

    actual = rounded_tiebreak_order(score, volatility, stocks, 4, True)

    assert actual.tolist() == [[1, 0]]


def test_order_overlap_rejects_invalid_width() -> None:
    order = np.array([[0, 1]], dtype=int)
    with pytest.raises(ValueError, match="invalid overlap"):
        order_overlap(order, order, 0)
