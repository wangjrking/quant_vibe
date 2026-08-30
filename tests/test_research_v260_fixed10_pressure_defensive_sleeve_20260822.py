from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_defensive_sleeve_20260822 import (
    ALPHA_SLOTS,
    CANDIDATES,
    DEFENSIVE_SLOTS,
    defensive_sleeve_order,
)


def test_candidate_budget_is_one_fixed_sleeve() -> None:
    assert CANDIDATES == ("production_order", "weak_market_8alpha_2defensive")
    assert ALPHA_SLOTS == 8
    assert DEFENSIVE_SLOTS == 2


def test_strong_market_order_is_unchanged() -> None:
    order = np.array([[0, 1, 2, 3]])
    actual = defensive_sleeve_order(
        order,
        np.array([[0.4, 0.3, 0.2, 0.1]]),
        np.array([False]),
        alpha_slots=2,
        defensive_slots=1,
        pool_size=4,
    )

    np.testing.assert_array_equal(actual, order)


def test_weak_market_preserves_alpha_and_adds_lowest_volatility_name() -> None:
    order = np.array([[0, 1, 2, 3, 4]])
    volatility = np.array([[0.5, 0.4, 0.3, 0.1, 0.2]])
    actual = defensive_sleeve_order(
        order,
        volatility,
        np.array([True]),
        alpha_slots=2,
        defensive_slots=1,
        pool_size=5,
    )

    np.testing.assert_array_equal(actual[0, :3], np.array([0, 1, 3]))
    np.testing.assert_array_equal(np.sort(actual[0]), np.arange(5))
