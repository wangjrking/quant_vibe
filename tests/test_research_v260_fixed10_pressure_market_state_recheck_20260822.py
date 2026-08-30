from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_market_state_recheck_20260822 import (
    CANDIDATES,
    pressure_limit_schedule,
)


def test_candidate_budget_is_binary_and_fixed() -> None:
    assert CANDIDATES == (
        "pressure_all_markets",
        "pressure_strong_market_only",
    )


def test_pressure_limit_schedule_uses_frozen_market_state() -> None:
    actual = pressure_limit_schedule([True, False, True, False])

    np.testing.assert_array_equal(actual, np.array([2, 1, 2, 1], dtype=np.int16))
