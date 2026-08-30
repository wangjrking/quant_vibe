from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as subject


def test_pressure_age_schedule_keeps_strong_market_at_four() -> None:
    strong = np.array([True, False, True, False], dtype=np.bool_)
    np.testing.assert_array_equal(
        subject.pressure_age_schedule(strong, 10),
        np.array([4, 10, 4, 10]),
    )


def test_pressure_age_schedule_is_integer_and_nonmutating() -> None:
    strong = np.array([False, True], dtype=np.bool_)
    before = strong.copy()
    actual = subject.pressure_age_schedule(strong, 6)
    assert np.issubdtype(actual.dtype, np.integer)
    np.testing.assert_array_equal(strong, before)
