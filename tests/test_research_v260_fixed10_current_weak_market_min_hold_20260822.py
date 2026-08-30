from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_weak_market_min_hold_20260822 as subject


def test_weak_market_min_hold_schedule_is_regime_only() -> None:
    strong = np.array([True, False, False, True], dtype=np.bool_)
    actual = subject.weak_market_min_hold_schedule(strong)
    np.testing.assert_array_equal(actual, np.array([4, 8, 8, 4]))


def test_schedule_does_not_mutate_input() -> None:
    strong = np.array([False, True], dtype=np.bool_)
    before = strong.copy()
    subject.weak_market_min_hold_schedule(strong)
    np.testing.assert_array_equal(strong, before)
