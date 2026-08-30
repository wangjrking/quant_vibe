from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_weak_market_exit_threshold_20260822 as subject


def test_exit_threshold_schedule_changes_only_weak_days() -> None:
    strong = np.array([True, False, True, False], dtype=np.bool_)
    actual = subject.exit_threshold_schedule(strong, 0.80)
    np.testing.assert_allclose(actual, np.array([0.85, 0.80, 0.85, 0.80]))


def test_exit_threshold_schedule_is_float_and_deterministic() -> None:
    strong = np.array([False, True], dtype=np.bool_)
    first = subject.exit_threshold_schedule(strong, 0.80)
    second = subject.exit_threshold_schedule(strong, 0.80)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, second)
