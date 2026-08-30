from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_exit_horizon_profit_optimization_20260822 as mod


def test_exit_horizons_are_fixed_and_include_current():
    assert mod.EXIT_HORIZONS == (
        "pure_10d",
        "1d_10pct",
        "3d_10pct",
        "5d_10pct",
    )


def test_raw_exit_score_uses_fixed_ten_percent_short_horizon():
    arrays = {
        "rank_1d": np.array([[1.0, 0.0]]),
        "rank_3d": np.array([[0.8, 0.2]]),
        "rank_5d": np.array([[0.6, 0.4]]),
        "rank_10d": np.array([[0.0, 1.0]]),
    }
    assert np.allclose(
        mod.raw_exit_score(arrays, "pure_10d"), [[0.0, 1.0]]
    )
    assert np.allclose(
        mod.raw_exit_score(arrays, "1d_10pct"), [[0.1, 0.9]]
    )
    assert np.allclose(
        mod.raw_exit_score(arrays, "3d_10pct"), [[0.08, 0.92]]
    )
    assert np.allclose(
        mod.raw_exit_score(arrays, "5d_10pct"), [[0.06, 0.94]]
    )
