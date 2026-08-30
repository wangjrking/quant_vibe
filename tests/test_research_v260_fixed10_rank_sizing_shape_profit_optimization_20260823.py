from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_shape_profit_optimization_20260823 as mod


def test_step_rank_multipliers_are_gross_neutral() -> None:
    order = np.arange(10, dtype=np.int64)[None, :]
    result = mod.step_rank_multipliers(order)
    assert np.allclose(result[0, :5], 1.10)
    assert np.allclose(result[0, 5:], 0.90)
    assert np.isclose(result.sum(), 10.0)


def test_step_rank_multipliers_follow_order_and_leave_refill_at_one() -> None:
    order = np.array([[3, 2, 4, 1, 5, 0]], dtype=np.int64)
    result = mod.step_rank_multipliers(order, positions=4)
    assert np.allclose(result[0, order[0, :2]], 1.10)
    assert np.allclose(result[0, order[0, 2:4]], 0.90)
    assert np.allclose(result[0, order[0, 4:]], 1.00)


def test_odd_position_count_has_neutral_middle() -> None:
    order = np.arange(5, dtype=np.int64)[None, :]
    result = mod.step_rank_multipliers(order, positions=5)
    assert np.allclose(result[0], [1.10, 1.10, 1.00, 0.90, 0.90])


def test_asymmetric_step_bounds_are_rejected() -> None:
    try:
        mod.step_rank_multipliers(
            np.arange(10, dtype=np.int64)[None, :],
            top_multiplier=1.10,
            bottom_multiplier=0.95,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("asymmetric gross-changing bounds were accepted")
