from __future__ import annotations

import numpy as np
import pytest

from research_v260_fixed10_refill_haircut_profit_optimization_20260823 import (
    refill_haircut_multipliers,
)


def test_top10_keeps_rank_schedule_and_refill_uses_existing_floor() -> None:
    order = np.array([[2, 0, 3, 1, 4, 5]], dtype=np.int64)
    actual = refill_haircut_multipliers(order, positions=4)
    expected_schedule = np.linspace(1.10, 0.90, 4)
    assert actual[0, order[0, :4]].tolist() == pytest.approx(
        expected_schedule.tolist()
    )
    assert actual[0, order[0, 4:]].tolist() == pytest.approx([0.90, 0.90])
    assert actual[0, order[0, :4]].sum() == pytest.approx(4.0)


def test_rows_are_deterministic() -> None:
    order = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    first = refill_haircut_multipliers(order, positions=2)
    second = refill_haircut_multipliers(order, positions=2)
    assert np.array_equal(first, second)


def test_invalid_shapes_and_bounds_fail() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        refill_haircut_multipliers(np.arange(4), positions=2)
    order = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="between two"):
        refill_haircut_multipliers(order, positions=1)
    with pytest.raises(ValueError, match="positive"):
        refill_haircut_multipliers(order, positions=2, bottom_multiplier=0.0)
