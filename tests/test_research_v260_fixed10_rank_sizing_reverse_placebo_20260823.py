from __future__ import annotations

import numpy as np
import pytest

import research_v260_fixed10_rank_sizing_reverse_placebo_20260823 as module


def test_reverse_schedule_preserves_reference_gross() -> None:
    order = np.array([[0, 1, 2, 3]], dtype=np.int64)
    multipliers = module.linear_rank_multipliers(order, 4, 0.9, 1.1)
    assert np.allclose(multipliers[0], [0.9, 0.9666666666666667, 1.0333333333333334, 1.1])
    assert multipliers.sum() == pytest.approx(4.0)


def test_forward_schedule_places_more_on_first_rank() -> None:
    order = np.array([[2, 0, 1]], dtype=np.int64)
    multipliers = module.linear_rank_multipliers(order, 3, 1.1, 0.9)
    assert multipliers[0, 2] == pytest.approx(1.1)
    assert multipliers[0, 1] == pytest.approx(0.9)


def test_placebo_rejects_nonpositive_multiplier() -> None:
    with pytest.raises(ValueError):
        module.linear_rank_multipliers(
            np.array([[0, 1]], dtype=np.int64), 2, 0.0, 2.0
        )


def test_placebo_requires_exact_reference_gross() -> None:
    with pytest.raises(ValueError):
        module.linear_rank_multipliers(
            np.array([[0, 1, 2]], dtype=np.int64), 3, 0.8, 1.0
        )
