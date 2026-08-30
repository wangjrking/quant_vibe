from __future__ import annotations

import numpy as np
import pytest

from quant.main.research_v260_fixed10_current_weak_market_vol_size_sell_priority_20260822 import (
    SIZE_PENALTIES,
    vol_size_priority_matrix,
)


def test_candidate_neighborhood_is_fixed() -> None:
    assert SIZE_PENALTIES == (0.0, 0.0125, 0.025, 0.05)


def test_size_term_only_changes_confirmed_weak_rows() -> None:
    current = np.array([[0.7, 0.8], [0.7, 0.8]])
    size = np.array([[0.1, 0.9], [0.1, 0.9]])
    result = vol_size_priority_matrix(current, size, np.array([False, True]), 0.025)
    np.testing.assert_array_equal(result[0], current[0])
    np.testing.assert_allclose(result[1], np.array([0.7025, 0.8225]))
    assert result[1, 0] < result[1, 1]


def test_missing_size_is_neutral_and_shape_mismatch_fails() -> None:
    result = vol_size_priority_matrix(
        np.array([[0.7]]), np.array([[np.nan]]), np.array([True]), 0.02
    )
    np.testing.assert_allclose(result, np.array([[0.71]]))
    with pytest.raises(ValueError, match="do not align"):
        vol_size_priority_matrix(
            np.zeros((2, 2)), np.zeros((2, 1)), np.array([True, False]), 0.02
        )
