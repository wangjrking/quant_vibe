import numpy as np
import pytest

from quant.main.research_v260_fixed10_decision_staleness_robustness_20260822 import (
    lag_rows,
)


def test_lag_rows_preserves_shape_and_uses_prior_sessions() -> None:
    values = np.array([[1, 2], [3, 4], [5, 6]])
    assert lag_rows(values, 1).tolist() == [[1, 2], [1, 2], [3, 4]]
    assert lag_rows(values, 2).tolist() == [[1, 2], [1, 2], [1, 2]]


def test_zero_lag_is_an_independent_equivalent_array() -> None:
    values = np.array([[1.0], [2.0]])
    result = lag_rows(values, 0)
    assert np.array_equal(result, values)
    assert result is not values


def test_invalid_lag_is_rejected() -> None:
    values = np.ones((2, 2))
    with pytest.raises(ValueError):
        lag_rows(values, -1)
    with pytest.raises(ValueError):
        lag_rows(values, 2)
