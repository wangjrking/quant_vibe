import numpy as np
import pytest

from quant.main.research_v260_fixed10_exit_timing_robustness_tradeoff_20260822 import (
    consecutive_below_mask,
)


def test_consecutive_below_mask_uses_only_current_and_prior_rows() -> None:
    score = np.array([[0.9], [0.8], [0.7], [0.9], [0.6]])
    assert consecutive_below_mask(score, 0.85, 1).ravel().tolist() == [
        False,
        True,
        True,
        False,
        True,
    ]
    assert consecutive_below_mask(score, 0.85, 2).ravel().tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]


def test_nonfinite_score_never_confirms() -> None:
    score = np.array([[0.8], [np.nan], [0.7]])
    assert not consecutive_below_mask(score, 0.85, 2).any()


def test_invalid_shape_or_days_is_rejected() -> None:
    with pytest.raises(ValueError):
        consecutive_below_mask(np.array([0.8]), 0.85, 2)
    with pytest.raises(ValueError):
        consecutive_below_mask(np.ones((2, 2)), 0.85, 0)
