from __future__ import annotations

import numpy as np
import pytest

import research_v260_fixed10_vol_priority_binary_simplification_20260822 as target


def test_binary_priority_changes_only_weak_high_volatility_cells():
    score = np.array([[0.90, 0.80], [0.70, 0.60]], dtype=float)
    volatility_rank = np.array([[0.80, 0.20], [0.90, 0.10]], dtype=float)
    weak = np.array([True, False])
    result = target.binary_priority_matrix(score, volatility_rank, weak, 0.75, 0.05)
    np.testing.assert_allclose(result, [[0.85, 0.80], [0.70, 0.60]])


def test_binary_priority_missing_volatility_is_not_penalized():
    score = np.array([[0.90, 0.80]], dtype=float)
    rank = np.array([[np.nan, 0.80]], dtype=float)
    result = target.binary_priority_matrix(score, rank, np.array([True]), 0.75)
    np.testing.assert_allclose(result, [[0.90, 0.75]])


def test_binary_priority_rejects_shape_and_threshold_errors():
    score = np.ones((2, 2))
    with pytest.raises(ValueError, match="do not align"):
        target.binary_priority_matrix(score, score, np.array([True]), 0.50)
    with pytest.raises(ValueError, match="between zero and one"):
        target.binary_priority_matrix(score, score, np.array([True, True]), 1.10)
