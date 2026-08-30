from __future__ import annotations

import numpy as np
import pytest

from research_v260_fixed10_score_distance_entry_sizing_20260823 import (
    score_distance_multipliers,
)


def test_score_distance_weights_follow_actual_score_gaps_and_preserve_gross() -> None:
    score = np.array([[5.0, 4.0, 3.0, 2.0]], dtype=np.float64)
    order = np.array([[0, 1, 2, 3]], dtype=np.int64)
    actual, diagnostics = score_distance_multipliers(score, order, positions=4)
    assert np.all(np.diff(actual[0]) < 0.0)
    assert actual[0].sum() == pytest.approx(4.0)
    assert actual[0].min() >= 0.90
    assert actual[0].max() <= 1.10
    assert diagnostics["active_days"] == 1


def test_equal_scores_fall_back_to_equalweight_without_failure() -> None:
    score = np.ones((1, 4), dtype=np.float64)
    order = np.array([[3, 2, 1, 0]], dtype=np.int64)
    actual, diagnostics = score_distance_multipliers(score, order, positions=4)
    assert np.array_equal(actual, np.ones_like(actual))
    assert diagnostics["tied_or_invalid_days"] == 1


def test_nonfinite_selected_score_falls_back_to_equalweight() -> None:
    score = np.array([[4.0, 3.0, np.nan, 1.0]], dtype=np.float64)
    order = np.array([[0, 1, 2, 3]], dtype=np.int64)
    actual, diagnostics = score_distance_multipliers(score, order, positions=4)
    assert np.array_equal(actual, np.ones_like(actual))
    assert diagnostics["tied_or_invalid_days"] == 1


def test_invalid_shapes_and_tilt_are_rejected() -> None:
    score = np.ones((1, 4), dtype=np.float64)
    order = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError, match="aligned"):
        score_distance_multipliers(score, order, positions=4)
    with pytest.raises(ValueError, match="maximum tilt"):
        score_distance_multipliers(score, np.array([order]), positions=4, maximum_tilt=1.0)
