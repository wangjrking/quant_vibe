from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from research_v260_fixed10_risk_event_score_tiebreak_20260822 import (
    score_tiebreak_order,
)


def test_empty_events_preserve_frozen_order_exactly() -> None:
    score = np.asarray([[0.91, 0.90, 0.80]])
    order = np.asarray([[0, 1, 2]])
    actual = score_tiebreak_order(
        score, order, np.zeros(score.shape, dtype=np.bool_), 0.05
    )
    assert np.array_equal(actual, order)


def test_event_stock_is_demoted_only_inside_existing_replacement_margin() -> None:
    score = np.asarray([[0.95, 0.91, 0.89]])
    order = np.asarray([[0, 1, 2]])
    events = np.asarray([[True, False, False]])
    actual = score_tiebreak_order(score, order, events, 0.05)
    assert actual.tolist() == [[1, 0, 2]]


def test_clearly_stronger_event_stock_remains_ahead() -> None:
    score = np.asarray([[0.95, 0.89, 0.88]])
    order = np.asarray([[0, 1, 2]])
    events = np.asarray([[True, False, False]])
    actual = score_tiebreak_order(score, order, events, 0.05)
    assert actual.tolist() == [[0, 1, 2]]


def test_equal_adjusted_score_deterministically_prefers_clean_stock() -> None:
    score = np.asarray([[0.95, 0.90, 0.80]])
    order = np.asarray([[0, 1, 2]])
    events = np.asarray([[True, False, False]])
    actual = score_tiebreak_order(score, order, events, 0.05)
    assert actual.tolist() == [[1, 0, 2]]


def test_invalid_shape_or_margin_fails_closed() -> None:
    score = np.asarray([[0.95, 0.90]])
    order = np.asarray([[0, 1]])
    with pytest.raises(ValueError, match="align"):
        score_tiebreak_order(score, order, np.zeros((2, 1), dtype=np.bool_), 0.05)
    with pytest.raises(ValueError, match="non-negative"):
        score_tiebreak_order(score, order, np.zeros_like(score, dtype=np.bool_), -0.1)
