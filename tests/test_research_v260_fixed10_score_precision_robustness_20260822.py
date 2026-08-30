import numpy as np
import pytest

from quant.main.research_v260_fixed10_score_precision_robustness_20260822 import (
    stable_descending_order,
    topn_overlap,
)


def test_stable_order_sorts_score_then_stock_code() -> None:
    score = np.array([[0.8, 0.9, 0.9, np.nan]])
    stocks = np.array(["000003.SZ", "000002.SZ", "000001.SZ", "000004.SZ"])
    assert stable_descending_order(score, stocks).tolist() == [[2, 1, 0, 3]]


def test_topn_overlap_ignores_order_inside_membership() -> None:
    left = np.array([[0, 1, 2]])
    right = np.array([[1, 0, 2]])
    assert topn_overlap(left, right, 2) == 1.0


def test_invalid_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        stable_descending_order(np.array([0.1, 0.2]), np.array(["a", "b"]))
    with pytest.raises(ValueError):
        topn_overlap(np.ones((1, 2)), np.ones((2, 1)), 1)
