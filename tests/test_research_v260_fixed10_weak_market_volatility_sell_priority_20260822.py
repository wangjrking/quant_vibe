from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as subject
from research_v260_runtime import fixed10_risk_event_v110 as runtime


def test_default_score_sell_order_matches_score_then_code() -> None:
    actual = runtime.order_score_sells(
        [0, 1, 2],
        np.array([0.70, 0.60, 0.60]),
        np.array(["B", "C", "A"]),
    )
    assert actual == [2, 1, 0]


def test_explicit_priority_changes_only_order() -> None:
    actual = runtime.order_score_sells(
        [0, 1, 2],
        np.array([0.50, 0.40, 0.30]),
        np.array(["A", "B", "C"]),
        np.array([0.10, 0.20, 0.30]),
    )
    assert actual == [0, 1, 2]


def test_weak_market_priority_leaves_strong_rows_unchanged() -> None:
    score = np.array([[0.8, 0.7], [0.8, 0.7]])
    vol = np.array([[1.0, 0.0], [1.0, 0.0]])
    actual = subject.sell_priority_matrix(score, vol, np.array([True, False]))
    np.testing.assert_allclose(actual[0], score[0])
    np.testing.assert_allclose(actual[1], np.array([0.7, 0.7]))


def test_priority_shape_mismatch_fails_closed() -> None:
    try:
        subject.sell_priority_matrix(
            np.zeros((2, 2)), np.zeros((1, 2)), np.array([True, False])
        )
    except ValueError:
        return
    raise AssertionError("shape mismatch was accepted")
