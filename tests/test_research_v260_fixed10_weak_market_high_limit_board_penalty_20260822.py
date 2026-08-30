from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_high_limit_board_penalty_20260822 as module


def test_board_penalty_only_changes_weak_market_order() -> None:
    score = np.array([[0.90, 0.88], [0.90, 0.88]])
    order = np.array([[0, 1], [0, 1]])
    result = module.weak_market_board_penalty_order(
        score,
        order,
        np.array([True, False]),
        np.array([False, True]),
        0.05,
    )
    assert np.array_equal(result[0], [0, 1])
    assert np.array_equal(result[1], [1, 0])


def test_zero_penalty_preserves_order() -> None:
    score = np.array([[0.9, 0.8]])
    order = np.array([[0, 1]])
    result = module.weak_market_board_penalty_order(
        score, order, np.array([True, False]), np.array([True]), 0.0
    )
    assert np.array_equal(result, order)
