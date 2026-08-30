from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_size_tilt_20260822 as module


def test_size_percentile_is_monotonic() -> None:
    values = np.array([[30.0, 10.0, 20.0]])
    result = module.size_percentile_matrix(values, np.ones_like(values, dtype=bool))
    assert np.allclose(result, [[1.0, 0.0, 0.5]])


def test_size_tilt_only_changes_weak_market_order() -> None:
    score = np.array([[0.90, 0.89], [0.90, 0.89]])
    order = np.array([[0, 1], [0, 1]])
    size = np.array([[0.0, 1.0], [0.0, 1.0]])
    result = module.weak_market_size_tilt_order(
        score, order, size, np.array([False, True]), 0.05
    )
    assert np.array_equal(result[0], [0, 1])
    assert np.array_equal(result[1], [1, 0])


def test_zero_penalty_is_byte_equivalent_order() -> None:
    score = np.array([[0.9, 0.8]])
    order = np.array([[0, 1]])
    size = np.array([[0.0, 1.0]])
    result = module.weak_market_size_tilt_order(
        score, order, size, np.array([True]), 0.0
    )
    assert np.array_equal(result, order)
