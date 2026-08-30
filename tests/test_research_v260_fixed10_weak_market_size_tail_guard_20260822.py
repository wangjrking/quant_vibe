from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_size_tail_guard_20260822 as module


def test_size_guard_changes_only_weak_market_rows() -> None:
    production = np.ones((2, 10), dtype=np.bool_)
    size = np.tile(np.arange(1.0, 11.0), (2, 1))
    clean = np.ones_like(production)
    result = module.size_tail_selection_mask(
        production, size, clean, np.array([False, True]), 0.20
    )
    assert result[0].all()
    assert not result[1, 0]
    assert not result[1, 1]
    assert result[1, 2:].all()


def test_zero_percentile_is_exact_baseline() -> None:
    production = np.array([[True, False], [False, True]])
    size = np.ones((2, 2))
    clean = np.ones_like(production)
    result = module.size_tail_selection_mask(
        production, size, clean, np.array([True, True]), 0.0
    )
    assert np.array_equal(result, production)


def test_size_guard_never_adds_ineligible_stock() -> None:
    production = np.array([[False] + [True] * 10])
    size = np.arange(1.0, 12.0)[None, :]
    clean = np.ones_like(production)
    result = module.size_tail_selection_mask(
        production, size, clean, np.array([True]), 0.20
    )
    assert not result[0, 0]
    assert np.all(result <= production)


def test_small_warmup_universe_preserves_production_mask() -> None:
    production = np.ones((1, 5), dtype=np.bool_)
    size = np.arange(1.0, 6.0)[None, :]
    clean = np.ones_like(production)
    result = module.size_tail_selection_mask(
        production, size, clean, np.array([True]), 0.20
    )
    assert np.array_equal(result, production)
