from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_turnover_tail_guard_20260822 as module


def test_turnover_guard_changes_only_weak_market_rows() -> None:
    production = np.ones((2, 10), dtype=np.bool_)
    turnover = np.tile(np.arange(1.0, 11.0), (2, 1))
    clean = np.ones_like(production)
    result = module.turnover_tail_selection_mask(
        production, turnover, clean, np.array([False, True]), 0.20
    )
    assert result[0].all()
    assert result[1, :8].all()
    assert not result[1, 8]
    assert not result[1, 9]


def test_turnover_guard_never_adds_ineligible_stock() -> None:
    production = np.array([[False] + [True] * 10])
    turnover = np.arange(1.0, 12.0)[None, :]
    clean = np.ones_like(production)
    result = module.turnover_tail_selection_mask(
        production, turnover, clean, np.array([True]), 0.10
    )
    assert not result[0, 0]
    assert np.all(result <= production)


def test_zero_tail_is_exact_baseline() -> None:
    production = np.array([[True, False]])
    result = module.turnover_tail_selection_mask(
        production,
        np.array([[1.0, 2.0]]),
        np.ones_like(production),
        np.array([True]),
        0.0,
    )
    assert np.array_equal(result, production)
