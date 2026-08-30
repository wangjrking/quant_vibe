from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_weak_market_defensive_rerank_20260822 as module


def test_trailing_volatility_is_scale_invariant_and_pit() -> None:
    close = np.asarray([[100.0], [101.0], [99.0], [102.0], [103.0], [100.0]])
    original = module.trailing_log_volatility(close, 3)
    scaled = module.trailing_log_volatility(close * 7.5, 3)
    np.testing.assert_allclose(original, scaled, equal_nan=True, atol=1e-15)

    poisoned = close.copy()
    poisoned[-1, 0] = 99999.0
    poisoned_result = module.trailing_log_volatility(poisoned, 3)
    np.testing.assert_allclose(
        original[:-1], poisoned_result[:-1], equal_nan=True, atol=0.0
    )


def test_defensive_order_changes_only_weak_day_pool() -> None:
    order = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]])
    volatility = np.asarray([[0.4, 0.1, 0.2, 0.3], [0.4, 0.1, 0.2, 0.3]])
    actual = module.defensive_order(order, volatility, np.asarray([False, True]), 3)
    np.testing.assert_array_equal(actual[0], order[0])
    np.testing.assert_array_equal(actual[1], np.asarray([1, 2, 0, 3]))


def test_bounded_score_order_changes_only_weak_day_pool() -> None:
    order = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]])
    score = np.asarray([[0.1, 0.4, 0.3, 0.9], [0.1, 0.4, 0.3, 0.9]])
    actual = module.bounded_score_order(order, score, np.asarray([False, True]), 3)
    np.testing.assert_array_equal(actual[0], order[0])
    np.testing.assert_array_equal(actual[1], np.asarray([1, 2, 0, 3]))


def test_candidate_budget_has_two_structural_alternatives() -> None:
    assert module.CANDIDATES == (
        "production_order",
        "weak_market_low_vol_top30",
        "weak_market_rank1d70_rank10d30_top30",
    )
    assert module.VOLATILITY_WINDOW == 20
    assert module.DEFENSIVE_POOL_SIZE == 30
