from __future__ import annotations

import numpy as np
import pytest

import research_v260_fixed10_hold_optimization_20260822 as module


def test_scalar_sell_threshold_becomes_daily_schedule() -> None:
    result = module.normalize_daily_float_override(
        0.85, 3, "sell_score_below_override"
    )
    assert result.dtype == np.float64
    assert result.tolist() == [0.85, 0.85, 0.85]


def test_daily_sell_threshold_is_preserved() -> None:
    result = module.normalize_daily_float_override(
        np.array([0.80, 0.85, 0.90]), 3, "sell_score_below_override"
    )
    assert result.tolist() == [0.80, 0.85, 0.90]


def test_sell_threshold_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="must match the trading calendar"):
        module.normalize_daily_float_override(
            np.array([0.80, 0.85]), 3, "sell_score_below_override"
        )
