from __future__ import annotations

import numpy as np
import pytest

import research_v260_fixed10_low_vol_entry_sizing_profit_optimization_20260823 as module


def test_low_volatility_receives_larger_target() -> None:
    order = np.array([[0, 1, 2]], dtype=np.int64)
    volatility = np.array([[0.3, 0.1, 0.2]], dtype=np.float64)
    result, availability = module.low_volatility_multipliers(
        order, volatility, positions=3
    )
    assert availability["missing_top10_volatility_observations"] == 0
    assert result[0, 1] == pytest.approx(1.10)
    assert result[0, 2] == pytest.approx(1.00)
    assert result[0, 0] == pytest.approx(0.90)


def test_missing_volatility_keeps_only_missing_name_at_equalweight() -> None:
    order = np.array([[0, 1, 2]], dtype=np.int64)
    volatility = np.array([[0.3, np.nan, 0.2]], dtype=np.float64)
    result, availability = module.low_volatility_multipliers(
        order, volatility, positions=3
    )
    assert availability["dates_with_any_missing_top10_volatility"] == 1
    assert result[0, 1] == pytest.approx(1.0)
    assert result[0, 2] == pytest.approx(1.1)
    assert result[0, 0] == pytest.approx(0.9)
    assert result.sum() == pytest.approx(3.0)


def test_low_vol_selection_is_profit_first() -> None:
    arms = {
        "equalweight": {"cumulative_return": 1.0},
        "global_score_rank_sizing": {"cumulative_return": 1.1},
        "low_volatility_rank_sizing": {"cumulative_return": 1.2},
    }
    assert module.select_by_pre2026_net_return(arms) == "low_volatility_rank_sizing"
