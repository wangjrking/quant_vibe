from __future__ import annotations

import pandas as pd
import pytest

from quant.main.research_v260_fixed10_drawdown_attribution_20260822 import (
    maximum_drawdown_episode,
    relative_window,
    worst_compound_window,
)


def sample_daily(returns: list[float]) -> pd.DataFrame:
    equity = pd.Series(returns).add(1.0).cumprod()
    return pd.DataFrame(
        {
            "date": [f"2022010{index + 1}" for index in range(len(returns))],
            "return": returns,
            "equity": equity,
        }
    )


def test_maximum_drawdown_episode_has_exact_peak_and_trough() -> None:
    actual = maximum_drawdown_episode(sample_daily([0.10, 0.10, -0.20, -0.10, 0.05]))
    assert actual["peak_date"] == "20220102"
    assert actual["trough_date"] == "20220104"
    assert actual["drawdown"] == pytest.approx(0.28)


def test_worst_compound_window_uses_compounding() -> None:
    actual = worst_compound_window(sample_daily([0.10, -0.10, -0.20]), 2)
    assert actual["start_date"] == "20220102"
    assert actual["end_date"] == "20220103"
    assert actual["return"] == pytest.approx(-0.28)


def test_relative_window_requires_same_dates() -> None:
    candidate = sample_daily([0.01, 0.02])
    production = sample_daily([0.00, 0.00])
    actual = relative_window(candidate, production, 2)
    assert actual["candidate_minus_production_return"] == pytest.approx(0.0302)
    production.loc[1, "date"] = "20220109"
    with pytest.raises(RuntimeError, match="dates differ"):
        relative_window(candidate, production, 2)
