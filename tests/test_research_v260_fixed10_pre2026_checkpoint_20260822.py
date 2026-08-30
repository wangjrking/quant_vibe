from __future__ import annotations

import pytest

from quant.main.research_v260_fixed10_pre2026_checkpoint_20260822 import (
    beats_production,
    key_deltas,
)


def test_beats_production_requires_return_risk_dominance() -> None:
    production = {"cagr": 0.5, "sharpe": 1.5, "max_drawdown": 0.2}
    assert beats_production(
        {"cagr": 0.6, "sharpe": 1.6, "max_drawdown": 0.19}, production
    )
    assert not beats_production(
        {"cagr": 0.6, "sharpe": 1.4, "max_drawdown": 0.19}, production
    )


def test_key_deltas_preserve_drawdown_direction() -> None:
    keys = {
        "cumulative_return": 2.0,
        "cagr": 0.4,
        "sharpe": 1.0,
        "max_drawdown": 0.3,
        "turnover_annualized": 10.0,
        "average_invested_ratio": 0.9,
        "average_positions": 10.0,
    }
    lower = {**keys, "max_drawdown": 0.2}
    assert key_deltas(lower, keys)["max_drawdown"] == pytest.approx(-0.1)
