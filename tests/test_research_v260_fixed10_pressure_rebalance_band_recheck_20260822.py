from quant.main.research_v260_fixed10_pressure_rebalance_band_recheck_20260822 import (
    REBALANCE_BANDS,
    practical_band_gates,
)


def test_rebalance_band_budget_is_coarse():
    assert REBALANCE_BANDS == (0.005, 0.010, 0.020)


def test_tiny_gain_with_worse_full_risk_is_not_practical():
    baseline = {
        "train_2022_2024": {"sharpe": 0.95, "cagr": 0.34},
        "metrics_0_30pct": {"sharpe": 1.33, "max_drawdown": 0.37},
    }
    candidate = {
        "train_2022_2024": {"sharpe": 0.951, "cagr": 0.343},
        "metrics_0_30pct": {"sharpe": 1.32, "max_drawdown": 0.38},
    }
    assert not all(practical_band_gates(candidate, baseline).values())
