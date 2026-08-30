from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rebalance_40d_refinement_20260822 as module


def test_candidate_intervals_are_coarse_and_predeclared() -> None:
    assert module.CANDIDATE_IDS == (
        "portfolio_rebalance_20d",
        "portfolio_rebalance_40d",
        "portfolio_rebalance_20d_strong_market_only",
        "portfolio_rebalance_5d_strong_market_only",
        "portfolio_topup_5d_never_trim",
        "portfolio_topup_5d_never_trim_score085",
    )
    assert module.candidate_id(20) == "portfolio_rebalance_20d"
    assert module.candidate_id(40) == "portfolio_rebalance_40d"
    assert module.MIN_CAGR_IMPROVEMENT == 0.005
    assert module.MIN_SHARPE_IMPROVEMENT == 0.01
    assert module.MIN_DRAWDOWN_IMPROVEMENT == 0.005


def test_strong_market_schedule_only_removes_rebalance_dates(monkeypatch) -> None:
    base = np.asarray([True, False, True, False])
    strong = np.asarray([False, True, True, False])
    monkeypatch.setattr(module.cadence, "rebalance_schedule", lambda length, interval: base)
    monkeypatch.setattr(module.regime, "strong_market_mask", lambda score, protocol: strong)
    actual = module.candidate_schedule(
        "portfolio_rebalance_20d_strong_market_only", np.zeros((4, 2)), {}
    )
    np.testing.assert_array_equal(actual, np.asarray([False, False, True, False]))
    actual_5d = module.candidate_schedule(
        "portfolio_rebalance_5d_strong_market_only", np.zeros((4, 2)), {}
    )
    np.testing.assert_array_equal(actual_5d, np.asarray([False, False, True, False]))
