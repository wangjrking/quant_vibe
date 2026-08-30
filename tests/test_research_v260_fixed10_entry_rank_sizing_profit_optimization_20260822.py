from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as mod


def test_entry_rank_multipliers_are_ordered_and_gross_neutral() -> None:
    order = np.array([[2, 0, 3, 1]], dtype=np.int64)
    weights = mod.entry_rank_multipliers(order, positions=4)
    ranked = weights[0, order[0]]
    assert np.all(np.diff(ranked) < 0.0)
    assert np.isclose(ranked.sum(), 4.0)
    assert np.isclose(ranked[0], mod.TOP_WEIGHT_MULTIPLIER)
    assert np.isclose(ranked[-1], mod.BOTTOM_WEIGHT_MULTIPLIER)


def test_non_top_names_keep_unit_multiplier() -> None:
    order = np.array([[4, 3, 2, 1, 0]], dtype=np.int64)
    weights = mod.entry_rank_multipliers(order, positions=3)
    assert np.allclose(weights[0, [1, 0]], 1.0)


def test_invalid_rank_shape_is_rejected() -> None:
    try:
        mod.entry_rank_multipliers(np.array([0, 1, 2]), positions=2)
    except ValueError:
        pass
    else:
        raise AssertionError("one-dimensional order was accepted")


def test_rank_multiplier_neighbors_remain_gross_neutral() -> None:
    order = np.arange(10, dtype=np.int64)[None, :]
    for top, bottom in ((1.10, 0.90), (1.20, 0.80), (1.30, 0.70)):
        weights = mod.entry_rank_multipliers(
            order,
            positions=10,
            top_multiplier=top,
            bottom_multiplier=bottom,
        )
        assert np.isclose(weights[0].sum(), 10.0)


def test_invalid_rank_multiplier_bounds_are_rejected() -> None:
    try:
        mod.entry_rank_multipliers(
            np.arange(10, dtype=np.int64)[None, :],
            top_multiplier=0.8,
            bottom_multiplier=1.2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ascending multiplier bounds were accepted")


def test_action_keys_include_signal_execution_action_and_stock() -> None:
    actions = pd.DataFrame(
        [
            {
                "signal_date": "20240102",
                "buy_date": "20240103",
                "action": "BUY",
                "stock_code": "000001.SZ",
            }
        ]
    )
    assert mod.action_keys(actions) == {
        ("20240102", "20240103", "BUY", "000001.SZ")
    }


def test_realized_buy_target_diagnostics_discloses_refill_equalweights() -> None:
    actions = pd.DataFrame(
        [
            {"action": "BUY", "target_pct": 0.11},
            {"action": "BUY", "target_pct": 0.10},
            {"action": "BUY", "target_pct": 0.09},
            {"action": "SELL", "target_pct": 0.00},
        ]
    )
    result = mod.realized_buy_target_diagnostics(actions)
    assert result["buy_actions"] == 3
    assert result["equalweight_10pct_actions"] == 1
    assert result["non_equalweight_actions"] == 2


def test_relative_path_diagnostics_aligns_periods_and_offsets() -> None:
    dates = pd.Series(
        pd.bdate_range("2024-01-02", periods=260).strftime("%Y%m%d")
    )
    baseline = np.zeros(260, dtype=np.float64)
    tilted = np.full(260, 0.001, dtype=np.float64)
    result = mod.relative_path_diagnostics(dates, tilted, baseline)
    assert result["monthly"]["positive_fraction"] == 1.0
    assert result["quarterly"]["positive_fraction"] == 1.0
    assert result["rolling_252"]["windows"] == 9
    assert all(
        item["tilted_minus_equalweight"] > 0.0
        for item in result["start_offsets"].values()
    )


def test_relative_path_diagnostics_rejects_misalignment() -> None:
    try:
        mod.relative_path_diagnostics(
            pd.Series(["20240102"]),
            np.array([0.0, 0.0]),
            np.array([0.0]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned relative path inputs were accepted")


def test_invested_exposure_attribution_separates_close_exposure_days() -> None:
    result = mod.invested_exposure_attribution(
        np.array([0.01, 0.02, 0.03]),
        np.array([0.00, 0.01, 0.02]),
        np.array([0.98, 0.97, 0.99]),
        np.array([0.98, 0.96, 0.99]),
        close_exposure_tolerance=0.0025,
    )
    assert result["close_exposure_days"] == 2
    assert result["close_exposure_log_excess"] > 0.0


def test_invested_exposure_attribution_rejects_misalignment() -> None:
    try:
        mod.invested_exposure_attribution(
            np.array([0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0]),
            np.array([1.0]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned exposure inputs were accepted")
