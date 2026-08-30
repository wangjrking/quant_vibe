from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_style_exposure_diagnostic_20260822 as module


def test_percentile_for_indices_uses_average_rank_for_ties() -> None:
    values = np.array([10.0, 20.0, 20.0, 40.0])
    mask = np.ones(4, dtype=np.bool_)
    result = module.percentile_for_indices(values, mask, [0, 1, 3])
    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_percentile_for_indices_excludes_masked_reference_values() -> None:
    values = np.array([10.0, 20.0, 30.0, 40.0])
    mask = np.array([True, False, True, True])
    result = module.percentile_for_indices(values, mask, [0, 2, 3])
    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_percentile_for_indices_clips_value_outside_reference_range() -> None:
    values = np.array([5.0, 10.0, 20.0, 30.0, 40.0])
    mask = np.array([False, True, True, True, False])
    result = module.percentile_for_indices(values, mask, [0, 4])
    assert np.allclose(result, [0.0, 1.0])


def test_percentile_for_indices_rejects_missing_held_value() -> None:
    values = np.array([10.0, np.nan, 30.0])
    mask = np.ones(3, dtype=np.bool_)
    try:
        module.percentile_for_indices(values, mask, [1])
    except ValueError as exc:
        assert "missing style value" in str(exc)
    else:
        raise AssertionError("missing held style value was accepted")


def test_build_daily_exposure_records_missing_values_without_imputation() -> None:
    observations = [
        {
            "signal_date": "20220104",
            "buy_date": "20220105",
            "positions_after_trades": {"A": {}, "B": {}},
        }
    ]
    arrays = {
        "stocks": np.array(["A", "B", "C"]),
        "dates": np.array(["20220104", "20220105"]),
        "signal_clean": np.array([[True, True, True], [True, True, True]]),
        "total_mv": np.array([[10.0, np.nan, 30.0], [10.0, 20.0, 30.0]]),
        "turnover_rate": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "amount": np.array([[100.0, 200.0, 300.0], [100.0, 200.0, 300.0]]),
    }
    result = module.build_daily_exposure(observations, arrays).iloc[0]
    assert result["observed_total_mv_positions"] == 1
    assert result["missing_total_mv_positions"] == 1
    assert result["mean_total_mv_percentile"] == 0.0
