from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research_v260_fixed10_residual_cash_sweep_robustness_20260823 as module


def test_paired_log_diagnostics_reports_years_and_rolling_windows() -> None:
    dates = pd.date_range("2022-01-01", periods=260, freq="D").strftime("%Y%m%d")
    frame = pd.DataFrame(
        {
            "date": dates,
            "current": np.zeros(260),
            "sweep": np.full(260, 0.001),
        }
    )
    actual = module.paired_log_diagnostics(frame)
    assert actual["total_relative_compound_excess"] > 0.0
    assert actual["rolling_log_excess"]["252"]["positive_fraction"] == 1.0
    assert actual["positive_year_count"] == 1


def test_paired_log_diagnostics_rejects_duplicate_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": ["20220101", "20220101"],
            "current": [0.0, 0.0],
            "sweep": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="unique dates"):
        module.paired_log_diagnostics(frame)


def test_paired_log_diagnostics_rejects_nonfinite_returns() -> None:
    frame = pd.DataFrame(
        {
            "date": ["20220101"],
            "current": [0.0],
            "sweep": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="invalid"):
        module.paired_log_diagnostics(frame)
