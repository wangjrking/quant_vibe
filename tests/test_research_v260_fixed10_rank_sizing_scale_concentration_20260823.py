from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_scale_concentration_20260823 as mod


def test_context_cash_rejects_nonpositive_value() -> None:
    class Context:
        protocol = {"execution": {"initial_cash": 1.0}}

    for value in (0.0, -1.0, float("nan")):
        try:
            mod.context_with_initial_cash(Context(), value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid initial cash was accepted")


def test_paired_excess_remains_positive_after_small_best_day_removal() -> None:
    dates = pd.Series(pd.bdate_range("2024-01-02", periods=30).strftime("%Y%m%d"))
    candidate = np.full(30, 0.002, dtype=np.float64)
    baseline = np.full(30, 0.001, dtype=np.float64)
    result = mod.paired_excess_concentration(dates, candidate, baseline)
    assert result["daily_removed_best"]["20"]["remaining_log_excess"] > 0.0


def test_paired_excess_rejects_misaligned_inputs() -> None:
    try:
        mod.paired_excess_concentration(
            pd.Series(["20240102"]),
            np.array([0.0, 0.0]),
            np.array([0.0]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned paired returns were accepted")
