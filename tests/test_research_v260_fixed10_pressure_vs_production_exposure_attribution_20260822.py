from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_vs_production_exposure_attribution_20260822 import (
    compound_return,
    summarize_exposure_bins,
)


def test_compound_return_is_multiplicative() -> None:
    assert np.isclose(compound_return([0.10, -0.10]), -0.01)


def test_exposure_bins_are_mutually_exclusive() -> None:
    frame = pd.DataFrame(
        {
            "invested_ratio_production": [0.49, 0.50, 0.89, 0.90],
            "return_candidate": [0.0] * 4,
            "return_production": [0.0] * 4,
            "log_excess": [0.0] * 4,
        }
    )

    actual = summarize_exposure_bins(frame)

    assert actual["production_below_50pct"]["days"] == 1
    assert actual["production_50_to_90pct"]["days"] == 2
    assert actual["production_at_least_90pct"]["days"] == 1
