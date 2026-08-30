from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_beta_exposure_diagnostic_20260822 as module


def test_build_exposure_uses_signal_date_without_imputing_missing_beta() -> None:
    observations = [
        {
            "signal_date": "20240102",
            "buy_date": "20240103",
            "positions_after_trades": {"A": {}, "B": {}},
        }
    ]
    arrays = {
        "stocks": np.array(["A", "B"]),
        "dates": np.array(["20240102", "20240103"]),
    }
    beta = np.array([[1.0, np.nan], [9.0, 9.0]])
    row = module.build_exposure(observations, arrays, beta).iloc[0]
    assert row["mean_market_beta"] == 1.0
    assert row["observed_beta_positions"] == 1
    assert row["missing_beta_positions"] == 1
