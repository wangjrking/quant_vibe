from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_rank_sizing_market_state_attribution_20260823 as mod


def test_state_summary_uses_paired_compounding() -> None:
    frame = pd.DataFrame(
        {
            "candidate_return": [0.02, -0.01],
            "baseline_return": [0.01, -0.01],
        }
    )
    result = mod.state_summary(frame)
    expected = np.log1p(0.02) - np.log1p(0.01)
    assert abs(result["candidate_minus_equalweight_log_excess"] - expected) < 1e-12


def test_state_summary_reports_day_fraction() -> None:
    frame = pd.DataFrame(
        {
            "candidate_return": [0.02, 0.00],
            "baseline_return": [0.01, 0.01],
        }
    )
    assert mod.state_summary(frame)["candidate_outperforms_day_fraction"] == 0.5
