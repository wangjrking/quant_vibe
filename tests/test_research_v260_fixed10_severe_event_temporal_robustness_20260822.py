import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_severe_event_temporal_robustness_20260822 import (
    aligned_log_return_attribution,
    blocked_baseline_buys,
)


def test_log_return_attribution_is_exact_and_concentration_is_bounded():
    baseline = pd.DataFrame({"date": ["20240102", "20240103"], "return": [0.01, -0.02]})
    candidate = pd.DataFrame({"date": ["20240102", "20240103"], "return": [0.02, -0.01]})
    _, audit = aligned_log_return_attribution(baseline, candidate)
    expected_ratio = (1.02 * 0.99) / (1.01 * 0.98)
    assert audit["terminal_wealth_ratio_candidate_over_baseline"] == pytest.approx(expected_ratio)
    assert audit["log_identity_error"] <= 1e-12
    assert 0.0 <= audit["top1_positive_day_share"] <= 1.0


def test_daily_date_mismatch_fails_closed():
    baseline = pd.DataFrame({"date": ["20240102"], "return": [0.0]})
    candidate = pd.DataFrame({"date": ["20240103"], "return": [0.0]})
    with pytest.raises(ValueError, match="do not align"):
        aligned_log_return_attribution(baseline, candidate)


def test_blocked_baseline_buys_uses_exact_signal_date_and_stock_key():
    actions = pd.DataFrame(
        [
            {"signal_date": "20240102", "buy_date": "20240103", "action": "BUY", "stock_code": "000001.SZ"},
            {"signal_date": "20240102", "buy_date": "20240103", "action": "BUY", "stock_code": "000002.SZ"},
            {"signal_date": "20240102", "buy_date": "20240103", "action": "SELL", "stock_code": "000001.SZ"},
        ]
    )
    block = np.array([[True, False]], dtype=np.bool_)
    result = blocked_baseline_buys(
        actions,
        np.array(["20240102"]),
        np.array(["000001.SZ", "000002.SZ"]),
        block,
    )
    assert result[["action", "stock_code"]].to_dict("records") == [
        {"action": "BUY", "stock_code": "000001.SZ"}
    ]


def test_block_matrix_shape_mismatch_fails_closed():
    actions = pd.DataFrame(columns=["signal_date", "buy_date", "action", "stock_code"])
    with pytest.raises(ValueError, match="does not align"):
        blocked_baseline_buys(
            actions,
            np.array(["20240102"]),
            np.array(["000001.SZ"]),
            np.zeros((2, 1), dtype=np.bool_),
        )
