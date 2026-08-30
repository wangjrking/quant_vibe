import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

import research_v260_fixed10_abnormal_announcement_proxy_20260822 as module


def sample_features():
    return pd.DataFrame(
        [
            {
                "signal_date": "20240102",
                "stock_code": "000001.SZ",
                "abnormal_count_1d": 1,
                "severe_count_5d": 0,
                "risk_warning_count_5d": 0,
            },
            {
                "signal_date": "20240103",
                "stock_code": "600000.SH",
                "abnormal_count_1d": 0,
                "severe_count_5d": 1,
                "risk_warning_count_5d": 1,
            },
        ]
    )


def test_combined_mask_uses_union_and_exact_keys():
    mask, audit = module.announcement_block_matrix(
        np.array(["20240102", "20240103"]),
        np.array(["000001.SZ", "600000.SH"]),
        sample_features(),
        ordinary_window=1,
        severe_window=5,
        warning_window=5,
    )
    assert mask.tolist() == [[True, False], [False, True]]
    assert audit["blocked_stock_dates"] == 2
    assert audit["component_stock_dates"] == {
        "abnormal": 1,
        "severe": 1,
        "risk_warning": 1,
    }


def test_disabled_rule_is_empty():
    mask, audit = module.announcement_block_matrix(
        np.array(["20240102"]),
        np.array(["000001.SZ"]),
        sample_features(),
        ordinary_window=0,
        severe_window=0,
        warning_window=0,
    )
    assert not mask.any()
    assert audit["blocked_stock_dates"] == 0


def test_duplicate_keys_fail_closed():
    frame = pd.concat([sample_features(), sample_features().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        module.announcement_block_matrix(
            np.array(["20240102", "20240103"]),
            np.array(["000001.SZ", "600000.SH"]),
            frame,
            ordinary_window=1,
            severe_window=5,
            warning_window=5,
        )


def test_unsupported_window_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        module.announcement_block_matrix(
            np.array(["20240102"]),
            np.array(["000001.SZ"]),
            sample_features(),
            ordinary_window=2,
            severe_window=0,
            warning_window=0,
        )
