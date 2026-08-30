from __future__ import annotations

import pandas as pd

from research_v260_rolling252_top10_ltr_reranker_10d_v1_20260817 import admitted_dates


def test_admitted_dates_require_mature_label_and_precede_prediction() -> None:
    source = pd.DataFrame({"trade_date": ["20220104", "20220105", "20220106", "20220107"]})
    end_map = {
        "20220104": "20220120",
        "20220105": "20220121",
        "20220106": "20220201",
        "20220107": "20220119",
    }
    assert admitted_dates(source, end_map, "20220122") == ["20220104", "20220105", "20220107"]


def test_admitted_dates_are_stably_sorted() -> None:
    source = pd.DataFrame({"trade_date": ["20220107", "20220104", "20220105"]})
    end_map = {date: "20220110" for date in source["trade_date"]}
    assert admitted_dates(source, end_map, "20220201") == ["20220104", "20220105", "20220107"]
