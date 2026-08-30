from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_stock_abnormal_announcement_l1_l2_20260822 as module


class AnnouncementL1L2Tests(unittest.TestCase):
    def source_item(
        self,
        title: str,
        category_code: str = module.ABNORMAL_COLUMN_CODE,
        category_name: str = module.ABNORMAL_COLUMN_NAME,
    ):
        return {
            "art_code": "AN1",
            "display_time": "2024-01-05 19:12:00:910",
            "notice_date": "2024-01-05 00:00:00",
            "title": title,
            "columns": [{"column_code": category_code, "column_name": category_name}],
            "codes": [
                {"ann_type": "A", "stock_code": "000001", "short_name": "A"},
                {"ann_type": "A", "stock_code": "600000", "short_name": "B"},
            ],
        }

    def test_abnormal_source_row_explodes_all_a_share_codes(self):
        frame = module.parse_source_items([self.source_item("x")], "2024-01-06T00:00:00+08:00")
        self.assertEqual(frame["stock_code"].tolist(), ["000001.SZ", "600000.SH"])
        self.assertEqual(frame["announcement_id"].nunique(), 1)

    def test_non_abnormal_category_is_ignored(self):
        frame = module.parse_source_items([self.source_item("x", "other", "other")], "2024-01-06T00:00:00+08:00")
        self.assertTrue(frame.empty)

    def test_event_is_never_visible_on_announcement_day(self):
        self.assertEqual(module.next_open_date("20240105", ["20240105", "20240108"]), "20240108")
        self.assertEqual(module.next_open_date("20240106", ["20240105", "20240108"]), "20240108")

    def test_missing_publication_time_is_preserved_not_invented(self):
        item = self.source_item("x")
        item["display_time"] = ""
        frame = module.parse_source_items([item], "2024-01-06T00:00:00+08:00")
        self.assertTrue(frame["published_at"].isna().all())
        self.assertFalse(frame["publication_time_available"].any())

    def test_event_classification_is_deterministic(self):
        severe, risk = module.classify_event("\u80a1\u7968\u4ea4\u6613\u4e25\u91cd\u5f02\u5e38\u6ce2\u52a8\u66a8\u98ce\u9669\u63d0\u793a")
        self.assertEqual(severe, "severe_abnormal_volatility")
        self.assertTrue(risk)

    def test_sparse_windows_use_trading_sessions(self):
        events = pd.DataFrame(
            [
                {
                    "stock_code": "000001.SZ",
                    "announcement_id": "AN1",
                    "available_signal_date": "20240108",
                    "event_type": "abnormal_volatility",
                    "risk_warning_in_title": False,
                }
            ]
        )
        dates = ["20240105", "20240108", "20240109", "20240110", "20240111"]
        features = module.build_signal_features(events, dates)
        self.assertEqual(features["signal_date"].tolist(), dates[1:])
        first = features.iloc[0]
        fourth = features.iloc[3]
        self.assertEqual(int(first["abnormal_count_1d"]), 1)
        self.assertEqual(int(fourth["abnormal_count_3d"]), 0)
        self.assertEqual(int(fourth["abnormal_count_5d"]), 1)


if __name__ == "__main__":
    unittest.main()
