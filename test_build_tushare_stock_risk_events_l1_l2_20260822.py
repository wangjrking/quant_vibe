from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tushare_stock_risk_events_l1_l2_20260822 as module


class TushareStockRiskEventTests(unittest.TestCase):
    def test_signal_features_stop_at_build_end_date(self):
        events = pd.DataFrame(
            [
                {
                    "event_id": "stk_shock:x",
                    "stock_code": "000001.SZ",
                    "source_api": "stk_shock",
                    "available_signal_date": "20260821",
                    "source_end_date": None,
                }
            ]
        )

        result = module.build_signal_features(
            events,
            ["20260820", "20260821", "20260824", "20260825"],
            end_date="20260821",
        )

        self.assertEqual(result["signal_date"].tolist(), ["20260821"])

    def test_date_normalization_accepts_documented_forms(self):
        self.assertEqual(module.normalize_yyyymmdd("2026-08-21"), "20260821")
        self.assertEqual(module.normalize_yyyymmdd("20260821"), "20260821")

    def test_next_open_session_is_strictly_after_event_date(self):
        dates = ["20260820", "20260821", "20260824"]
        self.assertEqual(module.next_open_date("20260821", dates), "20260824")
        self.assertEqual(module.next_open_date("20260822", dates), "20260824")

    def test_raw_normalization_deduplicates_exact_source_rows(self):
        spec = module.API_SPECS[0]
        source = pd.DataFrame(
            [["000001.SZ", "20260821", "A", "深交所", "x", "p"]] * 2,
            columns=spec.required_columns,
        )
        result = module.normalize_raw(spec, [source], "2026-08-22T00:00:00+08:00")
        self.assertEqual(len(result), 1)

    def test_l2_maps_three_apis_and_excludes_bj(self):
        shock = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260821",
                    "name": "A",
                    "trade_market": "深交所",
                    "reason": "x",
                    "period": "p",
                    "source_row_sha256": "1",
                },
                {
                    "ts_code": "000001.BJ",
                    "trade_date": "20260821",
                    "name": "B",
                    "trade_market": "北交所",
                    "reason": "x",
                    "period": "p",
                    "source_row_sha256": "2",
                },
            ]
        )
        high = pd.DataFrame(
            [{"ts_code": "600000.SH", "trade_date": "20260821", "name": "C", "trade_market": "上交所", "reason": "y", "period": "q", "source_row_sha256": "3"}]
        )
        alert = pd.DataFrame(
            [{"ts_code": "300001.SZ", "name": "D", "start_date": "20260821", "end_date": "20260828", "type": "重点", "source_row_sha256": "4"}]
        )
        result = module.build_l2_events(
            {"stk_shock": shock, "stk_high_shock": high, "stk_alert": alert},
            ["20260821", "20260824", "20260825", "20260826", "20260827", "20260828"],
        )
        self.assertEqual(set(result["source_api"]), {"stk_shock", "stk_high_shock", "stk_alert"})
        self.assertFalse(result["stock_code"].str.endswith(".BJ").any())
        self.assertTrue((result["available_signal_date"] == "20260824").all())

    def test_alert_active_uses_official_end_date(self):
        events = pd.DataFrame(
            [
                {
                    "event_id": "stk_alert:x",
                    "stock_code": "000001.SZ",
                    "source_api": "stk_alert",
                    "available_signal_date": "20260824",
                    "source_end_date": "20260825",
                }
            ]
        )
        result = module.build_signal_features(
            events, ["20260821", "20260824", "20260825", "20260826", "20260827"]
        )
        active = result.set_index("signal_date")["alert_active_count"].to_dict()
        self.assertEqual(active["20260824"], 1)
        self.assertEqual(active["20260825"], 1)
        self.assertEqual(active["20260826"], 0)

    def test_shock_windows_use_trading_sessions(self):
        events = pd.DataFrame(
            [
                {
                    "event_id": "stk_shock:x",
                    "stock_code": "000001.SZ",
                    "source_api": "stk_shock",
                    "available_signal_date": "20260824",
                    "source_end_date": None,
                }
            ]
        )
        result = module.build_signal_features(
            events, ["20260821", "20260824", "20260825", "20260826", "20260827"]
        ).set_index("signal_date")
        self.assertEqual(int(result.loc["20260824", "shock_count_1d"]), 1)
        self.assertEqual(int(result.loc["20260826", "shock_count_3d"]), 1)
        self.assertEqual(int(result.loc["20260827", "shock_count_3d"]), 0)


if __name__ == "__main__":
    unittest.main()
