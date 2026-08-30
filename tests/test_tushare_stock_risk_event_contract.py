from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import tushare_stock_risk_event_contract as contract


class TushareStockRiskEventContractTests(unittest.TestCase):
    def test_normalize_empty_frame_keeps_formal_schema(self) -> None:
        frame = contract.normalize_source_frame("stk_alert", pd.DataFrame())

        self.assertEqual(
            list(frame.columns),
            [
                "ts_code",
                "name",
                "start_date",
                "end_date",
                "type",
                "source_api",
                "source_row_sha256",
                "fetched_at",
            ],
        )
        self.assertTrue(frame.empty)

    def test_event_is_visible_only_on_next_open_session(self) -> None:
        shock = contract.normalize_source_frame(
            "stk_shock",
            pd.DataFrame(
                [["000001.SZ", "20260821", "A", "深交所", "reason", "period"]],
                columns=contract.API_SPEC_BY_NAME["stk_shock"].required_columns,
            ),
        )
        empty_high = contract.normalize_source_frame("stk_high_shock", pd.DataFrame())
        empty_alert = contract.normalize_source_frame("stk_alert", pd.DataFrame())
        dates = ["20260821", "20260824", "20260825"]
        events = contract.build_l2_events(
            {
                "stk_shock": shock,
                "stk_high_shock": empty_high,
                "stk_alert": empty_alert,
            },
            dates,
        )

        same_day = contract.build_signal_slice(events, dates, "20260821")
        next_day = contract.build_signal_slice(events, dates, "20260824")

        self.assertTrue(same_day.empty)
        self.assertEqual(int(next_day.iloc[0]["shock_count_1d"]), 1)

    def test_long_alert_remains_active_after_ten_sessions(self) -> None:
        dates = pd.bdate_range("2026-08-03", "2026-09-04").strftime("%Y%m%d").tolist()
        alert = contract.normalize_source_frame(
            "stk_alert",
            pd.DataFrame(
                [["000001.SZ", "A", dates[0], dates[-1], "重点"]],
                columns=contract.API_SPEC_BY_NAME["stk_alert"].required_columns,
            ),
        )
        events = contract.build_l2_events(
            {
                "stk_shock": contract.normalize_source_frame("stk_shock", pd.DataFrame()),
                "stk_high_shock": contract.normalize_source_frame("stk_high_shock", pd.DataFrame()),
                "stk_alert": alert,
            },
            dates,
        )

        result = contract.build_signal_slice(events, dates, dates[15])

        self.assertEqual(int(result.iloc[0]["alert_active_count"]), 1)
        self.assertEqual(int(result.iloc[0]["alert_start_count_10d"]), 0)


if __name__ == "__main__":
    unittest.main()
