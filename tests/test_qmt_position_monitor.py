import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


class QmtPositionMonitorTests(unittest.TestCase):
    def test_is_trading_window(self):
        from tools.qmt_position_monitor import is_trading_window

        tz = ZoneInfo("Asia/Shanghai")
        self.assertFalse(is_trading_window(datetime(2026, 6, 23, 9, 29, tzinfo=tz)))
        self.assertTrue(is_trading_window(datetime(2026, 6, 23, 9, 30, tzinfo=tz)))
        self.assertTrue(is_trading_window(datetime(2026, 6, 23, 14, 59, tzinfo=tz)))
        self.assertFalse(is_trading_window(datetime(2026, 6, 27, 10, 0, tzinfo=tz)))

    def test_build_monitor_snapshot_flags_core_risks(self):
        from tools.qmt_position_monitor import build_monitor_snapshot

        tz = ZoneInfo("Asia/Shanghai")
        strategy_context = {
            "strategy_entry": {"strategy_id": "prod_a", "name": "prod a"},
            "strategy_manifest": {"production_version": "v1"},
            "strategy_dir": "strategy_library/production/prod_a",
            "signal_path": "production_signals/prod_a_latest.csv",
            "signal_rows": [
                {
                    "signal_date": "20260622",
                    "buy_date": "20260623",
                    "stock_code": "000001.SZ",
                    "name": "A",
                    "rank": "1",
                    "target_pct": "0.30",
                },
                {
                    "signal_date": "20260622",
                    "buy_date": "20260623",
                    "stock_code": "000002.SZ",
                    "name": "B",
                    "rank": "2",
                    "target_pct": "0.20",
                },
            ],
        }
        holdings_summary = {
            "account_id": "acct",
            "account_type": "STOCK",
            "asset": {"total_asset": 100000.0, "market_value": 85000.0, "cash": 15000.0},
            "positions": [
                {
                    "stock_code": "000001.SZ",
                    "instrument_name": "A",
                    "volume": 1000,
                    "can_use_volume": 0,
                    "market_value": 50000.0,
                    "position_profit": -4000.0,
                    "profit_rate": -0.08,
                },
                {
                    "stock_code": "000009.SZ",
                    "instrument_name": "Z",
                    "volume": 1000,
                    "can_use_volume": 1000,
                    "market_value": 35000.0,
                    "position_profit": 1000.0,
                    "profit_rate": 0.03,
                },
            ],
        }
        ticks = {
            "000001.SZ": {"pct_chg": -0.06, "amplitude": 0.09, "spread": 0.001},
            "000009.SZ": {"pct_chg": 0.01, "amplitude": 0.02, "spread": 0.001},
        }

        snapshot = build_monitor_snapshot(
            strategy_context=strategy_context,
            holdings_summary=holdings_summary,
            ticks=ticks,
            now=datetime(2026, 6, 23, 10, 0, tzinfo=tz),
        )

        alert_types = {alert["type"] for alert in snapshot["alerts"]}
        self.assertIn("floating_loss", alert_types)
        self.assertIn("single_stock_pct", alert_types)
        self.assertIn("amplitude", alert_types)
        self.assertIn("position_gap", alert_types)
        self.assertIn("should_buy", alert_types)
        self.assertIn("should_sell", alert_types)
        self.assertEqual(snapshot["conclusion"], "重点关注")
        self.assertEqual(snapshot["consistency"]["target_missing_positions"], ["000002.SZ"])
        self.assertEqual(snapshot["consistency"]["non_target_positions"], ["000009.SZ"])


if __name__ == "__main__":
    unittest.main()
