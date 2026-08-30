import csv
import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from l7_buy_day_hard_gate import evaluate_buy_day_hard_gate, finalize_platform_signals


class L7BuyDayHardGateTests(unittest.TestCase):
    def test_missing_buy_day_market_row_keeps_signal_pending(self):
        result = evaluate_buy_day_hard_gate(
            signal_row={"stock_code": "000636.SZ", "buy_date": "20260630"},
            market_row=None,
        )

        self.assertEqual(result["status"], "pending_buy_day_hard_gate")
        self.assertFalse(result["ready_for_human_confirmation_execution"])
        self.assertFalse(result["buy_day_market_available"])
        self.assertIn("buy_day_market_row_missing", result["blockers"])

    def test_normal_buy_day_market_row_allows_human_confirmation(self):
        result = evaluate_buy_day_hard_gate(
            signal_row={"stock_code": "000636.SZ", "buy_date": "20260630"},
            market_row={
                "trade_date": "20260630",
                "stock_code": "000636.SZ",
                "name": "风华高科",
                "pre_close": 80.0,
                "open": 81.0,
                "limit_times": 0,
                "ST_TYPE": 0,
                "ST_TYPE_name": "",
            },
        )

        self.assertEqual(result["status"], "buy_day_hard_gate_passed")
        self.assertTrue(result["ready_for_human_confirmation_execution"])
        self.assertTrue(result["buy_day_market_available"])
        self.assertTrue(result["buy_day_hard_gate_complete"])
        self.assertEqual(result["blockers"], [])

    def test_st_like_buy_day_market_row_blocks_execution(self):
        result = evaluate_buy_day_hard_gate(
            signal_row={"stock_code": "000636.SZ", "buy_date": "20260630"},
            market_row={
                "trade_date": "20260630",
                "stock_code": "000636.SZ",
                "name": "ST风华",
                "pre_close": 80.0,
                "open": 81.0,
                "limit_times": 0,
                "ST_TYPE": 1,
                "ST_TYPE_name": "风险警示",
            },
        )

        self.assertEqual(result["status"], "buy_day_hard_gate_rejected")
        self.assertFalse(result["ready_for_human_confirmation_execution"])
        self.assertTrue(result["buy_day_st_rejected"])
        self.assertIn("buy_day_st_or_risk_warning", result["blockers"])

    def test_open_limit_up_buy_day_market_row_blocks_execution(self):
        result = evaluate_buy_day_hard_gate(
            signal_row={"stock_code": "000636.SZ", "buy_date": "20260630"},
            market_row={
                "trade_date": "20260630",
                "stock_code": "000636.SZ",
                "name": "风华高科",
                "pre_close": 80.0,
                "open": 88.0,
                "limit_times": 1,
                "ST_TYPE": 0,
                "ST_TYPE_name": "",
            },
        )

        self.assertEqual(result["status"], "buy_day_hard_gate_rejected")
        self.assertFalse(result["ready_for_human_confirmation_execution"])
        self.assertTrue(result["buy_day_open_limit_up_rejected"])
        self.assertIn("buy_day_open_limit_up", result["blockers"])

    def test_incomplete_buy_day_market_row_stays_pending_instead_of_rejecting(self):
        result = evaluate_buy_day_hard_gate(
            signal_row={"stock_code": "000636.SZ", "buy_date": "20260630"},
            market_row={
                "trade_date": "20260630",
                "stock_code": "000636.SZ",
                "name": "风华高科",
                "pre_close": 80.0,
                "ST_TYPE": 0,
                "ST_TYPE_name": "",
            },
        )

        self.assertEqual(result["status"], "pending_buy_day_hard_gate")
        self.assertFalse(result["ready_for_human_confirmation_execution"])
        self.assertFalse(result["buy_day_open_limit_up_rejected"])
        self.assertIn("buy_day_open_price_missing", result["blockers"])

    def test_finalize_platform_signals_writes_ready_package_when_market_row_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            platform_csv = root / "l7_platform_signals.csv"
            with platform_csv.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["strategy_id", "signal_date", "buy_date", "stock_code", "platform_symbol", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "strategy_id": "prod_a",
                        "signal_date": "20260629",
                        "buy_date": "20260630",
                        "stock_code": "000636.SZ",
                        "platform_symbol": "SZSE.000636",
                        "target_pct": "0.60000",
                    }
            )
            market_db = root / "market.duckdb"
            conn = duckdb.connect(str(market_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date TEXT,
                        stock_code TEXT,
                        name TEXT,
                        pre_close REAL,
                        open REAL,
                        limit_times REAL,
                        ST_TYPE REAL,
                        ST_TYPE_name TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO STOCK_DAILY_DATA VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("20260630", "000636.SZ", "风华高科", 80.0, 81.0, 0.0, 0.0, ""),
                )
                conn.commit()
            finally:
                conn.close()

            summary = finalize_platform_signals(
                platform_signals_path=platform_csv,
                market_db_path=market_db,
                output_dir=root / "out",
            )

            self.assertTrue(summary["ready_for_human_confirmation_execution"])
            summary_path = root / "out" / "buy_day_hard_gate_summary.json"
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "ready_for_human_confirmation_execution")
            self.assertEqual(persisted["report_path"], str(root / "out" / "buy_day_hard_gate_report.md"))
            with (root / "out" / "buy_day_hard_gate_platform_signals.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["status"], "buy_day_hard_gate_passed")
            self.assertEqual(rows[0]["ready_for_human_confirmation_execution"], "True")

    def test_finalize_platform_signals_uses_market_snapshot_before_eod_daily_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            platform_csv = root / "l7_platform_signals.csv"
            with platform_csv.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["strategy_id", "signal_date", "buy_date", "stock_code", "platform_symbol", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "strategy_id": "prod_a",
                        "signal_date": "20260629",
                        "buy_date": "20260630",
                        "stock_code": "000636.SZ",
                        "platform_symbol": "SZSE.000636",
                        "target_pct": "0.60000",
                    }
                )
            market_db = root / "market.duckdb"
            conn = duckdb.connect(str(market_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date TEXT,
                        stock_code TEXT,
                        name TEXT,
                        pre_close REAL,
                        open REAL,
                        limit_times REAL,
                        ST_TYPE REAL,
                        ST_TYPE_name TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()
            snapshot = root / "buy_day_snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "trade_date": "20260630",
                                "stock_code": "000636.SZ",
                                "stock_name": "风华高科",
                                "lastClose": 80.0,
                                "open": 81.0,
                                "limit_times": 0,
                                "ST_TYPE": 0,
                                "ST_TYPE_name": "",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = finalize_platform_signals(
                platform_signals_path=platform_csv,
                market_db_path=market_db,
                output_dir=root / "out",
                market_snapshot_path=snapshot,
            )

            self.assertTrue(summary["ready_for_human_confirmation_execution"])
            self.assertEqual(summary["gate_results"][0]["market_source"], "market_snapshot")
            with (root / "out" / "buy_day_hard_gate_platform_signals.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["hard_gate_market_source"], "market_snapshot")

    def test_market_snapshot_missing_open_or_last_close_stays_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            platform_csv = root / "l7_platform_signals.csv"
            with platform_csv.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["strategy_id", "signal_date", "buy_date", "stock_code", "platform_symbol", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "strategy_id": "prod_a",
                        "signal_date": "20260629",
                        "buy_date": "20260630",
                        "stock_code": "000636.SZ",
                        "platform_symbol": "SZSE.000636",
                        "target_pct": "0.60000",
                    }
                )
            market_db = root / "market.duckdb"
            conn = duckdb.connect(str(market_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date TEXT,
                        stock_code TEXT,
                        name TEXT,
                        pre_close REAL,
                        open REAL,
                        limit_times REAL,
                        ST_TYPE REAL,
                        ST_TYPE_name TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()
            snapshot = root / "incomplete_snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "trade_date": "20260630",
                                "stock_code": "000636.SZ",
                                "stock_name": "风华高科",
                                "lastClose": 80.0,
                                "limit_times": 0,
                                "ST_TYPE": 0,
                                "ST_TYPE_name": "",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = finalize_platform_signals(
                platform_signals_path=platform_csv,
                market_db_path=market_db,
                output_dir=root / "out",
                market_snapshot_path=snapshot,
            )

            self.assertFalse(summary["ready_for_human_confirmation_execution"])
            self.assertEqual(summary["status"], "pending_buy_day_hard_gate")
            self.assertEqual(summary["gate_results"][0]["market_source"], "market_snapshot")
            self.assertIn("buy_day_open_price_missing", summary["gate_results"][0]["blockers"])

    def test_market_snapshot_normalizes_symbol_alias_and_bare_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            platform_csv = root / "l7_platform_signals.csv"
            with platform_csv.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["strategy_id", "signal_date", "buy_date", "stock_code", "platform_symbol", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "strategy_id": "prod_a",
                        "signal_date": "20260629",
                        "buy_date": "20260630",
                        "stock_code": "000636.SZ",
                        "platform_symbol": "SZSE.000636",
                        "target_pct": "0.60000",
                    }
                )
            market_db = root / "market.duckdb"
            conn = duckdb.connect(str(market_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date TEXT,
                        stock_code TEXT,
                        name TEXT,
                        pre_close REAL,
                        open REAL,
                        limit_times REAL,
                        ST_TYPE REAL,
                        ST_TYPE_name TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()
            snapshot = root / "alias_snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "ticks": [
                            {
                                "date": "20260630",
                                "symbol": "000636",
                                "stock_name": "风华高科",
                                "lastClose": 80.0,
                                "openPrice": 81.0,
                                "limit_times": 0,
                                "ST_TYPE": 0,
                                "ST_TYPE_name": "",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = finalize_platform_signals(
                platform_signals_path=platform_csv,
                market_db_path=market_db,
                output_dir=root / "out",
                market_snapshot_path=snapshot,
            )

            self.assertTrue(summary["ready_for_human_confirmation_execution"])
            self.assertEqual(summary["gate_results"][0]["market_source"], "market_snapshot")
            self.assertEqual(summary["gate_results"][0]["market_row"]["stock_code"], "000636.SZ")


if __name__ == "__main__":
    unittest.main()
