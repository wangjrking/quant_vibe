import sqlite3
import tempfile
import unittest
import csv
import importlib.util
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, to_gm_symbol, write_gm_signals_csv
from selection_module import SelectionConfig


class GmSignalModuleTests(unittest.TestCase):
    def test_to_gm_symbol_converts_tushare_codes(self):
        self.assertEqual(to_gm_symbol("600000.SH"), "SHSE.600000")
        self.assertEqual(to_gm_symbol("000001.SZ"), "SZSE.000001")
        self.assertEqual(to_gm_symbol("430047.BJ"), "BJSE.430047")

    def test_build_gm_signal_rows_uses_next_trade_date(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(rows, SelectionConfig(top_k=1, min_pred_prob=0.01))

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal_date"], "20260102")
        self.assertEqual(signals[0]["buy_date"], "20260103")
        self.assertEqual(signals[0]["symbol"], "SHSE.600000")

    def test_build_gm_signal_rows_uses_market_trade_calendar_for_last_score_date(self):
        rows = [
            {
                "trade_date": "20260618",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260618",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]
        market_rows_by_trade_date = {
            "20260618": {
                "600000.SH": {
                    "stock_code": "600000.SH",
                    "name": "A",
                    "close": 10.0,
                },
                "000001.SZ": {
                    "stock_code": "000001.SZ",
                    "name": "B",
                    "close": 10.0,
                },
            },
            "20260622": {
                "600000.SH": {
                    "stock_code": "600000.SH",
                    "name": "A",
                    "pre_close": 10.0,
                    "open": 10.1,
                },
                "000001.SZ": {
                    "stock_code": "000001.SZ",
                    "name": "B",
                    "pre_close": 10.0,
                    "open": 10.1,
                },
            },
        }

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=1, min_pred_prob=0.01),
            market_rows_by_trade_date=market_rows_by_trade_date,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal_date"], "20260618")
        self.assertEqual(signals[0]["buy_date"], "20260622")
        self.assertEqual(signals[0]["stock_code"], "600000.SH")

    def test_write_gm_signals_csv_rejects_empty_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_gm_signals_csv([], Path(tmpdir) / "signals.csv")

    def test_write_gm_signals_csv_allows_later_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "signals.csv"
            write_gm_signals_csv(
                [
                    {"signal_date": "20260102", "stock_code": "600000.SH"},
                    {"signal_date": "20260102", "stock_code": "000001.SZ", "extra": "x"},
                ],
                output,
            )

            with output.open("r", newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                rows = list(reader)

        self.assertIn("extra", reader.fieldnames)
        self.assertEqual(rows[1]["extra"], "x")

    def test_write_gm_signals_csv_rejects_naked_front_adjusted_indicator_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "signals.csv"
            with self.assertRaisesRegex(ValueError, "naked front-adjusted indicator"):
                write_gm_signals_csv(
                    [
                        {"signal_date": "20260102", "stock_code": "600000.SH", "atr": 0.2},
                    ],
                    output,
                )

    def test_write_gm_signals_csv_rejects_naked_raw_derived_market_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "signals.csv"
            with self.assertRaisesRegex(ValueError, "naked raw-derived market fields"):
                write_gm_signals_csv(
                    [
                        {"signal_date": "20260102", "stock_code": "600000.SH", "pct_chg": 2.0},
                    ],
                    output,
                )

    @unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
    def test_load_market_rows_by_trade_date_supports_duckdb_stock_daily_table(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "market.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        name VARCHAR,
                        pre_close DOUBLE,
                        open DOUBLE,
                        close DOUBLE,
                        amount DOUBLE,
                        turnover_rate DOUBLE,
                        total_mv DOUBLE,
                        atr_qfq DOUBLE,
                        limit_times VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('20240604', '000001.SZ', '平安银行', 10.0, 10.1, 10.2, 1200000.0, 3.5, 900000.0, 0.2, NULL),
                    ('20240605', '000001.SZ', '平安银行', 10.1, 10.2, 10.3, 1300000.0, 3.6, 910000.0, 0.2, NULL)
                    """
                )

            grouped = load_market_rows_by_trade_date(db_path, "20240604", "20240605")

        self.assertEqual(sorted(grouped), ["20240604", "20240605"])
        self.assertEqual(grouped["20240604"]["000001.SZ"]["name"], "平安银行")
        self.assertEqual(grouped["20240605"]["000001.SZ"]["turnover_rate"], 3.6)

    def test_build_gm_signal_rows_can_attach_target_pct(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
            weight_mode="score",
            target_total_pct=0.98,
        )

        self.assertEqual(len(signals), 2)
        self.assertAlmostEqual(sum(signal["target_pct"] for signal in signals), 0.98, places=6)
        self.assertGreater(signals[0]["target_pct"], signals[1]["target_pct"])

    def test_build_gm_signal_rows_can_downweight_low_liquidity_targets(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
                "amount": 40000,
                "turnover_rate": 0.4,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.29,
                "close": 10.0,
                "atr_qfq": 0.2,
                "amount": 300000,
                "turnover_rate": 2.0,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
            weight_mode="equal",
            target_total_pct=0.98,
            liquidity_target_pct_enabled=True,
            liquidity_min_amount=100000,
            liquidity_min_turnover_rate=1.0,
            liquidity_mid_scale=0.8,
            liquidity_low_scale=0.6,
        )

        self.assertEqual(len(signals), 2)
        self.assertAlmostEqual(sum(signal["target_pct"] for signal in signals), 0.98, places=6)
        by_code = {signal["stock_code"]: signal for signal in signals}
        self.assertLess(by_code["600000.SH"]["target_pct"], by_code["000001.SZ"]["target_pct"])

    def test_build_gm_signal_rows_can_attach_per_signal_holding_days(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
            holding_days=3,
        )

        self.assertEqual(len(signals), 2)
        self.assertTrue(all(signal["holding_days"] == 3 for signal in signals))

    def test_build_gm_signal_rows_can_filter_by_prediction_quantile(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.90,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000003.SZ",
                "name": "D",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=3, min_pred_prob=None, min_pred_quantile=0.95),
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["stock_code"], "000002.SZ")

    def test_build_gm_signal_rows_can_cap_target_pct_by_expected_concurrency(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
                "pre_close": 9.8,
                "open": 10.0,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
                "pre_close": 10.0,
                "open": 10.0,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=1, min_pred_prob=0.01),
            holding_days=3,
            target_total_pct=0.98,
            max_positions=5,
        )

        self.assertEqual(len(signals), 1)
        self.assertAlmostEqual(signals[0]["target_pct"], 0.98 / 3.0, places=6)

    def test_build_gm_signal_rows_skips_unbuyable_next_day_limit_open(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
                "pre_close": 9.0,
                "open": 9.5,
            },
            {
                "trade_date": "20260103",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.10,
                "close": 11.0,
                "atr_qfq": 0.2,
                "pre_close": 10.0,
                "open": 11.0,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
                "pre_close": 9.8,
                "open": 10.0,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.15,
                "close": 10.2,
                "atr_qfq": 0.2,
                "pre_close": 10.0,
                "open": 10.1,
            },
            {
                "trade_date": "20260104",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
                "pre_close": 10.0,
                "open": 10.0,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
        )

        first_day_signals = [signal for signal in signals if signal["signal_date"] == "20260102"]
        self.assertEqual(len(first_day_signals), 1)
        self.assertEqual(first_day_signals[0]["stock_code"], "000001.SZ")

    def test_build_gm_signal_rows_uses_market_rows_for_next_day_unbuyable_filter(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        market_rows_by_trade_date = {
            "20260103": {
                "600000.SH": {
                    "stock_code": "600000.SH",
                    "name": "A",
                    "pre_close": 10.0,
                    "open": 11.0,
                },
                "000001.SZ": {
                    "stock_code": "000001.SZ",
                    "name": "B",
                    "pre_close": 10.0,
                    "open": 10.1,
                },
            }
        }

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
            market_rows_by_trade_date=market_rows_by_trade_date,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["stock_code"], "000001.SZ")

    def test_build_gm_signal_rows_treats_zero_limit_times_as_buyable(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "name": "A",
                "pred_prob": 0.30,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "B",
                "pred_prob": 0.20,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "C",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        market_rows_by_trade_date = {
            "20260103": {
                "600000.SH": {
                    "stock_code": "600000.SH",
                    "name": "A",
                    "pre_close": 10.0,
                    "open": 10.1,
                    "limit_times": "0",
                },
                "000001.SZ": {
                    "stock_code": "000001.SZ",
                    "name": "B",
                    "pre_close": 10.0,
                    "open": 10.1,
                    "limit_times": "1",
                },
            }
        }

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
            market_rows_by_trade_date=market_rows_by_trade_date,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["stock_code"], "600000.SH")

    def test_build_gm_signal_rows_uses_market_rows_for_signal_day_atr_filter(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "600000.SH",
                "pred_prob": 0.30,
                "atr_qfq": 0.20,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "pred_prob": 0.20,
                "atr_qfq": 0.20,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "pred_prob": 0.10,
                "atr_qfq": 0.20,
            },
        ]

        market_rows_by_trade_date = {
            "20260102": {
                "600000.SH": {
                    "stock_code": "600000.SH",
                    "name": "A",
                    "close": 20.0,
                },
                "000001.SZ": {
                    "stock_code": "000001.SZ",
                    "name": "B",
                    "close": 5.0,
                },
            },
            "20260103": {
                "600000.SH": {
                    "stock_code": "600000.SH",
                    "name": "A",
                    "pre_close": 20.0,
                    "open": 20.1,
                },
                "000001.SZ": {
                    "stock_code": "000001.SZ",
                    "name": "B",
                    "pre_close": 5.0,
                    "open": 5.1,
                },
            },
        }

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01, max_atr_ratio=0.03),
            market_rows_by_trade_date=market_rows_by_trade_date,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["stock_code"], "600000.SH")

    def test_build_gm_signal_rows_skips_bj_candidates(self):
        rows = [
            {
                "trade_date": "20260102",
                "stock_code": "430047.BJ",
                "name": "BJ",
                "pred_prob": 0.90,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260102",
                "stock_code": "000001.SZ",
                "name": "SZ",
                "pred_prob": 0.80,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
            {
                "trade_date": "20260103",
                "stock_code": "000002.SZ",
                "name": "NEXT",
                "pred_prob": 0.10,
                "close": 10.0,
                "atr_qfq": 0.2,
            },
        ]

        signals = build_gm_signal_rows(
            rows,
            SelectionConfig(top_k=2, min_pred_prob=0.01),
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["stock_code"], "000001.SZ")

    def test_load_market_rows_by_trade_date_includes_close_from_stock_daily_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "market.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date TEXT,
                        stock_code TEXT,
                        name TEXT,
                        pre_close REAL,
                        open REAL,
                        close REAL,
                        amount REAL,
                        turnover_rate REAL,
                        total_mv REAL,
                        atr_qfq REAL,
                        limit_times TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA (
                        trade_date, stock_code, name, pre_close, open, close,
                        amount, turnover_rate, total_mv, atr_qfq, limit_times
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("20260102", "600000.SH", "A", 9.8, 10.0, 10.2, 1000.0, 1.2, 200000.0, 0.24, None),
                )
                conn.commit()
            finally:
                conn.close()

            rows = load_market_rows_by_trade_date(db_path, "20260102", "20260102")

            self.assertEqual(rows["20260102"]["600000.SH"]["close"], 10.2)
            self.assertEqual(rows["20260102"]["600000.SH"]["atr_qfq"], 0.24)


if __name__ == "__main__":
    unittest.main()
