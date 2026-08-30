import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from light_factor_module import (
    build_light_factor_frame,
    find_unsupported_features,
    load_feature_list,
    read_raw_frame,
    split_light_factor_data,
)


class LightFactorModuleTests(unittest.TestCase):
    def test_load_feature_list_reads_json_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.json"
            path.write_text('{"features": ["close", "pb"]}', encoding="utf-8")

            self.assertEqual(load_feature_list(path), ["close", "pb"])

    def test_find_unsupported_features_rejects_legacy(self):
        unsupported = find_unsupported_features(["close", "legacy_close"], available_columns={"close"})

        self.assertEqual(unsupported, ["legacy_close"])

    def test_build_light_factor_frame_adds_forward_labels_and_rates(self):
        raw = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 13,
                "trade_date": [f"202601{i:02d}" for i in range(1, 14)],
                "name": ["Good"] * 13,
                "open": list(range(10, 23)),
                "close": list(range(11, 24)),
                "high": list(range(12, 25)),
                "low": list(range(9, 22)),
                "pre_close": list(range(10, 23)),
                "limit_times": [None] * 13,
                "st_type": [None] * 13,
                "industry": ["Bank"] * 13,
                "atr_qfq": [0.2] * 13,
                "pb": [1.0] * 13,
            }
        )

        result = build_light_factor_frame(raw, ["close_rate", "pb"], label="10d_yield_rate")

        first = result.iloc[0]
        second = result.iloc[1]
        self.assertTrue(pd.isna(first["close_rate"]))
        self.assertAlmostEqual(second["close_rate"], 12 / 11)
        self.assertAlmostEqual(first["10d_yield_rate"], (21 - 11) / 11)
        self.assertEqual(first["industry_encode"], 0)

    def test_build_light_factor_frame_uses_prior_qfq_close_not_raw_pre_close(self):
        raw = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 4,
                "trade_date": [f"202601{i:02d}" for i in range(1, 5)],
                "name": ["Good"] * 4,
                "open": [10.0, 11.0, 11.5, 12.0],
                "close": [10.0, 11.0, 12.1, 12.5],
                "high": [10.2, 11.2, 12.3, 12.7],
                "low": [9.8, 10.8, 11.9, 12.2],
                # Deliberately inconsistent raw pre_close values that should not drive qfq ratios.
                "pre_close": [9.5, 9.8, 10.0, 10.1],
                "limit_times": [None] * 4,
                "st_type": [None] * 4,
                "industry": ["Bank"] * 4,
                "atr_qfq": [0.2] * 4,
                "pb": [1.0] * 4,
            }
        )

        result = build_light_factor_frame(raw, ["close_rate", "open_rate", "high_rate", "low_rate"], label="10d_yield_rate")

        second = result.iloc[1]
        third = result.iloc[2]
        self.assertAlmostEqual(second["close_rate"], 11.0 / 10.0)
        self.assertAlmostEqual(second["open_rate"], 11.0 / 10.0)
        self.assertAlmostEqual(third["high_rate"], 12.3 / 11.0)
        self.assertAlmostEqual(third["low_rate"], 11.9 / 11.0)

    def test_build_light_factor_frame_supports_short_labels(self):
        raw = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 4,
                "trade_date": [f"202601{i:02d}" for i in range(1, 5)],
                "name": ["Good"] * 4,
                "open": [10.0, 10.5, 11.0, 11.2],
                "close": [10.2, 10.8, 11.1, 11.4],
                "high": [10.3, 10.9, 11.2, 11.5],
                "low": [9.9, 10.4, 10.9, 11.1],
                "pre_close": [10.0, 10.2, 10.8, 11.1],
                "limit_times": [None] * 4,
                "st_type": [None] * 4,
                "industry": ["Bank"] * 4,
                "atr_qfq": [0.2] * 4,
                "pb": [1.0] * 4,
            }
        )

        result = build_light_factor_frame(raw, ["close_rate", "pb"], label="executable_2d_open_return")

        first = result.iloc[0]
        self.assertAlmostEqual(first["2d_yield_rate"], (11.1 - 10.2) / 10.2)
        self.assertIn("executable_2d_open_return", result.columns)

    def test_build_light_factor_frame_supports_executable_5d_label(self):
        raw = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 7,
                "trade_date": [f"202601{i:02d}" for i in range(1, 8)],
                "name": ["Good"] * 7,
                "open": [10.0, 10.4, 10.6, 10.8, 10.9, 11.0, 11.2],
                "close": [10.1, 10.5, 10.7, 10.85, 11.0, 11.1, 11.25],
                "high": [10.2, 10.6, 10.8, 10.95, 11.1, 11.2, 11.3],
                "low": [9.9, 10.3, 10.5, 10.7, 10.8, 10.9, 11.0],
                "pre_close": [9.95, 10.1, 10.5, 10.7, 10.85, 11.0, 11.1],
                "limit_times": [None] * 7,
                "st_type": [None] * 7,
                "industry": ["Bank"] * 7,
                "atr_qfq": [0.2] * 7,
                "pb": [1.0] * 7,
            }
        )

        result = build_light_factor_frame(raw, ["close_rate", "pb"], label="executable_5d_open_return")

        first = result.iloc[0]
        expected = (11.2 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.4 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(first["executable_5d_open_return"], expected)

    def test_build_light_factor_frame_supports_executable_3d_label(self):
        raw = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 5,
                "trade_date": [f"202601{i:02d}" for i in range(1, 6)],
                "name": ["Good"] * 5,
                "open": [10.0, 10.2, 10.4, 10.5, 10.6],
                "close": [10.1, 10.3, 10.45, 10.55, 10.7],
                "high": [10.2, 10.4, 10.55, 10.65, 10.8],
                "low": [9.9, 10.1, 10.3, 10.4, 10.5],
                "pre_close": [9.95, 10.1, 10.3, 10.45, 10.55],
                "limit_times": [None] * 5,
                "st_type": [None] * 5,
                "industry": ["Bank"] * 5,
                "atr_qfq": [0.2] * 5,
                "pb": [1.0] * 5,
            }
        )

        result = build_light_factor_frame(raw, ["close_rate", "pb"], label="executable_3d_open_return")

        first = result.iloc[0]
        expected = (10.6 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.2 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(first["executable_3d_open_return"], expected)

    def test_build_light_factor_frame_supports_executable_1d_label(self):
        raw = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 3,
                "trade_date": [f"202601{i:02d}" for i in range(1, 4)],
                "name": ["Good"] * 3,
                "open": [10.0, 10.2, 10.3],
                "close": [10.1, 10.25, 10.35],
                "high": [10.2, 10.35, 10.45],
                "low": [9.9, 10.1, 10.2],
                "pre_close": [9.95, 10.1, 10.25],
                "limit_times": [None] * 3,
                "st_type": [None] * 3,
                "industry": ["Bank"] * 3,
                "atr_qfq": [0.2] * 3,
                "pb": [1.0] * 3,
            }
        )

        result = build_light_factor_frame(raw, ["close_rate", "pb"], label="executable_1d_open_return")

        first = result.iloc[0]
        expected = (10.3 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.2 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(first["executable_1d_open_return"], expected)

    def test_split_light_factor_data_returns_selected_columns(self):
        frame = pd.DataFrame(
            {
                "stock_code": ["000001.SZ", "000001.SZ"],
                "trade_date": ["20260101", "20260201"],
                "name": ["A", "A"],
                "close": [1.0, 2.0],
                "pb": [1.1, 1.2],
                "10d_yield_rate": [0.1, 0.2],
            }
        )

        train_x, train_y, test_x, test_y, train_data, test_data = split_light_factor_data(
            frame,
            features=["close", "pb"],
            label="10d_yield_rate",
            train_start="20260101",
            test_start="20260201",
        )

        self.assertEqual(train_x.columns.tolist(), ["close", "pb"])
        self.assertEqual(test_x.columns.tolist(), ["close", "pb"])
        self.assertEqual(train_y.tolist(), [0.1])
        self.assertEqual(test_data["trade_date"].tolist(), ["20260201"])

    def test_read_raw_frame_supports_duckdb_stock_daily_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "quant_production.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA AS
                    SELECT * FROM (
                        VALUES
                        ('20260101', '000001.SZ', '000001.SZ', 10.0, 10.2, 10.3, 9.9, 100.0, 1000.0, NULL, NULL),
                        ('20260102', '000001.SZ', '000001.SZ', 10.2, 10.4, 10.5, 10.1, 101.0, 1100.0, NULL, NULL)
                    ) AS t(
                        trade_date, stock_code, ts_code, open, close, high, low, amount, vol, name, industry
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE stock_basic_data AS
                    SELECT * FROM (
                        VALUES ('000001.SZ', 'PingAn', 'Bank')
                    ) AS t(ts_code, name, industry)
                    """
                )

            frame = read_raw_frame(
                db_path,
                start="20260101",
                end="20260102",
                needed_columns=["trade_date", "stock_code", "ts_code", "open", "close", "name", "industry"],
            )

            self.assertEqual(frame["trade_date"].tolist(), ["20260101", "20260102"])
            self.assertEqual(frame["stock_code"].tolist(), ["000001.SZ", "000001.SZ"])
            self.assertEqual(frame["name"].tolist(), ["PingAn", "PingAn"])
            self.assertEqual(frame["industry"].tolist(), ["Bank", "Bank"])


if __name__ == "__main__":
    unittest.main()
