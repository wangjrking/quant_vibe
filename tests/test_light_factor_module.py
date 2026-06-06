import tempfile
import unittest
from pathlib import Path

import pandas as pd

from light_factor_module import (
    build_light_factor_frame,
    find_unsupported_features,
    load_feature_list,
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
        self.assertAlmostEqual(first["close_rate"], 1.1)
        self.assertAlmostEqual(first["10d_yield_rate"], (21 - 11) / 11)
        self.assertEqual(first["industry_encode"], 0)

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


if __name__ == "__main__":
    unittest.main()
