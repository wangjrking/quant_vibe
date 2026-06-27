import unittest
import tempfile
from pathlib import Path

import pandas as pd

from fast_feature_selection import _read_parquet_date_range
from ai_module import prepare_training_label
from feature_selection_module import (
    FeatureSelectionConfig,
    prepare_selection_label,
    select_features,
)


class FeatureSelectionModuleTests(unittest.TestCase):
    def test_read_parquet_date_range_ignores_transient_tmp_parts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parts_dir = Path(temp_dir) / "production_factor_parts"
            parts_dir.mkdir()
            pd.DataFrame(
                [
                    {"trade_date": "20240102", "stock_code": "000001.SZ", "factor_a": 1.0},
                    {"trade_date": "20240103", "stock_code": "000002.SZ", "factor_a": 2.0},
                ]
            ).to_parquet(parts_dir / "production_factor_part_0000.parquet", index=False)
            (parts_dir / "production_factor_part_0001.parquet.tmp_20260624").write_bytes(b"partial")

            result = _read_parquet_date_range(
                parts_dir,
                ["trade_date", "stock_code", "factor_a"],
                "20240103",
                "20240103",
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["stock_code"], "000002.SZ")

    def test_select_features_prefers_predictive_non_leaky_factor(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260105"] * 4 + ["20260106"] * 4,
                "stock_code": list("ABCDEFGH"),
                "good_factor": [1, 2, 3, 4, 1, 2, 3, 4],
                "bad_factor": [4, 1, 4, 1, 4, 1, 4, 1],
                "post_open": [10, 10, 10, 10, 10, 10, 10, 10],
                "10d_yield_rate": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            }
        )

        selected = select_features(
            frame,
            FeatureSelectionConfig(label="10d_yield_rate", top_n=2, min_abs_ic=0.5),
        )

        self.assertIn("good_factor", selected)
        self.assertNotIn("post_open", selected)

    def test_prepare_training_label_builds_risk_adjusted_return(self):
        frame = pd.DataFrame(
            {
                "10d_yield_rate": [0.10, 0.05],
                "atr_qfq": [1.0, 0.5],
                "close": [20.0, 10.0],
            }
        )

        result = prepare_training_label(frame, "risk_adjusted_10d_yield_rate")

        self.assertAlmostEqual(result.loc[0, "risk_adjusted_10d_yield_rate"], 2.0)
        self.assertAlmostEqual(result.loc[1, "risk_adjusted_10d_yield_rate"], 1.0)

    def test_prepare_training_label_builds_executable_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post12_open": [11.0]})

        result = prepare_training_label(frame, "executable_10d_open_return")

        expected = (11.0 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_10d_open_return"], expected)

    def test_prepare_training_label_builds_executable_5d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post6_open": [10.8]})

        result = prepare_training_label(frame, "executable_5d_open_return")

        expected = (10.8 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_5d_open_return"], expected)

    def test_prepare_training_label_builds_daily_top_quantile_label(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260105"] * 10 + ["20260106"] * 10,
                "post_open": [10.0] * 20,
                "post6_open": list(range(10, 20)) + list(range(20, 10, -1)),
            }
        )

        result = prepare_training_label(frame, "executable_5d_open_top10")

        self.assertEqual(int(result["executable_5d_open_top10"].sum()), 2)
        self.assertEqual(int(result.loc[9, "executable_5d_open_top10"]), 1)
        self.assertEqual(int(result.loc[10, "executable_5d_open_top10"]), 1)

    def test_prepare_selection_label_builds_daily_top_quantile_label(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260105"] * 10,
                "post_open": [10.0] * 10,
                "post6_open": list(range(10, 20)),
            }
        )

        result = prepare_selection_label(frame, "executable_5d_open_top10")

        self.assertEqual(int(result["executable_5d_open_top10"].sum()), 1)

    def test_prepare_selection_label_builds_executable_5d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post6_open": [10.8]})

        result = prepare_selection_label(frame, "executable_5d_open_return")

        expected = (10.8 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_5d_open_return"], expected)

    def test_prepare_training_label_builds_executable_3d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post4_open": [10.6]})

        result = prepare_training_label(frame, "executable_3d_open_return")

        expected = (10.6 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_3d_open_return"], expected)

    def test_prepare_selection_label_builds_executable_3d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post4_open": [10.6]})

        result = prepare_selection_label(frame, "executable_3d_open_return")

        expected = (10.6 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_3d_open_return"], expected)

    def test_prepare_training_label_builds_executable_1d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post2_open": [10.3]})

        result = prepare_training_label(frame, "executable_1d_open_return")

        expected = (10.3 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_1d_open_return"], expected)

    def test_prepare_selection_label_builds_executable_1d_open_return(self):
        frame = pd.DataFrame({"post_open": [10.0], "post2_open": [10.3]})

        result = prepare_selection_label(frame, "executable_1d_open_return")

        expected = (10.3 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_1d_open_return"], expected)


if __name__ == "__main__":
    unittest.main()
