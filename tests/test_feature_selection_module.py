import unittest

import pandas as pd

from ai_module import prepare_training_label
from feature_selection_module import (
    FeatureSelectionConfig,
    prepare_selection_label,
    select_features,
)


class FeatureSelectionModuleTests(unittest.TestCase):
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
