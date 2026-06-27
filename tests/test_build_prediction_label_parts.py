import unittest

import pandas as pd

from build_prediction_label_parts import (
    available_labels,
    add_executable_labels,
    required_raw_columns,
    sanitize_existing_labels,
)


class BuildPredictionLabelPartsTest(unittest.TestCase):
    def test_available_labels_keeps_existing_and_computable_labels(self):
        raw_columns = [
            "trade_date",
            "stock_code",
            "10d_yield_rate",
            "post_open",
            "post6_open",
        ]

        labels = available_labels(raw_columns, ["10d_yield_rate", "executable_5d_open_return"])

        self.assertEqual(labels, ["10d_yield_rate", "executable_5d_open_return"])

    def test_required_raw_columns_reads_forward_open_inputs_for_computed_label(self):
        raw_columns = ["trade_date", "stock_code", "10d_yield_rate", "post_open", "post6_open"]

        columns = required_raw_columns(raw_columns, ["10d_yield_rate", "executable_5d_open_return"])

        self.assertEqual(columns, ["trade_date", "stock_code", "10d_yield_rate", "post_open", "post6_open"])

    def test_add_executable_labels_uses_existing_training_cost_formula(self):
        frame = pd.DataFrame({"post_open": [10.0], "post6_open": [10.8]})

        result = add_executable_labels(frame, ["executable_5d_open_return"])

        expected = (10.8 * (1 - 0.0003 - 0.0005 - 0.001)) / (10.0 * (1 + 0.0003 + 0.001)) - 1
        self.assertAlmostEqual(result.loc[0, "executable_5d_open_return"], expected)

    def test_required_raw_columns_reads_return_dependency_for_tag_labels(self):
        raw_columns = ["trade_date", "stock_code", "2d_tag", "2d_yield_rate"]

        columns = required_raw_columns(raw_columns, ["2d_tag"])

        self.assertEqual(columns, ["trade_date", "stock_code", "2d_tag", "2d_yield_rate"])

    def test_sanitize_existing_labels_nulls_unknown_future_tags(self):
        frame = pd.DataFrame(
            {
                "tag": [0, 1],
                "yield_rate": [float("nan"), 0.1],
                "2d_tag": [0, 1],
                "2d_yield_rate": [float("nan"), -0.2],
                "5d_tag": [0, 1],
                "5d_yield_rate": [float("nan"), 0.3],
            }
        )

        result = sanitize_existing_labels(frame, ["tag", "2d_tag", "5d_tag"])

        self.assertTrue(pd.isna(result.loc[0, "tag"]))
        self.assertEqual(result.loc[1, "tag"], 1.0)
        self.assertTrue(pd.isna(result.loc[0, "2d_tag"]))
        self.assertEqual(result.loc[1, "2d_tag"], 0.0)
        self.assertTrue(pd.isna(result.loc[0, "5d_tag"]))
        self.assertEqual(result.loc[1, "5d_tag"], 1.0)

    def test_sanitize_existing_labels_nulls_low_close_when_future_low_missing(self):
        frame = pd.DataFrame(
            {
                "low_close_yield_rate": [0.0, -0.1],
                "post_low": [float("nan"), 9.0],
            }
        )

        result = sanitize_existing_labels(frame, ["low_close_yield_rate"])

        self.assertTrue(pd.isna(result.loc[0, "low_close_yield_rate"]))
        self.assertEqual(result.loc[1, "low_close_yield_rate"], -0.1)


if __name__ == "__main__":
    unittest.main()
