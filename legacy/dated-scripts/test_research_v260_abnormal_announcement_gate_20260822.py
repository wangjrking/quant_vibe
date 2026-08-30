from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_v260_abnormal_announcement_gate_20260822 as module


class AnnouncementStrategyTests(unittest.TestCase):
    def test_feature_matrix_uses_exact_date_and_stock_keys(self):
        dates = np.asarray(["20240102", "20240103"])
        stocks = np.asarray(["000001.SZ", "600000.SH"])
        features = pd.DataFrame(
            [
                {"signal_date": "20240102", "stock_code": "000001.SZ", "abnormal_count_3d": 1},
                {"signal_date": "20240102", "stock_code": "999999.SZ", "abnormal_count_3d": 1},
                {"signal_date": "20240104", "stock_code": "600000.SH", "abnormal_count_3d": 1},
            ]
        )
        matrix, audit = module.feature_matrix(dates, stocks, features, 3)
        self.assertTrue(matrix[0, 0])
        self.assertEqual(int(matrix.sum()), 1)
        self.assertEqual(audit["matched_stock_dates"], 1)

    def test_zero_count_does_not_block_entry(self):
        dates = np.asarray(["20240102"])
        stocks = np.asarray(["000001.SZ"])
        features = pd.DataFrame(
            [{"signal_date": "20240102", "stock_code": "000001.SZ", "abnormal_count_1d": 0}]
        )
        matrix, _ = module.feature_matrix(dates, stocks, features, 1)
        self.assertFalse(matrix.any())

    def test_missing_window_column_fails_closed(self):
        with self.assertRaises(ValueError):
            module.feature_matrix(
                np.asarray(["20240102"]),
                np.asarray(["000001.SZ"]),
                pd.DataFrame([{"signal_date": "20240102", "stock_code": "000001.SZ"}]),
                5,
            )


if __name__ == "__main__":
    unittest.main()
