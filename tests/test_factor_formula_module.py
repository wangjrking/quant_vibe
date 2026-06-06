import math
import unittest

import pandas as pd

from data_process_module import group_factor_eng
from ta_compat import midpoint, midprice


def _sample_group(n=30):
    close = pd.Series([10.0 + i for i in range(n)])
    frame = pd.DataFrame(
        {
            "stock_code": ["000001.SZ"] * n,
            "trade_date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y%m%d"),
            "name": ["Sample"] * n,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "pre_close": close.shift(1),
            "vol": [100.0] * n,
            "amount": close * 10.0,
            "his_low": close - 2.0,
            "his_high": close + 2.0,
            "cost_5pct": close * 0.9,
            "cost_15pct": close * 0.95,
            "cost_50pct": close,
            "cost_85pct": close * 1.05,
            "cost_95pct": close * 1.1,
            "weight_avg": close,
            "fd_amount": [10.0] * n,
            "first_time": ["093000"] * n,
            "last_time": ["145500"] * n,
            "up_stat": ["1/2"] * n,
            "index_2000_close": [1000.0 + i for i in range(n)],
        }
    )
    return frame


class FactorFormulaModuleTests(unittest.TestCase):
    def test_midpoint_and_midprice_use_window_extremes(self):
        series = pd.Series([1.0, 3.0, 2.0])
        self.assertEqual(midpoint(series, length=3).iloc[-1], 2.0)
        high = pd.Series([10.0, 12.0, 11.0])
        low = pd.Series([8.0, 9.0, 7.0])
        self.assertEqual(midprice(high, low, length=3).iloc[-1], 9.5)

    def test_group_factor_eng_uses_correct_future_offsets_and_vwap(self):
        result = group_factor_eng(_sample_group())

        self.assertEqual(result.loc[0, "post6_close"], result.loc[6, "close"])
        self.assertNotEqual(result.loc[0, "post6_close"], result.loc[0, "post5_close"])
        self.assertAlmostEqual(result.loc[0, "6d_yield_rate"], (result.loc[6, "close"] - result.loc[0, "close"]) / result.loc[0, "close"])
        self.assertAlmostEqual(result.loc[0, "open_yield_rate"], (result.loc[1, "open"] - result.loc[0, "close"]) / result.loc[0, "close"])
        self.assertAlmostEqual(result.loc[0, "next_open_yield_rate"], result.loc[0, "open_yield_rate"])
        expected_vwap_proxy = (
            result.loc[0, "open"] + result.loc[0, "high"] + result.loc[0, "low"] + result.loc[0, "close"]
        ) / 4
        self.assertAlmostEqual(result.loc[0, "_ts_vwap"], expected_vwap_proxy)

    def test_imbalance_does_not_create_infinite_values_when_high_equals_low(self):
        frame = _sample_group()
        frame.loc[5, ["high", "low", "close"]] = 15.0
        result = group_factor_eng(frame)

        self.assertTrue(math.isfinite(result.loc[5, "_ts_imbalance"]))


if __name__ == "__main__":
    unittest.main()
