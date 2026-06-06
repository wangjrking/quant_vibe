import unittest

import pandas as pd

from market_filter_module import build_market_filter, build_multi_index_market_filter, merge_market_filter


class MarketFilterModuleTests(unittest.TestCase):
    def test_build_market_filter_uses_index_ma(self):
        index_data = pd.DataFrame(
            {
                "ts_code": ["IDX"] * 4,
                "trade_date": ["20260101", "20260102", "20260103", "20260104"],
                "close": [10.0, 11.0, 12.0, 10.0],
            }
        )

        result = build_market_filter(index_data, index_code="IDX", ma_window=2)

        self.assertEqual(result.iloc[0]["market_ok"], 0)
        self.assertEqual(result.iloc[1]["market_ok"], 1)
        self.assertEqual(result.iloc[-1]["market_ok"], 0)

    def test_merge_market_filter_defaults_missing_dates_to_off(self):
        rows = [{"trade_date": "20260101"}, {"trade_date": "20260103"}]
        market_filter = pd.DataFrame({"trade_date": ["20260101"], "market_ok": [1]})

        merged = merge_market_filter(rows, market_filter)

        self.assertEqual(merged[0]["market_ok"], 1)
        self.assertEqual(merged[1]["market_ok"], 0)

    def test_build_multi_index_market_filter_requires_all_indexes_by_default(self):
        index_data = pd.DataFrame(
            {
                "ts_code": ["A", "A", "B", "B"],
                "trade_date": ["20260101", "20260102", "20260101", "20260102"],
                "close": [10.0, 11.0, 10.0, 9.0],
            }
        )

        result = build_multi_index_market_filter(index_data, ["A", "B"], ma_window=2)

        self.assertEqual(result.iloc[-1]["market_ok"], 0)

    def test_build_multi_index_market_filter_can_use_any_mode(self):
        index_data = pd.DataFrame(
            {
                "ts_code": ["A", "A", "B", "B"],
                "trade_date": ["20260101", "20260102", "20260101", "20260102"],
                "close": [10.0, 11.0, 10.0, 9.0],
            }
        )

        result = build_multi_index_market_filter(index_data, ["A", "B"], ma_window=2, mode="any")

        self.assertEqual(result.iloc[-1]["market_ok"], 1)


if __name__ == "__main__":
    unittest.main()
