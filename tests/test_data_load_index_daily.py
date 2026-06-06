import unittest

import pandas as pd

from data_load_module import get_index_daily


class FakePro:
    def __init__(self):
        self.calls = []

    def query(self, name, ts_code, start_date, end_date):
        self.calls.append((name, ts_code, start_date, end_date))
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": [start_date], "close": [1.0]})


class DataLoadIndexDailyTests(unittest.TestCase):
    def test_get_index_daily_uses_requested_index_codes(self):
        pro = FakePro()

        result = get_index_daily("20260101", "20260102", pro, ["000300.SH", "000905.SH"])

        self.assertEqual([call[1] for call in pro.calls], ["000300.SH", "000905.SH"])
        self.assertEqual(result["ts_code"].tolist(), ["000300.SH", "000905.SH"])


if __name__ == "__main__":
    unittest.main()
