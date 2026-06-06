import tempfile
import unittest
from pathlib import Path

import pandas as pd

from raw_data_update_module import append_parquet_dedup, load_existing_stock_codes, next_start_date


class RawDataUpdateModuleTests(unittest.TestCase):
    def test_append_parquet_dedup_keeps_latest_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daily.parquet"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": ["20260101", "20260101"],
                    "close": [10.0, 20.0],
                }
            ).to_parquet(path, index=False)

            append_parquet_dedup(
                path,
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ", "000003.SZ"],
                        "trade_date": ["20260101", "20260102"],
                        "close": [11.0, 30.0],
                    }
                ),
                subset=["ts_code", "trade_date"],
            )

            result = pd.read_parquet(path).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ", "000002.SZ", "000003.SZ"])
        self.assertEqual(result.loc[0, "close"], 11.0)

    def test_next_start_date_returns_day_after_max_trade_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daily.parquet"
            pd.DataFrame({"trade_date": ["20260101", "20260103"]}).to_parquet(path, index=False)

            result = next_start_date(path, default_start="20100101")

        self.assertEqual(result, "20260104")

    def test_next_start_date_uses_default_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = next_start_date(Path(tmp) / "missing.parquet", default_start="20100101")

        self.assertEqual(result, "20100101")

    def test_load_existing_stock_codes_reads_unique_ts_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daily.parquet"
            pd.DataFrame({"ts_code": ["000001.SZ", "000001.SZ", "600000.SH"]}).to_parquet(path, index=False)

            result = load_existing_stock_codes(path)

        self.assertEqual(result, {"000001.SZ", "600000.SH"})


if __name__ == "__main__":
    unittest.main()
