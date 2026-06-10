import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_pool_module import build_all_a_stock_pool, filter_frame_by_stock_pool, load_stock_pool


class StockPoolModuleTests(unittest.TestCase):
    def test_load_stock_pool_deduplicates_and_excludes_bj(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            pd.DataFrame(
                {
                    "stock_code": [
                        "000001.SZ",
                        "000001.SZ",
                        "600000.SH",
                        "430001.BJ",
                        "",
                    ]
                }
            ).to_csv(path, index=False)

            pool = load_stock_pool(path)

        self.assertEqual(pool, {"000001.SZ", "600000.SH"})

    def test_filter_frame_by_stock_pool_keeps_matching_codes(self):
        frame = pd.DataFrame(
            {
                "stock_code": ["000001.SZ", "600000.SH", "000002.SZ"],
                "value": [1, 2, 3],
            }
        )

        result = filter_frame_by_stock_pool(frame, {"000001.SZ", "000002.SZ"})

        self.assertEqual(result["stock_code"].tolist(), ["000001.SZ", "000002.SZ"])

    def test_build_all_a_stock_pool_can_include_or_exclude_bj(self):
        frame = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600000.SH", "688001.SH", "430001.BJ", "BAD"],
                "market": ["主板", "主板", "科创板", "北交所", "其他"],
                "name": ["平安银行", "浦发银行", "科创测试", "北交测试", "坏数据"],
            }
        )

        include_bj = build_all_a_stock_pool(frame, include_bj=True)
        exclude_bj = build_all_a_stock_pool(frame, include_bj=False)

        self.assertEqual(include_bj, ["000001.SZ", "430001.BJ", "600000.SH", "688001.SH"])
        self.assertEqual(exclude_bj, ["000001.SZ", "600000.SH", "688001.SH"])


if __name__ == "__main__":
    unittest.main()
