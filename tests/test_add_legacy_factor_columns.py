import tempfile
import unittest
from pathlib import Path

import pandas as pd

from add_legacy_factor_columns import add_legacy_columns


class AddLegacyFactorColumnsTests(unittest.TestCase):
    def test_add_legacy_columns_adds_ranked_non_leaky_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            current = tmp / "current.parquet"
            legacy = tmp / "legacy.parquet"
            scores = tmp / "scores.csv"
            pd.DataFrame(
                {
                    "stock_code": ["A"],
                    "trade_date": ["20260101"],
                    "good": [1.0],
                    "10d_yield_rate": [0.2],
                }
            ).to_parquet(current, index=False)
            pd.DataFrame(
                {
                    "stock_code": ["A"],
                    "trade_date": ["20260101"],
                    "good": [9.0],
                    "10d_yield_rate": [0.5],
                }
            ).to_parquet(legacy, index=False)
            scores.write_text(
                "feature,mean_ic,abs_mean_ic\n10d_yield_rate,1,1\ngood,0.5,0.5\n",
                encoding="utf-8-sig",
            )

            added = add_legacy_columns(current, legacy, scores, top_n=2)
            result = pd.read_parquet(current)

        self.assertEqual(added, ["legacy_good"])
        self.assertEqual(result.loc[0, "legacy_good"], 9.0)
        self.assertNotIn("legacy_10d_yield_rate", result.columns)


if __name__ == "__main__":
    unittest.main()
