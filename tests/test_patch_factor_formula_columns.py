import tempfile
import unittest
from pathlib import Path

import pandas as pd

from patch_factor_formula_columns import patch_factor_columns


class PatchFactorFormulaColumnsTests(unittest.TestCase):
    def test_patch_factor_columns_repairs_deterministic_columns(self):
        frame = pd.DataFrame(
            {
                "stock_code": ["A"] * 20,
                "trade_date": [f"202601{i + 1:02d}" for i in range(20)],
                "close": [10.0 + i for i in range(20)],
                "open": [9.5 + i for i in range(20)],
                "high": [10.5 + i for i in range(20)],
                "low": [9.0 + i for i in range(20)],
                "post_open": [9.5 + i + 1 for i in range(20)],
                "post6_close": [0.0] * 20,
                "6d_yield_rate": [0.0] * 20,
                "open_yield_rate": [0.0] * 20,
                "midpoint": [0.0] * 20,
                "midprice": [0.0] * 20,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "factor.parquet"
            frame.to_parquet(path, index=False)
            result = patch_factor_columns(path)

        self.assertEqual(result.loc[0, "post6_close"], result.loc[6, "close"])
        self.assertAlmostEqual(result.loc[0, "open_yield_rate"], (result.loc[0, "post_open"] - result.loc[0, "close"]) / result.loc[0, "close"])
        self.assertAlmostEqual(result.loc[13, "midpoint"], (result.loc[0:13, "close"].max() + result.loc[0:13, "close"].min()) / 2)
        self.assertAlmostEqual(result.loc[13, "midprice"], (result.loc[0:13, "high"].max() + result.loc[0:13, "low"].min()) / 2)


if __name__ == "__main__":
    unittest.main()
