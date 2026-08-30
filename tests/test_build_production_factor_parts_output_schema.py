import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from build_production_factor_parts import build_production_part, production_raw_columns


class BuildProductionFactorPartsOutputSchemaTests(unittest.TestCase):
    def test_build_production_part_prefers_explicit_qfq_columns(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_parts_dir = root / "raw_parts"
            gtja_parts_dir = root / "gtja_parts"
            output_dir = root / "production_parts"
            raw_parts_dir.mkdir()
            gtja_parts_dir.mkdir()

            raw_part = raw_parts_dir / "raw_part_0000.parquet"
            gtja_part = gtja_parts_dir / "factor_standard_part_0000.parquet"

            pd.DataFrame(
                [
                    {
                        "trade_date": "20260102",
                        "stock_code": "000001.SZ",
                        "industry": "Bank",
                        "act_ent_type": "A",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "pre_close": 9.9,
                        "open_qfq": 5.0,
                        "high_qfq": 5.2,
                        "low_qfq": 4.9,
                        "close_qfq": 5.1,
                        "pre_close_qfq": 4.95,
                        "atr": 0.8,
                        "atr_qfq": 0.4,
                        "macd": 1.2,
                        "macd_qfq": 0.6,
                        "amount": 1000.0,
                        "vol": 200.0,
                        "turnover_rate": 0.1,
                    }
                ]
            ).to_parquet(raw_part, index=False)

            pd.DataFrame(
                [
                    {
                        "trade_date": "20260102",
                        "stock_code": "000001.SZ",
                        "gtja_alpha001": 0.123,
                    }
                ]
            ).to_parquet(gtja_part, index=False)

            raw_columns = production_raw_columns(pd.read_parquet(raw_part).columns.tolist())
            output_path = build_production_part(
                raw_parts_dir,
                gtja_part,
                output_dir,
                raw_columns,
                {"": 0, "Bank": 1},
                resume=False,
            )

            result = pd.read_parquet(output_path)

            self.assertIn("open_qfq", result.columns)
            self.assertIn("high_qfq", result.columns)
            self.assertIn("low_qfq", result.columns)
            self.assertIn("close_qfq", result.columns)
            self.assertIn("pre_close_qfq", result.columns)
            self.assertIn("atr_qfq", result.columns)
            self.assertIn("macd_qfq", result.columns)
            self.assertNotIn("open", result.columns)
            self.assertNotIn("high", result.columns)
            self.assertNotIn("low", result.columns)
            self.assertNotIn("close", result.columns)
            self.assertNotIn("pre_close", result.columns)
            self.assertNotIn("atr", result.columns)
            self.assertNotIn("macd", result.columns)
            self.assertIn("gtja_alpha001_qfq", result.columns)
            self.assertNotIn("gtja_alpha001", result.columns)
            self.assertIn("industry_encode", result.columns)


if __name__ == "__main__":
    unittest.main()
