import unittest
import sys
import types

import pandas  # noqa: F401
import pandas as pd

pyarrow_stub = types.ModuleType("pyarrow")
pyarrow_stub.parquet = types.ModuleType("pyarrow.parquet")
sys.modules.setdefault("pyarrow", pyarrow_stub)
sys.modules.setdefault("pyarrow.parquet", types.ModuleType("pyarrow.parquet"))
gtja_stub = types.ModuleType("gtja_alpha_workflow")
gtja_stub.GTJA_ALPHA_COLUMNS = ["gtja_alpha001"]
sys.modules.setdefault("gtja_alpha_workflow", gtja_stub)
from build_production_factor_parts import (
    KEY_COLUMNS,
    apply_industry_encode,
    extend_industry_encode_mapping,
    production_raw_columns,
)


class BuildProductionFactorPartsTests(unittest.TestCase):
    def test_production_raw_columns_excludes_source_limited_features(self):
        raw_columns = [
            *KEY_COLUMNS,
            "custom_factor",
            "turnover_rate",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "amount",
            "vol",
            "winner_rate",
            "cost_5pct",
            "std_cost_95pct",
            "his_high",
            "fd_amount",
            "limit_times",
            "first_time_int",
            "up_stat_rate",
            "st_type",
            "st_code",
            "ths_hot",
            "ths_rank",
            "dc_rank",
            "top_list",
            "post_open",
            "10d_yield_rate",
            "gtja_alpha001",
        ]

        selected = production_raw_columns(raw_columns)

        self.assertIn("custom_factor", selected)
        self.assertIn("turnover_rate", selected)
        self.assertIn("open", selected)
        self.assertIn("high", selected)
        self.assertIn("low", selected)
        self.assertIn("close", selected)
        self.assertIn("pre_close", selected)
        self.assertIn("amount", selected)
        self.assertIn("vol", selected)
        self.assertIn("gtja_alpha001", selected)
        self.assertNotIn("winner_rate", selected)
        self.assertNotIn("cost_5pct", selected)
        self.assertNotIn("std_cost_95pct", selected)
        self.assertNotIn("his_high", selected)
        self.assertNotIn("fd_amount", selected)
        self.assertNotIn("limit_times", selected)
        self.assertNotIn("first_time_int", selected)
        self.assertNotIn("up_stat_rate", selected)
        self.assertNotIn("st_type", selected)
        self.assertNotIn("st_code", selected)
        self.assertNotIn("ths_hot", selected)
        self.assertNotIn("ths_rank", selected)
        self.assertNotIn("dc_rank", selected)
        self.assertNotIn("top_list", selected)
        self.assertNotIn("post_open", selected)
        self.assertNotIn("10d_yield_rate", selected)

    def test_extend_industry_encode_mapping_only_appends_missing_values(self):
        mapping = {"": 0, "Bank": 1, "Tech": 2}

        updated = extend_industry_encode_mapping(mapping, ["Tech", "Energy", None, "Bank", "Auto"])

        self.assertEqual(updated[""], 0)
        self.assertEqual(updated["Bank"], 1)
        self.assertEqual(updated["Tech"], 2)
        self.assertEqual(updated["Auto"], 3)
        self.assertEqual(updated["Energy"], 4)

    def test_apply_industry_encode_uses_stable_mapping(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260618", "20260618", "20260618"],
                "stock_code": ["A", "B", "C"],
                "industry": ["Tech", None, "Bank"],
            }
        )

        result = apply_industry_encode(frame, {"": 0, "Bank": 1, "Tech": 2})

        self.assertEqual(result["industry"].tolist(), ["Tech", "", "Bank"])
        self.assertEqual(result["industry_encode"].tolist(), [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
