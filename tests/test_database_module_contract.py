import re
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database_module import sql


class DatabaseModuleContractTests(unittest.TestCase):
    def test_stock_daily_sql_exposes_explicit_qfq_market_price_columns(self):
        for alias in ("open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"):
            self.assertRegex(
                sql,
                rf"\bAS\s+{alias}\b|\bas\s+{alias}\b",
                msg=f"STOCK_DAILY_DATA build SQL must expose explicit qfq field: {alias}",
            )

    def test_stock_daily_sql_keeps_bare_market_prices_as_raw_columns(self):
        expected = {
            "open": r"\bT1\.open\s+AS\s+open\b",
            "high": r"\bT1\.high\s+AS\s+high\b",
            "low": r"\bT1\.low\s+AS\s+low\b",
            "close": r"\bT1\.close\s+AS\s+close\b",
            "pre_close": r"\bT1\.pre_close\s+AS\s+pre_close\b",
        }
        for alias, pattern in expected.items():
            self.assertRegex(
                sql,
                pattern,
                msg=f"STOCK_DAILY_DATA build SQL must keep bare {alias} as raw market price",
            )


if __name__ == "__main__":
    unittest.main()
