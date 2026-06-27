import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_daily_data_route import (
    connect_stock_daily_readonly,
    resolve_legacy_mixed_db_path,
    resolve_stock_daily_db_path,
)


class StockDailyDataRouteTests(unittest.TestCase):
    def test_default_l2_route_uses_dedicated_stock_daily_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            result = resolve_stock_daily_db_path(data_dir=data_dir)

            self.assertEqual(result, data_dir / "STOCK_DAILY_DATA.db")

    def test_environment_override_can_point_to_explicit_l2_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom_stock_daily.db"

            with patch.dict(os.environ, {"QUANT_STOCK_DAILY_DB": str(target)}):
                result = resolve_stock_daily_db_path()

            self.assertEqual(result, target)

    def test_legacy_mixed_db_path_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            result = resolve_legacy_mixed_db_path(data_dir=data_dir)

            self.assertEqual(result, data_dir / "odb.db")

    def test_readonly_connection_reads_stock_daily_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "STOCK_DAILY_DATA.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA(stock_code TEXT, trade_date TEXT)")
                conn.execute("INSERT INTO STOCK_DAILY_DATA VALUES ('000001.SZ', '20260616')")

            with closing(connect_stock_daily_readonly(db_path=db_path)) as conn:
                row = conn.execute("SELECT COUNT(*) FROM STOCK_DAILY_DATA").fetchone()

            self.assertEqual(row[0], 1)


if __name__ == "__main__":
    unittest.main()
