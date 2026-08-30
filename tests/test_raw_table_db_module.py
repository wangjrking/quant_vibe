import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw_table_db_module import raw_table_db_path, replace_raw_table_trade_range


class RawTableDbModuleTests(unittest.TestCase):
    def test_raw_table_db_path_defaults_to_duckdb_table_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            result = raw_table_db_path(data_dir, "adj_factor")

            self.assertEqual(
                result,
                data_dir / "production_assets" / "duckdb" / "l1_raw_tables" / "adj_factor.duckdb",
            )

    def test_raw_table_db_path_can_roll_back_to_legacy_odb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with patch.dict(os.environ, {"QUANT_RAW_DB_MODE": "legacy_odb", "QUANT_ALLOW_LEGACY_RAW_SQLITE": "1"}):
                result = raw_table_db_path(data_dir, "adj_factor")

            self.assertEqual(result, data_dir / "odb.db")

    def test_replace_raw_table_trade_range_writes_duckdb_table_file_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            frame = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": ["20260616", "20260616"],
                    "adj_factor": [1.0, 2.0],
                }
            )

            target_db = replace_raw_table_trade_range(data_dir, "adj_factor", frame, "20260616", "20260616")

            self.assertEqual(
                target_db,
                data_dir / "production_assets" / "duckdb" / "l1_raw_tables" / "adj_factor.duckdb",
            )
            self.assertFalse((data_dir / "odb.db").exists())
            self.assertFalse((data_dir / "raw_table_dbs" / "adj_factor.DB").exists())
            with duckdb.connect(str(target_db), read_only=True) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM adj_factor").fetchone()[0]
            self.assertEqual(rows, 2)

    def test_raw_table_db_path_can_roll_back_to_split_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with patch.dict(os.environ, {"QUANT_RAW_DB_MODE": "split", "QUANT_ALLOW_LEGACY_RAW_SQLITE": "1"}):
                result = raw_table_db_path(data_dir, "adj_factor")

            self.assertEqual(result, data_dir / "raw_table_dbs" / "adj_factor.DB")

    def test_raw_table_db_path_rejects_legacy_sqlite_without_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with patch.dict(os.environ, {"QUANT_RAW_DB_MODE": "split"}, clear=False):
                with self.assertRaises(ValueError):
                    raw_table_db_path(data_dir, "adj_factor")


if __name__ == "__main__":
    unittest.main()
