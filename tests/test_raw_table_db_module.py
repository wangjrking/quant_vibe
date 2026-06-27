import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw_table_db_module import raw_table_db_path, replace_raw_table_trade_range


class RawTableDbModuleTests(unittest.TestCase):
    def test_raw_table_db_path_defaults_to_split_table_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            result = raw_table_db_path(data_dir, "adj_factor")

            self.assertEqual(result, data_dir / "raw_table_dbs" / "adj_factor.DB")

    def test_raw_table_db_path_can_roll_back_to_legacy_odb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with patch.dict(os.environ, {"QUANT_RAW_DB_MODE": "legacy_odb"}):
                result = raw_table_db_path(data_dir, "adj_factor")

            self.assertEqual(result, data_dir / "odb.db")

    def test_replace_raw_table_trade_range_writes_only_target_table_db(self):
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

            self.assertEqual(target_db, data_dir / "raw_table_dbs" / "adj_factor.DB")
            self.assertFalse((data_dir / "odb.db").exists())
            with closing(sqlite3.connect(target_db)) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM adj_factor").fetchone()[0]
            self.assertEqual(rows, 2)


if __name__ == "__main__":
    unittest.main()
