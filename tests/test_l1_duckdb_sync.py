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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from l1_duckdb_sync import sync_raw_table_split_full_to_duckdb
from raw_table_db_module import replace_raw_table_full, replace_raw_table_trade_range


class L1DuckDBSyncTests(unittest.TestCase):
    def test_replace_full_writes_duckdb_table_file_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            duckdb_root = data_dir / "duckdb"
            frame = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": ["20260628", "20260628"],
                    "close": [10.0, 20.0],
                }
            )

            with patch.dict(os.environ, {"QUANT_L1_RAW_DUCKDB": str(duckdb_root / "l1_raw_tables")}):
                duckdb_path = replace_raw_table_full(data_dir, "daily_data", frame)
            with duckdb.connect(str(duckdb_root / "l1_raw_tables" / "daily_data.duckdb"), read_only=True) as conn:
                duck_rows = conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]

            self.assertEqual(duckdb_path, duckdb_root / "l1_raw_tables" / "daily_data.duckdb")
            self.assertFalse((data_dir / "raw_table_dbs" / "daily_data.DB").exists())
            self.assertEqual(duck_rows, 2)

    def test_replace_full_can_sync_split_rollback_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            duckdb_root = data_dir / "duckdb"
            frame = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": ["20260628", "20260628"],
                    "close": [10.0, 20.0],
                }
            )

            with patch.dict(
                os.environ,
                {
                    "QUANT_L1_RAW_DUCKDB": str(duckdb_root / "l1_raw_tables"),
                    "QUANT_RAW_DB_MODE": "split",
                },
            ):
                split_path = replace_raw_table_full(data_dir, "daily_data", frame)

            with closing(sqlite3.connect(split_path)) as conn:
                sqlite_rows = conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]
            with duckdb.connect(str(duckdb_root / "l1_raw_tables" / "daily_data.duckdb"), read_only=True) as conn:
                duck_rows = conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]

            self.assertEqual(sqlite_rows, 2)
            self.assertEqual(duck_rows, 2)

    def test_replace_trade_range_syncs_duckdb_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            duckdb_root = data_dir / "duckdb"
            initial = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20260627"],
                    "adj_factor": [1.0],
                }
            )
            replacement = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": ["20260628", "20260628"],
                    "adj_factor": [1.1, 2.2],
                }
            )

            with patch.dict(os.environ, {"QUANT_L1_RAW_DUCKDB": str(duckdb_root / "l1_raw_tables")}):
                replace_raw_table_full(data_dir, "adj_factor", initial)
                replace_raw_table_trade_range(
                    data_dir,
                    "adj_factor",
                    replacement,
                    "20260628",
                    "20260628",
                )

            with duckdb.connect(str(duckdb_root / "l1_raw_tables" / "adj_factor.duckdb"), read_only=True) as conn:
                total_rows = conn.execute("SELECT COUNT(*) FROM adj_factor").fetchone()[0]
                target_rows = conn.execute(
                    "SELECT COUNT(*) FROM adj_factor WHERE trade_date = '20260628'"
                ).fetchone()[0]

            self.assertEqual(total_rows, 3)
            self.assertEqual(target_rows, 2)

    def test_replace_trade_range_syncs_daily_data_duckdb_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            duckdb_root = data_dir / "duckdb"
            frame = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": ["20260628", "20260628"],
                    "close": [10.0, 20.0],
                }
            )

            with patch.dict(os.environ, {"QUANT_L1_RAW_DUCKDB": str(duckdb_root / "l1_raw_tables")}):
                replace_raw_table_trade_range(data_dir, "daily_data", frame, "20260628", "20260628")

            duckdb_path = duckdb_root / "l1_raw_tables" / "daily_data.duckdb"
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                duck_rows = conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]
                target_rows = conn.execute(
                    "SELECT COUNT(*) FROM daily_data WHERE trade_date = '20260628'"
                ).fetchone()[0]

            self.assertEqual(duck_rows, 2)
            self.assertEqual(target_rows, 2)

    def test_sync_split_full_to_duckdb_filters_bj_rows_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            duckdb_root = data_dir / "duckdb"
            split_root = data_dir / "raw_table_dbs"
            split_root.mkdir(parents=True)
            split_path = split_root / "finan_data_season.DB"
            with closing(sqlite3.connect(split_path)) as conn:
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ", "920001.BJ"],
                        "ann_date": ["20260630", "20260630"],
                        "end_date": ["20260331", "20260331"],
                        "update_flag": ["1", "1"],
                    }
                ).to_sql("finan_data_season", conn, if_exists="replace", index=False)
                conn.commit()

            with patch.dict(os.environ, {"QUANT_L1_RAW_DUCKDB": str(duckdb_root / "l1_raw_tables")}):
                duckdb_path = sync_raw_table_split_full_to_duckdb(data_dir, "finan_data_season")

            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                rows = conn.execute("SELECT ts_code FROM finan_data_season ORDER BY ts_code").fetchall()

            self.assertEqual(rows, [("000001.SZ",)])


if __name__ == "__main__":
    unittest.main()
