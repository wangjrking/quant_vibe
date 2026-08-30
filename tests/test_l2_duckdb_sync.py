import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l2_duckdb_sync import (
    count_stock_daily_duckdb_rows,
    sync_stock_daily_duckdb_table_full_to_duckdb,
    sync_stock_daily_frame_to_duckdb,
    sync_stock_daily_trade_range_to_duckdb,
)


class L2DuckDBSyncTests(unittest.TestCase):
    def _seed_sqlite(self, path: Path, rows: list[tuple[str, str, float]]) -> None:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE STOCK_DAILY_DATA(stock_code TEXT, trade_date TEXT, close REAL)")
            conn.executemany("INSERT INTO STOCK_DAILY_DATA VALUES (?, ?, ?)", rows)
            conn.commit()

    def test_sync_creates_duckdb_table_from_sqlite_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            sqlite_path = data_dir / "STOCK_DAILY_DATA.db"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._seed_sqlite(
                sqlite_path,
                [
                    ("000001.SZ", "20260627", 10.0),
                    ("000002.SZ", "20260627", 20.0),
                    ("000003.SZ", "20260628", 30.0),
                ],
            )

            result = sync_stock_daily_trade_range_to_duckdb(
                data_dir,
                "20260627",
                "20260627",
                sqlite_db_path=sqlite_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            self.assertEqual(result["target_total_rows"], 2)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                total_rows = conn.execute("SELECT COUNT(*) FROM STOCK_DAILY_DATA").fetchone()[0]
                target_rows = conn.execute(
                    "SELECT COUNT(*) FROM STOCK_DAILY_DATA WHERE trade_date = '20260627'"
                ).fetchone()[0]
            self.assertEqual(total_rows, 2)
            self.assertEqual(target_rows, 2)

    def test_sync_replaces_target_range_without_dropping_other_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            sqlite_path = data_dir / "STOCK_DAILY_DATA.db"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._seed_sqlite(
                sqlite_path,
                [
                    ("000001.SZ", "20260627", 10.0),
                    ("000002.SZ", "20260628", 20.0),
                ],
            )
            sync_stock_daily_trade_range_to_duckdb(
                data_dir,
                "20260627",
                "20260628",
                sqlite_db_path=sqlite_path,
                duckdb_path=duckdb_path,
            )

            with sqlite3.connect(sqlite_path) as conn:
                conn.execute("DELETE FROM STOCK_DAILY_DATA WHERE trade_date = '20260628'")
                conn.executemany(
                    "INSERT INTO STOCK_DAILY_DATA VALUES (?, ?, ?)",
                    [
                        ("000003.SZ", "20260628", 30.0),
                        ("000004.SZ", "20260628", 40.0),
                    ],
                )
                conn.commit()

            result = sync_stock_daily_trade_range_to_duckdb(
                data_dir,
                "20260628",
                "20260628",
                sqlite_db_path=sqlite_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            self.assertEqual(result["target_total_rows"], 3)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                old_rows = conn.execute(
                    "SELECT COUNT(*) FROM STOCK_DAILY_DATA WHERE trade_date = '20260627'"
                ).fetchone()[0]
                new_rows = conn.execute(
                    "SELECT COUNT(*) FROM STOCK_DAILY_DATA WHERE trade_date = '20260628'"
                ).fetchone()[0]
            self.assertEqual(old_rows, 1)
            self.assertEqual(new_rows, 2)

    def test_frame_sync_replaces_target_range_without_sqlite_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA(stock_code VARCHAR, trade_date VARCHAR, close DOUBLE)")
                conn.executemany(
                    "INSERT INTO STOCK_DAILY_DATA VALUES (?, ?, ?)",
                    [
                        ("000001.SZ", "20260627", 10.0),
                        ("000002.SZ", "20260628", 20.0),
                    ],
                )

            frame = pd.DataFrame(
                [
                    {"stock_code": "000003.SZ", "trade_date": "20260628", "close": 30.0},
                    {"stock_code": "000004.SZ", "trade_date": "20260628", "close": 40.0},
                ]
            )

            result = sync_stock_daily_frame_to_duckdb(
                data_dir,
                frame,
                "20260628",
                "20260628",
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            self.assertEqual(result["target_total_rows"], 3)
            self.assertEqual(result["source_mode"], "frame")
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                old_rows = conn.execute(
                    "SELECT COUNT(*) FROM STOCK_DAILY_DATA WHERE trade_date = '20260627'"
                ).fetchone()[0]
                new_codes = conn.execute(
                    "SELECT stock_code FROM STOCK_DAILY_DATA WHERE trade_date = '20260628' ORDER BY stock_code"
                ).fetchall()
            self.assertEqual(old_rows, 1)
            self.assertEqual(new_codes, [("000003.SZ",), ("000004.SZ",)])

    def test_frame_sync_retypes_numeric_blob_columns_before_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA(stock_code VARCHAR, trade_date VARCHAR, open BLOB)")
                conn.execute(
                    "INSERT INTO STOCK_DAILY_DATA VALUES ('000001.SZ', '20260627', CAST('10.5' AS BLOB))"
                )

            frame = pd.DataFrame(
                [{"stock_code": "000002.SZ", "trade_date": "20260628", "open": 20.5}]
            )

            result = sync_stock_daily_frame_to_duckdb(
                data_dir,
                frame,
                "20260628",
                "20260628",
                duckdb_path=duckdb_path,
            )

            self.assertEqual(result["target_rows"], 1)
            with duckdb.connect(str(duckdb_path), read_only=True) as conn:
                open_type = conn.execute("DESCRIBE STOCK_DAILY_DATA").fetchall()[2][1]
                values = conn.execute(
                    "SELECT stock_code, open FROM STOCK_DAILY_DATA ORDER BY stock_code"
                ).fetchall()
            self.assertEqual(open_type, "DOUBLE")
            self.assertEqual(values, [("000001.SZ", 10.5), ("000002.SZ", 20.5)])

    def test_count_rows_reads_duckdb_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            sqlite_path = data_dir / "STOCK_DAILY_DATA.db"
            duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._seed_sqlite(sqlite_path, [("000001.SZ", "20260629", 50.0)])
            sync_stock_daily_trade_range_to_duckdb(
                data_dir,
                "20260629",
                "20260629",
                sqlite_db_path=sqlite_path,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(
                count_stock_daily_duckdb_rows(data_dir, duckdb_path=duckdb_path),
                1,
            )

    def test_full_sync_copies_stock_daily_table_between_duckdb_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data_file"
            source_duckdb_path = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            target_duckdb_path = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            source_duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(source_duckdb_path)) as conn:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA(stock_code VARCHAR, trade_date VARCHAR, close DOUBLE)")
                conn.executemany(
                    "INSERT INTO STOCK_DAILY_DATA VALUES (?, ?, ?)",
                    [
                        ("000001.SZ", "20260629", 10.0),
                        ("000002.SZ", "20260630", 20.0),
                    ],
                )

            result = sync_stock_daily_duckdb_table_full_to_duckdb(
                data_dir,
                source_duckdb_path=source_duckdb_path,
                target_duckdb_path=target_duckdb_path,
            )

            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["target_rows"], 2)
            self.assertEqual(result["min_trade_date"], "20260629")
            self.assertEqual(result["max_trade_date"], "20260630")
            with duckdb.connect(str(target_duckdb_path), read_only=True) as conn:
                rows = conn.execute(
                    "SELECT stock_code, trade_date, close FROM STOCK_DAILY_DATA ORDER BY trade_date, stock_code"
                ).fetchall()
            self.assertEqual(
                rows,
                [("000001.SZ", "20260629", 10.0), ("000002.SZ", "20260630", 20.0)],
            )


if __name__ == "__main__":
    unittest.main()
