import sqlite3
import sys
import tempfile
import unittest
import shutil
import time
from pathlib import Path
from unittest.mock import patch
import json

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l1_raw_data_route import (
    BACKEND_DUCKDB,
    BACKEND_SPLIT,
    materialize_sqlite_temp_raw_table,
    read_raw_table_frame,
    resolve_l1_raw_backend,
    resolve_l1_raw_duckdb_path,
)


class L1RawDataRouteTests(unittest.TestCase):
    def test_backend_defaults_to_duckdb_without_registry(self):
        with patch.dict("os.environ", {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(resolve_l1_raw_backend(data_dir=tmp), BACKEND_DUCKDB)

    def test_backend_follows_registry_when_l1_mainline_is_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l1_duckdb",
                                "layer": "L1_raw_data",
                                "asset_type": "duckdb_table_files",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": "quant/data_file/production_assets/duckdb/l1_raw_tables/",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(resolve_l1_raw_backend(data_dir=data_dir), BACKEND_DUCKDB)

    def test_duckdb_route_uses_table_isolated_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            result = resolve_l1_raw_duckdb_path("daily_data", data_dir=data_dir)

            self.assertEqual(
                result,
                data_dir / "production_assets" / "duckdb" / "l1_raw_tables" / "daily_data.duckdb",
            )

    def test_read_raw_table_frame_reads_split_sqlite(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            data_dir = tmp
            raw_dir = data_dir / "raw_table_dbs"
            raw_dir.mkdir()
            db_path = raw_dir / "daily_data.DB"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE daily_data(ts_code TEXT, trade_date TEXT)")
                conn.execute("INSERT INTO daily_data VALUES ('000001.SZ', '20260628')")

            frame = read_raw_table_frame(data_dir, "daily_data", backend=BACKEND_SPLIT)

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["ts_code"], "000001.SZ")
        finally:
            time.sleep(0.05)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_read_raw_table_frame_reads_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            duckdb_path = data_dir / "l1_raw.duckdb"
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute("CREATE TABLE daily_data(ts_code TEXT, trade_date TEXT)")
                conn.execute("INSERT INTO daily_data VALUES ('000002.SZ', '20260628')")

            frame = read_raw_table_frame(
                data_dir,
                "daily_data",
                backend=BACKEND_DUCKDB,
                duckdb_path=duckdb_path,
            )

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["ts_code"], "000002.SZ")

    def test_materialize_sqlite_temp_raw_table_from_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            duckdb_path = data_dir / "l1_raw.duckdb"
            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute("CREATE TABLE daily_data(ts_code TEXT, trade_date TEXT)")
                conn.execute("INSERT INTO daily_data VALUES ('000003.SZ', '20260628')")

            with sqlite3.connect(":memory:") as target:
                materialize_sqlite_temp_raw_table(
                    target,
                    temp_name="daily_data",
                    table_name="daily_data",
                    data_dir=data_dir,
                    backend=BACKEND_DUCKDB,
                    duckdb_path=duckdb_path,
                )
                row = target.execute("SELECT COUNT(*), MIN(ts_code) FROM temp.daily_data").fetchone()

            self.assertEqual(row[0], 1)
            self.assertEqual(row[1], "000003.SZ")


if __name__ == "__main__":
    unittest.main()
