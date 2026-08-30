import os
import sqlite3
import sys
import tempfile
import unittest
import json
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_daily_data_route import (
    connect_stock_daily_readonly,
    resolve_stock_daily_backend,
    resolve_stock_daily_duckdb_path,
    resolve_legacy_mixed_db_path,
    resolve_stock_daily_db_path,
)


class StockDailyDataRouteTests(unittest.TestCase):
    def test_default_sqlite_route_is_disabled_without_explicit_legacy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with self.assertRaisesRegex(RuntimeError, "Implicit STOCK_DAILY_DATA\\.db fallback is disabled"):
                resolve_stock_daily_db_path(data_dir=data_dir)

    def test_backend_defaults_to_duckdb_without_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            self.assertEqual(resolve_stock_daily_backend(data_dir=data_dir), "duckdb")

    def test_backend_follows_registry_when_l2_mainline_is_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l2_duckdb",
                                "layer": "L2_stock_daily_base",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb::STOCK_DAILY_DATA",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(resolve_stock_daily_backend(data_dir=data_dir), "duckdb")

    def test_deprecated_stock_daily_backend_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with self.assertRaisesRegex(ValueError, "deprecated stock daily backend alias"):
                resolve_stock_daily_backend("sqlite", data_dir=data_dir)

    def test_default_duckdb_route_uses_production_duckdb_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            result = resolve_stock_daily_duckdb_path(data_dir=data_dir)

            self.assertEqual(result, data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb")

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

            with patch.dict(os.environ, {"QUANT_ALLOW_LEGACY_SQLITE_PARQUET_ROLLBACK": "1"}, clear=False):
                conn = connect_stock_daily_readonly(db_path=db_path, backend="legacy")
                with closing(conn):
                    row = conn.execute("SELECT COUNT(*) FROM STOCK_DAILY_DATA").fetchone()

            self.assertEqual(row[0], 1)

    def test_legacy_sqlite_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "STOCK_DAILY_DATA.db"
            db_path.write_bytes(b"")

            with patch.dict(os.environ, {"QUANT_ALLOW_LEGACY_SQLITE_PARQUET_ROLLBACK": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "explicit legacy SQLite/Parquet backend selection requires opt-in"):
                    connect_stock_daily_readonly(db_path=db_path, backend="legacy")

    def test_explicit_duckdb_selection_requires_duckdb_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with self.assertRaisesRegex(FileNotFoundError, "l2_stock_daily_data.duckdb"):
                connect_stock_daily_readonly(data_dir=data_dir, backend="duckdb")


if __name__ == "__main__":
    unittest.main()
