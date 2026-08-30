import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools.duckdb_migration_gate import check_report
from tools.materialize_duckdb_production_assets import materialize


@unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
class DuckDBMigrationToolsTests(unittest.TestCase):
    def test_materialize_sqlite_and_parquet_assets_and_gate_passes_with_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sqlite_db = root / "source.db"
            parquet_dir = root / "parts"
            report_dir = root / "reports"
            duckdb_root = root / "duckdb"
            audit_record = root / "audit.md"
            approval_audit = root / "approval.md"
            registry = root / "production_assets.json"
            audit_record.write_text("通过", encoding="utf-8")
            approval_audit.write_text("通过", encoding="utf-8")

            conn = sqlite3.connect(sqlite_db)
            try:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA(stock_code TEXT, trade_date TEXT, close REAL)")
                conn.execute("INSERT INTO STOCK_DAILY_DATA VALUES ('000001.SZ', '20260616', 10.5)")
                conn.commit()
            finally:
                conn.close()

            parquet_dir.mkdir()
            pd.DataFrame(
                {
                    "stock_code": ["000001.SZ"],
                    "trade_date": ["20260616"],
                    "factor_a": [1.2],
                }
            ).to_parquet(parquet_dir / "part-000.parquet", index=False)

            registry.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l2_test",
                                "track": "production",
                                "layer": "L2_stock_daily_base",
                                "asset_type": "sqlite_table",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{sqlite_db}::STOCK_DAILY_DATA",
                                "audit_record": str(audit_record),
                            },
                            {
                                "asset_id": "prod_l3_test",
                                "track": "production",
                                "layer": "L3_features",
                                "asset_type": "parquet_parts",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(parquet_dir),
                                "audit_record": str(audit_record),
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"QUANT_DUCKDB_ASSET_ROOT": str(duckdb_root)}):
                report = materialize(
                    registry_path=registry,
                    report_dir=report_dir,
                    batch_size=10,
                    layers=None,
                    dry_run=False,
                )

            self.assertEqual([asset["status"] for asset in report["assets"]], ["ok", "ok"])
            self.assertTrue((duckdb_root / "quant_production.duckdb").exists())

            errors = []
            check_report(report, str(approval_audit), errors)
            self.assertEqual(errors, [])

    def test_gate_fails_without_approval_audit(self):
        report = {
            "dry_run": False,
            "boundary": {
                "edits_production_registry": False,
                "switches_mainline_routes": False,
            },
            "duckdb_files": {},
            "assets": [],
        }
        errors = []

        check_report(report, None, errors)

        self.assertIn("approval audit record is required before DuckDB production switch", errors)

    def test_materialize_l1_raw_split_sqlite_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw_table_dbs"
            raw_dir.mkdir()
            sqlite_db = raw_dir / "daily_data.DB"
            report_dir = root / "reports"
            duckdb_root = root / "duckdb"
            audit_record = root / "audit.md"
            registry = root / "production_assets.json"
            audit_record.write_text("通过", encoding="utf-8")

            conn = sqlite3.connect(sqlite_db)
            try:
                conn.execute("CREATE TABLE daily_data(ts_code TEXT, trade_date TEXT, close REAL)")
                conn.execute("INSERT INTO daily_data VALUES ('000001.SZ', '20260628', 10.5)")
                conn.commit()
            finally:
                conn.close()

            registry.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l1_test",
                                "track": "production",
                                "layer": "L1_raw_data",
                                "asset_type": "raw_split_sqlite_dbs",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(raw_dir),
                                "audit_record": str(audit_record),
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"QUANT_DUCKDB_ASSET_ROOT": str(duckdb_root)}):
                report = materialize(
                    registry_path=registry,
                    report_dir=report_dir,
                    batch_size=10,
                    layers={"L1"},
                    dry_run=False,
                )

            self.assertEqual(report["assets"][0]["status"], "ok")
            self.assertTrue((duckdb_root / "quant_production.duckdb").exists())
            tables = report["assets"][0]["tables"]
            self.assertEqual(len(tables), 1)
            self.assertEqual(tables[0]["source_table"], "daily_data")
            self.assertEqual(tables[0]["target_table"], "daily_data")
            self.assertEqual(tables[0]["target_rows"], 1)

    def test_materialize_l4_sqlite_manifest_asset_preserves_table_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sqlite_db = root / "MODEL_PREDICTIONS.db"
            report_dir = root / "reports"
            duckdb_root = root / "duckdb"
            audit_record = root / "audit.md"
            registry = root / "production_assets.json"
            manifest_path = root / "executable_5d_open_return_l4_formal.json"
            audit_record.write_text("通过", encoding="utf-8")

            conn = sqlite3.connect(sqlite_db)
            try:
                conn.execute("CREATE TABLE stock_predict_data_demo(trade_date TEXT, stock_code TEXT, pred_prob REAL)")
                conn.execute("INSERT INTO stock_predict_data_demo VALUES ('20260628', '000001.SZ', 0.9)")
                conn.execute("CREATE TABLE stock_predict_data_extra(trade_date TEXT, stock_code TEXT, pred_prob REAL)")
                conn.execute("INSERT INTO stock_predict_data_extra VALUES ('20260628', '000002.SZ', 0.8)")
                conn.commit()
            finally:
                conn.close()

            manifest_path.write_text(
                json.dumps(
                    {
                        "approval_status": "approved_for_l5",
                        "source_type": "sqlite_table",
                        "db_path": str(sqlite_db),
                        "table": "stock_predict_data_demo",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            registry.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l4_test",
                                "track": "production",
                                "layer": "L4_model_prediction",
                                "asset_type": "sqlite_table_with_formal_manifests",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(sqlite_db),
                                "audit_record": str(audit_record),
                            },
                            {
                                "asset_id": "prod_l4_test_manifest",
                                "track": "production",
                                "layer": "L4_model_prediction",
                                "asset_type": "formal_prediction_manifest",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(manifest_path),
                                "input_assets": ["prod_l4_test"],
                                "audit_record": str(audit_record),
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"QUANT_DUCKDB_ASSET_ROOT": str(duckdb_root)}):
                report = materialize(
                    registry_path=registry,
                    report_dir=report_dir,
                    batch_size=10,
                    layers={"L4"},
                    dry_run=False,
                )

            tables = report["assets"][0]["tables"]
            self.assertEqual(len(tables), 1)
            self.assertEqual(tables[0]["source_table"], "stock_predict_data_demo")
            self.assertEqual(tables[0]["target_table"], "stock_predict_data_demo")

    def test_materialize_parquet_asset_can_resume_from_file_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_dir = root / "parts"
            report_dir = root / "reports"
            duckdb_root = root / "duckdb"
            audit_record = root / "audit.md"
            registry = root / "production_assets.json"
            audit_record.write_text("通过", encoding="utf-8")

            parquet_dir.mkdir()
            pd.DataFrame(
                {"trade_date": ["20260616"], "stock_code": ["000001.SZ"], "factor_a": [1.0]}
            ).to_parquet(parquet_dir / "part-000.parquet", index=False)
            pd.DataFrame(
                {"trade_date": ["20260617"], "stock_code": ["000002.SZ"], "factor_a": [2.0]}
            ).to_parquet(parquet_dir / "part-001.parquet", index=False)

            registry.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l3_test_resume",
                                "track": "production",
                                "layer": "L3_features",
                                "asset_type": "parquet_parts",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(parquet_dir),
                                "audit_record": str(audit_record),
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"QUANT_DUCKDB_ASSET_ROOT": str(duckdb_root)}):
                report = materialize(
                    registry_path=registry,
                    report_dir=report_dir,
                    batch_size=10,
                    layers={"L3_FEATURES"},
                    dry_run=False,
                )
                self.assertEqual(report["assets"][0]["tables"][0]["target_rows"], 2)

                import duckdb

                duckdb_path = duckdb_root / "quant_production.duckdb"
                with duckdb.connect(str(duckdb_path)) as conn:
                    conn.execute("DELETE FROM prod_l3_test_resume WHERE trade_date = '20260617'")

                resumed = materialize(
                    registry_path=registry,
                    report_dir=report_dir,
                    batch_size=10,
                    layers={"L3_FEATURES"},
                    dry_run=False,
                    resume_existing=True,
                )

            table_report = resumed["assets"][0]["tables"][0]
            self.assertEqual(table_report["existing_rows_before_resume"], 1)
            self.assertEqual(table_report["resumed_from_file_index"], 1)
            self.assertEqual(table_report["target_rows"], 2)


if __name__ == "__main__":
    unittest.main()
