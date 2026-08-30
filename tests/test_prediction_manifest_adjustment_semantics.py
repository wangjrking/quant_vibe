from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from adjustment_semantics import (
    default_adjustment_semantics,
    default_base_market_field_semantics,
    default_market_field_semantics,
)
from prediction_manifest import load_prediction_source_manifest, resolve_market_db_path


class PredictionManifestAdjustmentSemanticsTests(unittest.TestCase):
    def test_duckdb_manifest_requires_adjustment_semantics_when_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            duckdb_path = root / "quant_production.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_migration_manifest.json"
            duckdb_path.write_bytes(b"duckdb")
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": "pred_table",
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing adjustment_semantics"):
                load_prediction_source_manifest(manifest_path, require_approved=True, allow_legacy=False)

    def test_duckdb_manifest_requires_market_field_semantics_when_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            duckdb_path = root / "quant_production.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_migration_manifest.json"
            duckdb_path.write_bytes(b"duckdb")
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": "pred_table",
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing market_field_semantics"):
                load_prediction_source_manifest(manifest_path, require_approved=True, allow_legacy=False)

    def test_duckdb_manifest_accepts_valid_adjustment_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            duckdb_path = root / "quant_production.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_migration_manifest.json"
            duckdb_path.write_bytes(b"duckdb")
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": "pred_table",
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_market_field_semantics(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            source = load_prediction_source_manifest(
                manifest_path,
                require_approved=True,
                allow_legacy=False,
            )

            self.assertEqual(source["adjustment_semantics"], default_adjustment_semantics())
            self.assertEqual(source["market_field_semantics"], default_market_field_semantics())

    def test_duckdb_prediction_manifest_rejects_base_market_field_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            duckdb_path = root / "quant_production.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_migration_manifest.json"
            duckdb_path.write_bytes(b"duckdb")
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": "pred_table",
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_base_market_field_semantics(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema_kind must be strategy_output"):
                load_prediction_source_manifest(
                    manifest_path,
                    require_approved=True,
                    allow_legacy=False,
                )

    def test_resolve_market_db_path_rejects_sqlite_override_for_duckdb_source(self):
        source = {
            "source_type": "duckdb_table",
            "market_db_path": None,
        }

        with self.assertRaisesRegex(ValueError, "DuckDB file"):
            resolve_market_db_path(source, market_db_path="D:/tmp/STOCK_DAILY_DATA.db")

    def test_resolve_market_db_path_rejects_sqlite_manifest_market_db_for_duckdb_source(self):
        source = {
            "source_type": "duckdb_table",
            "market_db_path": Path("D:/tmp/STOCK_DAILY_DATA.db"),
        }

        with self.assertRaisesRegex(ValueError, "DuckDB file"):
            resolve_market_db_path(source, market_db_path=None)

    def test_resolve_market_db_path_allows_sqlite_override_for_legacy_reproduction(self):
        source = {
            "source_type": "sqlite_table",
            "market_db_path": None,
        }

        resolved = resolve_market_db_path(source, market_db_path="D:/tmp/STOCK_DAILY_DATA.db")
        self.assertEqual(resolved, Path("D:/tmp/STOCK_DAILY_DATA.db"))

    def test_resolve_market_db_path_rejects_implicit_sqlite_default_for_legacy_reproduction(self):
        source = {
            "source_type": "sqlite_table",
            "market_db_path": None,
        }

        with patch("prediction_manifest.resolve_stock_daily_backend", return_value="sqlite"):
            with self.assertRaisesRegex(ValueError, "must provide explicit market_db_path"):
                resolve_market_db_path(source, market_db_path=None)


if __name__ == "__main__":
    unittest.main()
