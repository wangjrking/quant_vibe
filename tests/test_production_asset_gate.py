import json
import tempfile
import unittest
from pathlib import Path
import sys

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from adjustment_semantics import (
    default_adjustment_semantics,
    default_base_market_field_semantics,
    default_market_field_semantics,
)
from tools.production_asset_gate import check_change, check_registry


class ProductionAssetGateTests(unittest.TestCase):
    def test_duckdb_asset_requires_migration_manifest_and_rollback_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            duckdb_file = root / "quant_production.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-switch",
                "change_type": "promote_duckdb",
                "layer": "L2",
                "owner_agent": "data-integration-agent",
                "audit_required": True,
                "audit_status": "passed",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::STOCK_DAILY_DATA",
                    "audit_record": str(audit),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertIn("change duckdb-switch DuckDB asset missing duckdb_migration_manifest file", errors)
            self.assertIn("change duckdb-switch DuckDB asset missing migration_source_asset", errors)

    def test_duckdb_asset_gate_accepts_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "quant_production.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-switch",
                "change_type": "promote_duckdb",
                "layer": "L2",
                "owner_agent": "data-integration-agent",
                "audit_required": True,
                "audit_status": "passed",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::STOCK_DAILY_DATA",
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l2_stock_daily_data_20260625",
                    "adjustment_semantics": default_adjustment_semantics(),
                    "market_field_semantics": default_base_market_field_semantics(),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertEqual(errors, [])

    def test_duckdb_asset_gate_accepts_approved_for_l5_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "quant_production.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-l4-switch",
                "change_type": "promote_duckdb",
                "layer": "L4",
                "owner_agent": "model-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "approved_for_l5",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::pred_table",
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l4_model_predictions_formal_20260625",
                    "adjustment_semantics": default_adjustment_semantics(),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertEqual(errors, [])

    def test_l4_duckdb_asset_requires_adjustment_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "quant_production.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-l4-switch",
                "change_type": "promote_duckdb",
                "layer": "L4",
                "owner_agent": "model-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "approved_for_l5",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::pred_table",
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l4_model_predictions_formal_20260625",
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertIn("change duckdb-l4-switch asset_after missing adjustment_semantics", errors)

    def test_l4_duckdb_asset_accepts_adjustment_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "quant_production.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-l4-switch",
                "change_type": "promote_duckdb",
                "layer": "L4",
                "owner_agent": "model-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "approved_for_l5",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::pred_table",
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l4_model_predictions_formal_20260625",
                    "adjustment_semantics": default_adjustment_semantics(),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertEqual(errors, [])

    def test_l2_duckdb_asset_requires_adjustment_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-l2-switch",
                "change_type": "promote_duckdb",
                "layer": "L2_stock_daily_base",
                "owner_agent": "data-integration-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::STOCK_DAILY_DATA",
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l2_stock_daily_data_20260625",
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertIn("change duckdb-l2-switch asset_after missing adjustment_semantics", errors)

    def test_l2_duckdb_asset_requires_base_market_field_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-l2-switch",
                "change_type": "promote_duckdb",
                "layer": "L2_stock_daily_base",
                "owner_agent": "data-integration-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::STOCK_DAILY_DATA",
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l2_stock_daily_data_20260625",
                    "adjustment_semantics": default_adjustment_semantics(),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertIn("change duckdb-l2-switch asset_after missing market_field_semantics", errors)

    def test_change_rejects_unreadable_adjustment_contract_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            semantics = default_adjustment_semantics()
            semantics["contract_rule"] = "???????????? qfq?? open/high/low/close/pre_close ????"
            change = {
                "change_id": "duckdb-l2-switch",
                "change_type": "promote_duckdb",
                "layer": "L2_stock_daily_base",
                "owner_agent": "data-integration-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file) + "::STOCK_DAILY_DATA",
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l2_stock_daily_data_20260625",
                    "adjustment_semantics": semantics,
                    "market_field_semantics": default_base_market_field_semantics(),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertIn(
                "change duckdb-l2-switch asset_after adjustment_semantics.contract_rule must explicitly describe qfq/front-adjusted semantics",
                errors,
            )

    def test_registry_allows_retired_legacy_reference_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            audit.write_text("閫氳繃", encoding="utf-8")
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                """
{
  "assets": [
    {
      "asset_id": "prod_l1_raw_split_dbs_20260625",
      "track": "production",
      "status": "retired_legacy_reference",
      "allowed_for_main_workflow": false,
      "superseded_by": "prod_l1_duckdb_20260628",
      "audit_record": "audit.md"
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertEqual(errors, [])

    def test_registry_active_l2_duckdb_asset_requires_adjustment_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l2_duckdb_stock_daily_data_split_20260701",
      "track": "production",
      "layer": "L2_stock_daily_base",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l2_stock_daily_data_20260628",
      "asset_path": "{duckdb_file.as_posix()}::STOCK_DAILY_DATA",
      "adjustment_semantics": {json.dumps(default_adjustment_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertIn("asset prod_l2_duckdb_stock_daily_data_split_20260701 missing market_field_semantics", errors)

    def test_registry_active_l2_duckdb_asset_accepts_base_market_field_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l2_duckdb_stock_daily_data_split_20260701",
      "track": "production",
      "layer": "L2_stock_daily_base",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l2_stock_daily_data_20260628",
      "asset_path": "{duckdb_file.as_posix()}::STOCK_DAILY_DATA",
      "adjustment_semantics": {json.dumps(default_adjustment_semantics(), ensure_ascii=False)},
      "market_field_semantics": {json.dumps(default_base_market_field_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertEqual(errors, [])

    def test_registry_active_l2_duckdb_asset_rejects_missing_market_qfq_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            import duckdb

            with duckdb.connect(str(duckdb_file)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA(
                        trade_date TEXT,
                        stock_code TEXT,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        pre_close DOUBLE,
                        atr_qfq DOUBLE
                    )
                    """
                )
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l2_duckdb_stock_daily_data_split_20260701",
      "track": "production",
      "layer": "L2_stock_daily_base",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l2_stock_daily_data_20260628",
      "asset_path": "{duckdb_file.as_posix()}::STOCK_DAILY_DATA",
      "adjustment_semantics": {json.dumps(default_adjustment_semantics(), ensure_ascii=False)},
      "market_field_semantics": {json.dumps(default_base_market_field_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertIn(
                "asset prod_l2_duckdb_stock_daily_data_split_20260701 schema contains naked market price columns without explicit qfq counterparts",
                "\n".join(errors),
            )

    def test_registry_active_l2_duckdb_asset_accepts_explicit_market_qfq_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            import duckdb

            with duckdb.connect(str(duckdb_file)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA(
                        trade_date TEXT,
                        stock_code TEXT,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        pre_close DOUBLE,
                        open_qfq DOUBLE,
                        high_qfq DOUBLE,
                        low_qfq DOUBLE,
                        close_qfq DOUBLE,
                        pre_close_qfq DOUBLE,
                        atr_qfq DOUBLE
                    )
                    """
                )
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l2_duckdb_stock_daily_data_split_20260701",
      "track": "production",
      "layer": "L2_stock_daily_base",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l2_stock_daily_data_20260628",
      "asset_path": "{duckdb_file.as_posix()}::STOCK_DAILY_DATA",
      "adjustment_semantics": {json.dumps(default_adjustment_semantics(), ensure_ascii=False)},
      "market_field_semantics": {json.dumps(default_base_market_field_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertEqual(errors, [])

    def test_registry_active_l3_feature_asset_rejects_missing_close_qfq_when_qfq_indicators_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l3_feature_current.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            import duckdb

            with duckdb.connect(str(duckdb_file)) as conn:
                conn.execute(
                    """
                    CREATE TABLE prod_l3_feature_current(
                        trade_date TEXT,
                        stock_code TEXT,
                        close DOUBLE,
                        atr_qfq DOUBLE
                    )
                    """
                )
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l3_feature_current",
      "track": "production",
      "layer": "L3_features",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l3_feature_legacy",
      "asset_path": "{duckdb_file.as_posix()}::prod_l3_feature_current",
      "adjustment_semantics": {json.dumps(default_adjustment_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertIn(
                "asset prod_l3_feature_current schema contains naked market price columns without explicit qfq counterparts: ['close_qfq']",
                errors,
            )

    def test_registry_active_l3_feature_asset_accepts_explicit_close_qfq(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l3_feature_current.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            import duckdb

            with duckdb.connect(str(duckdb_file)) as conn:
                conn.execute(
                    """
                    CREATE TABLE prod_l3_feature_current(
                        trade_date TEXT,
                        stock_code TEXT,
                        close DOUBLE,
                        close_qfq DOUBLE,
                        atr_qfq DOUBLE
                    )
                    """
                )
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l3_feature_current",
      "track": "production",
      "layer": "L3_features",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l3_feature_legacy",
      "asset_path": "{duckdb_file.as_posix()}::prod_l3_feature_current",
      "adjustment_semantics": {json.dumps(default_adjustment_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertEqual(errors, [])

    def test_registry_active_l3_feature_asset_rejects_naked_factor_when_qfq_counterpart_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l3_feature_current.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            import duckdb

            with duckdb.connect(str(duckdb_file)) as conn:
                conn.execute(
                    """
                    CREATE TABLE prod_l3_feature_current(
                        trade_date TEXT,
                        stock_code TEXT,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        pre_close DOUBLE,
                        open_qfq DOUBLE,
                        high_qfq DOUBLE,
                        low_qfq DOUBLE,
                        close_qfq DOUBLE,
                        pre_close_qfq DOUBLE,
                        atr DOUBLE,
                        atr_qfq DOUBLE
                    )
                    """
                )
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l3_feature_current",
      "track": "production",
      "layer": "L3_features",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l3_feature_legacy",
      "asset_path": "{duckdb_file.as_posix()}::prod_l3_feature_current",
      "adjustment_semantics": {json.dumps(default_adjustment_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertIn(
                "asset prod_l3_feature_current schema contains naked factor columns alongside explicit qfq counterparts: ['atr']",
                errors,
            )

    def test_registry_rejects_unreadable_adjustment_contract_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l2_stock_daily_data.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            semantics = default_adjustment_semantics()
            semantics["contract_rule"] = "???????????? qfq?? open/high/low/close/pre_close ????"
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l2_duckdb_stock_daily_data_split_20260701",
      "track": "production",
      "layer": "L2_stock_daily_base",
      "asset_type": "duckdb_table",
      "status": "production_active",
      "allowed_for_main_workflow": true,
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l2_stock_daily_data_20260628",
      "asset_path": "{duckdb_file.as_posix()}::STOCK_DAILY_DATA",
      "adjustment_semantics": {json.dumps(semantics, ensure_ascii=False)},
      "market_field_semantics": {json.dumps(default_base_market_field_semantics(), ensure_ascii=False)}
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertIn(
                "asset prod_l2_duckdb_stock_daily_data_split_20260701 adjustment_semantics.contract_rule must explicitly describe qfq/front-adjusted semantics",
                errors,
            )

    def test_l7_duckdb_asset_requires_strategy_market_field_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l7_delivery.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-l7-switch",
                "change_type": "promote_duckdb",
                "layer": "L7_trading_delivery",
                "owner_agent": "trading-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file),
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l7_duckdb_production_signals_20260628",
                    "adjustment_semantics": default_adjustment_semantics(),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertIn("change duckdb-l7-switch asset_after missing market_field_semantics", errors)

    def test_l7_duckdb_asset_accepts_strategy_market_field_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "l7_delivery.duckdb"
            audit.write_text("approved", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            change = {
                "change_id": "duckdb-l7-switch",
                "change_type": "promote_duckdb",
                "layer": "L7_trading_delivery",
                "owner_agent": "trading-agent",
                "audit_required": True,
                "audit_status": "approved",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_file),
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l7_duckdb_production_signals_20260628",
                    "adjustment_semantics": default_adjustment_semantics(),
                    "market_field_semantics": default_market_field_semantics(),
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertEqual(errors, [])

    def test_registry_allows_retired_duckdb_assets_without_active_status_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_file = root / "quant_production.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")
            registry = registry_dir / "production_assets.json"
            registry.write_text(
                f"""
{{
  "assets": [
    {{
      "asset_id": "prod_l1_duckdb_legacy",
      "track": "production",
      "asset_type": "duckdb_multi_table",
      "status": "retired_legacy_reference",
      "allowed_for_main_workflow": false,
      "superseded_by": "prod_l1_duckdb_new",
      "audit_record": "audit.md",
      "duckdb_migration_manifest": "duckdb_migration_manifest.json",
      "migration_source_asset": "prod_l1_raw_split_dbs_20260625",
      "asset_path": "{duckdb_file.as_posix()}"
    }}
  ]
}}
""".strip(),
                encoding="utf-8",
            )

            errors = check_registry(root)

            self.assertEqual(errors, [])

    def test_duckdb_table_files_asset_accepts_directory_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.md"
            manifest = root / "duckdb_migration_manifest.json"
            duckdb_dir = root / "l1_raw_tables"
            duckdb_dir.mkdir()
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            change = {
                "change_id": "duckdb-table-files-switch",
                "change_type": "promote_duckdb",
                "layer": "L1",
                "owner_agent": "data-ingestion-agent",
                "audit_required": True,
                "audit_status": "passed",
                "audit_record": str(audit),
                "asset_after": {
                    "track": "production",
                    "asset_type": "duckdb_table_files",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(duckdb_dir),
                    "audit_record": str(audit),
                    "duckdb_migration_manifest": str(manifest),
                    "migration_source_asset": "prod_l1_duckdb_20260628",
                },
            }
            errors = []

            check_change(root, change, errors)

            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()


