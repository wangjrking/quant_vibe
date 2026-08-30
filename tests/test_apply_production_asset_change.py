import json
import tempfile
import unittest
from pathlib import Path

from adjustment_semantics import default_adjustment_semantics, default_base_market_field_semantics
from tools.apply_production_asset_change import apply_change


class ApplyProductionAssetChangeTests(unittest.TestCase):
    def test_apply_change_replaces_asset_and_marks_previous_as_retired_legacy_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            report_dir = root / "reports"
            report_dir.mkdir()
            audit = report_dir / "audit.md"
            manifest = report_dir / "duckdb_manifest.json"
            duckdb_file = root / "quant_production.duckdb"
            audit.write_text("閫氳繃", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            duckdb_file.write_bytes(b"duckdb")

            registry_path = registry_dir / "production_assets.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l2_stock_daily_data_20260625",
                                "track": "production",
                                "layer": "L2_stock_daily_base",
                                "asset_type": "sqlite_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": "quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA",
                                "audit_record": "reports/audit.md",
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            change_path = root / "change.json"
            change_path.write_text(
                json.dumps(
                    {
                        "change_id": "promote_l2_duckdb",
                        "change_type": "promote_duckdb",
                        "layer": "L2",
                        "owner_agent": "data-integration-agent",
                        "audit_required": True,
                        "audit_status": "approved",
                        "audit_record": "reports/audit.md",
                        "asset_before": {
                            "asset_id": "prod_l2_stock_daily_data_20260625",
                            "asset_type": "sqlite_table",
                            "asset_path": "quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA",
                        },
                        "asset_after": {
                            "asset_id": "prod_l2_duckdb_stock_daily_data_20260628",
                            "track": "production",
                            "layer": "L2_stock_daily_base",
                            "asset_type": "duckdb_table",
                            "owner_agent": "data-integration-agent",
                            "status": "production_active",
                            "allowed_for_main_workflow": True,
                            "asset_path": str(duckdb_file) + "::STOCK_DAILY_DATA",
                            "audit_record": "reports/audit.md",
                            "duckdb_migration_manifest": "reports/duckdb_manifest.json",
                            "migration_source_asset": "prod_l2_stock_daily_data_20260625",
                            "adjustment_semantics": default_adjustment_semantics(),
                            "market_field_semantics": default_base_market_field_semantics(),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result_path = apply_change(root, change_path)
            updated = json.loads(result_path.read_text(encoding="utf-8"))

        assets = {asset["asset_id"]: asset for asset in updated["assets"]}
        self.assertEqual(assets["prod_l2_duckdb_stock_daily_data_20260628"]["status"], "production_active")
        self.assertTrue(assets["prod_l2_duckdb_stock_daily_data_20260628"]["allowed_for_main_workflow"])
        self.assertEqual(assets["prod_l2_stock_daily_data_20260625"]["status"], "retired_legacy_reference")
        self.assertFalse(assets["prod_l2_stock_daily_data_20260625"]["allowed_for_main_workflow"])
        self.assertEqual(
            assets["prod_l2_stock_daily_data_20260625"]["superseded_by"],
            "prod_l2_duckdb_stock_daily_data_20260628",
        )


if __name__ == "__main__":
    unittest.main()

