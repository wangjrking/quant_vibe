import json
import tempfile
import unittest
from pathlib import Path

from tools.standard_agent_architecture_check import check_asset_registry


class StandardAgentArchitectureCheckTests(unittest.TestCase):
    def test_check_asset_registry_allows_retired_legacy_reference_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_dir = root / "quant" / "data_file" / "asset_registry"
            registry_dir.mkdir(parents=True)
            (registry_dir / "manifest.schema.json").write_text(
                json.dumps(
                    {
                        "required_fields": [
                            "asset_id",
                            "track",
                            "layer",
                            "asset_type",
                            "owner_agent",
                            "status",
                            "allowed_for_main_workflow",
                            "asset_path",
                            "created_at",
                            "audit_record",
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (registry_dir / "experimental_assets.json").write_text(
                json.dumps({"audit_required_before_promotion": True, "assets": []}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "audit.md").write_text("通过", encoding="utf-8")
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "audit_required_before_change": True,
                        "assets": [
                            {
                                "asset_id": "prod_l1_raw_split_dbs_20260625",
                                "track": "production",
                                "layer": "L1_raw_data",
                                "asset_type": "raw_split_sqlite_dbs",
                                "owner_agent": "data-ingestion-agent",
                                "status": "retired_legacy_reference",
                                "allowed_for_main_workflow": False,
                                "asset_path": "quant/data_file/raw_table_dbs/",
                                "created_at": "2026-06-28T00:00:00+08:00",
                                "audit_record": "audit.md",
                                "superseded_by": "prod_l1_duckdb_20260628",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            errors: list[str] = []
            check_asset_registry(root, errors)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
