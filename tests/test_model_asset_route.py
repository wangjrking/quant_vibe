import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

from model_asset_route import (
    LEGACY_MODEL_ASSET_CHAIN_ENV,
    MODEL_FEATURE_MODE_LEGACY,
    MODEL_PREDICTION_MODE_LEGACY,
    RESEARCH_PREDICTION_APPROVAL_STATUS,
    RESEARCH_PREDICTION_ASSET_ROLE,
    enrich_research_prediction_manifest,
    legacy_model_asset_chain_opted_in,
    require_legacy_model_asset_chain_opt_in,
    resolve_legacy_mixed_factor_path,
    resolve_legacy_prediction_db_path,
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
    resolve_model_feature_path,
    resolve_model_label_duckdb_path,
    resolve_model_label_duckdb_table,
    resolve_model_label_path,
    resolve_model_prediction_duckdb_path,
    resolve_model_prediction_duckdb_table,
    resolve_model_prediction_db_path,
    resolve_prediction_run_dir,
    use_legacy_mixed_features,
    use_legacy_prediction_db,
)


class ModelAssetRouteTests(unittest.TestCase):
    def test_default_paths_point_to_split_assets_and_disable_implicit_prediction_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            self.assertEqual(resolve_model_feature_path(data_dir), data_dir / "production_factor_parts")
            self.assertEqual(resolve_model_label_path(data_dir), data_dir / "prediction_label_parts")
            with self.assertRaisesRegex(RuntimeError, "Implicit MODEL_PREDICTIONS\\.db fallback is disabled"):
                resolve_model_prediction_db_path(data_dir)
            self.assertEqual(resolve_prediction_run_dir(data_dir, label="x", output_table="pred_x"), data_dir / "model_predictions" / "pred_x")

    def test_default_duckdb_paths_point_to_layer_isolated_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            expected_feature = data_dir / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
            expected_label = data_dir / "production_assets" / "duckdb" / "l3_label_current.duckdb"
            expected_prediction = data_dir / "production_assets" / "duckdb" / "l4_predictions_current.duckdb"
            self.assertEqual(resolve_model_feature_duckdb_path(data_dir), expected_feature)
            self.assertEqual(resolve_model_label_duckdb_path(data_dir), expected_label)
            self.assertEqual(resolve_model_prediction_duckdb_path(data_dir), expected_prediction)

    def test_registry_driven_duckdb_paths_and_tables_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            expected = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l3_feature_duckdb",
                                "layer": "L3_features",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}::prod_l3_feature_table",
                            },
                            {
                                "asset_id": "prod_l3_label_duckdb",
                                "layer": "L3_labels",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}::prod_l3_label_table",
                            },
                            {
                                "asset_id": "prod_l4_prediction_duckdb",
                                "layer": "L4_model_prediction",
                                "asset_type": "duckdb_table",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}::prod_l4_prediction_table",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(resolve_model_feature_duckdb_path(data_dir), expected)
            self.assertEqual(resolve_model_feature_duckdb_table(data_dir), "prod_l3_feature_table")
            self.assertEqual(resolve_model_label_duckdb_path(data_dir), expected)
            self.assertEqual(resolve_model_label_duckdb_table(data_dir), "prod_l3_label_table")
            self.assertEqual(resolve_model_prediction_duckdb_path(data_dir), expected)
            self.assertEqual(resolve_model_prediction_duckdb_table(data_dir), "prod_l4_prediction_table")

    def test_parquet_or_sqlite_helpers_fail_closed_after_registry_switch_to_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            expected = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l3_feature_duckdb",
                                "layer": "L3_features",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}::prod_l3_feature_table",
                            },
                            {
                                "asset_id": "prod_l3_label_duckdb",
                                "layer": "L3_labels",
                                "asset_type": "duckdb_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}::prod_l3_label_table",
                            },
                            {
                                "asset_id": "prod_l4_prediction_duckdb",
                                "layer": "L4_model_prediction",
                                "asset_type": "duckdb_table",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}::prod_l4_prediction_table",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "L3 feature mainline is registered as DuckDB"):
                resolve_model_feature_path(data_dir)
            with self.assertRaisesRegex(RuntimeError, "L3 label mainline is registered as DuckDB"):
                resolve_model_label_path(data_dir)
            with self.assertRaisesRegex(RuntimeError, "L4 prediction mainline is registered as DuckDB"):
                resolve_model_prediction_db_path(data_dir)

    def test_active_l3_and_l4_sqlite_current_routes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            sqlite_path = data_dir / "legacy_current.db"
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l3_feature_sqlite_bad",
                                "layer": "L3_features",
                                "asset_type": "sqlite_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{sqlite_path.as_posix()}::prod_l3_feature_table",
                            },
                            {
                                "asset_id": "prod_l3_label_sqlite_bad",
                                "layer": "L3_labels",
                                "asset_type": "sqlite_table",
                                "status": "production_active",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{sqlite_path.as_posix()}::prod_l3_label_table",
                            },
                            {
                                "asset_id": "prod_l4_prediction_sqlite_bad",
                                "layer": "L4_model_prediction",
                                "asset_type": "sqlite_table",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{sqlite_path.as_posix()}::prod_l4_prediction_table",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "active L3_features mainline must not route to SQLite"):
                resolve_model_feature_duckdb_path(data_dir)
            with self.assertRaisesRegex(RuntimeError, "active L3_labels mainline must not route to SQLite"):
                resolve_model_label_duckdb_path(data_dir)
            with self.assertRaisesRegex(RuntimeError, "active L4 prediction mainline must not route to SQLite"):
                resolve_model_prediction_duckdb_path(data_dir)

    def test_l4_route_prefers_duckdb_table_over_formal_manifest_when_both_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            expected = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            manifest_path = data_dir / "config" / "prediction_manifests" / "formal.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}", encoding="utf-8")
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l4_formal_manifest",
                                "layer": "L4_model_prediction",
                                "asset_type": "formal_prediction_manifest",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(manifest_path),
                            },
                            {
                                "asset_id": "prod_l4_prediction_duckdb",
                                "layer": "L4_model_prediction",
                                "asset_type": "duckdb_table",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}::prod_l4_prediction_table",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(resolve_model_prediction_duckdb_path(data_dir), expected)
            self.assertEqual(resolve_model_prediction_duckdb_table(data_dir), "prod_l4_prediction_table")

    def test_l4_route_prefers_duckdb_bundle_over_formal_manifests_and_requires_manifest_level_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            expected = data_dir / "production_assets" / "duckdb" / "quant_production.duckdb"
            manifest_dir = data_dir / "config" / "prediction_manifests"
            manifest_dir.mkdir(parents=True)
            for name in ("3d.json", "5d.json", "10d.json"):
                (manifest_dir / name).write_text("{}", encoding="utf-8")
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l4_formal_manifest_3d",
                                "layer": "L4_model_prediction",
                                "asset_type": "formal_prediction_manifest",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(manifest_dir / "3d.json"),
                            },
                            {
                                "asset_id": "prod_l4_formal_manifest_5d",
                                "layer": "L4_model_prediction",
                                "asset_type": "formal_prediction_manifest",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(manifest_dir / "5d.json"),
                            },
                            {
                                "asset_id": "prod_l4_formal_manifest_10d",
                                "layer": "L4_model_prediction",
                                "asset_type": "formal_prediction_manifest",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(manifest_dir / "10d.json"),
                            },
                            {
                                "asset_id": "prod_l4_prediction_duckdb_bundle",
                                "layer": "L4_model_prediction",
                                "asset_type": "duckdb_prediction_bundle",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": f"{expected.as_posix()}",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(resolve_model_prediction_duckdb_path(data_dir), expected)
            with self.assertRaisesRegex(RuntimeError, "manifest-routed across multiple formal DuckDB assets"):
                resolve_model_prediction_duckdb_table(data_dir)

    def test_l4_route_fails_closed_when_active_mainline_is_manifest_routed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry_dir = data_dir / "asset_registry"
            registry_dir.mkdir(parents=True)
            manifest_dir = data_dir / "config" / "prediction_manifests"
            manifest_dir.mkdir(parents=True)
            for name in ("1d.json", "3d.json", "5d.json", "10d.json"):
                (manifest_dir / name).write_text("{}", encoding="utf-8")
            (registry_dir / "production_assets.json").write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "prod_l4_formal_manifest_1d",
                                "layer": "L4_model_prediction",
                                "asset_type": "formal_prediction_manifest",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(manifest_dir / "1d.json"),
                            },
                            {
                                "asset_id": "prod_l4_formal_manifest_3d",
                                "layer": "L4_model_prediction",
                                "asset_type": "formal_prediction_manifest",
                                "status": "approved_for_l5",
                                "allowed_for_main_workflow": True,
                                "asset_path": str(manifest_dir / "3d.json"),
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "manifest-routed and DuckDB-only"):
                resolve_model_prediction_db_path(data_dir)
            with self.assertRaisesRegex(RuntimeError, "manifest-routed across formal DuckDB assets"):
                resolve_model_prediction_duckdb_path(data_dir)
            with self.assertRaisesRegex(RuntimeError, "manifest-routed across multiple formal DuckDB assets"):
                resolve_model_prediction_duckdb_table(data_dir)

    def test_legacy_switches_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            self.assertTrue(use_legacy_mixed_features(MODEL_FEATURE_MODE_LEGACY))
            self.assertTrue(use_legacy_prediction_db(MODEL_PREDICTION_MODE_LEGACY))
            self.assertEqual(resolve_legacy_mixed_factor_path(data_dir), data_dir / "stock_factor_data.parquet")
            self.assertEqual(resolve_legacy_prediction_db_path(data_dir), data_dir / "odb.db")

    def test_env_override_can_point_prediction_db_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom_predictions.db"

            with patch.dict("os.environ", {"QUANT_MODEL_PREDICTION_DB": str(custom)}):
                self.assertEqual(resolve_model_prediction_db_path(), custom)

    def test_legacy_model_asset_chain_requires_explicit_opt_in(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertFalse(legacy_model_asset_chain_opted_in())
            with self.assertRaisesRegex(RuntimeError, "archived and disabled by default"):
                require_legacy_model_asset_chain_opt_in(reason="unit-test")

        with patch.dict("os.environ", {LEGACY_MODEL_ASSET_CHAIN_ENV: "1"}, clear=False):
            self.assertTrue(legacy_model_asset_chain_opted_in())
            require_legacy_model_asset_chain_opt_in(reason="unit-test")

    def test_enrich_research_prediction_manifest_sets_research_guardrails(self):
        manifest = enrich_research_prediction_manifest({"prediction_table": "pred_x"})

        self.assertEqual(manifest["asset_role"], RESEARCH_PREDICTION_ASSET_ROLE)
        self.assertEqual(manifest["model_track"], "research")
        self.assertEqual(manifest["approval_status"], RESEARCH_PREDICTION_APPROVAL_STATUS)
        self.assertTrue(manifest["promotion_requires_user_confirmation"])
        self.assertEqual(manifest["prediction_table"], "pred_x")


if __name__ == "__main__":
    unittest.main()
