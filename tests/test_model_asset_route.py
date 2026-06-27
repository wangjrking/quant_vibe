import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    resolve_model_feature_path,
    resolve_model_label_path,
    resolve_model_prediction_db_path,
    resolve_prediction_run_dir,
    use_legacy_mixed_features,
    use_legacy_prediction_db,
)


class ModelAssetRouteTests(unittest.TestCase):
    def test_default_paths_point_to_split_assets_and_independent_prediction_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            self.assertEqual(resolve_model_feature_path(data_dir), data_dir / "production_factor_parts")
            self.assertEqual(resolve_model_label_path(data_dir), data_dir / "prediction_label_parts")
            self.assertEqual(resolve_model_prediction_db_path(data_dir), data_dir / "model_predictions" / "MODEL_PREDICTIONS.db")
            self.assertEqual(resolve_prediction_run_dir(data_dir, label="x", output_table="pred_x"), data_dir / "model_predictions" / "pred_x")

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
