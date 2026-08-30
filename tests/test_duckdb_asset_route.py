import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckdb_asset_route import (
    DuckDBRouteError,
    resolve_asset_backend,
    resolve_duckdb_path,
    resolve_duckdb_root,
)


class DuckDBAssetRouteTests(unittest.TestCase):
    def test_default_root_is_production_assets_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            self.assertEqual(resolve_duckdb_root(data_dir), data_dir / "production_assets" / "duckdb")
            self.assertEqual(resolve_duckdb_path("L2", data_dir), data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb")
            self.assertEqual(resolve_duckdb_path("L3_FEATURES", data_dir), data_dir / "production_assets" / "duckdb" / "l3_feature_current.duckdb")
            self.assertEqual(resolve_duckdb_path("L3_LABELS", data_dir), data_dir / "production_assets" / "duckdb" / "l3_label_current.duckdb")
            self.assertEqual(resolve_duckdb_path("L4", data_dir), data_dir / "production_assets" / "duckdb" / "l4_predictions_current.duckdb")
            with self.assertRaises(DuckDBRouteError):
                resolve_duckdb_path("L1", data_dir)
            with self.assertRaises(DuckDBRouteError):
                resolve_duckdb_path("L5", data_dir)
            with self.assertRaises(DuckDBRouteError):
                resolve_duckdb_path("L6", data_dir)
            with self.assertRaises(DuckDBRouteError):
                resolve_duckdb_path("L7", data_dir)

    def test_env_root_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "duckdb_root"

            with patch.dict(os.environ, {"QUANT_DUCKDB_ASSET_ROOT": str(root)}):
                self.assertEqual(resolve_duckdb_path("L2"), root / "l2_stock_daily_data.duckdb")

    def test_backend_defaults_to_current_production_and_explicit_legacy_requires_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_asset_backend(), "duckdb")
            with self.assertRaisesRegex(RuntimeError, "explicit legacy SQLite/Parquet backend selection requires opt-in"):
                resolve_asset_backend("legacy")
            with self.assertRaisesRegex(ValueError, "deprecated local asset backend alias"):
                resolve_asset_backend("sqlite")

        with patch.dict(os.environ, {"QUANT_ALLOW_LEGACY_SQLITE_PARQUET_ROLLBACK": "1"}, clear=True):
            self.assertEqual(resolve_asset_backend("legacy"), "legacy")

    def test_explicit_duckdb_selection_is_allowed_without_legacy_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_asset_backend("duckdb"), "duckdb")

    def test_split_only_layers_never_allow_legacy_bundle_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with patch.dict(os.environ, {"QUANT_ALLOW_LEGACY_SQLITE_PARQUET_ROLLBACK": "1"}, clear=True):
                with self.assertRaises(DuckDBRouteError):
                    resolve_duckdb_path("L1", data_dir)
                with self.assertRaises(DuckDBRouteError):
                    resolve_duckdb_path("L5", data_dir)
                with self.assertRaises(DuckDBRouteError):
                    resolve_duckdb_path("L6", data_dir)
                with self.assertRaises(DuckDBRouteError):
                    resolve_duckdb_path("L7", data_dir)


if __name__ == "__main__":
    unittest.main()
