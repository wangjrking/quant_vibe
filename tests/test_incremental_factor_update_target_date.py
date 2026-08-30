import unittest
from unittest import mock

from incremental_factor_update_target_date import parse_args, resolve_legacy_parquet_parts_dirs
from project_paths import resolve_data_dir


class IncrementalFactorUpdateTargetDateTests(unittest.TestCase):
    def test_default_asset_paths_follow_repo_data_dir(self):
        args = parse_args(["--target-date", "20260617"])
        data_dir = resolve_data_dir()

        self.assertEqual(args.db_path, str(data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"))
        self.assertIsNone(args.raw_parts_dir)
        self.assertIsNone(args.production_parts_dir)

    def test_legacy_parquet_parts_are_fail_closed_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                resolve_legacy_parquet_parts_dirs(None, None, data_dir=resolve_data_dir())

    def test_legacy_parquet_parts_require_explicit_opt_in(self):
        data_dir = resolve_data_dir()
        with mock.patch.dict("os.environ", {"QUANT_ALLOW_LEGACY_L3_PARQUET_PARTS": "1"}):
            raw_dir, production_dir = resolve_legacy_parquet_parts_dirs(None, None, data_dir=data_dir)
        self.assertEqual(raw_dir, data_dir / "raw_factor_by_stock_parts")
        self.assertEqual(production_dir, data_dir / "production_factor_parts")


if __name__ == "__main__":
    unittest.main()
