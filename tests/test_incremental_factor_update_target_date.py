import unittest

from incremental_factor_update_target_date import parse_args
from project_paths import resolve_data_dir


class IncrementalFactorUpdateTargetDateTests(unittest.TestCase):
    def test_default_asset_paths_follow_repo_data_dir(self):
        args = parse_args(["--target-date", "20260617"])
        data_dir = resolve_data_dir()

        self.assertEqual(args.raw_parts_dir, str(data_dir / "raw_factor_by_stock_parts"))
        self.assertEqual(args.production_parts_dir, str(data_dir / "production_factor_parts"))


if __name__ == "__main__":
    unittest.main()
