import unittest
from pathlib import Path

from project_paths import DEFAULT_DATA_DIR, PROJECT_ROOT, QUANT_ROOT, resolve_data_dir, resolve_data_path


class ProjectPathsTests(unittest.TestCase):
    def test_default_data_dir_points_to_quant_data_file(self):
        self.assertEqual(PROJECT_ROOT.name, "main")
        self.assertEqual(QUANT_ROOT.name, "quant")
        self.assertEqual(DEFAULT_DATA_DIR, QUANT_ROOT / "data_file")
        self.assertEqual(resolve_data_dir(), QUANT_ROOT / "data_file")

    def test_resolve_data_path_supports_stock_pool_name_under_data_dir(self):
        self.assertEqual(
            resolve_data_path("stock_pool_all_a.csv"),
            (QUANT_ROOT / "data_file" / "stock_pool_all_a.csv").resolve(),
        )

    def test_resolve_data_path_accepts_legacy_data_file_relative_prefix(self):
        self.assertEqual(
            resolve_data_path("data_file/stock_pool_all_a.csv"),
            (QUANT_ROOT / "data_file" / "stock_pool_all_a.csv").resolve(),
        )

    def test_resolve_data_path_can_use_explicit_data_dir(self):
        custom_dir = Path(r"D:\tmp\custom_data_dir")
        self.assertEqual(
            resolve_data_path("stock_pool_all_a.csv", data_dir=custom_dir),
            (custom_dir / "stock_pool_all_a.csv").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
