import importlib.util
import tempfile
import unittest
from pathlib import Path

from duckdb_asset_route import resolve_duckdb_zone_path
from tools.init_duckdb_zone import initialize_duckdb_zone


@unittest.skipIf(importlib.util.find_spec("duckdb") is None, "duckdb is not installed")
class InitDuckDBZoneTests(unittest.TestCase):
    def test_initialize_creates_experiment_zone_file_with_metadata(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data_file"
            result = initialize_duckdb_zone("experiment", data_dir=data_dir)
            db_path = resolve_duckdb_zone_path("experiment", data_dir=data_dir, require_exists=True)

            self.assertEqual(Path(result["db_path"]), db_path)
            self.assertEqual(result["zone"], "experiment")
            with duckdb.connect(str(db_path), read_only=True) as conn:
                count = conn.execute("SELECT COUNT(*) FROM duckdb_zone_metadata").fetchone()[0]
                zone = conn.execute("SELECT zone FROM duckdb_zone_metadata LIMIT 1").fetchone()[0]

        self.assertEqual(count, 1)
        self.assertEqual(zone, "experiment")

    def test_initialize_creates_archive_zone_file_with_metadata(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data_file"
            result = initialize_duckdb_zone("archive", data_dir=data_dir)
            db_path = resolve_duckdb_zone_path("archive", data_dir=data_dir, require_exists=True)

            self.assertEqual(Path(result["db_path"]), db_path)
            with duckdb.connect(str(db_path), read_only=True) as conn:
                zone = conn.execute("SELECT zone FROM duckdb_zone_metadata LIMIT 1").fetchone()[0]

        self.assertEqual(zone, "archive")


if __name__ == "__main__":
    unittest.main()
