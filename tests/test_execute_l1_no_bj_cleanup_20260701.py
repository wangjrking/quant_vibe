import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execute_l1_no_bj_cleanup_20260701 import (
    EXPECTED_PYTHON,
    assert_materialized_duckdb_tables,
    ensure_expected_python,
    materialize_l1_duckdb_table_files,
    promote_staged_duckdb_root,
    write_failure_reports,
)


class ExecuteL1NoBjCleanupTests(unittest.TestCase):
    def test_ensure_expected_python_accepts_workspace_venv(self):
        with patch.object(sys, "executable", str(EXPECTED_PYTHON)):
            with patch.dict(os.environ, {}, clear=False):
                result = ensure_expected_python()

        self.assertEqual(result["status"], "workspace_venv_only")
        self.assertEqual(Path(result["expected_python"]), EXPECTED_PYTHON)

    def test_ensure_expected_python_rejects_other_interpreter(self):
        with patch.object(sys, "executable", r"C:\Users\wangj\.conda\envs\my_quant\python.exe"):
            with patch.dict(os.environ, {}, clear=False):
                with self.assertRaises(RuntimeError):
                    ensure_expected_python()

    def test_assert_materialized_duckdb_tables_requires_full_table_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            only_one = root / "daily_data.duckdb"
            with duckdb.connect(str(only_one)) as conn:
                conn.execute('CREATE TABLE "daily_data"(ts_code TEXT, trade_date TEXT)')

            with self.assertRaises(RuntimeError):
                assert_materialized_duckdb_tables(root)

    def test_promote_staged_duckdb_root_archives_previous_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            staging_root = tmp_path / "staging"
            final_root = tmp_path / "final"
            backup_root = tmp_path / "backup"
            staging_root.mkdir()
            final_root.mkdir()
            (staging_root / "daily_data.duckdb").write_text("new", encoding="utf-8")
            (final_root / "daily_data.duckdb").write_text("old", encoding="utf-8")

            result = promote_staged_duckdb_root(staging_root, final_root, backup_root)

            self.assertFalse(staging_root.exists())
            self.assertTrue(final_root.exists())
            self.assertEqual((final_root / "daily_data.duckdb").read_text(encoding="utf-8"), "new")
            self.assertIsNotNone(result["previous_root_backup"])
            archived = Path(result["previous_root_backup"])
            self.assertTrue(archived.exists())
            self.assertEqual((archived / "daily_data.duckdb").read_text(encoding="utf-8"), "old")

    def test_materialize_l1_duckdb_table_files_uses_active_source_table_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            source_root = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            source_root.mkdir(parents=True)
            backup_root = data_dir / "backups" / "case1"
            target_root = data_dir / "production_assets" / "duckdb" / "l1_raw_tables_build"

            source_path = source_root / "daily_data.duckdb"
            with duckdb.connect(str(source_path)) as conn:
                conn.execute('CREATE TABLE "daily_data"(ts_code TEXT, trade_date TEXT, close DOUBLE)')
                conn.execute(
                    'INSERT INTO "daily_data" VALUES '
                    "('000001.SZ', '20260630', 10.0), "
                    "('920001.BJ', '20260630', 11.0)"
                )

            results = materialize_l1_duckdb_table_files(
                data_dir,
                backup_root,
                new_duckdb_root=target_root,
            )

            self.assertIn("daily_data", results)
            self.assertEqual(results["daily_data"]["source_duckdb_path"], str(source_path))
            with duckdb.connect(str(target_root / "daily_data.duckdb"), read_only=True) as conn:
                rows = conn.execute('SELECT ts_code FROM "daily_data" ORDER BY ts_code').fetchall()
            self.assertEqual(rows, [("000001.SZ",)])

    def test_write_failure_reports_persists_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "reports").mkdir(parents=True, exist_ok=True)
            payload = {
                "error_type": "RuntimeError",
                "error_message": "boom",
                "backup_root": str(data_dir / "backup"),
                "staging_root": str(data_dir / "staging"),
                "final_root": str(data_dir / "final"),
                "traceback": "traceback body",
            }

            report_json, report_md = write_failure_reports(data_dir, payload)

            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            self.assertIn("RuntimeError", report_json.read_text(encoding="utf-8"))
            self.assertIn("traceback body", report_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
