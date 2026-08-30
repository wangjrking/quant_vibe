from __future__ import annotations

import sys
import inspect
import unittest
from pathlib import Path
from unittest import mock


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import integrate_l2_latest_and_recompute_qfq as l2


class RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params=None):
        self.calls.append((sql, params))
        return self


class L2QfqPerformancePlanTest(unittest.TestCase):
    def test_default_integration_does_not_run_historical_qfq_updates(self):
        source = inspect.getsource(l2.integrate_target_date_incremental)

        self.assertNotIn("recompute_qfq_prices_by_period(", source)
        self.assertNotIn("sync_qfq_indicators_by_period(", source)
        self.assertIn('"processing_scope": "target_trade_date_only"', source)
        self.assertIn('"historical_scan": False', source)

    def test_full_history_uses_year_periods(self):
        periods = l2.iter_year_ranges("20100104", "20260720")

        self.assertEqual(len(periods), 17)
        self.assertEqual(periods[0], ("20100104", "20101231"))
        self.assertEqual(periods[-1], ("20260101", "20260720"))

    def test_indicator_sync_updates_all_columns_once_per_year(self):
        conn = RecordingConnection()
        columns = [f"indicator_{index}_qfq" for index in range(74)]
        periods = [("20250101", "20251231"), ("20260101", "20260720")]

        transaction_count = l2.sync_qfq_indicators_by_period(conn, columns, periods)

        self.assertEqual(transaction_count, 2)
        self.assertEqual(sum(sql == "BEGIN TRANSACTION" for sql, _ in conn.calls), 2)
        updates = [(sql, params) for sql, params in conn.calls if "UPDATE STOCK_DAILY_DATA" in sql]
        self.assertEqual(len(updates), 2)
        for sql, _ in updates:
            self.assertIn('"indicator_0_qfq" = f."indicator_0_qfq"', sql)
            self.assertIn('"indicator_73_qfq" = f."indicator_73_qfq"', sql)

    def test_price_recompute_runs_once_per_year(self):
        conn = RecordingConnection()
        periods = [("20250101", "20251231"), ("20260101", "20260720")]

        transaction_count = l2.recompute_qfq_prices_by_period(conn, periods)

        self.assertEqual(transaction_count, 2)
        updates = [sql for sql, _ in conn.calls if "UPDATE STOCK_DAILY_DATA" in sql]
        self.assertEqual(len(updates), 2)

    def test_active_lock_gate_fails_before_processing(self):
        with mock.patch.object(l2.duckdb, "connect", side_effect=RuntimeError("held by PID 123")):
            with mock.patch.object(
                l2,
                "process_identity",
                return_value={"pid": 123, "create_time": 1.0, "command_line": ["unknown.exe"]},
            ):
                with self.assertRaisesRegex(RuntimeError, "write-lock gate failed"):
                    l2.require_duckdb_write_lock(Path("active.duckdb"))

    def test_runtime_gate_accepts_project_venv_and_project_duckdb(self):
        with mock.patch.object(l2.sys, "executable", str(l2.PROJECT_VENV_PYTHON)):
            with mock.patch.object(l2.sys, "prefix", str(l2.PROJECT_VENV)):
                result = l2.require_project_venv_runtime()

        self.assertTrue(result["passed"])
        self.assertTrue(result["duckdb_from_project_venv"])

    def test_runtime_gate_rejects_direct_conda_executor(self):
        with mock.patch.object(
            l2.sys,
            "executable",
            r"C:\Users\wangj\.conda\envs\my_quant\python.exe",
        ):
            with self.assertRaisesRegex(RuntimeError, "write-lock gate failed"):
                try:
                    l2.require_project_venv_runtime()
                except RuntimeError as exc:
                    raise RuntimeError(f"write-lock gate failed: {exc}") from exc

    def test_process_gate_distinguishes_project_writer_and_unknown_writer(self):
        project_writer = {
            "pid": 321,
            "create_time": 10.0,
            "command_line": [
                str(l2.PROJECT_VENV_PYTHON),
                str(MAIN_DIR / "integrate_l2_latest_and_recompute_qfq.py"),
            ],
            "parent_lineage": [],
        }
        unknown_writer = {
            "pid": 654,
            "create_time": 11.0,
            "command_line": [r"C:\Python\python.exe", "other_writer.py"],
            "parent_lineage": [],
        }

        project = l2.classify_lock_process(
            project_writer,
            current_pid=999,
            current_create_time=1.0,
        )
        unknown = l2.classify_lock_process(
            unknown_writer,
            current_pid=999,
            current_create_time=1.0,
        )

        self.assertEqual(project["role"], "project_venv_business_writer")
        self.assertTrue(project["fail_closed"])
        self.assertEqual(unknown["role"], "unknown_writer")
        self.assertTrue(unknown["fail_closed"])

    def test_process_gate_recognizes_project_readonly_collector(self):
        identity = {
            "pid": 777,
            "create_time": 12.0,
            "command_line": [
                str(l2.PROJECT_VENV_PYTHON),
                str(MAIN_DIR / "capture_l2_metrics.py"),
            ],
            "parent_lineage": [],
        }

        result = l2.classify_lock_process(
            identity,
            current_pid=999,
            current_create_time=1.0,
        )

        self.assertEqual(result["role"], "project_venv_readonly_collector")
        self.assertTrue(result["fail_closed"])

    def test_process_gate_recognizes_current_entry_by_pid_and_create_time(self):
        identity = {
            "pid": 100,
            "create_time": 20.0,
            "command_line": [str(l2.PROJECT_VENV_PYTHON), "entry.py"],
            "parent_lineage": [],
        }

        result = l2.classify_lock_process(
            identity,
            current_pid=100,
            current_create_time=20.0,
        )

        self.assertEqual(result["role"], "current_entry_preflight")
        self.assertFalse(result["fail_closed"])


if __name__ == "__main__":
    unittest.main()
