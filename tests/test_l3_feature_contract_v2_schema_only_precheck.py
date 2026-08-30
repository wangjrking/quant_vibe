from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock
import subprocess
import os

import l3_feature_contract_v2_schema_only_precheck as precheck


class SchemaOnlyPrecheckTests(unittest.TestCase):
    def test_runtime_gate_rejects_conda_provenance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "runtime provenance drift"):
            precheck._runtime_gate(
                {
                    "sys.executable": str(precheck.ISOLATED_RUNTIME.resolve()),
                    "sys.prefix": str(precheck.ISOLATED_RUNTIME.parent.resolve()),
                    "sys.base_prefix": "C:/Users/wangj/.conda/envs/my_quant",
                    "stdlib": "C:/Users/wangj/.conda/envs/my_quant/Lib",
                    "unittest": "C:/Users/wangj/.conda/envs/my_quant/Lib/unittest/__init__.py",
                    "duckdb": str(precheck.ISOLATED_RUNTIME.parent / "Lib/site-packages/duckdb/__init__.py"),
                }
            )

    def test_schema_summary_counts_qfq_columns(self) -> None:
        schema = [
            {"index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"index": 1, "name": "trade_date", "type": "DATE", "not_null": False},
            {"index": 2, "name": "open_qfq", "type": "DOUBLE", "not_null": False},
            {"index": 3, "name": "ema_qfq_10", "type": "DOUBLE", "not_null": False},
            {"index": 4, "name": "macdhist_qfq", "type": "DOUBLE", "not_null": False},
        ]
        summary = precheck._schema_summary(schema)
        self.assertEqual(summary["qfq_price_count"], 1)
        self.assertEqual(summary["qfq_technical_count"], 2)
        self.assertEqual(summary["qfq_total_count"], 3)

    def test_markdown_report_contains_hash_and_counts(self) -> None:
        payload = {
            "run_id": "demo-run",
            "status": "schema_only_precheck_passed",
            "target_trade_date": "20260804",
            "l2": {
                "path": "D:/demo/l2.duckdb",
                "table": "STOCK_DAILY_DATA",
                "file_state_before": {"sha256": "abc"},
            },
            "ordered_l2_schema_hash": "schemahash",
            "ordered_l2_schema_summary": {
                "column_count": 100,
                "key_order": ["stock_code", "trade_date"],
                "qfq_price_count": 5,
                "qfq_technical_count": 74,
                "qfq_total_count": 79,
            },
            "static_sql_safety_gate": {
                "no_l2_row_scan": True,
            },
        }
        text = precheck._markdown_report(payload)
        self.assertIn("schemahash", text)
        self.assertIn("74", text)
        self.assertIn("no_l2_row_scan", text)

    def test_quarantine_root_marks_reuse_prohibited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            (root / "evidence.txt").write_text("x", encoding="utf-8")
            result = precheck._quarantine_root(root, "boom")
            self.assertTrue(result["quarantined"])
            self.assertTrue(result["reuse_prohibited"])
            self.assertTrue(Path(result["path"]).exists())

    def test_run_schema_only_precheck_refuses_existing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            report = Path(tmp) / "report"
            root.mkdir()
            args = mock.Mock(
                isolated_root=str(root),
                report_dir=str(report),
                run_id="run",
                target_trade_date="20260804",
                expected_l2_sha256=precheck.EXPECTED_L2_SHA256,
            )
            with self.assertRaisesRegex(RuntimeError, "requires absent isolated root and report dir"):
                precheck.run_schema_only_precheck(args)

    def test_emit_runtime_identity_does_not_require_report_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "child.json"
            command = [
                str(precheck.ISOLATED_RUNTIME),
                str(Path(precheck.__file__).resolve()),
                "--run-id",
                "demo-run",
                "--isolated-root",
                str(Path(tmp) / "root"),
                "--emit-runtime-identity",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=str(precheck.PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
                env={**os.environ, "PYTHONPATH": str(precheck.PROJECT_ROOT / "quant/main")},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())

    def test_static_sql_safety_gate_rejects_row_scan_tokens(self) -> None:
        with mock.patch.object(precheck.Path, "read_text", return_value="SELECT COUNT(*) FROM STOCK_DAILY_DATA"):
            with self.assertRaisesRegex(RuntimeError, "forbidden row-scan SQL tokens"):
                precheck._static_sql_safety_gate()

    def test_runner_source_contains_no_forbidden_row_scan_tokens(self) -> None:
        source = Path(precheck.__file__).read_text(encoding="utf-8").upper()
        for token in precheck.FORBIDDEN_ROW_SCAN_SQL_TOKENS:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
