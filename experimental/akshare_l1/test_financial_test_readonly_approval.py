from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

import publish_financial_test_readonly_approval as publisher


class FinancialTestReadonlyApprovalTests(unittest.TestCase):
    def test_bundle_sets_identical_approval_metadata(self) -> None:
        manifest = {
            "assets": {
                "financial_indicator": {
                    "coverage": {},
                    "status": "experimental/test_pending",
                }
            }
        }
        checkpoint = {"stocks": {}, "resume_allowed": False}
        approved_manifest, approved_checkpoint = publisher.prepare_approved_bundle(
            manifest, checkpoint, "generation-1", "2026-07-18T15:00:00+08:00"
        )
        asset = approved_manifest["assets"]["financial_indicator"]
        for key, value in publisher.approval_fields("generation-1", "2026-07-18T15:00:00+08:00").items():
            self.assertEqual(asset[key], value)
            self.assertEqual(approved_checkpoint[key], value)
        self.assertTrue(asset["status"].startswith("experimental/test"))
        self.assertFalse(approved_checkpoint["resume_allowed"])

    def test_atomic_publish_stops_after_one_permission_error_without_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "metadata.json"
            with patch.object(publisher.os, "replace", side_effect=PermissionError(5, "Access is denied")) as replace:
                with self.assertRaises(PermissionError):
                    publisher.atomic_json_once(target, {"approved": True})
            self.assertEqual(replace.call_count, 1)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob("*.candidate")), [])

    def test_pointer_must_target_standard_files_in_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generation_dir = Path(temporary) / "generation-1"
            generation_dir.mkdir()
            pointer = {
                "generation_id": "generation-1",
                "manifest_path": str(generation_dir / "manifest.json"),
                "checkpoint_path": str(generation_dir / "financial_indicator_batch.json"),
                "db_sha256": publisher.EXPECTED_DB_SHA256,
                "code_domain_sha256": publisher.EXPECTED_DOMAIN_SHA256,
                "test_read_only_approval": "approved",
                "audit_task_id": publisher.AUDIT_TASK_ID,
                "audit_thread_id": publisher.AUDIT_THREAD_ID,
                "not_approved_for_production": True,
                "resume_allowed": False,
            }
            checks = publisher.validate_pointer(pointer, generation_dir, "generation-1")
            self.assertTrue(all(checks.values()), checks)
            pointer["manifest_path"] = str(Path(temporary) / "alternate.json")
            self.assertFalse(publisher.validate_pointer(pointer, generation_dir, "generation-1")["manifest_path"])

    def test_bundle_validator_rejects_one_of_four_approval_drifts(self) -> None:
        original_db = publisher.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                db = root / "financial_indicator.duckdb"
                connection = duckdb.connect(str(db))
                try:
                    connection.execute(
                        "CREATE TABLE financial_indicator AS SELECT '000001.SZ' ts_code, '20260331' report_date"
                    )
                finally:
                    connection.close()
                physical = publisher.contract.inspect_physical_db(db)
                manifest = {
                    "mode": "experimental/test",
                    "not_approved_for_production": True,
                    "assets": {
                        "financial_indicator": {
                            "status": "experimental/test_read_only_approved",
                            "not_approved_for_production": True,
                            "generation_id": "g1",
                            "db_sha256": physical["db_sha256"],
                            "code_domain_sha256": physical["code_domain_sha256"],
                            "rows": 1,
                            "stock_coverage": 1,
                            "test_read_only_approval": "approved",
                            "audit_task_id": publisher.AUDIT_TASK_ID,
                            "audit_thread_id": publisher.AUDIT_THREAD_ID,
                            "approved_at": "now",
                            "approval_scope": publisher.APPROVAL_SCOPE,
                            "production_approved": False,
                            "resume_allowed": False,
                        }
                    },
                }
                checkpoint = {
                    **manifest["assets"]["financial_indicator"],
                    "pool": {"total": 1},
                    "stocks": {"000001.SZ": {"status": "completed"}},
                }
                checkpoint["test_read_only_approval"] = "denied"
                manifest_path = root / "manifest.json"
                checkpoint_path = root / "checkpoint.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
                publisher.DB_PATH = db
                with patch.multiple(
                    publisher,
                    EXPECTED_ROWS=1,
                    EXPECTED_STOCKS=1,
                    EXPECTED_POOL=1,
                    EXPECTED_DB_SHA256=physical["db_sha256"],
                    EXPECTED_DOMAIN_SHA256=physical["code_domain_sha256"],
                ):
                    result = publisher.validate_approval_bundle(manifest_path, checkpoint_path, "g1", "now")
                self.assertFalse(result["valid"])
                self.assertFalse(result["checks"]["checkpoint_approval_fields"])
        finally:
            publisher.DB_PATH = original_db


if __name__ == "__main__":
    unittest.main()
