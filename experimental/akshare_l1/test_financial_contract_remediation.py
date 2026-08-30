from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

import akshare_l1_scaleout as scaleout
import remediate_financial_contract as remediation


class FinancialContractRemediationTests(unittest.TestCase):
    def test_atomic_json_retries_winerror5_with_unique_candidate(self) -> None:
        real_replace = os.replace
        sleeps: list[float] = []
        attempts = {"count": 0}
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "checkpoint.json"

            def flaky_replace(source: Path, destination: Path) -> None:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise PermissionError(5, "Access is denied")
                real_replace(source, destination)

            with patch.object(scaleout.os, "replace", side_effect=flaky_replace):
                scaleout.atomic_json(target, {"ok": True}, sleep=sleeps.append)

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(attempts["count"], 2)
            self.assertEqual(sleeps, [0.05])
            self.assertFalse(target.with_suffix(".json.tmp").exists())
            self.assertEqual(list(target.parent.glob("*.candidate")), [])

    def test_staged_and_running_reset_without_completed_drift(self) -> None:
        checkpoint = {
            "pool": {"total": 4},
            "stocks": {
                "000001.SZ": {"status": "completed", "rows": 1},
                "000002.SZ": {"status": "completed", "rows": 1},
                "600001.SH": {"status": "staged", "rows": 3, "source_rows": 3, "attempts": 1},
                "600002.SH": {"status": "running", "rows": 0, "started_at": "old", "attempts": 2},
            },
        }
        physical = {
            "codes": ["000001.SZ", "000002.SZ"],
            "rows": 2,
            "stock_coverage": 2,
            "db_sha256": "db",
            "code_domain_sha256": remediation.code_domain_sha256(["000001.SZ", "000002.SZ"]),
        }
        repaired, reset = remediation.prepare_checkpoint(checkpoint, physical, "generation", "now")
        self.assertEqual(remediation.status_codes(repaired, "completed"), physical["codes"])
        self.assertEqual(remediation.status_codes(repaired, "pending"), ["600001.SH", "600002.SH"])
        self.assertEqual(reset, {"staged": ["600001.SH"], "running": ["600002.SH"]})
        self.assertFalse(repaired["resume_allowed"])

    def test_three_way_generation_hash_domain_and_count_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = root / "financial_indicator.duckdb"
            checkpoint_path = root / "checkpoint.json"
            manifest_path = root / "manifest.json"
            connection = duckdb.connect(str(db))
            try:
                connection.execute(
                    "CREATE TABLE financial_indicator AS SELECT * FROM VALUES "
                    "('000001.SZ','20260331'),('600000.SH','20260331') AS t(ts_code,report_date)"
                )
            finally:
                connection.close()
            physical = remediation.inspect_physical_db(db)
            checkpoint = {
                "generation_id": "g1",
                "db_sha256": physical["db_sha256"],
                "code_domain_sha256": physical["code_domain_sha256"],
                "stocks": {code: {"status": "completed"} for code in physical["codes"]},
            }
            manifest = {
                "assets": {
                    "financial_indicator": {
                        "status": "experimental/test_not_approved_for_production",
                        "not_approved_for_production": True,
                        "generation_id": "g1",
                        "db_sha256": physical["db_sha256"],
                        "code_domain_sha256": physical["code_domain_sha256"],
                        "rows": physical["rows"],
                        "stock_coverage": physical["stock_coverage"],
                    }
                }
            }
            scaleout.atomic_json(checkpoint_path, checkpoint)
            scaleout.atomic_json(manifest_path, manifest)
            result = remediation.validate_contract(manifest_path, checkpoint_path, db)
            self.assertTrue(result["valid"], result["checks"])

    def test_legacy_tmp_is_quarantined_and_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "checkpoint.json"
            legacy_tmp = target.with_suffix(".json.tmp")
            legacy_tmp.write_text('{"staged": true}', encoding="utf-8")
            quarantined = remediation.quarantine_legacy_tmp(legacy_tmp, "g1")
            self.assertIsNotNone(quarantined)
            self.assertFalse(legacy_tmp.exists())
            self.assertTrue(quarantined.exists())
            scaleout.atomic_json(target, {"fresh": True})
            self.assertFalse(legacy_tmp.exists())
            self.assertEqual(json.loads(quarantined.read_text(encoding="utf-8")), {"staged": True})

    def test_contract_rejects_generation_mismatch(self) -> None:
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
            physical = remediation.inspect_physical_db(db)
            checkpoint = {
                "generation_id": "checkpoint-generation",
                "db_sha256": physical["db_sha256"],
                "code_domain_sha256": physical["code_domain_sha256"],
                "stocks": {"000001.SZ": {"status": "completed"}},
            }
            manifest = {
                "assets": {
                    "financial_indicator": {
                        "status": "experimental/test_not_approved_for_production",
                        "not_approved_for_production": True,
                        "generation_id": "manifest-generation",
                        "db_sha256": physical["db_sha256"],
                        "code_domain_sha256": physical["code_domain_sha256"],
                        "rows": 1,
                        "stock_coverage": 1,
                    }
                }
            }
            checkpoint_path = root / "checkpoint.json"
            manifest_path = root / "manifest.json"
            scaleout.atomic_json(checkpoint_path, checkpoint)
            scaleout.atomic_json(manifest_path, manifest)
            result = remediation.validate_contract(manifest_path, checkpoint_path, db)
            self.assertFalse(result["valid"])
            self.assertFalse(result["checks"]["generation_match"])


if __name__ == "__main__":
    unittest.main()
