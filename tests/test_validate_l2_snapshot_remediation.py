from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import duckdb

from rebuild_l2_stock_daily_duckdb_mainline import collect_output_metrics
from validate_l2_snapshot_remediation import _sha256, build_evidence


class SnapshotRemediationEvidenceTest(unittest.TestCase):
    def _create_db(self, path: Path) -> None:
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "CREATE TABLE STOCK_DAILY_DATA "
                "(stock_code VARCHAR, trade_date VARCHAR)"
            )
            conn.execute(
                "INSERT INTO STOCK_DAILY_DATA VALUES ('301583.SZ', '20260716')"
            )

    def _write_manifest(self, path: Path, snapshot: Path) -> None:
        payload = {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sha256": _sha256(snapshot),
            "table_metrics": collect_output_metrics(
                snapshot,
                target_trade_date="20260716",
                special_code="301583.SZ",
            ),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_existing_pre_mutation_snapshot_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            active = root / "active.duckdb"
            snapshot = root / "snapshot.duckdb"
            manifest = root / "manifest.json"
            validation = root / "validation.json"
            self._create_db(active)
            shutil.copy2(active, snapshot)
            time.sleep(0.05)
            self._write_manifest(manifest, snapshot)
            time.sleep(0.05)
            with duckdb.connect(str(active)) as conn:
                conn.execute(
                    "INSERT INTO STOCK_DAILY_DATA VALUES ('000001.SZ', '20260715')"
                )
            time.sleep(0.05)
            validation.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

            result = build_evidence(
                target_trade_date="20260716",
                active_path=active,
                snapshot_path=snapshot,
                original_manifest_path=manifest,
                validation_path=validation,
            )

            self.assertEqual(result["status"], "verified_existing_pre_mutation_snapshot")
            self.assertTrue(all(result["checks"].values()))
            self.assertFalse(result["active_touched_by_remediation"])

    def test_active_file_cannot_be_used_as_its_own_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            active = root / "active.duckdb"
            manifest = root / "manifest.json"
            validation = root / "validation.json"
            self._create_db(active)
            self._write_manifest(manifest, active)
            time.sleep(0.05)
            validation.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

            result = build_evidence(
                target_trade_date="20260716",
                active_path=active,
                snapshot_path=active,
                original_manifest_path=manifest,
                validation_path=validation,
            )

            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["checks"]["snapshot_path_distinct_from_active"])
            self.assertFalse(result["checks"]["snapshot_file_id_distinct_from_active"])


if __name__ == "__main__":
    unittest.main()
