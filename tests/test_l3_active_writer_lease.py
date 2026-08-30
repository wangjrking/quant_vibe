import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import l3_active_writer_lease as writer_lease
import l3_duckdb_sync
import scan_l3_write_bypass as bypass_scan


class ActiveL3WriterLeaseTests(unittest.TestCase):
    def test_shared_lease_is_exclusive_and_valid_for_protected_sink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "l3_feature_current.duckdb"
            lease_path = root / "writer_lease.json"
            with mock.patch.object(writer_lease, "PROTECTED_ACTIVE_L3_PATHS", {protected}):
                lease = writer_lease.acquire_active_l3_writer_lease(
                    workflow_run_id="run-1",
                    workspace=root / "workspace",
                    report_dir=root / "reports",
                    process_role="test_writer",
                    lease_path=lease_path,
                )
                validation = writer_lease.require_active_l3_writer_lease_for_path(
                    protected,
                    lease_path=lease_path,
                )
                self.assertEqual(validation["lease_nonce"], lease["lease_nonce"])
                with self.assertRaisesRegex(RuntimeError, "lease already exists"):
                    writer_lease.acquire_active_l3_writer_lease(
                        workflow_run_id="run-2",
                        workspace=root / "workspace-2",
                        report_dir=root / "reports-2",
                        process_role="test_writer_2",
                        lease_path=lease_path,
                    )
                release = writer_lease.release_active_l3_writer_lease(lease)
            self.assertTrue(release["lease_absent_after_release"])

    def test_protected_sink_without_lease_fails_before_writable_connect(self):
        with tempfile.TemporaryDirectory() as tmp:
            protected = Path(tmp) / "l3_feature_current.duckdb"
            lease_path = Path(tmp) / "missing_lease.json"
            with mock.patch.object(writer_lease, "PROTECTED_ACTIVE_L3_PATHS", {protected}), mock.patch.object(
                writer_lease, "ACTIVE_L3_WRITER_LEASE_PATH", lease_path
            ), mock.patch.object(l3_duckdb_sync, "require_active_l3_writer_lease_for_path", side_effect=lambda path: writer_lease.require_active_l3_writer_lease_for_path(path, lease_path=lease_path)), mock.patch(
                "duckdb.connect"
            ) as connect:
                with self.assertRaisesRegex(RuntimeError, "requires writer lease"):
                    l3_duckdb_sync._connect_writable(protected)
            connect.assert_not_called()

    def test_foreign_or_tampered_lease_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "l3_feature_current.duckdb"
            lease_path = root / "writer_lease.json"
            with mock.patch.object(writer_lease, "PROTECTED_ACTIVE_L3_PATHS", {protected}):
                lease = writer_lease.acquire_active_l3_writer_lease(
                    workflow_run_id="run-1",
                    workspace=root / "workspace",
                    report_dir=root / "reports",
                    process_role="test_writer",
                    lease_path=lease_path,
                )
                payload = json.loads(lease_path.read_text(encoding="utf-8"))
                payload["owner_pid"] = os.getpid() + 1000
                lease_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "not owned by the current process"):
                    writer_lease.require_active_l3_writer_lease_for_path(protected, lease_path=lease_path)
                lease_path.unlink()

    def test_non_active_candidate_path_does_not_require_global_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.duckdb"
            with mock.patch.object(writer_lease, "PROTECTED_ACTIVE_L3_PATHS", set()):
                result = writer_lease.require_active_l3_writer_lease_for_path(candidate)
            self.assertEqual(result["status"], "not_required")

    def test_static_bypass_scan_covers_required_entrypoints(self):
        result = bypass_scan.scan_l3_write_bypass()
        self.assertTrue(result["passed"], result)
        by_script = {item["script"]: item for item in result["entrypoints"]}
        for script in (
            "deliver_l3_target_date_duckdb_mainline.py",
            "l3_duckdb_sync.py",
            "refresh_l3_active_duckdb_full_delivery.py",
            "rebuild_l3_full_duckdb_mainline.py",
            "apply_production_asset_pair_change.py",
            "apply_production_asset_change.py",
        ):
            self.assertIn(script, by_script)
            self.assertTrue(by_script[script]["exists"])
        self.assertEqual(
            by_script["rebuild_l3_full_duckdb_mainline.py"]["active_write_policy"],
            "candidate_only_no_active_write",
        )
        self.assertEqual(result["unregistered_bypasses"], [])

    def test_static_bypass_scan_blocks_unregistered_direct_active_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            main_dir = Path(tmp)
            bypass = main_dir / "unregistered_writer.py"
            bypass.write_text(
                "import os\n"
                "target = 'production_assets/duckdb/l3_feature_current.duckdb'\n"
                "os.replace('candidate.duckdb', target)\n",
                encoding="utf-8",
            )
            with mock.patch.object(bypass_scan, "L3_MUTATION_ENTRY_REGISTRY", {}):
                result = bypass_scan.scan_l3_write_bypass(main_dir)
            self.assertFalse(result["passed"])
            self.assertEqual([item["script"] for item in result["unregistered_bypasses"]], [bypass.name])


if __name__ == "__main__":
    unittest.main()
