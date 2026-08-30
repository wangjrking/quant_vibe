import json
import tempfile
import time
import unittest
import json
from pathlib import Path
from unittest import mock

import pandas as pd

import l3_memory_bounded_v2_controlled_probe_runner_candidate_20260805_r3 as runner


class Revision3CandidateTests(unittest.TestCase):
    def test_process_gate_excludes_only_current_parent_chain(self):
        class FakeProcess:
            def __init__(self, pid, parent=None):
                self.pid = pid
                self._parent = parent

            def parent(self):
                return self._parent

        chain = FakeProcess(300, FakeProcess(200, FakeProcess(100, None)))
        with mock.patch.object(runner.psutil, "Process", return_value=chain), mock.patch.object(
            runner.os, "getpid", return_value=300
        ):
            self.assertEqual(runner.ancestor_pids(), {100, 200, 300})

    def test_child_identity_command_binds_run_and_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "child_runtime_identity.json"
            output.write_text(json.dumps({"pid": 123}), encoding="utf-8")
            completed = mock.Mock(returncode=0)
            with mock.patch.object(runner.subprocess, "run", return_value=completed) as run, mock.patch.object(
                runner, "runtime_gate"
            ):
                result = runner.spawn_and_verify_child(root, "synthetic-child-binding")
            command = run.call_args.args[0]
            self.assertEqual(result["pid"], 123)
            self.assertIn("--run-id", command)
            self.assertIn("synthetic-child-binding", command)
            self.assertIn("--isolated-root", command)
            self.assertIn(str(root.resolve()), command)

    def test_default_is_plan_only(self):
        result = runner.plan("synthetic-r3")
        self.assertFalse(result["canonical_l2_opened"])
        self.assertFalse(result["allow_probe_only"])

    def test_lease_is_atomic_and_mutual(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = runner.acquire_lease(root, "r3")
            with self.assertRaises(RuntimeError):
                runner.acquire_lease(root, "other")
            runner.release_lease(first)
            self.assertFalse((root / "exclusive_probe.lease.json").exists())

    def test_quarantine_moves_entire_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            (root / "workspace").mkdir(parents=True)
            (root / "staging").mkdir()
            (root / "report").mkdir()
            result = runner.quarantine(root, "synthetic_failure")
            self.assertTrue(result["quarantined"])
            self.assertTrue(result["reuse_prohibited"])
            self.assertFalse(root.exists())
            self.assertTrue(Path(result["path"], "staging").exists())

    def test_fake_lineage_is_rejected(self):
        with self.assertRaises(RuntimeError):
            runner.projection(["trade_date", "stock_code", "index_2000_post10_close"])

    def test_projection_has_no_select_star_and_checks_qfq(self):
        columns = ["trade_date", "stock_code", *sorted(runner.QFQ_PRICES)]
        columns.extend(f"tech_qfq_{i}" for i in range(74))
        result = runner.projection(columns)
        self.assertNotIn("*", result)
        self.assertEqual(len(result), 81)
        self.assertNotIn("SELECT *", runner.sql(result).upper())

    def test_sql_uses_varchar_trade_date_bounds(self):
        query = runner.sql(["trade_date", "stock_code"])
        self.assertIn("trade_date BETWEEN '20260803' AND '20260804'", query)
        self.assertNotIn("DATE '2026-08-03'", query)

    def test_plan_does_not_accept_external_state(self):
        result = runner.plan("r3")
        self.assertNotIn("lease", result)
        self.assertNotIn("child_identity", result)

    def test_recovery_gate_is_fail_closed_below_budget(self):
        before = {"rss_bytes": 10, "available_bytes": runner.REQUIRED_AVAILABLE, "captured_at_ns": 1}
        after = {"rss_bytes": 20, "available_bytes": runner.REQUIRED_AVAILABLE - 1, "captured_at_ns": 2}
        result = runner.recovery_gate(before, after)
        self.assertEqual(result["status"], "failed_closed")
        self.assertFalse(result["recovered"])

    def test_projection_consumes_lineage_rejection(self):
        with mock.patch.object(
            runner,
            "lineage_gate",
            return_value={"rejected_future_columns": ["unknown_shifted_field"]},
        ):
            with self.assertRaises(RuntimeError):
                runner.projection(["trade_date", "stock_code", "unknown_shifted_field"])

    def test_authorization_requires_explicit_owner_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "probe-root"
            auth = {
                "allow_probe_only": True,
                "scope": runner.PROBE_SCOPE,
                "runner_sha256": runner.sha256(Path(runner.__file__).resolve()),
                "grant_id": "synthetic",
                "approval_source": "commander-owner",
            }
            with self.assertRaises(RuntimeError):
                runner.validate_probe_authorization(auth, root, "synthetic")

    def test_owner_approval_replaces_external_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "probe-root"
            auth = {
                "allow_probe_only": True,
                "owner_approved": True,
                "approval_source": "commander-owner",
                "scope": runner.PROBE_SCOPE,
                "runner_sha256": runner.sha256(Path(runner.__file__).resolve()),
                "grant_id": "synthetic",
                "grant_nonce": "synthetic-owner-approval",
                "expires_at_epoch": time.time() + 3600,
                "run_id": "synthetic",
                "isolated_root": str(root.resolve()),
            }
            with mock.patch.object(runner, "audit_release_gate", return_value={"release_id": "synthetic-release"}), mock.patch.object(runner, "consume_grant"):
                result = runner.validate_probe_authorization(auth, root, "synthetic")
            self.assertEqual(result["approval_source"], "commander-owner")

    def test_runner_hash_binding_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "probe-root"
            auth = {
                "allow_probe_only": True,
                "owner_approved": True,
                "approval_source": "commander-owner",
                "scope": runner.PROBE_SCOPE,
                "runner_sha256": runner.sha256(Path(runner.__file__).resolve()).upper(),
                "grant_id": "synthetic",
                "grant_nonce": "synthetic-case-binding",
                "expires_at_epoch": time.time() + 3600,
                "run_id": "synthetic",
                "isolated_root": str(root.resolve()),
            }
            with mock.patch.object(runner, "audit_release_gate", return_value={"release_id": "synthetic-release"}), mock.patch.object(runner, "consume_grant"):
                result = runner.validate_probe_authorization(auth, root, "synthetic")
            self.assertEqual(result["grant_id"], "synthetic")

    def test_dependency_hash_binding_is_case_insensitive(self):
        uppercase = {path: value.upper() for path, value in runner.DEPENDENCY_SHA256.items()}
        with mock.patch.object(runner, "DEPENDENCY_SHA256", uppercase):
            observed = runner.dependency_gate()
        self.assertEqual(
            {path: value.lower() for path, value in observed.items()},
            {path: value.lower() for path, value in uppercase.items()},
        )

    def test_audit_release_is_required_before_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):
                runner.audit_release_gate(
                    {"grant_id": "synthetic", "grant_nonce": "nonce"},
                    "synthetic",
                    Path(temp),
                )

    def test_quality_contract_keeps_probe_distinct_from_feature_counts(self):
        columns = ["trade_date", "stock_code", *sorted(runner.QFQ_PRICES)]
        columns.extend(f"tech_qfq_{i}" for i in range(74))
        frame = pd.DataFrame(
            [["2026-08-03", "000001.SZ", *([1.0] * (len(columns) - 2))]],
            columns=columns,
        )
        quality = runner.validate_output(frame, columns)
        self.assertEqual(quality["source_rows"], quality["output_rows"])
        self.assertTrue(quality["key_domain_closed"])
        self.assertEqual(quality["qfq_contract"]["l2_source_technical"], 74)
        self.assertEqual(quality["qfq_contract"]["production_feature_technical"], 76)
        self.assertEqual(quality["qfq_contract"]["derived_technical"], 2)
        self.assertEqual(quality["qfq_contract"]["gtja_qfq"], 191)
        self.assertEqual(
            quality["qfq_contract"]["status"],
            "l2_input_contract_verified; downstream_feature_counts_require_l3_output_audit",
        )
        self.assertEqual(quality["source_null_profile"], quality["output_null_profile"])


if __name__ == "__main__":
    unittest.main()
