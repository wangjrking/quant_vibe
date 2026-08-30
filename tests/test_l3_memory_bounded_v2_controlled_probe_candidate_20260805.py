import tempfile
import unittest
from pathlib import Path

import l3_memory_bounded_v2_controlled_probe_candidate_20260805 as candidate


class ControlledProbeCandidateTests(unittest.TestCase):
    def test_scope_is_canonical_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = candidate.build_probe_scope(temp, "probe-20260805-r1")
            result = candidate.validate_scope(scope)
            self.assertTrue(result["passed"])
            self.assertEqual(scope.hash_bucket_count, 16)
            self.assertEqual(scope.hash_bucket_id, 3)
            self.assertEqual(scope.expected_l2_sha256, candidate.EXPECTED_L2_SHA256)

    def test_noncanonical_or_quarantine_scope_fails_closed(self):
        scope = candidate.ProbeScope(
            run_id="bad",
            workspace="D:/work/quarantine/workspace",
            staging_dir="D:/work/staging",
            report_dir="D:/work/report",
            l2_path="D:/other.duckdb",
            expected_l2_sha256="bad",
        )
        result = candidate.validate_scope(scope)
        self.assertFalse(result["passed"])
        self.assertIn("non_canonical_l2_path", result["errors"])
        self.assertIn("l2_sha256_not_locked", result["errors"])
        self.assertIn("forbidden_input_or_quarantine_path", result["errors"])

    def test_sql_is_registered_read_only_and_bucketed(self):
        with tempfile.TemporaryDirectory() as temp:
            sql = candidate.build_read_only_sql(candidate.build_probe_scope(temp, "r1"))
            self.assertIn("FROM STOCK_DAILY_DATA", sql)
            self.assertIn("BETWEEN DATE '2026-08-03'", sql)
            self.assertIn("MOD(ABS(HASH(stock_code)), 16) = 3", sql)
            self.assertNotIn("ATTACH", sql.upper())

    def test_future_and_negative_shift_columns_fail_closed(self):
        result = candidate.validate_projection_columns(
            ["open_qfq", "ema_qfq_10", "index_2000_post10_close", "foo_shift(-1)"]
        )
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["rejected"]), 2)

    def test_qfq_contract_is_explicit(self):
        result = candidate.validate_projection_columns(
            ["open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"]
            + [f"tech_qfq_{i}" for i in range(74)]
            + ["macdsignal_qfq", "macdhist_qfq"]
            + [f"alpha{i:03d}_qfq" for i in range(191)]
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["required_qfq_contract"]["l2_source_technical"], 74)
        self.assertEqual(result["required_qfq_contract"]["production_feature_technical"], 76)
        self.assertEqual(result["required_qfq_contract"]["gtja"], 191)

    def test_real_gate_requires_all_fingerprints_and_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = candidate.build_probe_scope(temp, "r1")
            result = candidate.validate_probe_contract(
                scope,
                runtime_passed=True,
                process_gate_passed=True,
                wal_gate_passed=True,
                lease_gate_passed=True,
                source_before={"sha256": "same"},
                source_after={"sha256": "changed"},
                output_columns=["open_qfq"],
            )
            self.assertFalse(result["passed"])
            self.assertIn("canonical_l2_fingerprint_drift_or_missing", result["errors"])

    def test_handoff_is_nonexecuting_and_fail_closed_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = candidate.build_probe_scope(temp, "r1")
            handoff = candidate.build_candidate_handoff(scope, "code-sha")
            self.assertTrue(handoff["ready_for_audit_review"])
            self.assertFalse(handoff["probe_started"])
            self.assertFalse(handoff["allow_probe_only"])
            self.assertFalse(handoff["allow_full_rebuild"])
            self.assertFalse(handoff["allow_l3"])


if __name__ == "__main__":
    unittest.main()
