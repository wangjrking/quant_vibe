import unittest

from l3_memory_bounded_v2_candidate import (
    MemoryBudgetContract,
    build_executor_contract,
    build_quarantine_manifest,
    build_partition_plan,
    build_fragment_metadata,
    estimate_peak_memory_bytes,
    validate_candidate_stage_state,
    validate_fragment_metrics,
    validate_global_candidate_metrics,
    validate_gtja_chunk_contract,
    validate_memory_budget_contract,
    validate_runtime_contract,
    validate_stage_transition,
    validate_fanin_equivalence,
    validate_fanin_merge_contract,
    validate_fingerprint_stability,
    validate_short_lived_worker_lifecycle,
    validate_worker_recovery,
    validate_fragment_metadata,
)


class L3MemoryBoundedV2CandidateTests(unittest.TestCase):
    def test_partition_plan_is_disjoint_and_excludes_bj(self):
        dates = ["20200102", "20200103", "20200106"]
        codes = ["000001.SZ", "600000.SH", "920001.BJ"]
        plan = build_partition_plan(dates, codes, date_chunk_days=2, stock_buckets=4)
        self.assertTrue(plan)
        self.assertTrue(all(item["date_start"] <= item["date_end"] for item in plan))
        self.assertTrue(all(item["stock_count"] >= 0 for item in plan))
        self.assertEqual(sum(item["stock_count"] for item in plan), 4)

    def test_memory_estimate_is_bounded_and_positive(self):
        estimate = estimate_peak_memory_bytes(max_rows_per_fragment=80000, selected_column_count=534)
        self.assertGreater(estimate, 0)
        self.assertLess(estimate, 8 * 1024**3)

    def test_fragment_gate_rejects_schema_or_future_leakage(self):
        metrics = {
            "source_rows": 10,
            "output_rows": 10,
            "duplicate_key_groups": 0,
            "bj_rows": 0,
            "schema_hash": "wrong",
            "column_count": 838,
            "future_or_label_columns": ["index_2000_post10_close"],
        }
        with self.assertRaisesRegex(ValueError, "schema_hash_mismatch"):
            validate_fragment_metrics(metrics, expected_schema_hash="expected", expected_columns=838)

    def test_fragment_gate_passes_only_clean_metadata(self):
        metrics = {
            "source_rows": 10,
            "output_rows": 10,
            "duplicate_key_groups": 0,
            "bj_rows": 0,
            "schema_hash": "expected",
            "column_count": 838,
            "future_or_label_columns": [],
            "write_active": False,
            "write_registry": False,
        }
        self.assertTrue(validate_fragment_metrics(metrics, expected_schema_hash="expected", expected_columns=838)["approved_for_merge"])

    def test_global_gate_requires_74_plus_2_qfq_contract(self):
        metrics = {
            "duplicate_key_groups": 0,
            "bj_rows": 0,
            "future_or_label_column_count": 0,
            "key_domain_missing": 0,
            "key_domain_extra": 0,
            "column_count": 838,
            "qfq_price_count": 5,
            "l2_source_qfq_technical_count": 74,
            "production_feature_qfq_technical_count": 76,
            "derived_qfq_technical_count": 2,
            "gtja_qfq_count": 191,
            "label_write_called": False,
            "active_switch_called": False,
            "registry_change": False,
            "production_paths_opened": False,
            "wal_present": False,
            "unknown_writer": False,
            "label_target_rows": 0,
            "label_max_trade_date": "20260616",
            "label_maturity_max_trade_date": "20260616",
        }
        fingerprints = {
            name: {
                "before": {"sha256": name, "size_bytes": 10, "mtime_ns": 20},
                "after": {"sha256": name, "size_bytes": 10, "mtime_ns": 20},
            }
            for name in ("active_feature", "active_label", "production_registry")
        }
        self.assertEqual(validate_global_candidate_metrics(metrics, fingerprints=fingerprints)["status"], "passed")

    def test_runtime_rejects_conda_and_accepts_project_venv(self):
        with self.assertRaisesRegex(RuntimeError, "runtime mismatch"):
            validate_runtime_contract(
                executable="C:/Users/wangj/.conda/envs/my_quant/python.exe",
                prefix="C:/Users/wangj/.conda/envs/my_quant",
            )
        self.assertEqual(
            validate_runtime_contract(
                executable="D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804/python.exe",
                prefix="D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804",
                base_prefix="D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804",
                stdlib_path="D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804/Lib",
                unittest_path="D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804/Lib/unittest/__init__.py",
            )["status"],
            "passed",
        )

    def test_runtime_rejects_conda_base_and_stdlib_provenance(self):
        with self.assertRaisesRegex(RuntimeError, "base_prefix mismatch"):
            validate_runtime_contract(
                executable="D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804/python.exe",
                prefix="D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804",
                base_prefix="C:/Users/wangj/.conda/envs/my_quant",
                stdlib_path="C:/Users/wangj/.conda/envs/my_quant/Lib",
                unittest_path="C:/Users/wangj/.conda/envs/my_quant/lib/unittest/__init__.py",
            )

    def test_memory_budget_preserves_64_gib_gate(self):
        result = validate_memory_budget_contract(MemoryBudgetContract())
        self.assertEqual(result["contract"]["dispatch_min_available_bytes"], 64 * 1024**3)
        with self.assertRaisesRegex(ValueError, "workers=1"):
            validate_memory_budget_contract(MemoryBudgetContract(workers=2))

    def test_stage_state_and_transition_fail_closed(self):
        self.assertEqual(validate_stage_transition("raw_fragment", "raw_fanin_merge")["status"], "passed")
        with self.assertRaisesRegex(ValueError, "stage regression"):
            validate_stage_transition("gtja_fragment", "raw_fragment")
        with self.assertRaisesRegex(ValueError, "active_write_forbidden"):
            validate_candidate_stage_state("feature_ctas", {"candidate_only": True, "active_write": True})

    def test_fanin_equivalence_requires_rows_key_hash_and_duplicates(self):
        result = validate_fanin_equivalence(
            source_rows=20,
            fragment_rows=20,
            merged_rows=20,
            source_key_hash="same",
            merged_key_hash="same",
            duplicate_key_groups=0,
        )
        self.assertTrue(result["approved_for_next_stage"])
        with self.assertRaisesRegex(ValueError, "key_hash_mismatch"):
            validate_fanin_equivalence(
                source_rows=20, fragment_rows=20, merged_rows=20,
                source_key_hash="a", merged_key_hash="b", duplicate_key_groups=0,
            )

    def test_gtja_chunk_requires_full_alpha_and_lookback(self):
        result = validate_gtja_chunk_contract(
            date_start="20200101", date_end="20200331", lookback_start="20100104",
            alpha_count=191, output_dates=60,
        )
        self.assertEqual(result["status"], "passed")
        with self.assertRaisesRegex(ValueError, "191"):
            validate_gtja_chunk_contract(
                date_start="20200101", date_end="20200331", lookback_start="20100104",
                alpha_count=190, output_dates=60,
            )

    def test_quarantine_manifest_and_executor_contract_are_candidate_only(self):
        quarantine = build_quarantine_manifest(run_id="run", workspace="w", reason="gate")
        self.assertTrue(quarantine["reuse_prohibited"])
        contract = build_executor_contract(run_id="run", workspace="w", report_dir="r")
        self.assertFalse(contract["execution_started"])
        self.assertTrue(contract["candidate_only"])
        self.assertFalse(contract["active_switch"])
        self.assertFalse(contract["production_paths_opened"])

    def test_runtime_and_worker_recovery_gates_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "system_hard_stop_crossed"):
            validate_worker_recovery(
                {
                    "rss_peak_bytes": 1,
                    "system_available_before_bytes": 64 * 1024**3,
                    "system_available_min_bytes": 19 * 1024**3,
                    "recovery_timeout_seconds": 1,
                    "recovery_completed": True,
                    "system_available_after_exit_bytes": 64 * 1024**3,
                }
            )
        events = [{"state": state, "pid": 1234, "residual": False} for state in
                  ("owned", "launching", "running", "completed", "joined", "validated")]
        self.assertEqual(validate_short_lived_worker_lifecycle(events)["status"], "passed")
        events[-1] = {"state": "validated", "pid": 1234, "residual": True}
        with self.assertRaisesRegex(ValueError, "residual"):
            validate_short_lived_worker_lifecycle(events)

    def test_fixture_fragment_metadata_and_fixed_fanin(self):
        columns = ["trade_date", "stock_code", "close_qfq"]
        metadata = build_fragment_metadata(
            fragment_id="fixture-0", source_rows=2, output_rows=2, columns=columns,
            key_hash="key", duplicate_key_groups=0, bj_rows=0,
            future_or_label_columns=[], shard_sha256="sha", size_bytes=100, mtime_ns=200,
        )
        self.assertTrue(validate_fragment_metadata(metadata, expected_columns=columns)["approved_for_merge"])
        self.assertEqual(
            validate_fanin_merge_contract(
                fan_in=8, child_count=2, source_rows=4, merged_rows=4,
                source_key_hash="key", merged_key_hash="key", duplicate_key_groups=0,
            )["fan_in"],
            8,
        )
        with self.assertRaisesRegex(ValueError, "fan-in must"):
            validate_fanin_merge_contract(
                fan_in=7, child_count=2, source_rows=4, merged_rows=4,
                source_key_hash="key", merged_key_hash="key", duplicate_key_groups=0,
            )

    def test_gtja_boundary_and_fingerprint_gates(self):
        self.assertEqual(
            validate_gtja_chunk_contract(
                date_start="20200101", date_end="20200331", lookback_start="20100104",
                alpha_count=191, output_dates=60, expected_output_dates=60,
                each_alpha_date_once=True, boundary_matches_reference=True,
            )["status"],
            "passed",
        )
        with self.assertRaisesRegex(ValueError, "boundary"):
            validate_gtja_chunk_contract(
                date_start="20200101", date_end="20200331", lookback_start="20100104",
                alpha_count=191, output_dates=60, expected_output_dates=60,
                each_alpha_date_once=True, boundary_matches_reference=False,
            )
        stable = {
            name: {
                "before": {"sha256": name, "size_bytes": 1, "mtime_ns": 2},
                "after": {"sha256": name, "size_bytes": 1, "mtime_ns": 2},
            }
            for name in ("active_feature", "active_label", "production_registry")
        }
        self.assertEqual(validate_fingerprint_stability(stable)["status"], "passed")
        stable["active_label"]["after"]["mtime_ns"] = 3
        with self.assertRaisesRegex(ValueError, "active_label_mtime_ns_changed"):
            validate_fingerprint_stability(stable)


if __name__ == "__main__":
    unittest.main()
