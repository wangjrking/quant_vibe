import json
import os
import tempfile
import unittest
from pathlib import Path

import l3_memory_bounded_v2_fast_candidate as fast


class FastCandidateTests(unittest.TestCase):
    def test_memory_contract_preserves_34_and_64_gib_gates(self):
        policy = fast.MemoryPolicy()
        self.assertEqual(policy.base_required_bytes, 34 * fast.GIB)
        self.assertEqual(policy.candidate_startup_gate_bytes, 34 * fast.GIB)
        self.assertEqual(policy.original_production_startup_gate_bytes, 64 * fast.GIB)
        self.assertEqual(policy.required_before_batch(2 * fast.GIB), 34 * fast.GIB)
        with self.assertRaises(RuntimeError):
            fast._memory_gate(policy, 7 * fast.GIB, 80 * fast.GIB)

    def test_runtime_gate_accepts_current_interpreter(self):
        identity = fast.runtime_identity(os.sys.executable)
        self.assertEqual(identity["runtime_provenance_status"], "passed")
        self.assertEqual(identity["sys_executable"], fast._norm_path(os.sys.executable))

    def test_runtime_gate_rejects_drift(self):
        with self.assertRaises(RuntimeError):
            fast.runtime_identity("C:/not-the-approved/python.exe")

    def test_synthetic_rows_are_no_bj_and_unique(self):
        rows = fast.synthetic_rows(5, 32)
        metrics = fast._fragment_metrics(rows)
        self.assertEqual(metrics["source_rows"], 160)
        self.assertEqual(metrics["duplicate_key_groups"], 0)
        self.assertEqual(metrics["bj_rows"], 0)
        self.assertEqual(metrics["future_label_columns"], [])

    def test_topology_a_is_bounded_to_three_workers(self):
        result = fast.run_benchmark(fast.BenchmarkConfig(stock_count=6, rows_per_stock=5, expected_executable=os.sys.executable))
        self.assertLessEqual(result["topology_a"]["run"]["max_active_processes"], 3)
        self.assertEqual(result["topology_a"]["run"]["source_rows"], 30)
        self.assertEqual(result["topology_a"]["run"]["output_rows"], 30)

    def test_topology_b_batches_multiple_stocks_with_one_worker(self):
        result = fast.run_benchmark(fast.BenchmarkConfig(stock_count=8, rows_per_stock=4, topology_b_batch_stock_count=4, expected_executable=os.sys.executable))
        self.assertEqual(result["topology_b"]["workers"], 1)
        self.assertEqual(result["topology_b"]["run"]["task_count"], 2)
        self.assertEqual(result["topology_b"]["run"]["max_active_processes"], 1)
        self.assertEqual(result["topology_b"]["run"]["source_rows"], 32)

    def test_topologies_are_equivalent(self):
        result = fast.run_benchmark(fast.BenchmarkConfig(stock_count=6, rows_per_stock=5, expected_executable=os.sys.executable))
        self.assertTrue(result["equivalence"]["passed"])
        self.assertEqual(result["topology_a"]["contract"], result["topology_b"]["contract"])

    def test_qfq_gtja_and_future_label_contract_is_explicit(self):
        result = fast.run_benchmark(fast.BenchmarkConfig(stock_count=4, rows_per_stock=3, expected_executable=os.sys.executable))
        contract = result["topology_b"]["contract"]
        self.assertEqual(contract["qfq_contract"]["l2_source_qfq_technical_count"], 74)
        self.assertEqual(contract["qfq_contract"]["production_feature_qfq_technical_count"], 76)
        self.assertEqual(contract["qfq_contract"]["derived_qfq_technical_count"], 2)
        self.assertEqual(contract["qfq_contract"]["gtja_qfq_count"], 191)
        self.assertEqual(contract["future_label_columns"], [])
        self.assertEqual(contract["label"]["write_called"], False)
        self.assertEqual(contract["label"]["target_trade_date_rows"], 0)

    def test_active_fingerprints_and_business_paths_are_candidate_only(self):
        result = fast.run_benchmark(fast.BenchmarkConfig(stock_count=3, rows_per_stock=2, expected_executable=os.sys.executable))
        self.assertFalse(result["business_inputs_read"])
        self.assertFalse(result["production_paths_opened"])
        self.assertTrue(result["active_fingerprints_unchanged"])
        self.assertFalse(result["candidate_written"])
        for value in result["active_fingerprints"].values():
            self.assertEqual(value, "synthetic-fixture")

    def test_forbidden_output_path_is_rejected(self):
        with self.assertRaises(RuntimeError):
            fast._assert_synthetic_only("D:/work/quant/quant_mcp/quant/data_file/production_assets/foo.json")

    def test_report_is_json_and_contains_no_production_write(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "fast_benchmark.json"
            result = fast.run_benchmark(fast.BenchmarkConfig(stock_count=3, rows_per_stock=2, expected_executable=os.sys.executable))
            report.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            loaded = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(loaded["equivalence"]["passed"])
            self.assertFalse(loaded["full_rebuild_started"])
            self.assertFalse(loaded["active_switch"])
            self.assertFalse(loaded["label_write"])
            self.assertFalse(loaded["registry_change"])


if __name__ == "__main__":
    unittest.main()
