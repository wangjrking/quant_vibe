import tempfile
import unittest
from pathlib import Path

import l3_memory_bounded_v2_controlled_probe_runner_candidate_20260805_r2 as runner


class ControlledProbeRunnerRevision2Tests(unittest.TestCase):
    def _config(self, root, execute=False):
        base = Path(root)
        return runner.ProbeConfig("run-r2", base / "workspace", base / "staging", base / "report", execute_probe=execute)

    def test_default_plan_only_never_opens_canonical_l2(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = runner.build_plan(self._config(temp), "D:/runtime/python.exe")
            self.assertEqual(plan["status"], "plan_only")
            self.assertFalse(plan["canonical_l2_opened"])
            self.assertTrue(plan["explicit_execute_probe_required"])
            self.assertFalse(plan["allow_probe_only"])

    def test_execute_switch_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "explicit --execute-probe"):
                runner.execute_probe(self._config(temp), "D:/runtime/python.exe", None, Path(temp) / "writer.marker")

    def test_route_hash_table_and_output_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            with self.assertRaisesRegex(RuntimeError, "canonical L2 SHA256 mismatch"):
                runner.validate_canonical_route(runner.ProbeConfig(**{**runner.asdict(config), "expected_l2_sha256": "bad"}))
            bad = runner.ProbeConfig("run-r2", Path(temp) / "quarantine", Path(temp) / "staging", Path(temp) / "report")
            with self.assertRaisesRegex(RuntimeError, "forbidden output path"):
                runner.assert_fresh_output_paths(bad)

    def test_report_workspace_must_be_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(temp)
            config.report_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be absent"):
                runner.assert_fresh_output_paths(config)

    def test_lease_and_wal_unknown_writer_gates_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "synthetic.duckdb"
            db.write_bytes(b"synthetic")
            wal = Path(str(db) + ".wal")
            wal.write_bytes(b"wal")
            marker = Path(temp) / "unknown_writer.marker"
            marker.write_text("writer", encoding="ascii")
            gate = runner.wal_writer_gate(db, marker)
            self.assertFalse(gate["passed"])
            self.assertFalse(runner.lease_gate(None, "run-r2")["passed"])

    def test_explicit_projection_has_no_select_star_and_rejects_future(self):
        schema = ["trade_date", "stock_code", *runner.QFQ_PRICE_COLUMNS]
        schema.extend(f"tech_qfq_{index:03d}" for index in range(74))
        columns = runner.explicit_projection(schema)
        sql = runner.projection_sql(self._config("D:/tmp/r2"), columns)
        self.assertIn("SELECT \"trade_date\"", sql)
        self.assertNotIn("SELECT *", sql.upper())
        with self.assertRaisesRegex(RuntimeError, "future/label"):
            runner.explicit_projection(schema + ["index_2000_post10_close"])

    def test_runtime_gate_rejects_conda_residue(self):
        identity = runner.runtime_identity()
        identity["sys_base_prefix"] = "C:/Users/test/.conda/envs/my_quant"
        gate = runner.runtime_gate(identity, identity["sys_executable"])
        self.assertFalse(gate["passed"])
        self.assertIn("conda_runtime_residue", gate["errors"])

    def test_candidate_flags_are_non_mutating(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = runner.build_plan(self._config(temp), "D:/runtime/python.exe")
            self.assertFalse(plan["active_switch"])
            self.assertFalse(plan["label_write"])
            self.assertFalse(plan["registry_change"])
            self.assertFalse(plan["allow_l3"])

    def test_failure_quarantines_entire_isolation_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "probe-root"
            config = runner.ProbeConfig("run-r2", root / "workspace", root / "staging", root / "report")
            config.workspace.mkdir(parents=True)
            config.staging_dir.mkdir()
            config.report_dir.mkdir()
            (config.staging_dir / "partial.parquet").write_bytes(b"partial")
            result = runner.quarantine_isolated_outputs(config, "synthetic_failure")
            self.assertTrue(result["quarantined"])
            self.assertTrue(result["reuse_prohibited"])
            self.assertFalse(root.exists())
            self.assertTrue(Path(result["path"]).exists())


if __name__ == "__main__":
    unittest.main()
