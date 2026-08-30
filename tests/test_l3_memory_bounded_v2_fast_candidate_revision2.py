import json
import os
import tempfile
import unittest
from pathlib import Path

import l3_memory_bounded_v2_fast_candidate_revision2 as revision


class Revision2FastCandidateTests(unittest.TestCase):
    def test_runtime_provenance_closed_rejects_conda_residue(self):
        with self.assertRaises(RuntimeError):
            revision.runtime_provenance_closed("C:/Users/example/.conda/python.exe")

    def test_negative_shift_and_future_denylist_fail_closed(self):
        result = revision.validate_future_lineage(["close_qfq", "index_2000_post10_close", "foo_shift(-1)"])
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["rejected"]), 2)

    def test_report_dir_nonempty_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report"
            path.mkdir()
            (path / "existing.json").write_text("{}", encoding="ascii")
            with self.assertRaises(RuntimeError):
                revision.assert_fresh_report_dir(path)

    def test_active_file_state_sampling_is_read_only_and_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = {}
            for name in ("l3_feature_current.duckdb", "l3_label_current.duckdb", "production_assets.json"):
                path = Path(temp) / name
                path.write_bytes(b"synthetic active state")
                paths[name] = str(path)
            before = revision.sample_active_file_state(paths)
            after = revision.sample_active_file_state(paths)
            revision.assert_active_file_state_unchanged(before, after)
            self.assertTrue(all(item["read_only_status_only"] for item in before.values()))

    def test_wal_gate_fails_closed_on_wal_and_unknown_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "synthetic.duckdb"
            db.write_bytes(b"synthetic")
            wal = Path(str(db) + ".wal")
            marker = Path(temp) / "unknown_writer.marker"
            wal.write_bytes(b"wal")
            wal_block = revision.writer_wal_gate(db, marker)
            self.assertFalse(wal_block["passed"])
            self.assertIn("wal_present", wal_block["reasons"])
            wal.unlink()
            marker.write_text("unknown", encoding="ascii")
            marker_block = revision.writer_wal_gate(db, marker)
            self.assertFalse(marker_block["passed"])
            self.assertIn("unknown_writer_marker_present", marker_block["reasons"])

    def test_synthetic_duckdb_subprocess_has_required_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            result = revision.run_duckdb_synthetic_gate(Path(temp), os.sys.executable)
            self.assertEqual(result["child_exit_code"], 0)
            self.assertEqual(str(result["settings"]["threads"]), "1")
            memory_limit = str(result["settings"]["memory_limit"]).upper()
            self.assertTrue("3GB" in memory_limit or "2.7 GIB" in memory_limit)
            self.assertTrue(result["wal_gate_fail_closed"]["wal_exists"])
            self.assertFalse(result["post_close_gate"]["passed"] is False)

    def test_runtime_and_duckdb_gate_report_is_candidate_only(self):
        with tempfile.TemporaryDirectory() as temp:
            result = revision.run_duckdb_synthetic_gate(Path(temp), os.sys.executable)
            self.assertFalse(result["business_inputs_read"])
            self.assertFalse(result["production_paths_opened"])
            self.assertTrue(result["unknown_writer_gate_fail_closed"]["reasons"])

    def test_real_lineage_accepts_named_qfq_columns(self):
        result = revision.validate_future_lineage(["open_qfq", "ema_qfq_10", "macdsignal_qfq"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["rejected"], [])

    def test_revision_report_has_no_business_write_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = {}
            for name in ("l3_feature_current.duckdb", "l3_label_current.duckdb", "production_assets.json"):
                path = Path(temp) / name
                path.write_bytes(b"synthetic active state")
                paths[name] = str(path)
            result = revision.run_revision2(Path(temp) / "evidence", os.sys.executable, paths)
            self.assertEqual(result["status"], "passed_candidate_only")
            self.assertFalse(result["business_inputs_read"])
            self.assertFalse(result["business_asset_written"])
            self.assertFalse(result["full_rebuild_started"])
            loaded = json.loads((Path(temp) / "evidence" / "revision2_synthetic_gate_report.json").read_text(encoding="utf-8"))
            self.assertTrue(loaded["active_status_sampling"]["unchanged"])


if __name__ == "__main__":
    unittest.main()
