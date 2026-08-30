import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import duckdb

import finalize_l3_target_date_candidate as recovery


class CandidateFinalizationPlanTest(unittest.TestCase):
    def test_runtime_sha256_is_in_process_and_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"approved-runtime-sha256")
            self.assertEqual(recovery._approved_runtime_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotIn("Start-Process", Path(recovery.__file__).read_text(encoding="utf-8"))

    def test_plan_checks_candidate_without_l2_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidate = root / "candidate.duckdb"; precheck = root / "precheck.json"
            with duckdb.connect(str(candidate)) as conn:
                conn.execute("CREATE TABLE feature(stock_code VARCHAR, trade_date VARCHAR, close DOUBLE)")
                conn.execute("INSERT INTO feature VALUES ('000001.SZ', '20260828', 1.0)")
            state = {"sha256": "active"}
            payload = {"workflow_run_id": "r1", "target_trade_date": "20260828", "l2": {"file_state": {"sha256": "l2"}, "qfq_technical_columns": []}, "active_feature_before": {"table": "feature", "file_state": state}, "target_expected_metrics": {"target_row_count": 1, "target_stock_count": 1}}
            precheck.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "expected active feature SHA"):
                recovery.build_finalization_plan(candidate, precheck, "20260828", "r1", "l2", "wrong")
            source = Path(recovery.__file__).read_text(encoding="utf-8")
            self.assertNotIn("_compute_l2_only_frames", source)
            self.assertNotIn("_compute_gtja_target", source)

    def test_plan_returns_complete_result_for_validated_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); precheck = root / "precheck.json"; candidate = root / "candidate.duckdb"
            candidate.write_bytes(b"candidate")
            precheck.write_text(json.dumps({"workflow_run_id":"r1","target_trade_date":"20260828","l2":{"file_state":{"sha256":"l2"},"qfq_technical_columns":[]},"active_feature_before":{"table":"feature","file_state":{"sha256":"active"}},"target_expected_metrics":{"target_row_count":1,"target_stock_count":1}}), encoding="utf-8")
            probe={"schema":[{"name":"stock_code"}]*838,"table_names":["feature"],"metrics":{"target_row_count":1,"target_stock_count":1,"duplicate_key_groups":0,"target_bj_row_count":0},"column_count":838,"file_state":{"sha256":"candidate"}}
            with mock.patch.object(recovery.delivery,"_asset_probe",return_value=probe), mock.patch.object(recovery.delivery,"_feature_schema_gate_summary",return_value={"ok":True,"details":{}}):
                plan=recovery.build_finalization_plan(candidate,precheck,"20260828","r1","l2","active")
            self.assertEqual(plan["status"],"candidate_finalization_plan_ready")
            self.assertFalse(plan["recompute_raw"])

    def test_finalization_gate_allows_only_bound_self_chain_and_writes_raw_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); workspace = root / "work"; report = root / "report"; workspace.mkdir(); report.mkdir()
            args = type("Args", (), {"workflow_run_id":"r1", "workspace_dir":str(workspace), "report_dir":str(report)})()
            escaped_workspace=str(workspace).replace("\\", "\\\\")
            self_chain={"cmdline":f"l3_candidate_finalization_driver.py r1 {escaped_workspace} {report}"}
            foreign={"command_line":f"driver r2 {workspace} {report}"}
            raw={"passed":False,"blocking_writers":[],"unknown_relevant_processes":[self_chain,foreign],"raw_marker":"complete"}
            with mock.patch.object(recovery.delivery,"_scan_l3_processes",return_value=raw):
                gate=recovery.finalization_process_gate(args,report)
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["allowed_finalization_self_chain"],[self_chain])
            self.assertEqual(self_chain["raw_cmdline_normalized"], f"l3_candidate_finalization_driver.py r1 {workspace} {report}")
            saved=json.loads((report/"l3_candidate_finalization_process_gate.json").read_text())
            self.assertEqual(saved["raw_marker"],"complete")
            child={"cmdline":f"finalize_l3_target_date_candidate.py r1 {workspace} {report}"}
            with mock.patch.object(recovery.delivery,"_scan_l3_processes",return_value={"passed":False,"blocking_writers":[],"unknown_relevant_processes":[child]}):
                child_gate=recovery.finalization_process_gate(args,report)
            self.assertTrue(child_gate["passed"])
            self.assertEqual(child_gate["allowed_finalization_self_chain"],[child])
            other_script={"cmdline":f"other_python_script.py r1 {workspace} {report}"}
            missing_report={"cmdline":f"finalize_l3_target_date_candidate.py r1 {workspace}"}
            with mock.patch.object(recovery.delivery,"_scan_l3_processes",return_value={"passed":False,"blocking_writers":[],"unknown_relevant_processes":[other_script,missing_report]}):
                denied=recovery.finalization_process_gate(args,report)
            self.assertFalse(denied["passed"])
            self.assertEqual(denied["allowed_finalization_self_chain"],[])
            self.assertEqual(denied["unknown_relevant_processes"],[other_script,missing_report])
            self.assertEqual(gate["unknown_relevant_processes"],[foreign])
