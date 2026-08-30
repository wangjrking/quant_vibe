import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import duckdb

import rebuild_l3_full_duckdb_mainline as full_l3


class RebuildL3FullDuckDBMainlineTests(unittest.TestCase):
    def test_precheck_mode_does_not_touch_active_assets_or_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            workspace = root / "workspace"
            snapshot = {
                "asset_id": "asset",
                "asset_path": "x.duckdb::t",
                "path": "x.duckdb",
                "table": "t",
                "file_state": {"path": "x", "size_bytes": 1, "mtime_ns": 1, "sha256": "a"},
                "metrics": {"row_count": 1},
                "schema_hash": "schema",
                "column_count": 1,
            }
            precheck = {
                "status": "precheck_passed",
                "target_trade_date": "20260717",
                "workflow_run_id": "run",
                "processing_mode": "full_history_rebuild",
                "l2": {"path": "l2", "table": "STOCK_DAILY_DATA", "metrics": {"row_count": 1}},
                "gates": {"ok": True},
            }
            registry_state = {"path": "registry", "size_bytes": 2, "mtime_ns": 2, "sha256": "b"}
            with mock.patch.object(full_l3, "_precheck", return_value=precheck), mock.patch.object(
                full_l3, "_active_snapshot", return_value=snapshot
            ), mock.patch.object(full_l3, "_file_state", return_value=registry_state), mock.patch.object(
                full_l3, "_validate_no_forbidden_runtime_paths"
            ):
                rc = full_l3.main(
                    [
                        "--target-trade-date", "20260717",
                        "--mode", "precheck",
                        "--workflow-run-id", "run",
                        "--report-dir", str(report_dir),
                        "--workspace-dir", str(workspace),
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads((report_dir / "l3_full_processing_20260717_precheck.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "precheck_passed_active_unchanged")
            self.assertTrue(all(payload["active_unchanged_gates"].values()))
            self.assertFalse(payload["active_switch_called"])

    def test_target_date_only_report_cannot_build_full_history_contract(self):
        args = Namespace(workflow_run_id="run", target_trade_date="20260717")
        report = {
            "processing_mode": "target_date_only",
            "candidate_feature": {"metrics": {"row_count": 1}},
            "candidate_label": {},
            "precheck": {"l2": {"metrics": {"row_count": 1}}},
        }
        with self.assertRaisesRegex(RuntimeError, "target-date-only"):
            full_l3._build_contract(report, args, [])

    def test_forbidden_20260716_quarantine_input_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            full_l3._validate_no_forbidden_runtime_paths([Path("runtime/20260716/quarantine")], "20260717")

    def test_l2_hash_drift_fails_before_opening_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                target_trade_date="20260717",
                workflow_run_id="run",
                expected_l2_sha256="expected",
                report_dir=str(Path(tmp) / "reports"),
                workspace_dir=str(Path(tmp) / "workspace"),
            )
            Path(args.report_dir).mkdir()
            asset = {"asset_id": full_l3.EXPECTED_L2_ASSET_ID}
            process_gate = {
                "status": "process_gate_passed",
                "passed": True,
                "blocking_writers": [],
                "unknown_relevant_processes": [],
                "allowed_current_chain": [],
                "allowed_readonly_observers": [],
            }
            with mock.patch.object(
                full_l3,
                "_active_registry_asset",
                return_value=(asset, full_l3.EXPECTED_L2_PATH.resolve(), full_l3.EXPECTED_L2_TABLE),
            ), mock.patch.object(full_l3, "_file_sha256", return_value="actual"), mock.patch.object(
                full_l3, "_scan_l3_processes", return_value=process_gate
            ), mock.patch.object(
                full_l3,
                "require_psutil",
                return_value=SimpleNamespace(
                    virtual_memory=lambda: SimpleNamespace(available=128 * 1024**3)
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA256 drift"):
                    full_l3._precheck(args)

    def test_qfq_gtja_future_leakage_schema_gates(self):
        l2_technical = [f"technical_{idx:03d}_qfq" for idx in range(73)] + ["ema_qfq_10"]
        good_columns = [
            "trade_date", "stock_code", *full_l3.FRONT_ADJUSTED_MARKET_PRICE_COLUMNS,
            *l2_technical, *[f"gtja_alpha{idx:03d}_qfq" for idx in range(1, 192)],
        ]
        good = full_l3._feature_schema_gate_summary(good_columns, l2_technical)
        self.assertTrue(all(value for key, value in good.items() if key != "details"))
        bad = full_l3._feature_schema_gate_summary([*good_columns, "open", "gtja_alpha001", "10d_yield_rate"], l2_technical)
        self.assertFalse(bad["gtja_naked_zero"])
        self.assertFalse(bad["naked_qfq_alias_zero"])
        self.assertFalse(bad["future_label_leakage_zero"])

    def test_source_limited_duplicate_column_preserves_l2_first_value(self):
        frame = pd.DataFrame([[1.5, 9.9, 2.5]], columns=["atr_qfq", "atr_qfq", "other"])
        result = full_l3._deduplicate_columns_preserve_source(frame)
        self.assertEqual(result.columns.tolist(), ["atr_qfq", "other"])
        self.assertEqual(result.loc[0, "atr_qfq"], 1.5)

    def test_text_schema_is_stable_across_numeric_and_named_enterprise_types(self):
        first = full_l3._normalize_types(pd.DataFrame({
            "trade_date": ["20260717"], "stock_code": ["000869.SZ"], "act_ent_type": [1]
        }))
        second = full_l3._normalize_types(pd.DataFrame({
            "trade_date": ["20260717"], "stock_code": ["002884.SZ"], "act_ent_type": ["natural_person"]
        }))
        self.assertEqual(str(first["act_ent_type"].dtype), "string")
        self.assertEqual(str(second["act_ent_type"].dtype), "string")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "text_schema.duckdb"
            with duckdb.connect(str(path)) as conn:
                conn.register("frame", first)
                conn.execute("CREATE TABLE raw_factor AS SELECT * FROM frame")
                conn.unregister("frame")
                conn.register("frame", second)
                conn.execute("INSERT INTO raw_factor BY NAME SELECT * FROM frame")
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM raw_factor").fetchone()[0], 2)
                column_type = conn.execute(
                    "SELECT data_type FROM information_schema.columns WHERE table_name='raw_factor' AND column_name='act_ent_type'"
                ).fetchone()[0]
                self.assertEqual(column_type, "VARCHAR")

    def test_raw_completion_validator_rejects_mismatch(self):
        task = {"task_id": "raw_bucket_0000", "codes": ["000001.SZ"]}
        full_l3._validate_raw_completion(
            task,
            {"source_rows": 1, "output_rows": 1, "duplicate_key_groups": 0, "code_count": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "row mismatch"):
            full_l3._validate_raw_completion(
                task,
                {"source_rows": 1, "output_rows": 0, "duplicate_key_groups": 0, "code_count": 1},
            )

    def test_label_maturity_does_not_forward_to_target(self):
        dates = [f"202601{day:02d}" for day in range(1, 32)]
        self.assertEqual(full_l3._compute_maturity_date(dates, horizon=22), "20260109")
        self.assertNotEqual(full_l3._compute_maturity_date(dates, horizon=22), dates[-1])

    def test_shrink_gate(self):
        full_l3._require_no_shrink(10, 10, stage="raw")
        with self.assertRaisesRegex(RuntimeError, "shrink"):
            full_l3._require_no_shrink(10, 9, stage="raw")

    def test_quarantine_on_fail_moves_candidate_and_writes_incident(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "workspace" / "candidates" / "feature.duckdb"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"candidate")
            incident = full_l3._quarantine(root / "workspace", RuntimeError("boom"), root / "reports", "20260717")
            self.assertFalse(candidate.exists())
            self.assertTrue(Path(incident["quarantined_candidate_paths"][0]).is_file())
            self.assertTrue((root / "reports" / "l3_full_processing_20260717_incident.json").is_file())
            self.assertFalse(incident["active_switch_called"])


if __name__ == "__main__":
    unittest.main()
