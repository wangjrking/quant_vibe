import json
import hashlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import deliver_l3_target_date_duckdb_mainline as delivery
import l3_active_writer_lease as active_writer_lease
import l3_process_gate as process_gate


class TargetDateDuckDBDeliveryTests(unittest.TestCase):
    def _args(self, root: Path, mode: str) -> list[str]:
        return [
            "--mode",
            mode,
            "--target-trade-date",
            "20260717",
            "--workflow-run-id",
            "incremental-trading-signal-20260717-L3-target-date-incremental-delivery",
            "--workspace-dir",
            str(root / "workspace"),
            "--report-dir",
            str(root / "reports"),
            "--expected-l2-sha256",
            "a" * 64,
        ]

    def test_forbidden_legacy_and_quarantine_inputs_fail_closed(self):
        forbidden = [
            r"D:\work\quant\quant_mcp\quant\data_file\production_factor_parts",
            r"D:\work\quant\quant_mcp\quant\data_file\stock_factor_data.parquet",
            r"D:\work\quant\quant_mcp\quant\data_file\runtime\quarantine\attempt-5",
            r"D:\work\quant\quant_mcp\quant\data_file\odb.db",
        ]
        for value in forbidden:
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "forbidden target-date input path"):
                delivery._assert_input_path(Path(value))

    def test_precheck_mode_never_calls_candidate_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            precheck = {"status": "precheck_passed"}
            with mock.patch.object(delivery, "_precheck", return_value=precheck), mock.patch.object(
                delivery, "execute"
            ) as execute:
                delivery.main(self._args(root, "precheck"))
            execute.assert_not_called()
            report = root / "reports" / "l3_target_date_20260717_precheck.json"
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")), precheck)

    def test_single_worker_recovery_keeps_raw_bucket_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = delivery.parse_args(
                self._args(Path(tmp), "execute")
                + ["--workers", "1", "--raw-buckets", "16"]
            )
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.raw_buckets, 16)
        self.assertEqual(args.target_trade_date, "20260717")
        self.assertEqual(args.lookback_trading_days, 300)

    def test_lookback_start_is_bounded_to_requested_trading_days(self):
        class Result:
            def fetchall(self):
                return [("20260717",), ("20260716",), ("20260715",)]

        class Connection:
            def execute(self, *_args, **_kwargs):
                return Result()

        start = delivery._resolve_lookback_start(Connection(), "STOCK_DAILY_DATA", "20260717", 3)
        self.assertEqual(start, "20260715")

    def test_mainline_pins_spawn_workers_to_current_python_executable(self):
        original_base = getattr(sys, "_base_executable", None)
        with mock.patch.object(delivery.multiprocessing, "set_executable") as set_executable:
            executable = delivery._configure_multiprocessing_executable()

        expected = str(Path(sys.executable).resolve())
        self.assertEqual(executable, expected)
        if hasattr(sys, "_base_executable"):
            self.assertEqual(sys._base_executable, expected)
            sys._base_executable = original_base
        set_executable.assert_called_once_with(expected)

    def test_single_worker_runs_raw_buckets_inline_without_process_pool(self):
        class FakeResult:
            def fetchall(self):
                return [("000001.SZ",), ("000002.SZ",)]

        class FakeConnection:
            def execute(self, *_args, **_kwargs):
                return FakeResult()

            def close(self):
                return None

        def compute_bucket(task):
            frame = pd.DataFrame(
                {
                    "trade_date": ["20260730"] * len(task["codes"]),
                    "stock_code": list(task["codes"]),
                }
            )
            return {
                "bucket_index": task["bucket_index"],
                "source_rows": len(frame),
                "output_rows": len(frame),
                "raw_target": frame.copy(),
                "gtja_input": frame.copy(),
            }

        args = delivery.parse_args(
            self._args(Path("root"), "execute")
            + ["--workers", "1", "--raw-buckets", "2"]
        )
        precheck = {"l2": {"path": "l2.duckdb", "table": "STOCK_DAILY_DATA"}}
        with mock.patch.object(delivery.duckdb, "connect", return_value=FakeConnection()), mock.patch.object(
            delivery, "_compute_bucket", side_effect=compute_bucket
        ) as worker, mock.patch.object(
            delivery.concurrent.futures, "ProcessPoolExecutor"
        ) as process_pool, mock.patch.object(
            delivery, "_append_progress"
        ) as append_progress:
            raw_target, gtja_input, metrics = delivery._compute_l2_only_frames(
                args,
                precheck,
                Path("progress.jsonl"),
            )

        process_pool.assert_not_called()
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(raw_target), 2)
        self.assertEqual(len(gtja_input), 2)
        bucket_events = [
            call
            for call in append_progress.call_args_list
            if len(call.args) > 1 and call.args[1] == "raw_bucket_done"
        ]
        self.assertEqual(len(bucket_events), 2)
        self.assertTrue(
            all(call.kwargs["execution_mode"] == "inline_single_worker" for call in bucket_events)
        )

    def test_verified_copy_flushes_hashes_and_atomically_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.duckdb"
            destination = root / "candidate.duckdb"
            payload = (b"verified-copy" * 257) + b"!"
            source.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            result = delivery._copy_file_verified(source, destination, expected, chunk_bytes=31)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(result["sha256"], expected)
            self.assertEqual(result["source_sha256_before"], expected)
            self.assertEqual(result["source_sha256_after"], expected)
            self.assertEqual(result["copied_bytes"], len(payload))
            self.assertEqual(result["source_before"]["size_bytes"], len(payload))
            self.assertEqual(result["source_after"]["sha256"], expected)
            self.assertEqual(result["destination"]["sha256"], expected)
            self.assertEqual(result["destination"]["mtime_ns"], result["source_after"]["mtime_ns"])
            self.assertGreaterEqual(result["copy_duration_seconds"], 0.0)
            self.assertFalse(list(root.glob("candidate.duckdb.copying-*.tmp")))

    def test_verified_copy_rejects_corrupted_destination_without_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.duckdb"
            destination = root / "candidate.duckdb"
            source.write_bytes(b"source-bytes")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            real_hash = delivery._file_sha256

            def corrupt_temp_hash(path):
                if ".copying-" in Path(path).name:
                    return "0" * 64
                return real_hash(Path(path))

            with mock.patch.object(delivery, "_file_sha256", side_effect=corrupt_temp_hash):
                with self.assertRaisesRegex(RuntimeError, "destination hash mismatch"):
                    delivery._copy_file_verified(source, destination, expected, chunk_bytes=4)

            self.assertFalse(destination.exists())
            self.assertEqual(len(list(root.glob("candidate.duckdb.copying-*.tmp"))), 1)

    def test_snapshot_candidate_rejects_active_or_candidate_wal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "active.duckdb"
            destination = root / "candidate.duckdb"
            source.write_bytes(b"active")
            expected_state = delivery._file_state(source)

            active_wal = Path(f"{source}.wal")
            active_wal.write_bytes(b"wal")
            with self.assertRaisesRegex(RuntimeError, "active feature WAL"):
                delivery._snapshot_active_feature_candidate_base(source, destination, expected_state)
            active_wal.unlink()

            candidate_wal = Path(f"{destination}.wal")
            candidate_wal.write_bytes(b"wal")
            with self.assertRaisesRegex(RuntimeError, "candidate base WAL"):
                delivery._snapshot_active_feature_candidate_base(source, destination, expected_state)
            self.assertFalse(destination.exists())

    def test_writer_lease_is_exclusive_owned_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = delivery.parse_args(self._args(root, "candidate"))
            lease_path = root / "lease.json"
            with mock.patch.object(delivery, "_active_feature_lease_path", return_value=lease_path):
                lease = delivery._acquire_active_feature_writer_lease(args, root / "active.duckdb")
                self.assertTrue(lease_path.is_file())
                self.assertTrue(lease["lease_nonce"])
                with self.assertRaisesRegex(RuntimeError, "writer lease already exists"):
                    delivery._acquire_active_feature_writer_lease(args, root / "active.duckdb")
                release = delivery._release_active_feature_writer_lease(lease)
            self.assertEqual(release["status"], "released")
            self.assertTrue(release["lease_absent_after_release"])
            self.assertFalse(lease_path.exists())

    def test_execute_snapshots_before_long_compute_and_releases_lease_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = delivery.parse_args(self._args(root, "candidate"))
            active = root / "active.duckdb"
            active.write_bytes(b"active")
            state = delivery._file_state(active)
            precheck = {
                "active_feature_before": {
                    "path": str(active),
                    "table": "feature",
                    "schema": [{"name": "trade_date"}, {"name": "stock_code"}],
                    "file_state": state,
                }
            }
            lease_path = root / "lease.json"
            order = []

            def snapshot(*_args, **_kwargs):
                order.append("snapshot")
                return {
                    "candidate": state,
                    "copy_duration_seconds": 0.1,
                }

            def compute(*_args, **_kwargs):
                order.append("compute")
                raise RuntimeError("synthetic long-compute failure")

            (root / "workspace").mkdir()
            with mock.patch.object(delivery, "_active_feature_lease_path", return_value=lease_path), mock.patch.object(
                delivery,
                "_scan_l3_processes",
                return_value={"passed": True, "blocking_writers": [], "unknown_relevant_processes": []},
            ), mock.patch.object(
                delivery, "_snapshot_active_feature_candidate_base", side_effect=snapshot
            ), mock.patch.object(
                delivery, "_compute_l2_only_frames", side_effect=compute
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic long-compute failure"):
                    delivery.execute(args, precheck, apply_active=False)

            self.assertEqual(order, ["snapshot", "compute"])
            self.assertTrue((root / "workspace").is_dir())
            self.assertFalse(lease_path.exists())
            release_path = root / "reports" / "l3_target_date_20260717_writer_lease_release.json"
            self.assertEqual(json.loads(release_path.read_text(encoding="utf-8"))["status"], "released")
            incident = json.loads(
                (root / "reports" / "l3_target_date_delivery_20260717_incident.json").read_text(encoding="utf-8")
            )
            self.assertEqual(incident["status"], "failed_closed")
            self.assertEqual(incident["candidate_base_snapshot"]["copy_duration_seconds"], 0.1)

    def test_verified_copy_rejects_source_drift_without_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.duckdb"
            destination = root / "candidate.duckdb"
            source.write_bytes(b"source-bytes")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()

            with mock.patch.object(
                delivery,
                "_file_sha256",
                side_effect=[expected, "1" * 64],
            ):
                with self.assertRaisesRegex(RuntimeError, "source hash drifted during copy"):
                    delivery._copy_file_verified(source, destination, expected, chunk_bytes=4)

            self.assertFalse(destination.exists())

    def test_candidate_base_snapshot_records_stable_source_and_candidate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "active.duckdb"
            candidate = root / "workspace" / "candidate.duckdb"
            source.write_bytes(b"active-feature" * 127)
            expected = delivery._file_state(source)

            result = delivery._snapshot_active_feature_candidate_base(source, candidate, expected)

            self.assertEqual(result["status"], "verified_candidate_base_ready")
            self.assertEqual(result["source_before"]["sha256"], expected["sha256"])
            self.assertEqual(result["source_after"]["sha256"], expected["sha256"])
            self.assertEqual(result["candidate"]["sha256"], expected["sha256"])
            self.assertEqual(candidate.read_bytes(), source.read_bytes())
            self.assertIn("before_long_compute", result["protocol"])

    def test_candidate_base_snapshot_rejects_wal_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "active.duckdb"
            source.write_bytes(b"active-feature")
            Path(f"{source}.wal").write_bytes(b"writer-present")

            with self.assertRaisesRegex(RuntimeError, "WAL is present"):
                delivery._snapshot_active_feature_candidate_base(
                    source,
                    root / "candidate.duckdb",
                    delivery._file_state(source),
                )

    def test_writer_lease_is_exclusive_and_released_only_by_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "active.duckdb"
            source.write_bytes(b"active-feature")
            args = delivery.parse_args(self._args(root, "candidate"))
            lease_dir = root / "leases"
            with mock.patch.object(delivery, "ACTIVE_FEATURE_LEASE_DIR", lease_dir):
                lease = delivery._acquire_active_feature_writer_lease(args, source)
                with self.assertRaisesRegex(RuntimeError, "writer lease already exists"):
                    delivery._acquire_active_feature_writer_lease(args, source)
                delivery._release_active_feature_writer_lease(lease)
            self.assertFalse((lease_dir / "l3_feature_current.writer_lease.json").exists())

    def test_execute_snapshots_candidate_before_long_compute(self):
        source = inspect.getsource(delivery.execute)
        self.assertLess(
            source.index("_snapshot_active_feature_candidate_base"),
            source.index("_compute_l2_only_frames"),
        )
        self.assertLess(
            source.index("_pre_active_switch_gate"),
            source.index("_atomic_feature_switch"),
        )

    def test_pre_switch_gate_rechecks_writer_wal_lease_and_asset_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = delivery.parse_args(self._args(root, "execute"))
            active = root / "active.duckdb"
            label = root / "label.duckdb"
            l2 = root / "l2.duckdb"
            registry = root / "production_assets.json"
            candidate = root / "candidate.duckdb"
            for path, payload in (
                (active, b"active"),
                (label, b"label"),
                (l2, b"l2"),
                (registry, b"registry"),
                (candidate, b"candidate"),
            ):
                path.write_bytes(payload)
            precheck = {
                "active_feature_before": {"path": str(active), "file_state": delivery._file_state(active)},
                "active_label_before": {"path": str(label), "file_state": delivery._file_state(label)},
                "l2": {"path": str(l2), "file_state": delivery._file_state(l2)},
                "registry_before": delivery._file_state(registry),
            }
            candidate_validation = {"file_state": delivery._file_state(candidate)}
            evidence = root / "pre_switch_gate.json"
            lease_path = root / "lease.json"
            process_scan = {"passed": True, "blocking_writers": [], "unknown_relevant_processes": []}
            with mock.patch.object(delivery, "_active_feature_lease_path", return_value=lease_path), mock.patch.object(
                delivery, "REGISTRY_PATH", registry
            ), mock.patch.object(delivery, "_scan_l3_processes", return_value=process_scan), mock.patch.object(
                active_writer_lease, "PROTECTED_ACTIVE_L3_PATHS", {active, label, registry}
            ):
                lease = delivery._acquire_active_feature_writer_lease(args, active)
                gate = delivery._pre_active_switch_gate(
                    args,
                    precheck,
                    candidate,
                    candidate_validation,
                    lease,
                    {"files": []},
                    evidence,
                )
                self.assertTrue(gate["passed"])
                self.assertEqual(gate["lease_validation"]["lease_nonce"], lease["lease_nonce"])
                snapshot = root / "rollback.duckdb"
                switch = delivery._atomic_feature_switch(
                    candidate,
                    snapshot,
                    active,
                    precheck["active_feature_before"]["file_state"]["sha256"],
                    pre_switch_gate=gate,
                    expected_lease_nonce=lease["lease_nonce"],
                    final_gate_recheck=lambda: gate,
                )
                delivery._release_active_feature_writer_lease(lease)
            self.assertTrue(switch["active_switch_called"])
            self.assertEqual(active.read_bytes(), b"candidate")
            self.assertEqual(snapshot.read_bytes(), b"active")

    def test_pre_switch_gate_fails_closed_for_unknown_writer_or_wal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = delivery.parse_args(self._args(root, "execute"))
            active = root / "active.duckdb"
            label = root / "label.duckdb"
            l2 = root / "l2.duckdb"
            registry = root / "production_assets.json"
            candidate = root / "candidate.duckdb"
            for path in (active, label, l2, registry, candidate):
                path.write_bytes(path.name.encode("ascii"))
            precheck = {
                "active_feature_before": {"path": str(active), "file_state": delivery._file_state(active)},
                "active_label_before": {"path": str(label), "file_state": delivery._file_state(label)},
                "l2": {"path": str(l2), "file_state": delivery._file_state(l2)},
                "registry_before": delivery._file_state(registry),
            }
            candidate_validation = {"file_state": delivery._file_state(candidate)}
            lease_path = root / "lease.json"
            Path(f"{active}.wal").write_bytes(b"wal")
            process_scan = {
                "passed": False,
                "blocking_writers": [],
                "unknown_relevant_processes": [{"pid": 999, "reason": "unknown writer"}],
            }
            evidence = root / "failed_gate.json"
            with mock.patch.object(delivery, "_active_feature_lease_path", return_value=lease_path), mock.patch.object(
                delivery, "REGISTRY_PATH", registry
            ), mock.patch.object(delivery, "_scan_l3_processes", return_value=process_scan), mock.patch.object(
                active_writer_lease, "PROTECTED_ACTIVE_L3_PATHS", {active, label, registry}
            ):
                lease = delivery._acquire_active_feature_writer_lease(args, active)
                with self.assertRaisesRegex(RuntimeError, "pre-active-switch gate failed"):
                    delivery._pre_active_switch_gate(
                        args,
                        precheck,
                        candidate,
                        candidate_validation,
                        lease,
                        {"files": []},
                        evidence,
                    )
                delivery._release_active_feature_writer_lease(lease)
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertTrue(payload["process_scan"]["unknown_relevant_processes"])
            self.assertTrue(payload["wal_paths"]["active_feature"])
            self.assertEqual(active.read_bytes(), b"active.duckdb")

    def test_atomic_switch_rejects_missing_gate_or_wrong_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.duckdb"
            candidate = root / "candidate.duckdb"
            active.write_bytes(b"active")
            candidate.write_bytes(b"candidate")
            active_hash = delivery._file_sha256(active)
            with self.assertRaisesRegex(RuntimeError, "requires a passed pre-switch gate"):
                delivery._atomic_feature_switch(
                    candidate,
                    root / "rollback.duckdb",
                    active,
                    active_hash,
                    pre_switch_gate={"passed": False},
                    expected_lease_nonce="lease",
                    final_gate_recheck=lambda: {"passed": True, "status": "passed"},
                )
            self.assertEqual(active.read_bytes(), b"active")

    def test_external_writer_appearing_after_pre_gate_blocks_atomic_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.duckdb"
            candidate = root / "candidate.duckdb"
            active.write_bytes(b"active")
            candidate.write_bytes(b"candidate")
            active_state = delivery._file_state(active)
            initial_gate = {
                "status": "passed",
                "passed": True,
                "lease_validation": {"lease_nonce": "lease"},
                "asset_state_checks": {
                    "active_feature": {"matches": True, "actual": active_state},
                },
            }
            external_writer_gate = {
                "status": "failed_closed",
                "passed": False,
                "lease_validation": {"lease_nonce": "lease"},
                "process_scan": {
                    "passed": False,
                    "blocking_writers": [{"pid": 998, "reason": "external L3 writer appeared after pre-gate"}],
                    "unknown_relevant_processes": [],
                },
            }
            with self.assertRaisesRegex(RuntimeError, "final interlock did not pass"):
                delivery._atomic_feature_switch(
                    candidate,
                    root / "rollback.duckdb",
                    active,
                    active_state["sha256"],
                    pre_switch_gate=initial_gate,
                    expected_lease_nonce="lease",
                    final_gate_recheck=lambda: external_writer_gate,
                )
            self.assertEqual(active.read_bytes(), b"active")
            self.assertEqual(candidate.read_bytes(), b"candidate")
            self.assertFalse((root / "rollback.duckdb").exists())

    def test_code_provenance_captures_four_complete_read_only_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for index in range(4):
                source = root / f"source_{index}.py"
                source.write_text(f"VALUE = {index}\n", encoding="utf-8")
                sources.append(source)
            report_dir = root / "reports"
            with mock.patch.object(delivery, "CODE_PROVENANCE_SOURCES", tuple(sources)):
                result = delivery._capture_code_provenance(report_dir, "run-provenance")
                with self.assertRaisesRegex(RuntimeError, "overwrite is prohibited"):
                    delivery._capture_code_provenance(report_dir, "run-provenance")
            self.assertEqual(result["file_count"], 4)
            self.assertEqual(result["diff_line_count"], 0)
            self.assertEqual(delivery._file_sha256(Path(result["manifest_path"])), result["manifest_sha256"])
            for item in result["files"]:
                self.assertEqual(item["source_before"]["sha256"], item["source_after"]["sha256"])
                self.assertEqual(item["source_after"]["sha256"], item["snapshot"]["sha256"])
                self.assertEqual(item["source_after"]["size_bytes"], item["snapshot"]["size_bytes"])
                self.assertEqual(item["source_after"]["mtime_ns"], item["snapshot"]["mtime_ns"])
                self.assertEqual(item["diff_line_count"], 0)

    def test_candidate_mode_never_requests_active_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            report_dir.mkdir()
            baseline = {
                "target_expected_metrics": {"target_row_count": 5195, "target_stock_count": 5195},
                "l2": {"file_state": {"sha256": "l2"}},
                "active_feature_before": {"file_state": {"sha256": "feature"}},
                "active_label_before": {"file_state": {"sha256": "label"}},
                "registry_before": {"sha256": "registry"},
            }
            (report_dir / "l3_target_date_20260717_precheck.json").write_text(
                json.dumps(baseline), encoding="utf-8"
            )
            result = {
                "status": "ready_for_audit_review",
                "candidate_feature": {"metrics": {"target_row_count": 5195, "target_stock_count": 5195}},
                "active_label_after": {"metrics": {"max_trade_date": "20260616"}},
                "report_path": "candidate.json",
                "contract_path": "contract.json",
            }
            with mock.patch.object(delivery, "_precheck", return_value=baseline), mock.patch.object(
                delivery, "execute", return_value=result
            ) as execute:
                delivery.main(self._args(root, "candidate"))
            self.assertFalse(execute.call_args.kwargs["apply_active"])

    def test_workspace_gate_allows_authoritative_empty_execute_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            state, gates = delivery._workspace_gate(workspace, "authoritative_empty")
        self.assertTrue(state["exists"])
        self.assertEqual(state["entries"], [])
        self.assertTrue(gates["workspace_authoritative_root_exists"])
        self.assertTrue(gates["workspace_authoritative_root_empty"])

    def test_execute_mode_creates_authoritative_workspace_before_live_precheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            report_dir.mkdir()
            baseline = {
                "target_expected_metrics": {"target_row_count": 5195, "target_stock_count": 5195},
                "l2": {"file_state": {"sha256": "l2"}},
                "active_feature_before": {"file_state": {"sha256": "feature"}},
                "active_label_before": {"file_state": {"sha256": "label"}},
                "registry_before": {"sha256": "registry"},
            }
            (report_dir / "l3_target_date_20260717_precheck.json").write_text(
                json.dumps(baseline), encoding="utf-8"
            )
            result = {
                "status": "ready_for_audit_review",
                "candidate_feature": {"metrics": {"target_row_count": 5195, "target_stock_count": 5195}},
                "active_label_after": {"metrics": {"max_trade_date": "20260616"}},
                "report_path": "candidate.json",
                "contract_path": "contract.json",
            }

            def live_precheck(args, suffix="", workspace_policy="absent"):
                self.assertEqual(suffix, "_execute")
                self.assertEqual(workspace_policy, "authoritative_empty")
                workspace = Path(args.workspace_dir)
                self.assertTrue(workspace.exists())
                self.assertEqual(list(workspace.iterdir()), [])
                return baseline

            with mock.patch.object(delivery, "_precheck", side_effect=live_precheck), mock.patch.object(
                delivery, "execute", return_value=result
            ) as execute:
                delivery.main(self._args(root, "candidate"))

            startup = json.loads(
                (report_dir / "l3_target_date_20260717_execute_startup_root.json").read_text(encoding="utf-8")
            )
            self.assertEqual(startup["status"], "authoritative_workspace_created_for_execute")
            self.assertFalse(startup["workspace_gate"]["exists"])
            self.assertTrue(startup["workspace_after_create_gates"]["workspace_authoritative_root_empty"])
            self.assertFalse(execute.call_args.kwargs["apply_active"])

    def test_execute_mode_rejects_preexisting_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            workspace = root / "workspace"
            report_dir.mkdir()
            workspace.mkdir()
            (workspace / "unexpected.txt").write_text("junk", encoding="utf-8")
            baseline = {
                "target_expected_metrics": {"target_row_count": 5195, "target_stock_count": 5195},
                "l2": {"file_state": {"sha256": "l2"}},
                "active_feature_before": {"file_state": {"sha256": "feature"}},
                "active_label_before": {"file_state": {"sha256": "label"}},
                "registry_before": {"sha256": "registry"},
            }
            (report_dir / "l3_target_date_20260717_precheck.json").write_text(
                json.dumps(baseline), encoding="utf-8"
            )
            with mock.patch.object(delivery, "_precheck") as precheck, mock.patch.object(delivery, "execute") as execute:
                with self.assertRaisesRegex(RuntimeError, "workspace startup root must be absent"):
                    delivery.main(self._args(root, "candidate"))
            precheck.assert_not_called()
            execute.assert_not_called()
            startup = json.loads(
                (report_dir / "l3_target_date_20260717_execute_startup_root.json").read_text(encoding="utf-8")
            )
            self.assertEqual(startup["status"], "failed_closed_startup_root")
            self.assertEqual(startup["workspace_gate"]["entries"], ["unexpected.txt"])

    def test_target_frame_gates_use_dynamic_l2_target_counts(self):
        feature = pd.DataFrame(
            {
                "trade_date": ["20260715", "20260715"],
                "stock_code": ["000001.SZ", "000002.SZ"],
                "open_qfq": [1.0, 2.0],
                "high_qfq": [1.0, 2.0],
                "low_qfq": [1.0, 2.0],
                "close_qfq": [1.0, 2.0],
                "pre_close_qfq": [1.0, 2.0],
                "technical_001_qfq": [1.0, 2.0],
                "gtja_alpha001_qfq": [1.0, 2.0],
            }
        )
        precheck = {
            "l2": {"qfq_technical_columns": ["technical_001_qfq"]},
            "target_expected_metrics": {"target_row_count": 2, "target_stock_count": 2},
        }
        with mock.patch.object(
            delivery,
            "_feature_schema_gate_summary",
            return_value={
                "gtja_qfq_count_191": True,
                "gtja_naked_zero": True,
                "naked_qfq_alias_zero": True,
                "future_label_leakage_zero": True,
                "details": {},
            },
        ):
            result = delivery._target_frame_gates(feature, precheck)
        self.assertTrue(result["gates"]["rows_match_l2_target"])
        self.assertTrue(result["gates"]["stocks_match_l2_target"])

    def test_candidate_validation_uses_dynamic_target_row_expectation(self):
        precheck = {
            "l2": {"qfq_technical_columns": ["technical_001_qfq"]},
            "target_expected_metrics": {"target_row_count": 2, "target_stock_count": 2},
            "active_feature_before": {
                "metrics": {"row_count": 10, "target_row_count": 3},
                "schema_hash": "schema",
                "historical_key_metrics": {"rows": 7},
            },
        }
        probe = {
            "table_names": ["feature"],
            "metrics": {
                "row_count": 9,
                "target_row_count": 2,
                "target_stock_count": 2,
                "duplicate_key_groups": 0,
                "bj_row_count": 0,
                "target_bj_row_count": 0,
            },
            "column_count": 839,
            "schema_hash": "schema",
            "historical_key_metrics": {"rows": 7},
            "schema": [{"name": "trade_date"}, {"name": "stock_code"}],
        }
        with mock.patch.object(delivery, "_asset_probe", return_value=probe), mock.patch.object(
            delivery,
            "_feature_schema_gate_summary",
            return_value={
                "gtja_qfq_count_191": True,
                "gtja_naked_zero": True,
                "naked_qfq_alias_zero": True,
                "future_label_leakage_zero": True,
                "details": {},
            },
        ), mock.patch.object(
            delivery,
            "_source_qfq_audit",
            return_value={"mismatch_total": 0, "null_pattern_preserved": True},
        ):
            result = delivery._validate_candidate(Path("candidate.duckdb"), "feature", "20260715", precheck)
        self.assertTrue(result["gates"]["rows_expected"])
        self.assertTrue(result["gates"]["target_rows_match_l2_target"])

    def _legacy_test_markdown_uses_requested_target_trade_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.md"
            args = delivery.parse_args(self._args(Path(tmp), "candidate") + [])
            args.target_trade_date = "20260720"
            report = {
                "candidate_feature": {
                    "path": "candidate.duckdb",
                    "table": "feature",
                    "column_count": 839,
                    "metrics": {
                        "target_row_count": 5197,
                        "target_stock_count": 5197,
                        "duplicate_key_groups": 0,
                        "target_bj_row_count": 0,
                    },
                },
                "active_label_after": {
                    "path": "label.duckdb",
                    "table": "label",
                    "metrics": {"max_trade_date": "20260616", "target_row_count": 0},
                },
                "candidate_validation": {
                    "qfq_source_audit": {
                        "mismatch_total": 0,
                        "l2_null_total": 288,
                        "feature_null_total": 288,
                    }
                },
                "active_feature_after": {"path": "active.duckdb", "table": "feature"},
            }
            delivery._write_markdown(output, report, args)
            text = output.read_text(encoding="utf-8")
            self.assertIn("# 20260720 L3 目标日增量交付报告", text)
            self.assertIn("替换 `20260720` 截面", text)
            self.assertNotIn("替换 `20260717` 截面", text)

    def test_contract_reflects_candidate_and_execute_delivery_state(self):
        base_report = {
            "precheck": {"gates": {}, "l2": {"path": "l2.duckdb", "table": "l2"}},
            "target_frame_validation": {"gates": {}},
            "candidate_validation": {"gates": {}},
            "post_switch_gates": {},
            "progress_path": "progress.jsonl",
            "candidate_feature": {"path": "candidate.duckdb", "table": "feature"},
            "active_feature_after": {"path": "active.duckdb", "table": "feature"},
            "active_label_after": {"path": "label.duckdb", "table": "label"},
            "rollback_snapshot": {"path": None},
        }
        args = delivery.parse_args(self._args(Path("root"), "candidate"))
        for active_switch_called, expected_path, expected_candidate_only in (
            (False, "candidate.duckdb", True),
            (True, "active.duckdb", False),
        ):
            with self.subTest(active_switch_called=active_switch_called), tempfile.TemporaryDirectory() as tmp:
                report = dict(base_report, active_switch_called=active_switch_called)
                with mock.patch.object(delivery, "build_layer_handoff_contract", return_value={}) as builder, mock.patch.object(
                    delivery, "validate_layer_handoff_contract", return_value=[]
                ):
                    delivery._build_contract(report, args, Path(tmp) / "contract.json")
                kwargs = builder.call_args.kwargs
                self.assertEqual(kwargs["active_output_assets"][0], f"{expected_path}::feature")
                self.assertEqual(kwargs["boundaries"]["candidate_only"], expected_candidate_only)
                self.assertEqual(kwargs["boundaries"]["active_switch_called"], active_switch_called)

    def test_markdown_uses_requested_target_trade_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.md"
            args = delivery.parse_args(self._args(Path(tmp), "candidate"))
            args.target_trade_date = "20260720"
            report = {
                "active_switch_called": True,
                "candidate_feature": {
                    "column_count": 839,
                    "metrics": {
                        "target_row_count": 5197,
                        "target_stock_count": 5197,
                        "duplicate_key_groups": 0,
                        "target_bj_row_count": 0,
                    },
                },
                "active_feature_after": {"path": "active.duckdb", "table": "feature"},
                "active_label_after": {
                    "path": "label.duckdb",
                    "table": "label",
                    "metrics": {"max_trade_date": "20260616", "target_row_count": 0},
                },
                "candidate_validation": {
                    "qfq_source_audit": {"mismatch_total": 0, "l2_null_total": 288, "feature_null_total": 288}
                },
                "rollback_snapshot": {"path": "rollback.duckdb"},
            }
            delivery._write_markdown(output, report, args)
            text = output.read_text(encoding="utf-8")
            self.assertIn("# 20260720 L3 \u76ee\u6807\u65e5\u589e\u91cf\u4ea4\u4ed8\u62a5\u544a", text)
            self.assertIn("\u5df2\u5b8c\u6210\u53d7\u63a7 active \u539f\u5b50\u5207\u6362", text)
            self.assertNotIn("20260717", text)

    def test_process_gate_registry_classifies_delivery_modes(self):
        spec = process_gate.L3_MUTATION_ENTRY_REGISTRY["deliver_l3_target_date_duckdb_mainline.py"]
        self.assertEqual(spec["policy"], "mode_switch")
        self.assertEqual(spec["readonly_modes"], {"precheck"})
        self.assertEqual(spec["writer_modes"], {"candidate", "execute"})

    def test_active_raw_derived_cci_is_restored_without_allowing_naked_qfq_fields(self):
        raw = pd.DataFrame(
            {
                "trade_date": ["20260717"],
                "stock_code": ["000001.SZ"],
                "cci": [1.25],
                "cci_qfq": [2.5],
                "macd": [3.0],
            }
        )
        feature = raw[["trade_date", "stock_code", "cci_qfq"]].copy()
        result, restored = delivery._restore_active_raw_derived_columns(
            feature,
            raw,
            ["trade_date", "stock_code", "cci", "cci_qfq", "macd"],
        )
        self.assertEqual(list(restored), ["cci"])
        self.assertEqual(restored["cci"]["source_columns"], ["high", "low", "close"])
        self.assertEqual(restored["cci"]["source_semantics"], "raw_market_price")
        self.assertEqual(float(result.loc[0, "cci"]), 1.25)
        self.assertNotIn("macd", result.columns)

    def test_compatibility_allowlist_rejects_prices_qfq_alias_gtja_future_and_unknown(self):
        raw = pd.DataFrame(
            {
                "trade_date": ["20260717"],
                "stock_code": ["000001.SZ"],
                "open": [10.0],
                "atr": [1.0],
                "alpha_001": [2.0],
                "post1_yield_rate": [3.0],
                "unknown_factor": [4.0],
            }
        )
        feature = raw[["trade_date", "stock_code"]].copy()
        result, restored = delivery._restore_active_raw_derived_columns(feature, raw, list(raw.columns))
        self.assertEqual(restored, {})
        for column in ("open", "atr", "alpha_001", "post1_yield_rate", "unknown_factor"):
            self.assertNotIn(column, result.columns)

    def test_production_raw_columns_rejects_index_post10_shift_field(self):
        columns = [
            "trade_date",
            "stock_code",
            "close_qfq",
            "index_2000_close",
            "index_2000_post10_close",
            "macdsignal_qfq",
        ]
        selected = delivery.production_raw_columns(columns)
        self.assertNotIn("index_2000_post10_close", selected)
        self.assertTrue(delivery.is_future_or_label_column("index_2000_post10_close"))

    def test_unknown_active_schema_column_fails_closed(self):
        raw = pd.DataFrame(
            {
                "trade_date": ["20260717"],
                "stock_code": ["000001.SZ"],
                "industry": ["银行"],
                "unknown_factor": [4.0],
            }
        )
        gtja = raw[["trade_date", "stock_code"]].copy()

        def encode(frame, _mapping):
            result = frame.copy()
            result["industry_encode"] = 1
            return result

        with mock.patch.object(
            delivery, "production_raw_columns", return_value=["trade_date", "stock_code", "industry"]
        ), mock.patch.object(delivery, "GTJA_ALPHA_COLUMNS", []), mock.patch.object(
            delivery, "load_industry_encode_mapping", return_value={"银行": 1}
        ), mock.patch.object(delivery, "apply_industry_encode", side_effect=encode):
            with self.assertRaisesRegex(RuntimeError, "unknown_factor"):
                delivery._build_feature_target(
                    raw,
                    gtja,
                    ["trade_date", "stock_code", "industry", "industry_encode", "unknown_factor"],
                )

    def test_v2_feature_contract_excludes_semantic_future_from_target_frame(self):
        raw = pd.DataFrame(
            {
                "trade_date": ["20260804"],
                "stock_code": ["000001.SZ"],
                "open_qfq": [10.0],
                "high_qfq": [11.0],
                "low_qfq": [9.0],
                "close_qfq": [10.5],
                "pre_close_qfq": [10.1],
                "macd_qfq": [0.25],
                "index_2000_post10_close": [1234.5],
            }
        )
        gtja = raw[["trade_date", "stock_code"]].copy()
        active_columns = [
            "trade_date",
            "stock_code",
            "open_qfq",
            "high_qfq",
            "low_qfq",
            "close_qfq",
            "pre_close_qfq",
            "macd_qfq",
            "index_2000_post10_close",
        ]

        feature, meta = delivery._build_feature_target(
            raw,
            gtja,
            active_columns,
            feature_contract_version=delivery.FEATURE_CONTRACT_V2,
        )

        self.assertNotIn("index_2000_post10_close", feature.columns)
        self.assertEqual(meta["feature_contract_version"], delivery.FEATURE_CONTRACT_V2)
        self.assertEqual(meta["excluded_semantic_future_columns"], ["index_2000_post10_close"])
        self.assertEqual(list(feature.columns), active_columns[:-1])

    def test_v2_feature_contract_drops_forbidden_columns_from_candidate_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "candidate.duckdb"
            with delivery.duckdb.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE feature (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        close_qfq DOUBLE,
                        index_2000_post10_close DOUBLE
                    )
                    """
                )
                before_schema = delivery._schema(conn, "feature")
                precheck = {"active_feature_before": {"schema": before_schema}}
                contract = delivery._expected_feature_contract(precheck, delivery.FEATURE_CONTRACT_V2)
                result = delivery._apply_feature_contract_schema(conn, "feature", contract)
                after_schema = delivery._schema(conn, "feature")

        self.assertEqual(result["dropped_columns"], ["index_2000_post10_close"])
        self.assertEqual([item["name"] for item in after_schema], ["trade_date", "stock_code", "close_qfq"])
        self.assertEqual(delivery._schema_hash(after_schema), contract["schema_hash"])

    def test_v2_precheck_accepts_known_legacy_future_only_on_active_schema(self):
        columns = [
            "trade_date",
            "stock_code",
            "open_qfq",
            "high_qfq",
            "low_qfq",
            "close_qfq",
            "pre_close_qfq",
        ] + [f"tech_{index:03d}_qfq" for index in range(74)] + [
            f"gtja_alpha{index:03d}_qfq" for index in range(1, 192)
        ] + [f"factor_{index:03d}" for index in range(566)] + ["index_2000_post10_close"]
        gate = delivery._feature_schema_gate_for_contract(
            columns,
            [f"tech_{index:03d}_qfq" for index in range(74)],
            delivery.FEATURE_CONTRACT_V2,
        )
        self.assertTrue(gate["semantic_future_columns_known"])
        self.assertTrue(gate["v2_candidate_schema_838"])
        self.assertEqual(gate["details"]["semantic_future_columns_to_remove"], ["index_2000_post10_close"])
        self.assertEqual(gate["details"]["unexpected_future_columns"], [])

    def test_feature_schema_version_gate_binds_v2_to_838_and_excludes_future_lineage(self):
        probe = {"column_count": 838}
        gate = {
            "v2_candidate_schema_838": True,
            "future_label_leakage_zero": True,
            "details": {
                "v2_candidate_columns": ["trade_date", "stock_code"],
            },
        }
        result = delivery._feature_schema_version_gate(probe, gate, delivery.FEATURE_CONTRACT_V2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_column_count"], 838)
        self.assertTrue(result["index_2000_post10_close_absent"])
        self.assertTrue(result["future_lineage_zero"])

    def test_feature_schema_version_gate_keeps_v1_839_rejection_explicit(self):
        result = delivery._feature_schema_version_gate(
            {"column_count": 838},
            {"details": {}},
            delivery.FEATURE_CONTRACT_ACTIVE,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["expected_column_count"], 839)
        self.assertTrue(result["legacy_v1_schema_rejected_when_not_839"])

    def test_v2_schema_version_gate_rejects_future_lineage(self):
        result = delivery._feature_schema_version_gate(
            {"column_count": 839},
            {
                "v2_candidate_schema_838": True,
                "future_label_leakage_zero": False,
                "details": {
                    "v2_candidate_columns": ["trade_date", "index_2000_post10_close"],
                },
            },
            delivery.FEATURE_CONTRACT_V2,
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["index_2000_post10_close_absent"])
        self.assertFalse(result["future_lineage_zero"])

    def test_v2_precheck_uses_target_date_memory_contract_and_writes_failure_evidence(self):
        class FakePsutil:
            @staticmethod
            def virtual_memory():
                return type("VirtualMemory", (), {"available": delivery.TARGET_DATE_V2_MIN_AVAILABLE_BYTES - 1})()

        class FakeDisk:
            free = delivery.MIN_FREE_BYTES + 1

        l2_schema = (
            [{"name": "stock_code"}, {"name": "trade_date"}]
            + [{"name": column} for column in ["open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"]]
            + [{"name": f"tech_{index:02d}_qfq"} for index in range(delivery.QFQ_TECHNICAL_EXPECTED_COUNT)]
        )
        feature_schema = [{"name": "stock_code"}, {"name": "trade_date"}] + [
            {"name": f"feature_{index:03d}"} for index in range(837)
        ]
        metrics = {
            "target_row_count": 1,
            "target_stock_count": 1,
            "bj_row_count": 0,
            "target_bj_row_count": 0,
            "duplicate_key_groups": 0,
            "max_trade_date": "20260803",
        }

        def probe(path, table, _target_date):
            schema = l2_schema if table == "STOCK_DAILY_DATA" else feature_schema
            if table == "LABEL":
                schema = [{"name": "stock_code"}, {"name": "trade_date"}]
                table_metrics = dict(metrics, target_row_count=0, max_trade_date="20260716")
            else:
                table_metrics = dict(metrics)
            return {
                "path": str(path),
                "table": table,
                "file_state": {"sha256": "a" * 64, "size_bytes": 1, "mtime_ns": 1},
                "metrics": table_metrics,
                "historical_key_metrics": {},
                "schema": schema,
                "schema_hash": "schema",
                "column_count": len(schema),
                "table_names": [table],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = delivery.parse_args(
                self._args(root, "precheck")
                + ["--feature-contract-version", delivery.FEATURE_CONTRACT_V2]
            )
            args.report_dir = str(root / "reports")
            args.workspace_dir = str(root / "workspace")
            with mock.patch.dict(sys.modules, {"psutil": FakePsutil}), mock.patch.object(
                delivery.shutil, "disk_usage", return_value=FakeDisk()
            ), mock.patch.object(
                delivery, "_active_registry_asset", side_effect=[
                    ({"asset_id": delivery.EXPECTED_L2_ASSET_ID}, delivery.EXPECTED_L2_PATH.resolve(), delivery.EXPECTED_L2_TABLE),
                    ({}, root / "feature.duckdb", "FEATURE"),
                    ({}, root / "label.duckdb", "LABEL"),
                ]
            ), mock.patch.object(delivery, "_file_sha256", return_value="a" * 64), mock.patch.object(
                delivery, "_asset_probe", side_effect=probe
            ), mock.patch.object(
                delivery, "_file_state", return_value={"sha256": "b" * 64, "size_bytes": 1, "mtime_ns": 1}
            ), mock.patch.object(
                delivery, "_feature_schema_gate_summary", return_value={"qfq_price_5": True, "qfq_technical_74": True, "gtja_qfq_191": True, "details": {}}
            ), mock.patch.object(
                delivery, "_target_expected_metrics", return_value={"target_row_count": 1, "target_stock_count": 1}
            ), mock.patch.object(
                delivery, "_scan_l3_processes", return_value={"passed": True}
            ):
                with self.assertRaisesRegex(RuntimeError, "available_memory_at_least_32gib_target_date_v2"):
                    delivery._precheck(args)

            failure = root / "reports" / "l3_target_date_20260717_precheck_failed.json"
            payload = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed_closed_precheck")
            self.assertIn("available_memory_at_least_32gib_target_date_v2", payload["failed_gates"])
            self.assertEqual(
                payload["resources"]["memory_contract"]["profile_id"],
                "target_date_incremental_v2_single_worker_32gib",
            )


if __name__ == "__main__":
    unittest.main()
