from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import rebuild_l3_memory_bounded_v2_candidate as candidate


class MemoryBoundedV2CandidateTests(unittest.TestCase):
    def test_streaming_policy_matches_formula(self) -> None:
        policy = candidate._streaming_policy()
        self.assertEqual(policy.workers, 1)
        self.assertEqual(policy.worker_rss_hard_bytes, 6 * 1024**3)
        self.assertEqual(policy.parent_rss_budget_bytes, 4 * 1024**3)
        self.assertEqual(policy.system_hard_stop_bytes, 20 * 1024**3)
        self.assertEqual(policy.startup_available_bytes, 34 * 1024**3)
        self.assertEqual(policy.dispatch_available_bytes, 34 * 1024**3)
        self.assertEqual(policy.duckdb_memory_limit, "3GB")
        contract = candidate._streaming_resource_contract(policy)
        self.assertEqual(contract["topology"]["raw_stock_buckets"], 1024)
        self.assertEqual(contract["topology"]["gtja_date_chunk_days"], 5)
        self.assertEqual(contract["topology"]["gtja_alpha_batch_size"], 1)
        self.assertTrue(contract["safety"]["original_64_gib_profile_unchanged"])

    def test_parse_defaults_are_streaming_v2(self) -> None:
        args = candidate.parse_args(
            [
                "--target-trade-date",
                "20260804",
                "--workflow-run-id",
                "test-run",
                "--report-dir",
                "C:/tmp/report",
            ]
        )
        self.assertEqual((args.raw_buckets, args.gtja_date_chunk_days, args.gtja_alpha_batch_size, args.workers), (1024, 5, 1, 1))
        self.assertEqual(args.feature_contract_version, "v2")

    def test_main_rejects_non_streaming_overrides_before_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one worker"):
            candidate.main(
                [
                    "--target-trade-date",
                    "20260804",
                    "--workflow-run-id",
                    "test-run",
                    "--workers",
                    "2",
                    "--report-dir",
                    "C:/tmp/report",
                ]
            )

    def test_runtime_provenance_gate_is_closed_to_audited_copy(self) -> None:
        evidence = candidate._runtime_provenance_gate()
        self.assertTrue(evidence["sys.executable"].endswith("runtime_candidates\\my_quant_copy_20260804\\python.exe"))
        self.assertIn("runtime_candidates\\my_quant_copy_20260804", evidence["stdlib"])

    def test_negative_shift_lineage_is_scanned_and_unknown_is_rejected(self) -> None:
        lineage = candidate._negative_shift_lineage_from_source()
        self.assertIn("index_2000_post10_close", lineage)
        with self.assertRaises(ValueError):
            candidate.validate_future_lineage({"unregistered_feature": "source.shift(-3)"})

    def test_build_rejects_nonempty_report_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            report_dir.mkdir()
            (report_dir / "old.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "report directory prohibits reuse"):
                candidate.main(
                    [
                        "--target-trade-date",
                        "20260804",
                        "--mode",
                        "build",
                        "--workflow-run-id",
                        "test-run",
                        "--report-dir",
                        str(report_dir),
                    ]
                )
            incident_path = report_dir / "l3_full_processing_20260804_incident.json"
            self.assertTrue(incident_path.is_file())
            incident = json.loads(incident_path.read_text(encoding="utf-8"))
            self.assertEqual(incident["status"], "failed_closed_startup_gate")
            self.assertIn("report directory prohibits reuse", incident["error"])

    def test_build_allows_only_same_run_passed_precheck_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            report_dir.mkdir()
            target_date = "20260804"
            run_id = "same-run"
            precheck = {
                "status": "precheck_passed_active_unchanged",
                "workflow_run_id": run_id,
                "target_trade_date": target_date,
                "feature_contract_version": "v2",
                "l2": {"sha256": "a" * 64, "expected_sha256": "a" * 64},
            }
            (report_dir / f"l3_full_processing_{target_date}_precheck.json").write_text(
                json.dumps(precheck), encoding="utf-8"
            )
            (report_dir / f"l3_full_processing_{target_date}_process_gate.json").write_text("{}", encoding="utf-8")
            (report_dir / f"l3_full_processing_{target_date}_progress.jsonl").write_text("", encoding="utf-8")
            (report_dir / f"l3_full_processing_{target_date}_outer_cleanup.json").write_text("{}", encoding="utf-8")
            args = candidate.parse_args(
                [
                    "--target-trade-date", target_date,
                    "--mode", "build",
                    "--workflow-run-id", run_id,
                    "--report-dir", str(report_dir),
                    "--expected-l2-sha256", "a" * 64,
                ]
            )
            self.assertTrue(candidate._allow_same_run_precheck_continuation(report_dir, args))
            precheck["status"] = "precheck_passed"
            (report_dir / f"l3_full_processing_{target_date}_precheck.json").write_text(
                json.dumps(precheck), encoding="utf-8"
            )
            self.assertFalse(candidate._allow_same_run_precheck_continuation(report_dir, args))
            precheck["status"] = "precheck_passed_active_unchanged"
            (report_dir / f"l3_full_processing_{target_date}_precheck.json").write_text(
                json.dumps(precheck), encoding="utf-8"
            )
            (report_dir / "foreign.json").write_text("{}", encoding="utf-8")
            self.assertFalse(candidate._allow_same_run_precheck_continuation(report_dir, args))

    def test_precheck_mode_does_not_write_partial_passed_precheck_before_active_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            workspace_dir = Path(tmp) / "workspace"
            original_runtime_gate = candidate._runtime_provenance_gate
            original_precheck = candidate._precheck
            original_active_snapshot = candidate._active_snapshot
            original_file_state = candidate._file_state
            original_cleanup = candidate.cleanup_descendants_for_run
            try:
                candidate._runtime_provenance_gate = lambda: {"test_runtime": True}
                candidate._precheck = lambda args: {
                    "status": "precheck_passed",
                    "target_trade_date": args.target_trade_date,
                    "workflow_run_id": args.workflow_run_id,
                    "l2": {"sha256": "a" * 64, "expected_sha256": "a" * 64},
                    "gates": {},
                }
                feature_before = {
                    "asset_id": "feature",
                    "asset_path": "feature.duckdb::table",
                    "file_state": {"sha256": "feature-before"},
                    "metrics": {"row_count": 1},
                    "schema_hash": "schema",
                }
                label_snapshot = {
                    "asset_id": "label",
                    "asset_path": "label.duckdb::table",
                    "file_state": {"sha256": "label"},
                    "metrics": {"row_count": 1},
                    "schema_hash": "schema",
                }
                feature_after = {
                    **feature_before,
                    "file_state": {"sha256": "feature-after"},
                }
                snapshots = iter([feature_before, label_snapshot, feature_after, label_snapshot])
                candidate._active_snapshot = lambda layer, target_date: next(snapshots)
                candidate._file_state = lambda path: {"sha256": "registry"}
                candidate.cleanup_descendants_for_run = lambda **kwargs: {"residual_processes_after": []}
                rc = candidate.main(
                    [
                        "--target-trade-date",
                        "20260804",
                        "--mode",
                        "precheck",
                        "--workflow-run-id",
                        "partial-precheck-test",
                        "--report-dir",
                        str(report_dir),
                        "--workspace-dir",
                        str(workspace_dir),
                        "--expected-l2-sha256",
                        "a" * 64,
                    ]
                )
            finally:
                candidate._runtime_provenance_gate = original_runtime_gate
                candidate._precheck = original_precheck
                candidate._active_snapshot = original_active_snapshot
                candidate._file_state = original_file_state
                candidate.cleanup_descendants_for_run = original_cleanup
            self.assertEqual(rc, 1)
            self.assertFalse((report_dir / "l3_full_processing_20260804_precheck.json").exists())
            incident = json.loads((report_dir / "l3_full_processing_20260804_incident.json").read_text(encoding="utf-8"))
            self.assertIn("precheck touched active state", incident["error"])

    def _make_shard(self, root: Path, name: str, rows: list[tuple[str, str, float]]) -> Path:
        path = root / name
        with duckdb.connect(str(path)) as conn:
            conn.execute("CREATE TABLE raw_factor(stock_code VARCHAR, trade_date VARCHAR, value DOUBLE)")
            conn.executemany("INSERT INTO raw_factor VALUES (?, ?, ?)", rows)
        return path

    def test_ordered_schema_rejects_reversed_missing_and_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "key order mismatch"):
            candidate._ordered_columns(["trade_date", "stock_code", "value"], context="reversed")
        with self.assertRaisesRegex(RuntimeError, "missing key columns"):
            candidate._ordered_columns(["stock_code", "value"], context="missing")
        with self.assertRaisesRegex(RuntimeError, "duplicate columns"):
            candidate._ordered_columns(["stock_code", "trade_date", "trade_date"], context="duplicate")

    def test_schema_contract_rejects_type_drift(self) -> None:
        expected = [
            {"name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"name": "trade_date", "type": "VARCHAR", "not_null": False},
        ]
        observed = [
            {"name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"name": "trade_date", "type": "DATE", "not_null": False},
        ]
        with self.assertRaisesRegex(RuntimeError, "schema contract mismatch"):
            candidate._assert_schema_contract(observed, expected, context="type drift")

    def test_schema_contract_rejects_missing_or_extra_columns(self) -> None:
        expected = [{"name": "stock_code", "type": "VARCHAR", "not_null": False}]
        with self.assertRaisesRegex(RuntimeError, "schema contract mismatch"):
            candidate._assert_schema_contract(
                expected + [{"name": "trade_date", "type": "VARCHAR", "not_null": False}],
                expected,
                context="extra column",
            )
        with self.assertRaisesRegex(RuntimeError, "schema contract mismatch"):
            candidate._assert_schema_contract([], expected, context="missing column")

    def test_load_expected_schema_accepts_schema_lock_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l2_schema_lock.json"
            schema = [
                {"index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
                {"index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
            ]
            path.write_text(json.dumps({"ordered_l2_schema": schema}), encoding="utf-8")
            self.assertEqual(candidate._load_expected_schema(str(path)), schema)

    def test_schema_hash_accepts_schema_lock_index_key(self) -> None:
        lock_schema = [
            {"index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
        ]
        pragma_schema = [
            {"column_index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"column_index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
        ]
        self.assertEqual(candidate._schema_hash(pragma_schema), candidate._schema_hash(lock_schema))

    def test_raw_output_contract_allows_group_factor_derived_columns(self) -> None:
        source_schema = [
            {"column_index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"column_index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
            {"column_index": 2, "name": "close", "type": "DOUBLE", "not_null": False},
        ]
        contract = candidate._raw_factor_output_contract(
            source_schema,
            ["stock_code", "trade_date", "close", "macd_qfq"],
            context="synthetic factor engineering",
        )
        self.assertEqual(contract["input_schema_hash"], candidate._schema_hash(source_schema))
        self.assertEqual(contract["derived_columns"], ["macd_qfq"])
        self.assertEqual(contract["output_columns"][:3], ["stock_code", "trade_date", "close"])

    def test_raw_output_contract_allows_loop_generated_alpha158_columns(self) -> None:
        source_schema = [
            {"column_index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"column_index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
            {"column_index": 2, "name": "close", "type": "DOUBLE", "not_null": False},
        ]
        contract = candidate._raw_factor_output_contract(
            source_schema,
            [
                "stock_code",
                "trade_date",
                "close",
                "alpha158_roc5",
                "alpha158_ma10",
                "alpha158_rank20",
            ],
            context="synthetic alpha158 factor engineering",
        )
        self.assertEqual(
            contract["derived_columns"],
            ["alpha158_roc5", "alpha158_ma10", "alpha158_rank20"],
        )

    def test_raw_output_contract_models_pre_close_reinsert_order(self) -> None:
        source_schema = [
            {"column_index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"column_index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
            {"column_index": 2, "name": "close", "type": "DOUBLE", "not_null": False},
            {"column_index": 3, "name": "pre_close", "type": "DOUBLE", "not_null": False},
            {"column_index": 4, "name": "pre_close_qfq", "type": "DOUBLE", "not_null": False},
        ]
        contract = candidate._raw_factor_output_contract(
            source_schema,
            ["stock_code", "trade_date", "close", "pre_close_qfq", "pre_close", "macd_qfq"],
            context="synthetic pre_close reorder factor engineering",
        )
        self.assertEqual(
            contract["output_columns"],
            ["stock_code", "trade_date", "close", "pre_close_qfq", "pre_close", "macd_qfq"],
        )

    def test_raw_output_contract_rejects_legacy_pre_close_input_order(self) -> None:
        source_schema = [
            {"column_index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"column_index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
            {"column_index": 2, "name": "close", "type": "DOUBLE", "not_null": False},
            {"column_index": 3, "name": "pre_close", "type": "DOUBLE", "not_null": False},
            {"column_index": 4, "name": "pre_close_qfq", "type": "DOUBLE", "not_null": False},
        ]
        with self.assertRaisesRegex(RuntimeError, "raw output order mismatch"):
            candidate._raw_factor_output_contract(
                source_schema,
                ["stock_code", "trade_date", "close", "pre_close", "pre_close_qfq", "macd_qfq"],
                context="synthetic stale pre_close order factor engineering",
            )

    def test_raw_output_contract_rejects_unknown_derived_column(self) -> None:
        source_schema = [
            {"column_index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"column_index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
            {"column_index": 2, "name": "close", "type": "DOUBLE", "not_null": False},
        ]
        with self.assertRaisesRegex(RuntimeError, "unapproved derived columns"):
            candidate._raw_factor_output_contract(
                source_schema,
                ["stock_code", "trade_date", "close", "alpha158_not_a_real_factor"],
                context="synthetic bad factor engineering",
            )

    def test_raw_output_contract_normalizes_l2_uppercase_columns_for_pandas_raw(self) -> None:
        source_schema = [
            {"column_index": 0, "name": "stock_code", "type": "VARCHAR", "not_null": False},
            {"column_index": 1, "name": "trade_date", "type": "VARCHAR", "not_null": False},
            {"column_index": 2, "name": "ST_TYPE", "type": "VARCHAR", "not_null": False},
            {"column_index": 3, "name": "ST_TYPE_name", "type": "VARCHAR", "not_null": False},
            {"column_index": 4, "name": "close", "type": "DOUBLE", "not_null": False},
        ]
        raw_source_schema = candidate._schema_with_lowercase_names(source_schema, context="synthetic L2")
        contract = candidate._raw_factor_output_contract(
            raw_source_schema,
            ["stock_code", "trade_date", "st_type", "st_type_name", "close", "macd_qfq"],
            context="synthetic factor engineering",
        )
        self.assertEqual(
            contract["output_columns"],
            ["stock_code", "trade_date", "st_type", "st_type_name", "close", "macd_qfq"],
        )

    def test_raw_bucket_materializes_l2_plus_approved_derived_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            l2_path = root / "synthetic_l2.duckdb"
            output_path = root / "raw_bucket.duckdb"
            with duckdb.connect(str(l2_path)) as conn:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA(stock_code VARCHAR, trade_date VARCHAR, close DOUBLE)")
                conn.executemany(
                    "INSERT INTO STOCK_DAILY_DATA VALUES (?, ?, ?)",
                    [("000001.SZ", "20260803", 10.0), ("000001.SZ", "20260804", 11.0)],
                )

            original_gate = candidate._runtime_provenance_gate
            original_engine = candidate.group_factor_eng
            try:
                candidate._runtime_provenance_gate = lambda: None

                def fake_engine(frame, *, include_future_labels=True):
                    result = frame.copy()
                    result["macd_qfq"] = result["close"] * 0.0
                    return result

                candidate.group_factor_eng = fake_engine
                result = candidate._build_raw_bucket(
                    {
                        "output_path": str(output_path),
                        "l2_path": str(l2_path),
                        "l2_table": "STOCK_DAILY_DATA",
                        "codes": ["000001.SZ"],
                        "bucket_index": 0,
                        "temp_directory": str(root / "duckdb_tmp"),
                        "duckdb_memory_limit": "3GB",
                    }
                )
            finally:
                candidate._runtime_provenance_gate = original_gate
                candidate.group_factor_eng = original_engine
            self.assertEqual(result["source_rows"], 2)
            self.assertEqual(result["output_rows"], 2)
            self.assertEqual(result["raw_output_derived_columns"], ["macd_qfq"])
            with duckdb.connect(str(output_path), read_only=True) as conn:
                self.assertEqual(
                    [row[0] for row in conn.execute("DESCRIBE raw_factor").fetchall()],
                    ["stock_code", "trade_date", "close", "macd_qfq"],
                )

    def test_hierarchical_merge_preserves_rows_and_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = [
                self._make_shard(root, f"shard_{i:02d}.duckdb", [(f"2026080{i % 4 + 1}", f"00000{i}.SZ", float(i))])
                for i in range(10)
            ]
            output = root / "merged.duckdb"
            result = candidate._merge_shards(
                shards,
                output,
                "raw_factor",
                "raw_factor",
                workspace=root / "workspace",
                policy=candidate._streaming_policy(),
            )
            self.assertEqual(result["row_count"], 10)
            self.assertEqual(result["duplicate_key_groups"], 0)
            self.assertGreaterEqual(result["merge_levels"], 2)
            with duckdb.connect(str(output), read_only=True) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM raw_factor").fetchone()[0], 10)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM (SELECT trade_date, stock_code FROM raw_factor GROUP BY 1,2)").fetchone()[0], 10)

    def test_hierarchical_merge_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = [
                self._make_shard(root, "shard_a.duckdb", [("20260804", "000001.SZ", 1.0)]),
                self._make_shard(root, "shard_b.duckdb", [("20260804", "000001.SZ", 2.0)]),
            ]
            with self.assertRaises(RuntimeError):
                candidate._merge_shards(
                    shards,
                    root / "merged.duckdb",
                    "raw_factor",
                    "raw_factor",
                    workspace=root / "workspace",
                    policy=candidate._streaming_policy(),
                )

    def test_hierarchical_merge_rejects_reordered_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._make_shard(root, "shard_a.duckdb", [("000001.SZ", "20260804", 1.0)])
            second = root / "shard_b.duckdb"
            with duckdb.connect(str(second)) as conn:
                conn.execute("CREATE TABLE raw_factor(trade_date VARCHAR, stock_code VARCHAR, value DOUBLE)")
                conn.execute("INSERT INTO raw_factor VALUES ('20260804', '000002.SZ', 2.0)")
            with self.assertRaisesRegex(RuntimeError, "schema contract mismatch|schema order drift|key order mismatch"):
                candidate._merge_shards(
                    [first, second],
                    root / "merged.duckdb",
                    "raw_factor",
                    "raw_factor",
                    workspace=root / "workspace",
                    policy=candidate._streaming_policy(),
                )

    def test_future_field_is_excluded_from_v2_allowlist(self) -> None:
        selected = candidate.candidate_production_raw_columns(
            ["trade_date", "stock_code", "close_qfq", "index_2000_post10_close"]
        )
        self.assertNotIn("index_2000_post10_close", selected)

    def test_feature_projection_is_explicit_and_not_select_star(self) -> None:
        source = inspect.getsource(candidate._build_feature_candidate)
        self.assertIn("SELECT {', '.join([*raw_exprs, *gtja_exprs])}", source)
        self.assertNotIn("SELECT * FROM rawsrc.raw_factor", source)


if __name__ == "__main__":
    unittest.main()
