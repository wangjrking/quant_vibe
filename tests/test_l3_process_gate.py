import json
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import l3_process_gate as gate
import rebuild_l3_full_duckdb_mainline as full_l3


RUN_ID = "incremental-trading-signal-20260717-L3-full-processing-candidate-staging-attempt-4"
WORKSPACE = r"D:\work\quant\quant_mcp\quant\data_file\runtime\agent_workspaces\factor-agent\work\attempt4"
REPORT_DIR = r"D:\work\quant\quant_mcp\quant\data_file\reports\attempt4"
UNIFIED = r"D:\work\quant\quant_mcp\.venv\Scripts\python.exe"
SCRIPT = r"D:\work\quant\quant_mcp\quant\main\rebuild_l3_full_duckdb_mainline.py"


def command(mode="build", run_id=RUN_ID, workspace=WORKSPACE, report_dir=REPORT_DIR):
    return [
        SCRIPT,
        "--mode", mode,
        "--workflow-run-id", run_id,
        "--workspace-dir", workspace,
        "--report-dir", report_dir,
    ]


def process(pid, ppid, *, name="python.exe", exe=r"C:\Python\python.exe", cmdline=None, **extra):
    return {
        "pid": pid,
        "ppid": ppid,
        "name": name,
        "exe": exe,
        "cmdline": cmdline or [],
        "create_time": 1.0,
        **extra,
    }


def classify(records, expected_workers=None, protected=()):
    return gate.classify_process_gate(
        records,
        current_pid=200,
        workflow_run_id=RUN_ID,
        workspace=WORKSPACE,
        report_dir=REPORT_DIR,
        unified_python_launcher=UNIFIED,
        expected_workers=expected_workers,
        protected_l3_paths=protected,
    )


class L3ProcessGateTests(unittest.TestCase):
    def current_chain(self):
        current_command = [r"C:\Python\python.exe", *command()]
        launcher_command = [UNIFIED, *command()]
        shell_command = ["powershell.exe", "-Command", "&", UNIFIED, *command()]
        return [
            process(50, 1, name="powershell.exe", exe=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", cmdline=shell_command),
            process(100, 50, exe=UNIFIED, cmdline=launcher_command),
            process(200, 100, cmdline=current_command),
        ]

    def test_actual_powershell_and_unified_python_wrapper_fixture_is_allowed(self):
        result = classify(self.current_chain())
        self.assertTrue(result["passed"])
        self.assertEqual({item["pid"] for item in result["allowed_current_chain"]}, {50, 100, 200})

    def test_shell_wrapper_without_workflow_flags_is_allowed_only_in_current_chain(self):
        records = [
            process(
                50,
                1,
                name="powershell.exe",
                exe=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                cmdline=["powershell.exe", "-NoProfile", "-Command", "&", UNIFIED, SCRIPT],
            ),
            process(200, 50, cmdline=[r"C:\Python\python.exe", *command()]),
        ]
        result = classify(records)
        self.assertTrue(result["passed"])
        self.assertEqual(result["unknown_relevant_processes"], [])
        self.assertEqual(result["allowed_current_chain"][0]["process_role"], "legal_shell_launcher_ancestor")

        foreign = records + [
            process(60, 1, name="powershell.exe", exe="powershell.exe", cmdline=["powershell.exe", "-Command", "&", UNIFIED, SCRIPT]),
        ]
        blocked = classify(foreign)
        self.assertFalse(blocked["passed"])
        self.assertEqual(blocked["blocking_writers"], [])
        self.assertEqual(blocked["unknown_relevant_processes"][0]["process_role"], "unknown_l3_mutation_entry")

    def test_foreign_powershell_launcher_is_blocked(self):
        records = self.current_chain() + [
            process(60, 1, name="powershell.exe", exe="powershell.exe", cmdline=["powershell.exe", *command()])
        ]
        result = classify(records)
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["process_role"], "foreign_shell_launcher")

    def test_independent_same_run_writer_is_blocked(self):
        records = self.current_chain() + [process(300, 1, cmdline=["python.exe", *command()])]
        result = classify(records)
        self.assertEqual([item["pid"] for item in result["blocking_writers"]], [300])

    def test_dash_wal_evidence_for_active_feature_is_blocking(self):
        active = r"D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\l3_feature_current.duckdb"
        wal = process(
            501,
            1,
            name="duckdb_wal_evidence",
            cmdline=["duckdb_wal", f"{active}-wal"],
            open_files=[{"path": f"{active}-wal", "mode": "w"}],
        )
        result = classify(self.current_chain() + [wal], protected=[active])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["process_role"], "l3_lock_holder")

    def test_old_run_writer_is_blocked(self):
        records = self.current_chain() + [
            process(301, 1, cmdline=["python.exe", *command(run_id="old-run", workspace=WORKSPACE + "_old")])
        ]
        result = classify(records)
        self.assertEqual([item["pid"] for item in result["blocking_writers"]], [301])

    def test_registered_worker_with_exact_parent_run_workspace_and_task_is_allowed(self):
        worker = process(400, 200, cmdline=["python.exe", "-c", "from multiprocessing.spawn import spawn_main"])
        expected = {400: {
            "pid": 400,
            "create_time_ns": 1_000_000_000,
            "parent_pid": 200,
            "run_nonce": "run-nonce",
            "worker_nonce": "worker-nonce",
            "workflow_run_id": RUN_ID,
            "workspace": WORKSPACE,
            "task_id": "raw_bucket_0000",
        }}
        result = classify(self.current_chain() + [worker], expected_workers=expected)
        self.assertTrue(result["passed"])
        self.assertEqual(next(item for item in result["allowed_current_chain"] if item["pid"] == 400)["task_id"], "raw_bucket_0000")

    def test_worker_with_wrong_parent_is_blocked(self):
        worker = process(401, 50, cmdline=["python.exe", "-c", "from multiprocessing.spawn import spawn_main"])
        expected = {401: {
            "pid": 401,
            "parent_pid": 200,
            "workflow_run_id": RUN_ID,
            "workspace": WORKSPACE,
            "task_id": "raw_bucket_0001",
        }}
        result = classify(self.current_chain() + [worker], expected_workers=expected)
        self.assertFalse(result["passed"])

    def test_unregistered_orchestrator_child_is_blocked(self):
        child = process(402, 200, cmdline=["python.exe", *command(workspace=WORKSPACE + "_child")])
        result = classify(self.current_chain() + [child])
        self.assertEqual(result["blocking_writers"][0]["process_role"], "unexpected_orchestrator_child")

    def test_allowlisted_readonly_observer_with_distinct_paths_is_allowed(self):
        observer = process(500, 1, cmdline=["python.exe", *command(mode="precheck", workspace=WORKSPACE + "_observer", report_dir=REPORT_DIR + "_observer")])
        result = classify(self.current_chain() + [observer])
        self.assertTrue(result["passed"])
        self.assertEqual([item["pid"] for item in result["allowed_readonly_observers"]], [500])

    def test_readonly_observer_path_collision_is_blocked(self):
        observer = process(501, 1, cmdline=["python.exe", *command(mode="precheck", workspace=WORKSPACE, report_dir=REPORT_DIR + "_observer")])
        result = classify(self.current_chain() + [observer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["process_role"], "readonly_observer_path_collision")

    def test_unregistered_spawn_process_is_blocked(self):
        worker = process(502, 1, cmdline=["python.exe", "-c", "from multiprocessing.spawn import spawn_main"])
        result = classify(self.current_chain() + [worker])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["process_role"], "unregistered_or_foreign_worker")

    def test_access_denied_is_unknown_and_fail_closed(self):
        inaccessible = process(503, 1, name="unknown", cmdline=[], access_error="AccessDenied")
        result = classify(self.current_chain() + [inaccessible])
        self.assertFalse(result["passed"])
        self.assertEqual([item["pid"] for item in result["unknown_relevant_processes"]], [503])

    def test_active_l3_write_handle_is_blocked(self):
        active_path = r"D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\l3_feature_current.duckdb"
        holder = process(504, 1, cmdline=["python.exe", "writer.py"], open_files=[{"path": active_path, "mode": "r+"}])
        result = classify(self.current_chain() + [holder], protected=[active_path])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["process_role"], "l3_lock_holder")

    def test_pair_change_execute_without_open_handle_is_blocked(self):
        pair_script = r"D:\work\quant\quant_mcp\quant\main\tools\apply_production_asset_pair_change.py"
        writer = process(505, 1, cmdline=["python.exe", pair_script, "--pair-change", "plan.json", "--execute"])
        result = classify(self.current_chain() + [writer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["pid"], 505)

    def test_pair_change_module_execute_without_open_handle_is_blocked(self):
        writer = process(512, 1, cmdline=[
            "python.exe", "-m", "tools.apply_production_asset_pair_change",
            "--pair-change", "plan.json", "--execute",
        ])
        result = classify(self.current_chain() + [writer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["pid"], 512)

    def test_pair_change_fully_qualified_module_execute_is_blocked(self):
        writer = process(513, 1, cmdline=[
            "python.exe", "-m", "quant.main.tools.apply_production_asset_pair_change",
            "--pair-change", "plan.json", "--execute",
        ])
        result = classify(self.current_chain() + [writer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["pid"], 513)

    def test_pair_change_fully_qualified_module_dry_run_is_readonly(self):
        observer = process(514, 1, cmdline=[
            "python.exe", "-m", "quant.main.tools.apply_production_asset_pair_change",
            "--pair-change", "plan.json",
        ])
        result = classify(self.current_chain() + [observer])
        self.assertTrue(result["passed"])
        self.assertEqual(result["allowed_readonly_observers"][0]["pid"], 514)

    def test_pair_change_fully_qualified_module_dry_run_output_is_unknown(self):
        observer = process(515, 1, cmdline=[
            "python.exe", "-m", "quant.main.tools.apply_production_asset_pair_change",
            "--pair-change", "plan.json", "--dry-run-output", r"D:\tmp\pair.json",
        ])
        result = classify(self.current_chain() + [observer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["unknown_relevant_processes"][0]["pid"], 515)

    def test_l3_sync_fully_qualified_module_is_blocked(self):
        writer = process(516, 1, cmdline=[
            "python.exe", "-m", "quant.main.l3_duckdb_sync", "--sync",
        ])
        result = classify(self.current_chain() + [writer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["pid"], 516)

    def test_single_asset_change_fully_qualified_module_is_blocked(self):
        writer = process(517, 1, cmdline=[
            "python.exe", "-m", "quant.main.tools.apply_production_asset_change", "--execute",
        ])
        result = classify(self.current_chain() + [writer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["pid"], 517)

    def test_unregistered_related_python_with_mutation_flag_is_unknown(self):
        writer = process(518, 1, cmdline=[
            "python.exe", "-m", "quant.main.unregistered_l3_writer", "--replace",
        ])
        result = classify(self.current_chain() + [writer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["unknown_relevant_processes"][0]["pid"], 518)
        self.assertEqual(result["unknown_relevant_processes"][0]["process_role"], "unknown_l3_mutation_entry")

    def test_unrelated_shell_text_with_mutation_flag_is_not_relevant(self):
        shell = process(
            519,
            1,
            name="powershell.exe",
            exe="powershell.exe",
            cmdline=["powershell.exe", "-Command", "Write-Output --execute"],
        )
        result = classify(self.current_chain() + [shell])
        self.assertTrue(result["passed"])
        self.assertNotIn(519, {
            item["pid"]
            for group in (
                "allowed_current_chain",
                "allowed_readonly_observers",
                "blocking_writers",
                "unknown_relevant_processes",
            )
            for item in result[group]
        })

    def test_unrelated_l2_reader_under_quant_repo_is_not_misclassified_as_l3_writer(self):
        reader = process(
            520,
            1,
            cmdline=[
                "python.exe",
                r"D:\work\quant\quant_mcp\quant\main\run_juejin_signal_backtest.py",
                "--market-db",
                r"D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\l2_stock_daily_data.duckdb",
            ],
        )
        result = classify(self.current_chain() + [reader])
        self.assertTrue(result["passed"])
        self.assertNotIn(520, {
            item["pid"]
            for group in (
                "allowed_current_chain",
                "allowed_readonly_observers",
                "blocking_writers",
                "unknown_relevant_processes",
            )
            for item in result[group]
        })

    def test_pair_change_dry_run_is_classified_readonly(self):
        pair_script = r"D:\work\quant\quant_mcp\quant\main\tools\apply_production_asset_pair_change.py"
        observer = process(506, 1, cmdline=["python.exe", pair_script, "--pair-change", "plan.json"])
        result = classify(self.current_chain() + [observer])
        self.assertTrue(result["passed"])
        self.assertEqual(result["allowed_readonly_observers"][0]["process_role"], "allowlisted_pair_change_dry_run")

    def test_pair_change_dry_run_with_output_is_unknown_until_path_is_proven(self):
        pair_script = r"D:\work\quant\quant_mcp\quant\main\tools\apply_production_asset_pair_change.py"
        observer = process(511, 1, cmdline=[
            "python.exe", pair_script, "--pair-change", "plan.json",
            "--dry-run-output", r"D:\tmp\pair_dry_run.json",
        ])
        result = classify(self.current_chain() + [observer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["unknown_relevant_processes"][0]["pid"], 511)

    def test_registry_atomic_replace_process_is_unknown_and_blocked(self):
        registry = r"D:\work\quant\quant_mcp\quant\data_file\asset_registry\production_assets.json"
        unknown = process(507, 1, cmdline=["python.exe", "-c", f"os.replace(temp, r'{registry}')"])
        result = classify(self.current_chain() + [unknown], protected=[registry])
        self.assertFalse(result["passed"])
        self.assertEqual(result["unknown_relevant_processes"][0]["process_role"], "unknown_l3_mutation_entry")

    def test_l3_duckdb_sync_entry_is_blocked_without_open_handle(self):
        script = r"D:\work\quant\quant_mcp\quant\main\l3_duckdb_sync.py"
        writer = process(508, 1, cmdline=["python.exe", script])
        result = classify(self.current_chain() + [writer])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["pid"], 508)

    def test_unregistered_process_with_l3_mutation_path_is_unknown_and_blocked(self):
        active = r"D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\l3_feature_current.duckdb"
        unknown = process(509, 1, cmdline=["python.exe", "mystery_writer.py", "--sync", active])
        result = classify(self.current_chain() + [unknown], protected=[active])
        self.assertFalse(result["passed"])
        self.assertEqual(result["unknown_relevant_processes"][0]["pid"], 509)

    def test_registry_write_handle_is_protected(self):
        registry = r"D:\work\quant\quant_mcp\quant\data_file\asset_registry\production_assets.json"
        holder = process(510, 1, cmdline=["python.exe", "writer.py"], open_files=[{"path": registry, "mode": "w"}])
        result = classify(self.current_chain() + [holder], protected=[registry])
        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_writers"][0]["process_role"], "l3_lock_holder")

    def test_mutation_entry_registry_contains_all_known_l3_writers(self):
        required = {
            "rebuild_l3_full_duckdb_mainline.py",
            "refresh_l3_active_duckdb_full_delivery.py",
            "deliver_l3_target_date_duckdb_mainline.py",
            "l3_duckdb_sync.py",
            "build_production_factor_parts.py",
            "build_production_factor_raw_gtja_parts.py",
            "build_prediction_label_parts.py",
            "incremental_factor_update_target_date.py",
            "incremental_prediction_label_update_target_date.py",
            "run_incremental_factor_update_chunked.py",
            "rebuild_raw_factor_by_stock.py",
            "apply_production_asset_pair_change.py",
            "apply_production_asset_change.py",
        }
        self.assertTrue(required.issubset(gate.L3_MUTATION_ENTRY_REGISTRY))

    def test_all_direct_l3_sync_and_pair_change_callers_are_registered(self):
        main_dir = Path(__file__).resolve().parents[1]
        pattern = re.compile(
            r"sync_(?:feature|label)_parts_(?:full|target_date)_to_duckdb\(|"
            r"execute_pair_registry_change\("
        )
        discovered = set()
        for path in main_dir.rglob("*.py"):
            if "tests" in path.parts:
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                discovered.add(path.name)
        self.assertTrue(discovered.issubset(gate.L3_MUTATION_ENTRY_REGISTRY), sorted(discovered - gate.L3_MUTATION_ENTRY_REGISTRY.keys()))

    def test_process_gate_json_is_written_before_later_precheck_failure_for_pass_and_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            for passed in (True, False):
                report_dir = Path(tmp) / str(passed)
                workspace = report_dir / "workspace"
                report_dir.mkdir(parents=True)
                args = Namespace(
                    target_trade_date="20260717",
                    workflow_run_id=RUN_ID,
                    workspace_dir=str(workspace),
                    report_dir=str(report_dir),
                    expected_l2_sha256="expected",
                )
                scan = {
                    "status": "process_gate_passed" if passed else "process_gate_blocked",
                    "passed": passed,
                    "blocking_writers": [] if passed else [{"pid": 999}],
                    "unknown_relevant_processes": [],
                    "allowed_current_chain": [],
                    "allowed_readonly_observers": [],
                }
                with mock.patch.object(full_l3, "_scan_l3_processes", return_value=scan), mock.patch.object(
                    full_l3, "require_psutil"
                ) as require_psutil, mock.patch.object(
                    full_l3, "_active_registry_asset", side_effect=RuntimeError("later gate")
                ):
                    require_psutil.return_value.virtual_memory.return_value.available = 80 * 1024**3
                    with self.assertRaisesRegex(RuntimeError, "later gate"):
                        full_l3._precheck(args)
                gate_path = report_dir / "l3_full_processing_20260717_process_gate.json"
                self.assertTrue(gate_path.is_file())
                self.assertEqual(json.loads(gate_path.read_text(encoding="utf-8"))["passed"], passed)

    def test_psutil_missing_fails_closed_and_writes_process_gate_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "reports"
            report_dir.mkdir()
            args = Namespace(
                target_trade_date="20260717",
                workflow_run_id=RUN_ID,
                workspace_dir=str(Path(tmp) / "workspace"),
                report_dir=str(report_dir),
                expected_l2_sha256="expected",
            )
            with mock.patch.object(full_l3, "_scan_l3_processes", side_effect=RuntimeError("psutil is required")):
                with self.assertRaisesRegex(RuntimeError, "psutil is required"):
                    full_l3._precheck(args)
            payload = json.loads((report_dir / "l3_full_processing_20260717_process_gate.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["status"], "process_gate_scan_failed")


if __name__ == "__main__":
    unittest.main()
