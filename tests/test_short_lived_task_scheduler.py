import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import psutil

import rebuild_l3_full_duckdb_mainline as full_l3
import short_lived_task_scheduler as short_lived


def _tiny_success_worker(task):
    time.sleep(float(task.get("delay", 0.02)))
    output = Path(task["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((task["task_id"] * 10).encode("ascii"))
    return {
        "task_id": task["task_id"],
        "path": str(output),
        "source_rows": 1,
        "output_rows": 1,
        "duplicate_key_groups": 0,
    }


def _tiny_failure_or_sleep_worker(task):
    if task.get("fail"):
        raise RuntimeError("intentional child failure")
    time.sleep(5)
    return _tiny_success_worker(task)


def _tiny_no_completion_worker(task):
    output = Path(task["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"no-completion")
    os._exit(0)


def _tiny_hash_tamper_worker(task):
    output = Path(task["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"before")

    def tamper():
        time.sleep(0.2)
        output.write_bytes(b"after-tamper")

    threading.Thread(target=tamper, daemon=False).start()
    return {
        "task_id": task["task_id"],
        "path": str(output),
        "source_rows": 1,
        "output_rows": 1,
        "duplicate_key_groups": 0,
    }


def _validate_tiny_result(_task, result):
    if result["source_rows"] != result["output_rows"] or result["duplicate_key_groups"]:
        raise RuntimeError("tiny result validation failed")


class _FakeMemoryInfo:
    rss = 1


class _FakePsutilProcess:
    def __init__(self, module, pid):
        self.module = module
        self.pid = pid

    def create_time(self):
        value = self.module.create_times.get(self.pid)
        if isinstance(value, BaseException):
            raise value
        if value is None:
            raise psutil.NoSuchProcess(self.pid)
        return value

    def ppid(self):
        return self.module.parents.get(self.pid, 0)

    def memory_info(self):
        return _FakeMemoryInfo()

    def children(self, recursive=True):
        return []

    def name(self):
        return "python.exe"

    def cmdline(self):
        return ["python.exe", "worker"]


class _FakePsutil:
    NoSuchProcess = psutil.NoSuchProcess
    AccessDenied = psutil.AccessDenied
    TimeoutExpired = psutil.TimeoutExpired

    def __init__(self, parent_pid, parent_create_time_ns, child_specs):
        self.parents = {parent_pid: os.getppid()}
        self.create_times = {parent_pid: parent_create_time_ns / 1_000_000_000}
        for pid, create_time_ns, child_parent in child_specs:
            self.parents[pid] = child_parent
            self.create_times[pid] = create_time_ns / 1_000_000_000

    def Process(self, pid):
        return _FakePsutilProcess(self, pid)

    @staticmethod
    def virtual_memory():
        return type("Memory", (), {"available": 100 * short_lived.GIB})()


class _FakeSpawnProcess:
    def __init__(self, pid, *, terminate_fails=False):
        self._assigned_pid = pid
        self.pid = None
        self.sentinel = object()
        self.exitcode = None
        self.alive = False
        self.terminate_fails = terminate_fails
        self.terminate_called = False
        self.kill_called = False

    def start(self):
        self.pid = self._assigned_pid
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_called = True
        if self.terminate_fails:
            raise RuntimeError("terminate failed")
        self.alive = False
        self.exitcode = -15

    def kill(self):
        self.kill_called = True
        self.alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        return None


class _FakeContext:
    def __init__(self, processes):
        self.processes = iter(processes)

    def Process(self, target, args):
        return next(self.processes)


class ShortLivedTaskSchedulerTests(unittest.TestCase):
    def _policy(self, workers=2):
        return short_lived.MemoryPolicy(
            workers=workers,
            worker_rss_hard_bytes=2 * short_lived.GIB,
            parent_rss_budget_bytes=2 * short_lived.GIB,
            startup_available_bytes=1,
            dispatch_available_bytes=1,
            system_hard_stop_bytes=1,
            recovery_timeout_seconds=5,
            recovery_tolerance_bytes=64 * short_lived.GIB,
            baseline_drop_limit_bytes=64 * short_lived.GIB,
            sample_interval_seconds=0.05,
        )

    def _tasks(self, workspace, count, delays=None):
        logs = workspace / "logs"
        outputs = workspace / "outputs"
        delays = delays or [0.02] * count
        return [
            {
                "task_id": f"task_{index}",
                "stage": "probe",
                "delay": delays[index],
                "output_path": str(outputs / f"task_{index}.bin"),
                "completion_path": str(logs / f"task_{index}.json"),
                "failure_path": str(logs / f"task_{index}_failure.json"),
            }
            for index in range(count)
        ]

    def _scheduler(self, workspace, worker, events, workers=2):
        return short_lived.ShortLivedTaskScheduler(
            stage="probe",
            worker_fn=worker,
            workspace=workspace,
            quarantine_root=workspace.parent / "quarantine",
            logger=lambda stage, **payload: events.append({"stage": stage, **payload}),
            policy=self._policy(workers),
            result_validator=_validate_tiny_result,
        )

    def _fake_dispatch_scheduler(self, workspace, processes, child_specs):
        scheduler = self._scheduler(workspace, _tiny_success_worker, [], workers=2)
        scheduler.psutil = _FakePsutil(scheduler.parent_pid, scheduler.parent_create_time_ns, child_specs)
        scheduler.context = _FakeContext(processes)
        return scheduler

    def test_worker_nonce_is_unique_and_max_active_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            events = []
            summary = self._scheduler(workspace, _tiny_success_worker, events).run(self._tasks(workspace, 4))
            self.assertEqual(len(summary["worker_pids"]), 4)
            self.assertEqual(summary["worker_lifecycle_count"], 4)
            self.assertEqual(summary["max_active_observed"], 2)
            nonces = [item["worker_nonce"] for item in summary["worker_lifecycles"]]
            self.assertEqual(len(set(nonces)), 4)
            for index in range(4):
                completion = json.loads((workspace / "logs" / f"task_{index}.json").read_text(encoding="utf-8"))
                self.assertEqual(completion["task_id"], f"task_{index}")
                self.assertEqual(completion["stage"], "probe")
                identity = completion["worker_identity"]
                for field in (
                    "run_nonce", "worker_nonce", "task_id", "pid", "create_time_ns",
                    "parent_pid", "parent_create_time_ns", "stage", "lifecycle_sequence",
                ):
                    self.assertIsNotNone(identity[field])
                self.assertEqual(identity["state"], "completed")

    def test_pid_27576_sequential_reuse_is_allowed_after_validated_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            pid = 27576
            first = _FakeSpawnProcess(pid)
            second = _FakeSpawnProcess(pid)
            scheduler = self._fake_dispatch_scheduler(
                workspace,
                [first, second],
                [(pid, 1_000_000_000, os.getpid())],
            )
            launching, active, active_pid_index = {}, {}, {}
            first_task, second_task = self._tasks(workspace, 2)
            scheduler._dispatch(first_task, launching, active, active_pid_index)
            first_nonce = active_pid_index[pid]
            first.alive = False
            first.exitcode = 0
            active[first_nonce]["identity"]["state"] = "completed"
            active.pop(first_nonce)
            active_pid_index.pop(pid)
            scheduler.psutil.create_times[pid] = 2.0
            scheduler._dispatch(second_task, launching, active, active_pid_index)
            second_nonce = active_pid_index[pid]
            self.assertNotEqual(first_nonce, second_nonce)
            self.assertTrue(active[second_nonce]["identity"]["pid_reuse_allowed"])
            scheduler._cleanup_all_owned(launching, active, active_pid_index, reason="test_cleanup")

    def test_simultaneously_active_pid_collision_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            pid = 27576
            first = _FakeSpawnProcess(pid)
            second = _FakeSpawnProcess(pid)
            scheduler = self._fake_dispatch_scheduler(
                workspace,
                [first, second],
                [(pid, 1_000_000_000, os.getpid())],
            )
            launching, active, active_pid_index = {}, {}, {}
            first_task, second_task = self._tasks(workspace, 2)
            scheduler._dispatch(first_task, launching, active, active_pid_index)
            with self.assertRaisesRegex(RuntimeError, "simultaneous active worker PID collision"):
                scheduler._dispatch(second_task, launching, active, active_pid_index)
            self.assertEqual(len(launching), 1)
            scheduler._cleanup_all_owned(launching, active, active_pid_index, reason="test_collision_cleanup")

    def test_same_pid_and_create_time_with_different_active_nonce_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            pid = 27576
            scheduler = self._fake_dispatch_scheduler(
                workspace,
                [_FakeSpawnProcess(pid), _FakeSpawnProcess(pid)],
                [(pid, 1_000_000_000, os.getpid())],
            )
            launching, active, active_pid_index = {}, {}, {}
            tasks = self._tasks(workspace, 2)
            scheduler._dispatch(tasks[0], launching, active, active_pid_index)
            old_nonce = active_pid_index[pid]
            with self.assertRaisesRegex(RuntimeError, old_nonce):
                scheduler._dispatch(tasks[1], launching, active, active_pid_index)
            scheduler._cleanup_all_owned(launching, active, active_pid_index, reason="test_identity_collision")

    def test_worker_identity_read_failure_fails_closed_and_cleans_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            pid = 27576
            process = _FakeSpawnProcess(pid)
            scheduler = self._fake_dispatch_scheduler(
                workspace,
                [process],
                [(pid, 1_000_000_000, os.getpid())],
            )
            scheduler.psutil.create_times[pid] = psutil.AccessDenied(pid)
            launching, active, active_pid_index = {}, {}, {}
            with self.assertRaisesRegex(RuntimeError, "unable to read process identity"):
                scheduler._dispatch(self._tasks(workspace, 1)[0], launching, active, active_pid_index)
            self.assertEqual(len(launching), 1)
            cleanup = scheduler._cleanup_all_owned(
                launching, active, active_pid_index, reason="test_identity_read_cleanup"
            )
            self.assertTrue(cleanup["cleanup_completed"])
            self.assertTrue(process.terminate_called)

    def test_parent_identity_mismatch_after_start_is_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            pid = 27576
            process = _FakeSpawnProcess(pid)
            scheduler = self._fake_dispatch_scheduler(workspace, [process], [(pid, 1_000_000_000, 99999)])
            launching, active, active_pid_index = {}, {}, {}
            with self.assertRaisesRegex(RuntimeError, "worker parent identity mismatch"):
                scheduler._dispatch(self._tasks(workspace, 1)[0], launching, active, active_pid_index)
            scheduler._cleanup_all_owned(launching, active, active_pid_index, reason="test_parent_cleanup")
            self.assertFalse(process.is_alive())

    def test_completion_worker_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            pid = 27576
            process = _FakeSpawnProcess(pid)
            scheduler = self._fake_dispatch_scheduler(
                workspace,
                [process],
                [(pid, 1_000_000_000, os.getpid())],
            )
            launching, active, active_pid_index = {}, {}, {}
            task = self._tasks(workspace, 1)[0]
            scheduler._dispatch(task, launching, active, active_pid_index)
            nonce = active_pid_index[pid]
            record = active[nonce]
            process.alive = False
            process.exitcode = 0
            bad_identity = {**record["identity"], "worker_nonce": "wrong", "state": "completed"}
            Path(task["completion_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(task["completion_path"]).write_text(
                json.dumps({"pid": pid, "worker_identity": bad_identity}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "completion worker identity mismatch"):
                scheduler._validate_completion(record, {})
            scheduler._cleanup_all_owned(launching, active, active_pid_index, reason="test_completion_cleanup")

    def test_terminate_failure_escalates_to_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            process = _FakeSpawnProcess(27576, terminate_fails=True)
            process.start()
            scheduler = self._scheduler(workspace, _tiny_success_worker, [], workers=1)
            identity = {
                "run_nonce": scheduler.run_nonce,
                "worker_nonce": "nonce",
                "task_id": "task",
                "pid": process.pid,
                "create_time_ns": 1,
            }
            evidence = scheduler._terminate_process_record(
                {"process": process, "identity": identity}, reason="test_kill_escalation"
            )
            self.assertTrue(evidence["terminate_attempted"])
            self.assertTrue(evidence["kill_attempted"])
            self.assertTrue(process.kill_called)
            self.assertFalse(evidence["residual_alive"])

    def test_residual_process_prevents_worker_registry_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            scheduler = self._scheduler(workspace, _tiny_success_worker, [], workers=1)
            residual = {"pid": 27576, "run_nonce": scheduler.run_nonce}
            with mock.patch.object(scheduler, "_lineage_residuals", side_effect=[[], [residual]]):
                with self.assertRaisesRegex(RuntimeError, "unable to stop all owned child processes"):
                    scheduler._cleanup_all_owned({}, {}, {}, reason="test_residual")
            registry = json.loads(scheduler.worker_registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["cleanup_status"], "failed_residual")
            self.assertEqual(registry["residual_processes"], [residual])

    def test_outer_orchestrator_cleanup_terminates_unregistered_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence_path = Path(tmp) / "outer_cleanup.json"
            parent = psutil.Process(os.getpid())
            started_ns = int(time.time() * 1_000_000_000)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                payload = short_lived.cleanup_descendants_for_run(
                    run_nonce="probe-run",
                    parent_pid=os.getpid(),
                    parent_create_time_ns=int(parent.create_time() * 1_000_000_000),
                    started_create_time_ns=started_ns,
                    evidence_path=evidence_path,
                    reason="focused_test",
                    timeout_seconds=2.0,
                )
                self.assertTrue(payload["cleanup_completed"])
                self.assertTrue(any(item["pid"] == child.pid for item in payload["descendants_before"]))
                child.wait(timeout=2)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=2)

    def test_first_completed_replenishes_exactly_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            events = []
            self._scheduler(workspace, _tiny_success_worker, events).run(
                self._tasks(workspace, 4, delays=[0.1, 0.5, 0.1, 0.1])
            )
            stages = [event["stage"] for event in events]
            third_dispatch = [index for index, stage in enumerate(stages) if stage == "probe_task_dispatched"][2]
            first_done = stages.index("probe_task_done")
            self.assertGreater(third_dispatch, first_done)
            self.assertLessEqual(max(event.get("active_count", 0) for event in events), 2)

    def test_first_failure_stops_dispatch_and_clears_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            events = []
            tasks = self._tasks(workspace, 4)
            tasks[0]["fail"] = True
            scheduler = self._scheduler(workspace, _tiny_failure_or_sleep_worker, events)
            with self.assertRaises(short_lived.ShortLivedSchedulerError) as raised:
                scheduler.run(tasks)
            quarantine = Path(raised.exception.quarantine_path)
            manifest = json.loads((quarantine / "quarantine_manifest.json").read_text(encoding="utf-8"))
            self.assertLessEqual(len(manifest["worker_pids"]), 2)
            self.assertTrue(all(not psutil.pid_exists(pid) for pid in manifest["worker_pids"]))
            self.assertFalse(manifest["approved_for_candidate"])

    def test_missing_completion_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            scheduler = self._scheduler(workspace, _tiny_no_completion_worker, [], workers=1)
            with self.assertRaisesRegex(short_lived.ShortLivedSchedulerError, "completion JSON is missing"):
                scheduler.run(self._tasks(workspace, 1))

    def test_output_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            scheduler = self._scheduler(workspace, _tiny_hash_tamper_worker, [], workers=1)
            with self.assertRaisesRegex(short_lived.ShortLivedSchedulerError, "output (size|hash) mismatch"):
                scheduler.run(self._tasks(workspace, 1))

    def test_psutil_missing_fails_closed(self):
        with mock.patch.object(short_lived, "_psutil", None):
            with self.assertRaisesRegex(RuntimeError, "psutil is required"):
                short_lived.require_psutil()

    def test_memory_gate_matrix(self):
        policy = self._policy()
        self.assertIn("startup_available", short_lived.memory_gate_failures(
            policy, phase="startup", parent_rss_bytes=1, system_available_bytes=0
        ))
        self.assertIn("dispatch_available", short_lived.memory_gate_failures(
            policy, phase="dispatch", parent_rss_bytes=1, system_available_bytes=0
        ))
        self.assertIn("system_hard_stop", short_lived.memory_gate_failures(
            policy, phase="runtime", parent_rss_bytes=1, system_available_bytes=0
        ))
        self.assertIn("worker_rss_hard", short_lived.memory_gate_failures(
            policy, phase="runtime", parent_rss_bytes=1, system_available_bytes=2, worker_rss_bytes=3 * short_lived.GIB
        ))
        self.assertIn("parent_rss_budget", short_lived.memory_gate_failures(
            policy, phase="runtime", parent_rss_bytes=3 * short_lived.GIB, system_available_bytes=2
        ))

    def test_existing_task_artifact_prohibits_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            tasks = self._tasks(workspace, 1)
            Path(tasks[0]["output_path"]).parent.mkdir(parents=True)
            Path(tasks[0]["output_path"]).write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "prohibits resume"):
                self._scheduler(workspace, _tiny_success_worker, []).run(tasks)

    def test_raw_and_gtja_use_shared_short_lived_entry(self):
        raw_source = inspect.getsource(full_l3._build_raw_stage)
        gtja_source = inspect.getsource(full_l3._build_gtja_stage)
        self.assertIn("_run_short_lived_tasks", raw_source)
        self.assertIn("_run_short_lived_tasks", gtja_source)

    def test_gtja_long_pool_and_as_completed_defects_are_closed(self):
        source = inspect.getsource(full_l3._build_gtja_stage)
        self.assertNotIn("ProcessPoolExecutor", source)
        self.assertNotIn("as_completed", source)
        self.assertIn("completion_path", source)


if __name__ == "__main__":
    unittest.main()
