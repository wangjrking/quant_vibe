from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import os
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from typing import Any, Callable


GIB = 1024**3

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover - exercised through a patched dependency test
    _psutil = None


def require_psutil():
    if _psutil is None:
        raise RuntimeError("psutil is required for L3 short-lived process memory governance")
    return _psutil


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def cleanup_descendants_for_run(
    *,
    run_nonce: str,
    parent_pid: int,
    parent_create_time_ns: int,
    started_create_time_ns: int,
    evidence_path: Path,
    reason: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Stop every descendant created by this orchestrator before closeout."""
    psutil = require_psutil()
    try:
        parent = psutil.Process(parent_pid)
        if _process_create_time_ns(psutil, parent_pid) != parent_create_time_ns:
            raise RuntimeError("orchestrator parent identity changed before descendant cleanup")
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
        raise RuntimeError(f"unable to enumerate orchestrator descendants: {type(error).__name__}") from error

    descendants = []
    processes = []
    for child in children:
        try:
            create_time_ns = _process_create_time_ns(psutil, child.pid)
            if create_time_ns + 1_000_000_000 < started_create_time_ns:
                continue
            descendants.append({
                "run_nonce": run_nonce,
                "pid": int(child.pid),
                "ppid": int(child.ppid()),
                "create_time_ns": create_time_ns,
                "name": child.name(),
                "cmdline": child.cmdline(),
            })
            processes.append(child)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError(f"unable to inspect orchestrator descendant: pid={child.pid}") from error

    actions = []
    for process, identity in zip(processes, descendants):
        action = {**identity, "terminate_attempted": True, "kill_attempted": False}
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            action["already_exited"] = True
        except psutil.AccessDenied as error:
            action["terminate_error"] = type(error).__name__
        actions.append(action)
    _, alive = psutil.wait_procs(processes, timeout=timeout_seconds)
    alive_by_pid = {process.pid: process for process in alive}
    for action in actions:
        process = alive_by_pid.get(action["pid"])
        if process is None:
            action["joined_after_terminate"] = True
            continue
        action["joined_after_terminate"] = False
        action["kill_attempted"] = True
        try:
            process.kill()
        except psutil.NoSuchProcess:
            action["already_exited_before_kill"] = True
        except psutil.AccessDenied as error:
            action["kill_error"] = type(error).__name__
    if alive:
        _, alive = psutil.wait_procs(alive, timeout=timeout_seconds)

    residuals = []
    try:
        current_children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
        raise RuntimeError(f"unable to rescan orchestrator descendants: {type(error).__name__}") from error
    for child in current_children:
        try:
            create_time_ns = _process_create_time_ns(psutil, child.pid)
            if create_time_ns + 1_000_000_000 < started_create_time_ns:
                continue
            residuals.append({
                "run_nonce": run_nonce,
                "pid": int(child.pid),
                "ppid": int(child.ppid()),
                "create_time_ns": create_time_ns,
                "name": child.name(),
                "cmdline": child.cmdline(),
            })
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError(f"unable to validate orchestrator descendant exit: pid={child.pid}") from error

    payload = {
        "run_nonce": run_nonce,
        "parent_pid": parent_pid,
        "parent_create_time_ns": parent_create_time_ns,
        "started_create_time_ns": started_create_time_ns,
        "reason": reason,
        "descendants_before": descendants,
        "cleanup_actions": actions,
        "residual_processes_after": residuals,
        "cleanup_completed": not residuals and not alive,
        "generated_at": now_iso(),
    }
    atomic_json(evidence_path, payload)
    if not payload["cleanup_completed"]:
        raise RuntimeError(f"orchestrator descendant cleanup left residual processes: {residuals}")
    return payload


def memory_gate_failures(
    policy: "MemoryPolicy",
    *,
    phase: str,
    parent_rss_bytes: int,
    system_available_bytes: int,
    worker_rss_bytes: int = 0,
) -> list[str]:
    failures = []
    if parent_rss_bytes > policy.parent_rss_budget_bytes:
        failures.append("parent_rss_budget")
    if phase == "startup" and system_available_bytes < policy.startup_available_bytes:
        failures.append("startup_available")
    if phase == "dispatch" and system_available_bytes < policy.dispatch_available_bytes:
        failures.append("dispatch_available")
    if phase == "runtime" and system_available_bytes < policy.system_hard_stop_bytes:
        failures.append("system_hard_stop")
    if phase == "runtime" and worker_rss_bytes > policy.worker_rss_hard_bytes:
        failures.append("worker_rss_hard")
    return failures


def validate_output_evidence(payload: dict[str, Any], output_path: Path) -> tuple[int, str]:
    if not output_path.is_file():
        raise RuntimeError(f"task output is missing: {output_path}")
    actual_size = int(output_path.stat().st_size)
    actual_hash = file_sha256(output_path)
    if actual_size != int(payload.get("output_size_bytes", -1)):
        raise RuntimeError(f"task output size mismatch: expected={payload.get('output_size_bytes')} actual={actual_size}")
    if actual_hash != payload.get("output_sha256"):
        raise RuntimeError(f"task output hash mismatch: expected={payload.get('output_sha256')} actual={actual_hash}")
    return actual_size, actual_hash


@dataclass(frozen=True)
class MemoryPolicy:
    workers: int = 2
    worker_rss_hard_bytes: int = 24 * GIB
    parent_rss_budget_bytes: int = 4 * GIB
    startup_available_bytes: int = 64 * GIB
    dispatch_available_bytes: int = 40 * GIB
    system_hard_stop_bytes: int = 20 * GIB
    recovery_timeout_seconds: int = 120
    recovery_tolerance_bytes: int = 2 * GIB
    baseline_drop_limit_bytes: int = 8 * GIB
    baseline_completion_count: int = 3
    sample_interval_seconds: float = 2.0
    duckdb_memory_limit: str = "12GB"

    def __post_init__(self) -> None:
        if self.workers < 1 or self.workers > 2:
            raise ValueError("short-lived scheduler workers must be 1 or 2")
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")


class ShortLivedSchedulerError(RuntimeError):
    def __init__(self, message: str, *, quarantine_path: str | None = None):
        super().__init__(message)
        self.quarantine_path = quarantine_path


def _completion_path(task: dict[str, Any]) -> Path:
    return Path(task["completion_path"])


def _failure_path(task: dict[str, Any]) -> Path:
    value = task.get("failure_path")
    if value:
        return Path(value)
    completion = _completion_path(task)
    return completion.with_name(f"{completion.stem}_failure.json")


def _output_path(task: dict[str, Any], result: dict[str, Any] | None = None) -> Path:
    value = (result or {}).get("path") or task.get("output_path")
    if not value:
        raise RuntimeError("task output path is missing")
    return Path(value)


def _thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _process_create_time_ns(psutil_module, pid: int) -> int:
    try:
        return int(float(psutil_module.Process(pid).create_time()) * 1_000_000_000)
    except (psutil_module.NoSuchProcess, psutil_module.AccessDenied) as error:
        raise RuntimeError(f"unable to read process identity: pid={pid} error={type(error).__name__}") from error


def _child_entry(
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    task: dict[str, Any],
    policy: MemoryPolicy,
    owned_identity: dict[str, Any],
) -> None:
    psutil = require_psutil()
    _thread_env()
    pid = os.getpid()
    process = psutil.Process(pid)
    parent_pid = os.getppid()
    worker_identity = {
        **owned_identity,
        "pid": pid,
        "create_time_ns": _process_create_time_ns(psutil, pid),
        "parent_pid": parent_pid,
        "parent_create_time_ns": _process_create_time_ns(psutil, parent_pid),
        "state": "child_running",
    }
    stats_lock = threading.Lock()
    stop_event = threading.Event()
    rss_start = int(process.memory_info().rss)
    available_start = int(psutil.virtual_memory().available)
    stats = {"rss_peak": rss_start, "system_available_min": available_start}

    def monitor() -> None:
        while not stop_event.wait(policy.sample_interval_seconds):
            try:
                rss = int(process.memory_info().rss)
                available = int(psutil.virtual_memory().available)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            with stats_lock:
                stats["rss_peak"] = max(stats["rss_peak"], rss)
                stats["system_available_min"] = min(stats["system_available_min"], available)

    monitor_thread = threading.Thread(target=monitor, name=f"l3-memory-monitor-{pid}", daemon=True)
    monitor_thread.start()
    result: dict[str, Any] | None = None
    try:
        result = worker_fn(task)
        gc.collect()
        rss_end = int(process.memory_info().rss)
        available_after_work = int(psutil.virtual_memory().available)
        with stats_lock:
            rss_peak = max(stats["rss_peak"], rss_end)
            available_min = min(stats["system_available_min"], available_after_work)
        if rss_peak > policy.worker_rss_hard_bytes:
            raise RuntimeError(
                f"worker RSS hard limit exceeded: peak={rss_peak} limit={policy.worker_rss_hard_bytes}; "
                "increase raw-buckets to 512 before retry"
            )
        if available_min < policy.system_hard_stop_bytes:
            raise RuntimeError(
                f"system available hard limit crossed in worker: minimum={available_min} limit={policy.system_hard_stop_bytes}"
            )
        output_path = _output_path(task, result)
        if not output_path.is_file():
            raise RuntimeError(f"task output is missing: {output_path}")
        completion_identity = {**worker_identity, "state": "completed"}
        payload = {
            **result,
            "status": "completed",
            "task_id": task.get("task_id"),
            "stage": task.get("stage"),
            "pid": pid,
            "worker_identity": completion_identity,
            "exit_code": 0,
            "rss_start_bytes": rss_start,
            "rss_peak_bytes": rss_peak,
            "rss_end_bytes": rss_end,
            "system_available_before_bytes": available_start,
            "system_available_min_bytes": available_min,
            "system_available_after_work_bytes": available_after_work,
            "output_size_bytes": int(output_path.stat().st_size),
            "output_sha256": file_sha256(output_path),
            "memory_gate_status": "child_passed_parent_pending",
            "generated_at": now_iso(),
        }
        atomic_json(_completion_path(task), payload)
    except BaseException as error:
        try:
            gc.collect()
            rss_end = int(process.memory_info().rss)
            available_after_work = int(psutil.virtual_memory().available)
            with stats_lock:
                rss_peak = max(stats["rss_peak"], rss_end)
                available_min = min(stats["system_available_min"], available_after_work)
            failure_identity = {**worker_identity, "state": "failed"}
            failure = {
                "status": "failed",
                "task_id": task.get("task_id"),
                "stage": task.get("stage"),
                "pid": pid,
                "worker_identity": failure_identity,
                "exit_code": 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "rss_start_bytes": rss_start,
                "rss_peak_bytes": rss_peak,
                "rss_end_bytes": rss_end,
                "system_available_before_bytes": available_start,
                "system_available_min_bytes": available_min,
                "system_available_after_work_bytes": available_after_work,
                "memory_gate_status": "failed",
                "approved_for_candidate": False,
                "approved_for_active": False,
                "approved_for_downstream": False,
                "generated_at": now_iso(),
            }
            atomic_json(_failure_path(task), failure)
        finally:
            stop_event.set()
            monitor_thread.join(timeout=max(1.0, policy.sample_interval_seconds * 2))
        raise SystemExit(1)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=max(1.0, policy.sample_interval_seconds * 2))


class ShortLivedTaskScheduler:
    def __init__(
        self,
        *,
        stage: str,
        worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
        workspace: Path,
        quarantine_root: Path,
        logger: Callable[..., None],
        workflow_run_id: str | None = None,
        run_nonce: str | None = None,
        policy: MemoryPolicy | None = None,
        result_validator: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        self.stage = stage
        self.worker_fn = worker_fn
        self.workspace = Path(workspace)
        self.quarantine_root = Path(quarantine_root)
        self.logger = logger
        self.workflow_run_id = workflow_run_id
        self.run_nonce = run_nonce or uuid.uuid4().hex
        self.policy = policy or MemoryPolicy()
        self.result_validator = result_validator
        self.psutil = require_psutil()
        self.context = multiprocessing.get_context("spawn")
        self.timeline: list[dict[str, Any]] = []
        self.max_active_observed = 0
        self.worker_pids: list[int] = []
        self.worker_lifecycles: list[dict[str, Any]] = []
        self.cleanup_events: list[dict[str, Any]] = []
        self.lifecycle_sequence = 0
        self.parent_pid = os.getpid()
        self.parent_create_time_ns = _process_create_time_ns(self.psutil, self.parent_pid)
        self.scheduler_started_create_time_ns = int(time.time() * 1_000_000_000)
        self.worker_registry_path = self.workspace / "logs" / "short_lived_worker_registry.json"

    def _write_worker_registry(
        self,
        launching: dict[str, dict[str, Any]],
        active: dict[str, dict[str, Any]],
        *,
        cleanup_status: str = "not_started",
        residual_processes: list[dict[str, Any]] | None = None,
    ) -> None:
        atomic_json(self.worker_registry_path, {
            "workflow_run_id": self.workflow_run_id,
            "run_nonce": self.run_nonce,
            "workspace": str(self.workspace.resolve()),
            "stage": self.stage,
            "parent_pid": self.parent_pid,
            "parent_create_time_ns": self.parent_create_time_ns,
            "launching_workers": [
                {
                    **record["identity"],
                    "workflow_run_id": self.workflow_run_id,
                    "workspace": str(self.workspace.resolve()),
                    "started_at": record["started_at"],
                }
                for record in launching.values()
            ],
            "active_workers": [
                {
                    **record["identity"],
                    "workflow_run_id": self.workflow_run_id,
                    "workspace": str(self.workspace.resolve()),
                    "started_at": record["started_at"],
                }
                for record in active.values()
            ],
            "cleanup_status": cleanup_status,
            "residual_processes": residual_processes or [],
            "updated_at": now_iso(),
        })

    def _memory_sample(self, active: dict[int, dict[str, Any]]) -> dict[str, Any]:
        parent_rss = int(self.psutil.Process(os.getpid()).memory_info().rss)
        available = int(self.psutil.virtual_memory().available)
        workers = []
        for record in active.values():
            process = record["process"]
            try:
                rss = int(self.psutil.Process(process.pid).memory_info().rss)
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                rss = 0
            record["rss_peak_parent"] = max(record["rss_peak_parent"], rss)
            record["system_available_min_parent"] = min(record["system_available_min_parent"], available)
            workers.append({
                "pid": process.pid,
                "rss_bytes": rss,
                "task_id": record["task"].get("task_id"),
                "worker_nonce": record["identity"]["worker_nonce"],
                "lifecycle_sequence": record["identity"]["lifecycle_sequence"],
            })
        sample = {
            "time": now_iso(),
            "parent_rss_bytes": parent_rss,
            "system_available_bytes": available,
            "workers": workers,
        }
        self.timeline.append(sample)
        return sample

    def _check_runtime_gates(self, active: dict[int, dict[str, Any]]) -> None:
        sample = self._memory_sample(active)
        parent_failures = memory_gate_failures(
            self.policy,
            phase="runtime",
            parent_rss_bytes=sample["parent_rss_bytes"],
            system_available_bytes=sample["system_available_bytes"],
        )
        if parent_failures:
            raise RuntimeError(f"runtime memory gate failed: {parent_failures}")
        for worker in sample["workers"]:
            worker_failures = memory_gate_failures(
                self.policy,
                phase="runtime",
                parent_rss_bytes=sample["parent_rss_bytes"],
                system_available_bytes=sample["system_available_bytes"],
                worker_rss_bytes=worker["rss_bytes"],
            )
            if "worker_rss_hard" in worker_failures:
                raise RuntimeError(
                    f"worker RSS hard limit exceeded: pid={worker['pid']} rss={worker['rss_bytes']} "
                    f"limit={self.policy.worker_rss_hard_bytes}; increase raw-buckets to 512 before retry"
                )

    def _assert_dispatch_gate(self) -> int:
        parent_rss = int(self.psutil.Process(os.getpid()).memory_info().rss)
        available = int(self.psutil.virtual_memory().available)
        failures = memory_gate_failures(
            self.policy,
            phase="dispatch",
            parent_rss_bytes=parent_rss,
            system_available_bytes=available,
        )
        if failures:
            raise RuntimeError(f"dispatch memory gate failed: {failures}")
        return available

    def _dispatch(
        self,
        task: dict[str, Any],
        launching: dict[str, dict[str, Any]],
        active: dict[str, dict[str, Any]],
        active_pid_index: dict[int, str],
    ) -> None:
        available_before = self._assert_dispatch_gate()
        self.lifecycle_sequence += 1
        worker_nonce = uuid.uuid4().hex
        identity = {
            "run_nonce": self.run_nonce,
            "worker_nonce": worker_nonce,
            "task_id": task.get("task_id"),
            "pid": None,
            "create_time_ns": None,
            "parent_pid": self.parent_pid,
            "parent_create_time_ns": self.parent_create_time_ns,
            "stage": self.stage,
            "lifecycle_sequence": self.lifecycle_sequence,
            "state": "owned_pre_start",
            "pid_reuse_allowed": False,
        }
        process = self.context.Process(target=_child_entry, args=(self.worker_fn, task, self.policy, identity))
        record = {
            "process": process,
            "task": task,
            "identity": identity,
            "available_before": available_before,
            "rss_peak_parent": 0,
            "system_available_min_parent": available_before,
            "started_at": now_iso(),
        }
        self.worker_lifecycles.append(identity)
        launching[worker_nonce] = record
        self._write_worker_registry(launching, active)
        process.start()
        identity["state"] = "launching"
        identity["pid"] = int(process.pid)
        self._write_worker_registry(launching, active)
        child_process = self.psutil.Process(process.pid)
        identity["create_time_ns"] = _process_create_time_ns(self.psutil, process.pid)
        self._write_worker_registry(launching, active)
        actual_parent_pid = int(child_process.ppid())
        if actual_parent_pid != self.parent_pid:
            raise RuntimeError(
                f"worker parent identity mismatch: pid={process.pid} expected_parent={self.parent_pid} actual_parent={actual_parent_pid}"
            )
        if _process_create_time_ns(self.psutil, actual_parent_pid) != self.parent_create_time_ns:
            raise RuntimeError(f"worker parent create_time mismatch: pid={process.pid} parent={actual_parent_pid}")
        if process.pid in active_pid_index:
            raise RuntimeError(
                f"simultaneous active worker PID collision: pid={process.pid} "
                f"existing_worker_nonce={active_pid_index[process.pid]} new_worker_nonce={worker_nonce}"
            )
        identity["pid_reuse_allowed"] = any(
            lifecycle.get("pid") == process.pid and lifecycle.get("state") == "completed"
            for lifecycle in self.worker_lifecycles
        )
        self.worker_pids.append(process.pid)
        identity["state"] = "active"
        active[worker_nonce] = launching.pop(worker_nonce)
        active_pid_index[process.pid] = worker_nonce
        self._write_worker_registry(launching, active)
        self.max_active_observed = max(self.max_active_observed, len(active))
        self.logger(
            f"{self.stage}_task_dispatched",
            task_id=task.get("task_id"),
            pid=process.pid,
            run_nonce=self.run_nonce,
            worker_nonce=worker_nonce,
            lifecycle_sequence=self.lifecycle_sequence,
            pid_reuse_allowed=identity["pid_reuse_allowed"],
            active_count=len(active),
            system_available_before_bytes=available_before,
        )

    def _wait_for_recovery(self, record: dict[str, Any], active: dict[str, dict[str, Any]]) -> int:
        target = min(
            self.policy.dispatch_available_bytes,
            max(0, int(record["available_before"]) - self.policy.recovery_tolerance_bytes),
        )
        deadline = time.monotonic() + self.policy.recovery_timeout_seconds
        while True:
            available = int(self.psutil.virtual_memory().available)
            self.timeline.append({
                "time": now_iso(),
                "event": "worker_exit_recovery",
                "pid": record["process"].pid,
                "system_available_bytes": available,
                "target_bytes": target,
            })
            if available >= target:
                return available
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"memory did not recover after worker exit: pid={record['process'].pid} available={available} target={target}"
                )
            time.sleep(min(2.0, self.policy.sample_interval_seconds))

    def _validate_completion(self, record: dict[str, Any], active: dict[str, dict[str, Any]]) -> dict[str, Any]:
        process = record["process"]
        task = record["task"]
        completion_path = _completion_path(task)
        if process.exitcode != 0:
            raise RuntimeError(f"task process exited nonzero: task={task.get('task_id')} pid={process.pid} exit={process.exitcode}")
        if not completion_path.is_file():
            raise RuntimeError(f"task completion JSON is missing: {completion_path}")
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if int(payload.get("pid", -1)) != process.pid:
            raise RuntimeError(f"completion PID mismatch: expected={process.pid} actual={payload.get('pid')}")
        actual_identity = payload.get("worker_identity") or {}
        expected_identity = record["identity"]
        identity_fields = (
            "run_nonce",
            "worker_nonce",
            "task_id",
            "pid",
            "create_time_ns",
            "parent_pid",
            "parent_create_time_ns",
            "stage",
            "lifecycle_sequence",
        )
        identity_mismatches = {
            field: {"expected": expected_identity.get(field), "actual": actual_identity.get(field)}
            for field in identity_fields
            if actual_identity.get(field) != expected_identity.get(field)
        }
        if identity_mismatches:
            raise RuntimeError(f"completion worker identity mismatch: {identity_mismatches}")
        if actual_identity.get("state") != "completed":
            raise RuntimeError(f"completion worker state mismatch: {actual_identity.get('state')}")
        output_path = _output_path(task, payload)
        actual_size, actual_hash = validate_output_evidence(payload, output_path)
        if self.result_validator is not None:
            self.result_validator(task, payload)
        after_exit = self._wait_for_recovery(record, active)
        payload.update({
            "exit_code": int(process.exitcode),
            "rss_peak_bytes": max(int(payload.get("rss_peak_bytes", 0)), int(record["rss_peak_parent"])),
            "system_available_min_bytes": min(
                int(payload.get("system_available_min_bytes", record["system_available_min_parent"])),
                int(record["system_available_min_parent"]),
            ),
            "system_available_after_exit_bytes": after_exit,
            "output_size_bytes": actual_size,
            "output_sha256": actual_hash,
            "memory_gate_status": "passed",
            "worker_identity": {**expected_identity, "state": "completed"},
            "validated_at": now_iso(),
        })
        atomic_json(completion_path, payload)
        return payload

    def _terminate_process_record(self, record: dict[str, Any], *, reason: str) -> dict[str, Any]:
        process = record["process"]
        identity = record["identity"]
        evidence = {
            "run_nonce": self.run_nonce,
            "worker_nonce": identity["worker_nonce"],
            "task_id": identity["task_id"],
            "pid": process.pid,
            "create_time_ns": identity.get("create_time_ns"),
            "reason": reason,
            "terminate_attempted": False,
            "terminate_error": None,
            "join_after_terminate": False,
            "kill_attempted": False,
            "kill_error": None,
            "join_after_kill": False,
            "residual_alive": False,
        }
        try:
            if process.pid is not None and process.is_alive():
                evidence["terminate_attempted"] = True
                try:
                    process.terminate()
                except BaseException as error:
                    evidence["terminate_error"] = f"{type(error).__name__}: {error}"
                process.join(timeout=10.0)
                evidence["join_after_terminate"] = not process.is_alive()
                if process.is_alive():
                    evidence["kill_attempted"] = True
                    try:
                        process.kill()
                    except BaseException as error:
                        evidence["kill_error"] = f"{type(error).__name__}: {error}"
                    process.join(timeout=10.0)
                    evidence["join_after_kill"] = not process.is_alive()
            elif process.pid is not None:
                process.join(timeout=0)
                evidence["join_after_terminate"] = True
        finally:
            evidence["residual_alive"] = bool(process.pid is not None and process.is_alive())
            evidence["completed_at"] = now_iso()
        self.cleanup_events.append(evidence)
        return evidence

    def _lineage_residuals(self) -> list[dict[str, Any]]:
        try:
            parent = self.psutil.Process(self.parent_pid)
            if _process_create_time_ns(self.psutil, self.parent_pid) != self.parent_create_time_ns:
                raise RuntimeError("scheduler parent identity changed during residual scan")
            children = parent.children(recursive=True)
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied) as error:
            raise RuntimeError(f"unable to scan scheduler descendants: {type(error).__name__}") from error
        residuals = []
        for child in children:
            try:
                create_time_ns = _process_create_time_ns(self.psutil, child.pid)
                if create_time_ns + 1_000_000_000 < self.scheduler_started_create_time_ns:
                    continue
                residuals.append({
                    "run_nonce": self.run_nonce,
                    "pid": int(child.pid),
                    "ppid": int(child.ppid()),
                    "create_time_ns": create_time_ns,
                    "name": child.name(),
                    "cmdline": child.cmdline(),
                })
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        return residuals

    def _terminate_lineage_residuals(self, residuals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        actions = []
        for residual in residuals:
            action = {**residual, "terminate_attempted": False, "kill_attempted": False, "residual_alive": False}
            try:
                process = self.psutil.Process(residual["pid"])
                if _process_create_time_ns(self.psutil, residual["pid"]) != residual["create_time_ns"]:
                    action["identity_changed"] = True
                    actions.append(action)
                    continue
                action["terminate_attempted"] = True
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except self.psutil.TimeoutExpired:
                    action["kill_attempted"] = True
                    process.kill()
                    try:
                        process.wait(timeout=10.0)
                    except self.psutil.TimeoutExpired:
                        action["residual_alive"] = True
            except self.psutil.NoSuchProcess:
                pass
            except self.psutil.AccessDenied as error:
                action["error"] = type(error).__name__
                action["residual_alive"] = True
            actions.append(action)
        self.cleanup_events.extend(actions)
        return actions

    def _cleanup_all_owned(
        self,
        launching: dict[str, dict[str, Any]],
        active: dict[str, dict[str, Any]],
        active_pid_index: dict[int, str],
        *,
        reason: str,
    ) -> dict[str, Any]:
        records = {**launching, **active}
        direct_actions = [self._terminate_process_record(record, reason=reason) for record in records.values()]
        lineage_before = self._lineage_residuals()
        lineage_actions = self._terminate_lineage_residuals(lineage_before)
        lineage_after = self._lineage_residuals()
        direct_residuals = [item for item in direct_actions if item["residual_alive"]]
        cleanup = {
            "run_nonce": self.run_nonce,
            "reason": reason,
            "direct_actions": direct_actions,
            "lineage_residuals_before": lineage_before,
            "lineage_actions": lineage_actions,
            "lineage_residuals_after": lineage_after,
            "direct_residuals_after": direct_residuals,
            "cleanup_completed": not direct_residuals and not lineage_after,
            "generated_at": now_iso(),
        }
        if not cleanup["cleanup_completed"]:
            self._write_worker_registry(
                launching,
                active,
                cleanup_status="failed_residual",
                residual_processes=[*direct_residuals, *lineage_after],
            )
            raise RuntimeError(
                f"unable to stop all owned child processes: direct={direct_residuals} lineage={lineage_after}"
            )
        launching.clear()
        active.clear()
        active_pid_index.clear()
        self._write_worker_registry({}, {}, cleanup_status="completed", residual_processes=[])
        return cleanup

    def _quarantine(
        self,
        error: BaseException,
        launching: dict[str, dict[str, Any]],
        active: dict[str, dict[str, Any]],
        active_pid_index: dict[int, str],
    ) -> Path:
        cleanup = self._cleanup_all_owned(
            launching,
            active,
            active_pid_index,
            reason=f"scheduler_failure:{type(error).__name__}",
        )
        failure_path = self.workspace / "logs" / f"{self.stage}_scheduler_failure.json"
        failure = {
            "status": "failed_closed",
            "stage": self.stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "run_nonce": self.run_nonce,
            "parent_pid": self.parent_pid,
            "parent_create_time_ns": self.parent_create_time_ns,
            "cleanup": cleanup,
            "worker_pids": self.worker_pids,
            "worker_lifecycles": self.worker_lifecycles,
            "max_active_observed": self.max_active_observed,
            "memory_timeline": self.timeline,
            "approved_for_candidate": False,
            "approved_for_active": False,
            "approved_for_downstream": False,
            "reuse_prohibited": True,
            "generated_at": now_iso(),
        }
        atomic_json(failure_path, failure)
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.quarantine_root / f"{self.workspace.name}_{self.stage}_failed_{suffix}"
        if target.exists():
            raise RuntimeError(f"quarantine target already exists: {target}")
        os.replace(self.workspace, target)
        manifest = {
            **failure,
            "status": "quarantine_evidence_only",
            "quarantine_path": str(target),
            "in_place_resume_prohibited": True,
        }
        atomic_json(target / "quarantine_manifest.json", manifest)
        return target

    def run(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        if not tasks:
            raise ValueError("short-lived scheduler requires at least one task")
        if self.policy.workers > 2:
            raise RuntimeError("short-lived scheduler refuses workers greater than 2")
        for task in tasks:
            for path in (_completion_path(task), _failure_path(task), Path(task["output_path"])):
                if path.exists():
                    raise RuntimeError(f"existing task artifact prohibits resume: {path}")
        startup_available = int(self.psutil.virtual_memory().available)
        startup_parent_rss = int(self.psutil.Process(os.getpid()).memory_info().rss)
        startup_failures = memory_gate_failures(
            self.policy,
            phase="startup",
            parent_rss_bytes=startup_parent_rss,
            system_available_bytes=startup_available,
        )
        if startup_failures:
            raise RuntimeError(f"startup memory gate failed: {startup_failures}")
        started_at = now_iso()
        pending = iter(tasks)
        launching: dict[str, dict[str, Any]] = {}
        active: dict[str, dict[str, Any]] = {}
        active_pid_index: dict[int, str] = {}
        results: list[dict[str, Any]] = []
        completed_effective_available: list[int] = []
        cleanup: dict[str, Any] | None = None
        self._write_worker_registry(launching, active)
        try:
            for _ in range(min(self.policy.workers, len(tasks))):
                self._dispatch(next(pending), launching, active, active_pid_index)
            while active or launching:
                if launching:
                    raise RuntimeError(f"worker remained in launching state after dispatch: {list(launching)}")
                self._check_runtime_gates(active)
                ready = wait_connections(
                    [record["process"].sentinel for record in active.values()],
                    timeout=self.policy.sample_interval_seconds,
                )
                if not ready:
                    continue
                for sentinel in ready:
                    worker_nonce, record = next(
                        (nonce, item)
                        for nonce, item in active.items()
                        if item["process"].sentinel == sentinel
                    )
                    process = record["process"]
                    process.join(timeout=0)
                    record["identity"]["state"] = "exited_pending_validation"
                    self._write_worker_registry(launching, active)
                    other_active = {nonce: item for nonce, item in active.items() if nonce != worker_nonce}
                    payload = self._validate_completion(record, other_active)
                    record["identity"]["state"] = "completed"
                    active.pop(worker_nonce)
                    active_pid_index.pop(int(process.pid), None)
                    self._write_worker_registry(launching, active)
                    results.append(payload)
                    effective_available = int(payload["system_available_after_exit_bytes"]) + sum(
                        int(self.psutil.Process(item["process"].pid).memory_info().rss)
                        for item in active.values()
                        if item["process"].is_alive()
                    )
                    completed_effective_available.append(effective_available)
                    if len(completed_effective_available) >= self.policy.baseline_completion_count:
                        drop = startup_available - completed_effective_available[-1]
                        if drop > self.policy.baseline_drop_limit_bytes:
                            raise RuntimeError(
                                f"available memory baseline dropped after completions: drop={drop} "
                                f"limit={self.policy.baseline_drop_limit_bytes}"
                            )
                    self.logger(
                        f"{self.stage}_task_done",
                        task_id=record["task"].get("task_id"),
                        pid=process.pid,
                        run_nonce=self.run_nonce,
                        worker_nonce=worker_nonce,
                        lifecycle_sequence=record["identity"]["lifecycle_sequence"],
                        pid_reuse_allowed=record["identity"]["pid_reuse_allowed"],
                        active_count=len(active),
                        rss_peak_bytes=payload["rss_peak_bytes"],
                        system_available_after_exit_bytes=payload["system_available_after_exit_bytes"],
                    )
                    next_task = next(pending, None)
                    if next_task is not None:
                        self._dispatch(next_task, launching, active, active_pid_index)
            cleanup = self._cleanup_all_owned(
                launching,
                active,
                active_pid_index,
                reason="scheduler_success_residual_scan",
            )
        except BaseException as error:
            quarantine_path = self._quarantine(error, launching, active, active_pid_index)
            raise ShortLivedSchedulerError(str(error), quarantine_path=str(quarantine_path)) from error
        return {
            "stage": self.stage,
            "run_nonce": self.run_nonce,
            "parent_pid": self.parent_pid,
            "parent_create_time_ns": self.parent_create_time_ns,
            "started_at": started_at,
            "completed_at": now_iso(),
            "task_count": len(tasks),
            "results": results,
            "worker_pids": self.worker_pids,
            "unique_worker_pid_count": len(set(self.worker_pids)),
            "worker_lifecycle_count": len(self.worker_lifecycles),
            "worker_lifecycles": self.worker_lifecycles,
            "max_active_observed": self.max_active_observed,
            "startup_available_bytes": startup_available,
            "memory_timeline": self.timeline,
            "cleanup": cleanup,
            "policy": asdict(self.policy),
        }
