"""Candidate-only contract for classifying a trusted runtime shell ancestor.

This module is intentionally not imported by the production process gate. It
provides a narrow, fixture-testable contract for a later audited integration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


SHELL_NAMES = {"powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe"}
PYTHON_NAMES = {"python", "python.exe", "pythonw", "pythonw.exe"}


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return os.path.normcase(os.path.abspath(value.strip("\"'")))


def _command(record: dict[str, Any]) -> str:
    value = record.get("cmdline") or []
    return value if isinstance(value, str) else " ".join(str(part) for part in value)


def _flag(command: str, name: str) -> str | None:
    tokens = command.replace("=", " ").split()
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1].strip("\"'")
    return None


def _is_shell(record: dict[str, Any]) -> bool:
    return str(record.get("name") or "").lower() in SHELL_NAMES or Path(
        str(record.get("exe") or "")
    ).name.lower() in SHELL_NAMES


def _is_python(record: dict[str, Any]) -> bool:
    return str(record.get("name") or "").lower() in PYTHON_NAMES or Path(
        str(record.get("exe") or "")
    ).name.lower() in PYTHON_NAMES


def _lineage(records: list[dict[str, Any]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    by_pid = {int(item["pid"]): item for item in records}
    children: dict[int, list[int]] = {}
    for item in records:
        children.setdefault(int(item.get("ppid") or 0), []).append(int(item["pid"]))
    ancestors: dict[int, list[int]] = {}
    descendants: dict[int, list[int]] = {}
    for pid in by_pid:
        current = pid
        seen = {pid}
        chain: list[int] = []
        while int(by_pid.get(current, {}).get("ppid") or 0) and int(
            by_pid.get(current, {}).get("ppid") or 0
        ) not in seen:
            parent = int(by_pid[current]["ppid"])
            chain.append(parent)
            seen.add(parent)
            if parent not in by_pid:
                break
            current = parent
        ancestors[pid] = chain

        found: list[int] = []
        pending = list(children.get(pid, []))
        seen_children: set[int] = set()
        while pending:
            child = pending.pop(0)
            if child in seen_children:
                continue
            seen_children.add(child)
            found.append(child)
            pending.extend(children.get(child, []))
        descendants[pid] = found
    return ancestors, descendants


def classify_runtime_shell_ancestor(
    records: Iterable[dict[str, Any]],
    *,
    current_pid: int,
    approved_runtime: str,
    script_path: str,
    workflow_run_id: str,
    workspace: str,
    report_dir: str,
) -> dict[str, Any]:
    """Return an allow/block decision for one runtime shell ancestor.

    Allow requires exact executable and command-role evidence at every link:
    current process and its Python shim use the approved runtime, the shell is
    a real ancestor with a closed descendant path, all identities have a
    create_time, and run/workspace/report flags match exactly. Anything
    missing or inconsistent is unknown and fail-closed.
    """

    rows = [dict(item) for item in records]
    by_pid = {int(item["pid"]): item for item in rows}
    if len(by_pid) != len(rows):
        return {"passed": False, "status": "blocked_unknown", "reason": "duplicate PID identity"}
    current = by_pid.get(int(current_pid))
    if current is None:
        return {"passed": False, "status": "blocked_unknown", "reason": "current PID missing"}
    ancestors, descendants = _lineage(rows)
    runtime = _norm(approved_runtime)
    script = _norm(script_path)
    expected_workspace = _norm(workspace)
    expected_report = _norm(report_dir)
    current_exe = _norm(str(current.get("exe") or ""))
    if not _is_python(current) or current_exe != runtime:
        return {
            "passed": False,
            "status": "blocked_unknown",
            "reason": "current executable is not the approved runtime",
            "current_pid": current_pid,
            "actual_executable": current.get("exe"),
        }
    if current.get("create_time") is None:
        return {"passed": False, "status": "blocked_unknown", "reason": "current create_time missing"}

    chain = ancestors.get(int(current_pid), [])
    for pid in chain:
        identity = by_pid.get(pid)
        if identity is None or identity.get("create_time") is None:
            return {"passed": False, "status": "blocked_unknown", "reason": "ancestor identity/create_time incomplete", "pid": pid}
        if not _is_shell(identity):
            continue
        command = _command(identity)
        command_norm = command.lower().replace("\\", "/")
        required = {
            "runtime": runtime.replace("\\", "/"),
            "script": script.replace("\\", "/"),
        }
        role_ok = all(value in command_norm for value in required.values())
        role_ok = role_ok and _flag(command, "--workflow-run-id") == workflow_run_id
        role_ok = role_ok and _norm(_flag(command, "--workspace-dir")) == expected_workspace
        role_ok = role_ok and _norm(_flag(command, "--report-dir")) == expected_report
        role_ok = role_ok and int(current_pid) in descendants.get(pid, [])
        if not role_ok:
            return {
                "passed": False,
                "status": "blocked_unknown",
                "reason": "shell ancestor command role, lineage, or identity mismatch",
                "pid": pid,
                "ancestors": chain,
                "descendants": descendants.get(pid, []),
            }
        return {
            "passed": True,
            "status": "approved_current_runtime_shell_ancestor",
            "process_role": "approved_runtime_shell_launcher_ancestor",
            "pid": pid,
            "current_pid": current_pid,
            "ancestors": chain,
            "descendants": descendants.get(pid, []),
            "runtime": runtime,
            "workflow_run_id": workflow_run_id,
            "workspace": expected_workspace,
            "report_dir": expected_report,
            "identity_evidence": {
                "shell_create_time": identity.get("create_time"),
                "current_create_time": current.get("create_time"),
                "current_executable": current_exe,
            },
        }
    return {"passed": False, "status": "blocked_unknown", "reason": "no shell ancestor in verified current lineage"}
