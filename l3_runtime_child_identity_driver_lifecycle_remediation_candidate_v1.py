"""Candidate-only deterministic driver lifecycle contract.

This module is intentionally outside the formal L3 entrypoints.  It provides
an auditable Python-native replacement for wrappers that lose a driver's exit
code while preserving stdout/stderr as separate evidence channels.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


def _failed(reason: str, evidence: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "passed": False,
        "status": "failed_closed",
        "reason": reason,
        "evidence": dict(evidence),
        **extra,
    }


def _deterministic_returncode(process: Any) -> int | None:
    """Read a completed process return code without accepting an unknown value."""

    returncode = getattr(process, "returncode", None)
    if returncode is None:
        poll = getattr(process, "poll", None)
        if callable(poll):
            returncode = poll()
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        return None
    return returncode


def _cleanup_after_timeout(process: Any, timeout_seconds: float) -> dict[str, Any]:
    """Terminate a timed-out driver and collect its final streams."""

    kill = getattr(process, "kill", None)
    if not callable(kill):
        return {
            "cleanup_status": "kill_unavailable",
            "returncode": _deterministic_returncode(process),
            "stdout": "",
            "stderr": "",
        }
    try:
        kill()
    except Exception as error:  # pragma: no cover - defensive process boundary
        return {
            "cleanup_status": "kill_exception",
            "cleanup_error": repr(error),
            "returncode": _deterministic_returncode(process),
            "stdout": "",
            "stderr": "",
        }
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except Exception as error:  # pragma: no cover - defensive process boundary
        return {
            "cleanup_status": "communicate_exception",
            "cleanup_error": repr(error),
            "returncode": _deterministic_returncode(process),
            "stdout": "",
            "stderr": "",
        }
    return {
        "cleanup_status": "killed_and_collected",
        "returncode": _deterministic_returncode(process),
        "stdout": stdout or "",
        "stderr": stderr or "",
    }


def run_driver_lifecycle(
    runtime_executable: str,
    driver_script: str,
    *,
    driver_args: Sequence[str] = (),
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run a short driver with deterministic exit-state classification.

    A zero exit code plus valid JSON stdout is required.  Non-empty stderr is
    retained as diagnostic evidence and is not itself a failure.  A missing
    return code after communication is always fail-closed.
    """

    command = [runtime_executable, driver_script, *driver_args]
    evidence: dict[str, Any] = {
        "command": command,
        "timeout_seconds": timeout_seconds,
        "stdout": "",
        "stderr": "",
        "stderr_present": False,
        "returncode": None,
    }
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except FileNotFoundError as error:
        return _failed("driver_start_file_not_found", evidence, error=repr(error))
    except OSError as error:
        return _failed("driver_start_os_error", evidence, error=repr(error))
    except Exception as error:  # pragma: no cover - defensive process boundary
        return _failed("driver_start_exception", evidence, error=repr(error))

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        cleanup = _cleanup_after_timeout(process, timeout_seconds)
        evidence.update(
            {
                "stdout": cleanup["stdout"],
                "stderr": cleanup["stderr"],
                "stderr_present": bool(cleanup["stderr"]),
                "returncode": cleanup["returncode"],
                "timeout_error": repr(error),
                "cleanup_status": cleanup["cleanup_status"],
            }
        )
        if cleanup.get("cleanup_error"):
            evidence["cleanup_error"] = cleanup["cleanup_error"]
        reason = (
            "driver_timeout_cleanup_incomplete"
            if cleanup["returncode"] is None
            else "driver_timeout"
        )
        return _failed(reason, evidence)
    except Exception as error:
        evidence["communicate_error"] = repr(error)
        evidence["returncode"] = _deterministic_returncode(process)
        return _failed("driver_communicate_exception", evidence)

    evidence.update(
        {
            "stdout": stdout or "",
            "stderr": stderr or "",
            "stderr_present": bool(stderr),
            "returncode": _deterministic_returncode(process),
        }
    )
    returncode = evidence["returncode"]
    if returncode is None:
        return _failed("driver_returncode_unavailable", evidence)
    if returncode != 0:
        return _failed("driver_returncode_nonzero", evidence)

    payload_text = evidence["stdout"].strip()
    if not payload_text:
        return _failed("driver_structured_stdout_missing", evidence)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        return _failed(
            "driver_structured_stdout_invalid_json", evidence, parse_error=str(error)
        )
    if not isinstance(payload, dict):
        return _failed("driver_structured_stdout_not_object", evidence)
    return {
        "passed": True,
        "status": "driver_completed",
        "reason": "zero_exit_valid_json_stderr_diagnostic_only",
        "payload": payload,
        "evidence": evidence,
    }


def normalize_runtime_path(path: str) -> str:
    """Return a stable path form for evidence comparisons."""

    return str(Path(path)).replace("\\", "/").lower().rstrip("/")
