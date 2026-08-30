"""Candidate-only child runtime identity capture.

The formal L3 entry does not import this module.  It provides a narrow,
testable subprocess contract that keeps stdout/stderr/exit-code semantics
separate from PowerShell native-command error handling.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


IDENTITY_CODE = (
    "import json, os, sys, sysconfig, unittest; "
    "print(json.dumps({"
    "'pid': os.getpid(), 'ppid': os.getppid(), "
    "'sys_executable': sys.executable, 'sys_prefix': sys.prefix, "
    "'sys_base_prefix': sys.base_prefix, "
    "'stdlib': sysconfig.get_paths().get('stdlib'), "
    "'unittest': unittest.__file__, 'sys_path': sys.path"
    "}, ensure_ascii=False))"
)


def _failed(reason: str, evidence: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "passed": False,
        "status": "failed_closed",
        "reason": reason,
        "evidence": dict(evidence),
        **extra,
    }


def interpret_child_identity_result(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    expected_executable: str,
    expected_runtime_root: str,
) -> dict[str, Any]:
    """Validate a completed child result without treating stderr as failure."""

    evidence = {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stderr_present": bool(stderr),
    }
    if returncode != 0:
        return _failed("child_returncode_nonzero", evidence)
    payload_text = stdout.strip()
    if not payload_text:
        return _failed("structured_stdout_missing", evidence)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        return _failed("structured_stdout_invalid_json", evidence, parse_error=str(error))
    if not isinstance(payload, dict):
        return _failed("structured_stdout_not_object", evidence)

    expected_executable_norm = str(Path(expected_executable)).replace("\\", "/").lower()
    expected_root_norm = str(Path(expected_runtime_root)).replace("\\", "/").rstrip("/").lower()
    actual_executable = str(payload.get("sys_executable") or "").replace("\\", "/")
    actual_root = str(payload.get("sys_prefix") or "").replace("\\", "/").rstrip("/")
    if actual_executable.lower() != expected_executable_norm:
        return _failed("child_executable_mismatch", evidence, result=payload)
    if actual_root.lower() != expected_root_norm:
        return _failed("child_prefix_mismatch", evidence, result=payload)
    if str(payload.get("sys_base_prefix") or "").replace("\\", "/").rstrip("/").lower() != expected_root_norm:
        return _failed("child_base_prefix_mismatch", evidence, result=payload)
    for field in ("stdlib", "unittest"):
        value = str(payload.get(field) or "").replace("\\", "/").lower()
        if not value.startswith(expected_root_norm + "/"):
            return _failed(f"child_{field}_outside_runtime", evidence, result=payload)
    return {
        "passed": True,
        "status": "child_identity_accepted",
        "reason": "zero_exit_valid_identity_stderr_diagnostic_only",
        "result": payload,
        "evidence": evidence,
    }


def capture_child_runtime_identity(
    runtime_executable: str,
    *,
    expected_runtime_root: str,
    timeout_seconds: float = 30.0,
    extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Capture one short-lived child identity with deterministic cleanup."""

    command = [runtime_executable, "-c", IDENTITY_CODE, *extra_args]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        process.kill()
        stdout, stderr = process.communicate()
        return _failed(
            "child_identity_timeout",
            {
                "command": command,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "stderr_present": bool(stderr),
                "timeout_seconds": timeout_seconds,
            },
        )
    result = interpret_child_identity_result(
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        expected_executable=runtime_executable,
        expected_runtime_root=expected_runtime_root,
    )
    result["command"] = command
    result["child_pid"] = process.pid
    return result

