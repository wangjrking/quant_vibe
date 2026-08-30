"""Candidate-only subprocess result contract for L3 preflight wrappers.

This module is not wired into the formal L3 entry. It makes native stderr
diagnostic-only when the child returns zero and emits valid structured JSON.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Sequence


def run_structured_preflight_command(command: Sequence[str]) -> dict[str, Any]:
    """Run a candidate preflight command with explicit stdout/stderr handling."""

    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = (completed.stdout or "").strip()
    stderr = completed.stderr or ""
    evidence: dict[str, Any] = {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": stderr,
        "stderr_present": bool(stderr),
    }
    if completed.returncode != 0:
        return {
            "passed": False,
            "status": "failed_closed",
            "reason": "child_returncode_nonzero",
            "evidence": evidence,
        }
    if not stdout:
        return {
            "passed": False,
            "status": "failed_closed",
            "reason": "structured_stdout_missing",
            "evidence": evidence,
        }
    try:
        structured = json.loads(stdout)
    except json.JSONDecodeError as error:
        return {
            "passed": False,
            "status": "failed_closed",
            "reason": "structured_stdout_invalid_json",
            "parse_error": str(error),
            "evidence": evidence,
        }
    if not isinstance(structured, dict):
        return {
            "passed": False,
            "status": "failed_closed",
            "reason": "structured_stdout_not_object",
            "evidence": evidence,
        }
    if structured.get("passed") is False:
        return {
            "passed": False,
            "status": "failed_closed",
            "reason": "structured_preflight_failed",
            "result": structured,
            "evidence": evidence,
        }
    return {
        "passed": True,
        "status": "preflight_result_accepted",
        "reason": "zero_exit_and_valid_structured_result",
        "result": structured,
        "evidence": evidence,
    }
