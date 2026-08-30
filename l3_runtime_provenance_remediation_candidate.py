"""Candidate-only runtime provenance remediation and preflight gates.

This module discovers local interpreter candidates and classifies supplied
runtime evidence. It never installs, copies, rewrites, or launches a runtime
and never opens a business asset.
"""

from __future__ import annotations

import hashlib
import os
import sys
import sysconfig
import unittest as _unittest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REMEDIATION_ID = "l3-runtime-provenance-remediation-20260804"
PROJECT_ROOT = "D:/work/quant/quant_mcp"
PROJECT_VENV = f"{PROJECT_ROOT}/.venv"
CONDA_MARKERS = ("/.conda/", "/conda/", "anaconda", "miniconda")
REQUIRED_PACKAGES = ("duckdb", "pyarrow", "xgboost")


def _norm(value: str | None) -> str:
    return str(value or "").replace("\\", "/")


def _under(path: str, root: str) -> bool:
    value = _norm(path).lower().rstrip("/")
    base = _norm(root).lower().rstrip("/")
    return value == base or value.startswith(base + "/")


def _has_conda_marker(path: str) -> bool:
    value = _norm(path).lower()
    return any(marker in value for marker in CONDA_MARKERS)


def collect_current_runtime_evidence() -> dict[str, Any]:
    """Collect process metadata only; no project or business asset is opened."""

    modules: dict[str, Any] = {}
    for name in REQUIRED_PACKAGES:
        try:
            module = __import__(name)
            modules[name] = {
                "version": str(getattr(module, "__version__", "unknown")),
                "file": _norm(getattr(module, "__file__", "")),
            }
        except Exception as exc:  # pragma: no cover - depends on host packages
            modules[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "sys_executable": _norm(sys.executable),
        "sys_prefix": _norm(sys.prefix),
        "sys_base_prefix": _norm(getattr(sys, "base_prefix", "")),
        "stdlib_path": _norm(sysconfig.get_paths().get("stdlib", "")),
        "unittest_path": _norm(getattr(_unittest, "__file__", "")),
        "sys_path": [_norm(item) for item in sys.path],
        "packages": modules,
    }


def validate_self_contained_runtime(
    evidence: Mapping[str, Any],
    *,
    runtime_root: str,
    expected_executable: str | None = None,
) -> dict[str, Any]:
    """Require interpreter, stdlib and all relevant imports under one root."""

    root = _norm(runtime_root)
    errors: list[str] = []
    executable = _norm(evidence.get("sys_executable"))
    prefix = _norm(evidence.get("sys_prefix"))
    base_prefix = _norm(evidence.get("sys_base_prefix"))
    if expected_executable and executable.lower() != _norm(expected_executable).lower():
        errors.append("executable_mismatch")
    if not _under(executable, root) or not executable.lower().endswith("python.exe"):
        errors.append("executable_outside_runtime_root")
    for field in ("sys_prefix", "sys_base_prefix", "stdlib_path", "unittest_path"):
        if not _under(_norm(evidence.get(field)), root):
            errors.append(f"{field}_outside_runtime_root")
        if _has_conda_marker(_norm(evidence.get(field))):
            errors.append(f"{field}_conda_provenance")
    if prefix.lower() != base_prefix.lower():
        errors.append("base_prefix_mismatch")
    for item in evidence.get("sys_path", []):
        if item and not _under(_norm(item), root):
            errors.append("sys_path_outside_runtime_root")
    for name in REQUIRED_PACKAGES:
        info = evidence.get("packages", {}).get(name, {})
        package_file = _norm(info.get("file"))
        if info.get("error"):
            errors.append(f"{name}_import_error")
        elif not _under(package_file, root) or _has_conda_marker(package_file):
            errors.append(f"{name}_provenance_mismatch")
    if errors:
        return {
            "status": "failed_closed",
            "runtime_root": root,
            "errors": sorted(set(errors)),
            "probe_authorized": False,
        }
    return {"status": "passed", "runtime_root": root, "errors": [], "probe_authorized": False}


def discover_local_python_candidates(search_roots: Sequence[str]) -> list[dict[str, Any]]:
    """Read-only discovery of local python.exe files; never launches or copies them."""

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in search_roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("python.exe"):
            normalized = _norm(str(path))
            if normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            candidates.append(
                {
                    "path": normalized,
                    "under_conda": _has_conda_marker(normalized),
                    "under_project_venv": _under(normalized, PROJECT_VENV),
                    "candidate_status": "rejected_conda_or_existing_venv"
                    if _has_conda_marker(normalized) or _under(normalized, PROJECT_VENV)
                    else "requires_runtime_probe",
                }
            )
    return sorted(candidates, key=lambda item: item["path"])


def build_runtime_remediation_manifest(
    *,
    current_evidence: Mapping[str, Any],
    discovered_candidates: Sequence[Mapping[str, Any]],
    runtime_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a no-action remediation manifest suitable for read-only audit."""

    payload = {
        "remediation_id": REMEDIATION_ID,
        "status": "failed_closed" if runtime_validation.get("status") != "passed" else "candidate_runtime_not_authorized",
        "candidate_runtime_available": bool(discovered_candidates),
        "current_runtime_validation": dict(runtime_validation),
        "current_runtime": dict(current_evidence),
        "discovered_candidates": [dict(item) for item in discovered_candidates],
        "candidate_runtime_created": False,
        "current_venv_modified": False,
        "download_performed": False,
        "business_inputs_read": False,
        "business_assets_written": False,
        "production_paths_opened": False,
        "probe_started": False,
        "full_rebuild_started": False,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "allow_probe_only": False,
        "allow_full_rebuild": False,
        "allow_L3": False,
        "allow_next_layer_continue": False,
        "rollback_cleanup": {
            "action_taken": False,
            "delete_or_modify_current_venv": False,
            "cleanup_required": False,
            "instruction": "Do not modify current .venv; remove only a separately created candidate directory after audit decision."
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(
        repr(sorted((str(key), repr(value)) for key, value in payload.items())).encode("utf-8")
    ).hexdigest()
    return payload

