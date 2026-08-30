from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "quant" / "data_file"
ACTIVE_L3_FEATURE_PATH = (DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb").resolve()
ACTIVE_L3_LABEL_PATH = (DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb").resolve()
PRODUCTION_REGISTRY_PATH = (DATA_DIR / "asset_registry" / "production_assets.json").resolve()
ACTIVE_L3_WRITER_LEASE_PATH = (
    DATA_DIR
    / "runtime"
    / "agent_workspaces"
    / "factor-agent"
    / "leases"
    / "l3_active_assets.writer_lease.json"
).resolve()
PROTECTED_ACTIVE_L3_PATHS = {
    ACTIVE_L3_FEATURE_PATH,
    ACTIVE_L3_LABEL_PATH,
    PRODUCTION_REGISTRY_PATH,
}


def _now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def is_protected_active_l3_path(path: str | Path) -> bool:
    resolved = Path(path).resolve()
    return any(_same_path(resolved, protected) for protected in PROTECTED_ACTIVE_L3_PATHS)


def acquire_active_l3_writer_lease(
    *,
    workflow_run_id: str,
    workspace: str | Path,
    report_dir: str | Path,
    process_role: str,
    lease_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(lease_path).resolve() if lease_path is not None else ACTIVE_L3_WRITER_LEASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "held",
        "acquired_at": _now(),
        "lease_nonce": uuid.uuid4().hex,
        "owner_pid": os.getpid(),
        "owner_parent_pid": os.getppid(),
        "workflow_run_id": str(workflow_run_id),
        "workspace": str(Path(workspace).resolve()),
        "report_dir": str(Path(report_dir).resolve()),
        "process_role": str(process_role),
        "protected_paths": sorted(str(path) for path in PROTECTED_ACTIVE_L3_PATHS),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"active L3 writer lease already exists; fail closed: {path}") from error
    return {"path": str(path), **payload}


def validate_active_l3_writer_lease(
    lease: dict[str, Any],
    *,
    workflow_run_id: str,
    workspace: str | Path,
    report_dir: str | Path,
    process_role: str,
    protected_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(str(lease.get("path") or ""))
    if not path.is_file():
        raise RuntimeError(f"active L3 writer lease is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "held",
        "lease_nonce": lease.get("lease_nonce"),
        "owner_pid": os.getpid(),
        "workflow_run_id": str(workflow_run_id),
        "workspace": str(Path(workspace).resolve()),
        "report_dir": str(Path(report_dir).resolve()),
        "process_role": str(process_role),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if protected_path is not None and not is_protected_active_l3_path(protected_path):
        mismatches["protected_path"] = {
            "expected": "registered active L3 feature/label/registry path",
            "actual": str(Path(protected_path).resolve()),
        }
    if mismatches:
        raise RuntimeError(f"active L3 writer lease ownership or scope mismatch: {mismatches}")
    return {
        "status": "valid",
        "validated_at": _now(),
        "path": str(path),
        "lease_nonce": payload["lease_nonce"],
        "owner_pid": payload["owner_pid"],
        "workflow_run_id": payload["workflow_run_id"],
        "process_role": payload["process_role"],
        "protected_path": str(Path(protected_path).resolve()) if protected_path is not None else None,
    }


def require_active_l3_writer_lease_for_path(
    path: str | Path,
    *,
    lease_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not is_protected_active_l3_path(resolved):
        return {"status": "not_required", "path": str(resolved)}
    actual_lease_path = Path(lease_path).resolve() if lease_path is not None else ACTIVE_L3_WRITER_LEASE_PATH
    if not actual_lease_path.is_file():
        raise RuntimeError(f"active L3 mutation requires writer lease before opening writable asset: {resolved}")
    payload = json.loads(actual_lease_path.read_text(encoding="utf-8"))
    if payload.get("status") != "held" or int(payload.get("owner_pid") or -1) != os.getpid():
        raise RuntimeError(
            "active L3 mutation lease is not owned by the current process: "
            f"path={resolved}, lease_owner={payload.get('owner_pid')}, current_pid={os.getpid()}"
        )
    if not payload.get("lease_nonce") or not payload.get("workflow_run_id"):
        raise RuntimeError(f"active L3 mutation lease is incomplete: {actual_lease_path}")
    return {
        "status": "valid",
        "validated_at": _now(),
        "path": str(actual_lease_path),
        "lease_nonce": payload["lease_nonce"],
        "owner_pid": payload["owner_pid"],
        "workflow_run_id": payload["workflow_run_id"],
        "process_role": payload.get("process_role"),
        "protected_path": str(resolved),
    }


def release_active_l3_writer_lease(lease: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(lease.get("path") or ""))
    validation = validate_active_l3_writer_lease(
        lease,
        workflow_run_id=str(lease.get("workflow_run_id") or ""),
        workspace=str(lease.get("workspace") or ""),
        report_dir=str(lease.get("report_dir") or ""),
        process_role=str(lease.get("process_role") or ""),
    )
    path.unlink()
    deadline = time.monotonic() + 2.0
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if path.exists():
        raise RuntimeError(f"active L3 writer lease remained after release: {path}")
    return {
        **validation,
        "status": "released",
        "released_at": _now(),
        "lease_absent_after_release": True,
    }
