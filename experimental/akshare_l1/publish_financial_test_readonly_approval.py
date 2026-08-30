from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import akshare_l1_experimental as base
import remediate_financial_contract as contract


TASK_ID = "investment-platform-test-data-gapfill-20260718-publish-test-readonly-approval"
AUDIT_TASK_ID = "investment-platform-test-data-gapfill-20260718-site-gate-reaudit"
AUDIT_THREAD_ID = "019ed091-89b0-73c2-b60b-e2ba1d6ac6d9"
APPROVAL = "approved"
APPROVAL_SCOPE = "experimental/test only"
EXPECTED_ROWS = 241688
EXPECTED_STOCKS = 3528
EXPECTED_POOL = 5210
EXPECTED_DB_SHA256 = "1135c6c8f3ad38e9cf91b35ffbd1cbaa0d42355ed2b7710dc842ac18e3ef6ce1"
EXPECTED_DOMAIN_SHA256 = "630cd1ddbb593d844692008ff320b4ffda3e13bd225a10d23f974e350be60320"

ROOT = base.DEFAULT_OUTPUT_ROOT
DB_PATH = ROOT / "financial_indicator" / "financial_indicator.duckdb"
CHECKPOINT_PATH = ROOT / "checkpoints" / "financial_indicator_batch.json"
MANIFEST_PATH = ROOT / "manifest.json"
GENERATION_ROOT = ROOT / "metadata_generations"
CURRENT_POINTER = GENERATION_ROOT / "current.json"
REPORT_JSON = base.DEFAULT_REPORT_ROOT / f"{TASK_ID}.json"
REPORT_MD = REPORT_JSON.with_suffix(".md")
HASH_JSON = base.DEFAULT_REPORT_ROOT / f"{TASK_ID}_hashes.json"
SERVER_PATH = base.PROJECT_ROOT / "site" / "backend" / "server.py"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transient_files(root: Path) -> list[str]:
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".candidate") or path.name.endswith(".tmp"))
    )


def atomic_text_once(path: Path, text: str) -> Path:
    """Publish once; a Windows sharing violation is a hard stop for this task."""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid.uuid4().hex}.candidate")
    try:
        with candidate.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(candidate, path)
    except Exception:
        if candidate.exists():
            candidate.unlink()
        raise
    return path


def atomic_json_once(path: Path, payload: dict[str, Any]) -> Path:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    json.loads(serialized)
    return atomic_text_once(path, serialized)


def site_gate() -> dict[str, Any]:
    os.environ["QUANT_ENABLE_AKSHARE_EXPERIMENTAL"] = "1"
    os.environ["QUANT_ENABLE_AKSHARE_EXPERIMENTAL_FINANCIALS"] = "1"
    spec = importlib.util.spec_from_file_location(
        f"investment_site_financial_approval_gate_{uuid.uuid4().hex}", SERVER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load test-site contract gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CACHE.clear()
    return module._akshare_financial_contract_gate()


def status_counts(checkpoint: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for state in (checkpoint.get("stocks") or {}).values():
        status = str(state.get("status") or "missing")
        result[status] = result.get(status, 0) + 1
    return dict(sorted(result.items()))


def approval_fields(generation_id: str, approved_at: str) -> dict[str, Any]:
    return {
        "test_read_only_approval": APPROVAL,
        "audit_task_id": AUDIT_TASK_ID,
        "audit_thread_id": AUDIT_THREAD_ID,
        "approved_at": approved_at,
        "generation_id": generation_id,
        "approval_scope": APPROVAL_SCOPE,
        "not_approved_for_production": True,
        "production_approved": False,
        "resume_allowed": False,
    }


def prepare_approved_bundle(
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    generation_id: str,
    approved_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    approved_manifest = copy.deepcopy(manifest)
    approved_checkpoint = copy.deepcopy(checkpoint)
    fields = approval_fields(generation_id, approved_at)
    asset = approved_manifest["assets"]["financial_indicator"]
    asset.update(fields)
    asset["status"] = "experimental/test_read_only_approved_partial_3528_of_5210_not_approved_for_production"
    asset["resume_policy"] = "resume disabled; this approval permits test-site read-only display only"
    asset["coverage"]["resume_allowed"] = False
    asset["coverage"]["resume_blocked_reason"] = "test read-only approval does not authorize source resume"
    approved_checkpoint.update(fields)
    approved_checkpoint["status"] = "experimental/test_read_only_approved_metadata_only_not_approved_for_production"
    approved_checkpoint["resume_blocked_reason"] = "test read-only approval does not authorize source resume"
    approved_checkpoint["updated_at"] = approved_at
    approved_manifest["financial_contract_generation"] = {
        **fields,
        "published_at": approved_at,
        "status": "experimental/test_read_only_approved_not_approved_for_production",
    }
    approved_manifest["test_read_only_approval"] = APPROVAL
    approved_manifest["audit_task_id"] = AUDIT_TASK_ID
    approved_manifest["audit_thread_id"] = AUDIT_THREAD_ID
    approved_manifest["approved_at"] = approved_at
    approved_manifest["generated_at"] = approved_at
    approved_manifest["not_approved_for_production"] = True
    approved_manifest["production_approved"] = False
    approved_manifest["production_assets_touched"] = False
    approved_manifest["route_registry_touched"] = False
    approved_manifest["l2_l8_triggered"] = False
    return approved_manifest, approved_checkpoint


def validate_approval_bundle(
    manifest_path: Path,
    checkpoint_path: Path,
    generation_id: str,
    approved_at: str,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    checkpoint = load_json(checkpoint_path)
    asset = manifest["assets"]["financial_indicator"]
    physical = contract.inspect_physical_db(DB_PATH)
    base_validation = contract.validate_contract(manifest_path, checkpoint_path, DB_PATH)
    expected_fields = approval_fields(generation_id, approved_at)
    checks = {
        "base_contract_valid": base_validation["valid"],
        "fixed_rows": physical["rows"] == EXPECTED_ROWS,
        "fixed_stocks": physical["stock_coverage"] == EXPECTED_STOCKS,
        "fixed_db_sha256": physical["db_sha256"] == EXPECTED_DB_SHA256,
        "fixed_domain_sha256": physical["code_domain_sha256"] == EXPECTED_DOMAIN_SHA256,
        "pool_total": int(checkpoint["pool"]["total"]) == EXPECTED_POOL,
        "status_counts": status_counts(checkpoint) == {"completed": EXPECTED_STOCKS, "pending": EXPECTED_POOL - EXPECTED_STOCKS},
        "asset_approval_fields": all(asset.get(key) == value for key, value in expected_fields.items()),
        "checkpoint_approval_fields": all(checkpoint.get(key) == value for key, value in expected_fields.items()),
        "manifest_experimental_only": manifest.get("mode") == "experimental/test" and manifest.get("not_approved_for_production") is True,
        "asset_experimental_only": str(asset.get("status") or "").startswith("experimental/test") and asset.get("not_approved_for_production") is True,
        "checkpoint_experimental_only": str(checkpoint.get("status") or "").startswith("experimental/test") and checkpoint.get("not_approved_for_production") is True,
        "resume_disabled": asset.get("resume_allowed") is False and checkpoint.get("resume_allowed") is False,
        "no_staged_running": not contract.status_codes(checkpoint, "staged") and not contract.status_codes(checkpoint, "running"),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "base_checks": base_validation["checks"],
        "physical": {key: value for key, value in physical.items() if key != "codes"},
        "checkpoint_status_counts": status_counts(checkpoint),
    }


def validate_pointer(pointer: dict[str, Any], generation_dir: Path, generation_id: str) -> dict[str, bool]:
    return {
        "generation_id": pointer.get("generation_id") == generation_id,
        "manifest_path": Path(str(pointer.get("manifest_path") or "")).resolve() == (generation_dir / "manifest.json").resolve(),
        "checkpoint_path": Path(str(pointer.get("checkpoint_path") or "")).resolve() == (generation_dir / "financial_indicator_batch.json").resolve(),
        "db_sha256": pointer.get("db_sha256") == EXPECTED_DB_SHA256,
        "code_domain_sha256": pointer.get("code_domain_sha256") == EXPECTED_DOMAIN_SHA256,
        "approval": pointer.get("test_read_only_approval") == APPROVAL,
        "audit_task_id": pointer.get("audit_task_id") == AUDIT_TASK_ID,
        "audit_thread_id": pointer.get("audit_thread_id") == AUDIT_THREAD_ID,
        "experimental_only": pointer.get("not_approved_for_production") is True,
        "resume_disabled": pointer.get("resume_allowed") is False,
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Experimental financial test read-only approval",
        "",
        f"- status: {report['status']}",
        f"- generation_id: {report['generation_id']}",
        f"- approval: {report['four_place_approval']['all_approved']}",
        f"- rows / stocks / pool: {EXPECTED_ROWS} / {EXPECTED_STOCKS} / {EXPECTED_POOL}",
        f"- checkpoint: {report['checkpoint_status_counts']}",
        f"- current pointer switched: {report['atomic_switch']['current_pointer_switched']}",
        f"- final site gate valid: {report['atomic_switch']['final_gate']['valid']}",
        f"- DB unchanged: {report['db_unchanged']}",
        f"- production tree unchanged: {report['production_tree_unchanged']}",
        "- scope: experimental/test read-only display only",
        "- production approval: false",
        "- source fetch / resume / L2-L8: false / false / false",
    ]
    atomic_text_once(REPORT_MD, "\n".join(lines) + "\n")


def run() -> dict[str, Any]:
    transients_before = transient_files(ROOT)
    if transients_before:
        raise RuntimeError(f"stop: candidate/tmp residue exists: {transients_before}")
    production_before = contract.tree_metadata_snapshot(base.PRODUCTION_ROOT)
    physical_before = contract.inspect_physical_db(DB_PATH)
    if (
        physical_before["rows"] != EXPECTED_ROWS
        or physical_before["stock_coverage"] != EXPECTED_STOCKS
        or physical_before["db_sha256"] != EXPECTED_DB_SHA256
        or physical_before["code_domain_sha256"] != EXPECTED_DOMAIN_SHA256
    ):
        raise RuntimeError("stop: physical DB contract differs from audited approval baseline")
    root_manifest_before = load_json(MANIFEST_PATH)
    root_checkpoint_before = load_json(CHECKPOINT_PATH)
    if status_counts(root_checkpoint_before) != {"completed": EXPECTED_STOCKS, "pending": EXPECTED_POOL - EXPECTED_STOCKS}:
        raise RuntimeError("stop: checkpoint status domain differs from approved baseline")
    gate_before = site_gate()
    if gate_before.get("valid"):
        raise RuntimeError("stop: test-site gate was already valid before this approval generation")

    approved_at = now_iso()
    generation_id = (
        "akshare-financial-test-readonly-approved-"
        + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        + "-"
        + EXPECTED_DOMAIN_SHA256[:12]
    )
    generation_dir = GENERATION_ROOT / generation_id
    generation_dir.mkdir(parents=True, exist_ok=False)
    generation_manifest = generation_dir / "manifest.json"
    generation_checkpoint = generation_dir / "financial_indicator_batch.json"
    generation_validation = generation_dir / "validation.json"
    approved_manifest, approved_checkpoint = prepare_approved_bundle(
        root_manifest_before, root_checkpoint_before, generation_id, approved_at
    )
    atomic_json_once(generation_checkpoint, approved_checkpoint)
    atomic_json_once(generation_manifest, approved_manifest)
    candidate_validation = validate_approval_bundle(
        generation_manifest, generation_checkpoint, generation_id, approved_at
    )
    if not candidate_validation["valid"]:
        raise RuntimeError(f"stop: candidate generation validation failed: {candidate_validation['checks']}")
    atomic_json_once(
        generation_validation,
        {
            "task_id": TASK_ID,
            "generation_id": generation_id,
            "validated_at": now_iso(),
            "approval": APPROVAL,
            "audit_task_id": AUDIT_TASK_ID,
            "audit_thread_id": AUDIT_THREAD_ID,
            "validation": candidate_validation,
        },
    )
    gate_after_candidate = site_gate()
    if gate_after_candidate.get("valid"):
        raise RuntimeError("stop: unpublished candidate unexpectedly opened the site gate")
    if contract.inspect_physical_db(DB_PATH)["db_sha256"] != EXPECTED_DB_SHA256:
        raise RuntimeError("stop: DB changed before root metadata switch")
    if contract.tree_metadata_snapshot(base.PRODUCTION_ROOT) != production_before:
        raise RuntimeError("stop: production assets changed before root metadata switch")

    atomic_json_once(CHECKPOINT_PATH, approved_checkpoint)
    gate_after_checkpoint = site_gate()
    if gate_after_checkpoint.get("valid"):
        raise RuntimeError("stop: checkpoint-only intermediate state did not fail closed")
    atomic_json_once(MANIFEST_PATH, approved_manifest)
    gate_after_manifest = site_gate()
    if gate_after_manifest.get("valid"):
        raise RuntimeError("stop: root metadata switched before current pointer but gate opened")
    if contract.inspect_physical_db(DB_PATH)["db_sha256"] != EXPECTED_DB_SHA256:
        raise RuntimeError("stop: DB changed before current pointer switch")
    if contract.tree_metadata_snapshot(base.PRODUCTION_ROOT) != production_before:
        raise RuntimeError("stop: production assets changed before current pointer switch")

    pointer = {
        "task_id": TASK_ID,
        "generation_id": generation_id,
        "published_at": approved_at,
        "approved_at": approved_at,
        "test_read_only_approval": APPROVAL,
        "audit_task_id": AUDIT_TASK_ID,
        "audit_thread_id": AUDIT_THREAD_ID,
        "approval_scope": APPROVAL_SCOPE,
        "manifest_path": str(generation_manifest),
        "checkpoint_path": str(generation_checkpoint),
        "validation_path": str(generation_validation),
        "db_path": str(DB_PATH),
        "db_sha256": EXPECTED_DB_SHA256,
        "code_domain_sha256": EXPECTED_DOMAIN_SHA256,
        "resume_allowed": False,
        "not_approved_for_production": True,
        "production_approved": False,
        "status": "experimental/test_read_only_approved_not_approved_for_production",
    }
    atomic_json_once(CURRENT_POINTER, pointer)
    final_gate = site_gate()
    pointer_checks = validate_pointer(load_json(CURRENT_POINTER), generation_dir, generation_id)
    root_validation = validate_approval_bundle(MANIFEST_PATH, CHECKPOINT_PATH, generation_id, approved_at)
    generation_validation_after = validate_approval_bundle(
        generation_manifest, generation_checkpoint, generation_id, approved_at
    )
    physical_after = contract.inspect_physical_db(DB_PATH)
    production_after = contract.tree_metadata_snapshot(base.PRODUCTION_ROOT)
    transients_after = transient_files(ROOT)
    if not final_gate.get("valid"):
        raise RuntimeError(f"stop: final test-site gate is blocked: {final_gate}")
    if not root_validation["valid"] or not generation_validation_after["valid"]:
        raise RuntimeError("stop: root or generation approval bundle failed final validation")
    if not all(pointer_checks.values()):
        raise RuntimeError(f"stop: current pointer validation failed: {pointer_checks}")
    if physical_after["db_sha256"] != physical_before["db_sha256"]:
        raise RuntimeError("stop: physical DB changed during metadata approval")
    if production_after != production_before:
        raise RuntimeError("stop: production assets changed during metadata approval")
    if transients_after:
        raise RuntimeError(f"stop: candidate/tmp residue after publish: {transients_after}")

    root_manifest_after = load_json(MANIFEST_PATH)
    root_checkpoint_after = load_json(CHECKPOINT_PATH)
    generation_manifest_after = load_json(generation_manifest)
    generation_checkpoint_after = load_json(generation_checkpoint)
    four_values = {
        "root_manifest_asset": root_manifest_after["assets"]["financial_indicator"].get("test_read_only_approval"),
        "root_checkpoint": root_checkpoint_after.get("test_read_only_approval"),
        "generation_manifest_asset": generation_manifest_after["assets"]["financial_indicator"].get("test_read_only_approval"),
        "generation_checkpoint": generation_checkpoint_after.get("test_read_only_approval"),
    }
    report = {
        "task_id": TASK_ID,
        "status": "completed_experimental_test_read_only_approved",
        "generated_at": now_iso(),
        "generation_id": generation_id,
        "approved_at": approved_at,
        "audit_task_id": AUDIT_TASK_ID,
        "audit_thread_id": AUDIT_THREAD_ID,
        "approval_scope": APPROVAL_SCOPE,
        "four_place_approval": {
            **four_values,
            "all_approved": all(value == APPROVAL for value in four_values.values()),
        },
        "physical": {key: value for key, value in physical_after.items() if key != "codes"},
        "checkpoint_status_counts": status_counts(root_checkpoint_after),
        "atomic_switch": {
            "candidate_gate": gate_after_candidate,
            "checkpoint_only_gate": gate_after_checkpoint,
            "root_metadata_before_pointer_gate": gate_after_manifest,
            "current_pointer_switched": True,
            "pointer_checks": pointer_checks,
            "final_gate": final_gate,
            "intermediate_states_fail_closed": all(
                not state.get("valid")
                for state in (gate_after_candidate, gate_after_checkpoint, gate_after_manifest)
            ),
        },
        "root_validation": root_validation,
        "generation_validation": generation_validation_after,
        "db_unchanged": physical_after["db_sha256"] == physical_before["db_sha256"],
        "production_tree_unchanged": production_after == production_before,
        "candidate_tmp_residue": transients_after,
        "source_calls": 0,
        "resume_allowed": False,
        "test_site_read_only_consumption_allowed": True,
        "production_assets_touched": False,
        "route_registry_touched": False,
        "l2_l8_triggered": False,
        "production_approved": False,
        "generation_paths": {
            "directory": str(generation_dir),
            "manifest": str(generation_manifest),
            "checkpoint": str(generation_checkpoint),
            "validation": str(generation_validation),
            "current_pointer": str(CURRENT_POINTER),
        },
        "evidence_paths": {
            "report_json": str(REPORT_JSON),
            "report_markdown": str(REPORT_MD),
            "hash_json": str(HASH_JSON),
        },
    }
    atomic_json_once(REPORT_JSON, report)
    write_markdown(report)
    hash_paths = [
        DB_PATH,
        CHECKPOINT_PATH,
        MANIFEST_PATH,
        generation_checkpoint,
        generation_manifest,
        generation_validation,
        CURRENT_POINTER,
        REPORT_JSON,
        REPORT_MD,
        Path(__file__),
        SERVER_PATH,
    ]
    atomic_json_once(
        HASH_JSON,
        {
            "task_id": TASK_ID,
            "generated_at": now_iso(),
            "algorithm": "SHA256",
            "files": {
                str(path): {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in hash_paths
            },
            "note": "hash evidence intentionally does not self-hash",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish audited experimental financial test-site read-only approval metadata"
    )
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
