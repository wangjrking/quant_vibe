from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

import akshare_l1_experimental as base
import akshare_l1_scaleout as scaleout


TASK_ID = "investment-platform-test-data-gapfill-20260718-contract-remediation"
ROOT = base.DEFAULT_OUTPUT_ROOT
DB_PATH = ROOT / "financial_indicator" / "financial_indicator.duckdb"
CHECKPOINT_PATH = ROOT / "checkpoints" / "financial_indicator_batch.json"
LEGACY_TMP_PATH = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
MANIFEST_PATH = ROOT / "manifest.json"
GENERATION_ROOT = ROOT / "metadata_generations"
CURRENT_POINTER = GENERATION_ROOT / "current.json"
REPORT_JSON = base.DEFAULT_REPORT_ROOT / f"{TASK_ID}.json"
REPORT_MD = REPORT_JSON.with_suffix(".md")
HASH_JSON = base.DEFAULT_REPORT_ROOT / f"{TASK_ID}_hashes.json"
PRODUCTION_ROOT = base.PRODUCTION_ROOT


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_domain_sha256(codes: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(codes))).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_metadata_snapshot(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    records = [
        f"{path.relative_to(root).as_posix()}|{path.stat().st_size}|{path.stat().st_mtime_ns}"
        for path in files
    ]
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return {"root": str(root), "file_count": len(files), "metadata_sha256": digest}


def inspect_physical_db(path: Path) -> dict[str, Any]:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
        ).fetchall()]
        rows = int(connection.execute("SELECT COUNT(*) FROM financial_indicator").fetchone()[0])
        codes = [str(row[0]) for row in connection.execute(
            "SELECT DISTINCT ts_code FROM financial_indicator ORDER BY ts_code"
        ).fetchall()]
        min_date, max_date = connection.execute(
            "SELECT MIN(report_date), MAX(report_date) FROM financial_indicator"
        ).fetchone()
        duplicate_groups = int(connection.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT ts_code, report_date, COUNT(*) AS c FROM financial_indicator "
            "GROUP BY ts_code, report_date HAVING COUNT(*) > 1)"
        ).fetchone()[0])
        bj_rows = int(connection.execute(
            "SELECT COUNT(*) FROM financial_indicator WHERE UPPER(ts_code) LIKE '%.BJ'"
        ).fetchone()[0])
        key_nulls = connection.execute(
            "SELECT COUNT(*) FILTER (WHERE ts_code IS NULL), "
            "COUNT(*) FILTER (WHERE report_date IS NULL) FROM financial_indicator"
        ).fetchone()
    finally:
        connection.close()
    return {
        "path": str(path),
        "tables": tables,
        "rows": rows,
        "codes": codes,
        "stock_coverage": len(codes),
        "min_date": str(min_date),
        "max_date": str(max_date),
        "duplicate_key_groups": duplicate_groups,
        "bj_rows": bj_rows,
        "natural_key_null_counts": {"ts_code": int(key_nulls[0]), "report_date": int(key_nulls[1])},
        "db_sha256": file_sha256(path),
        "code_domain_sha256": code_domain_sha256(codes),
    }


def status_codes(checkpoint: dict[str, Any], status: str) -> list[str]:
    return sorted(
        str(code)
        for code, state in (checkpoint.get("stocks") or {}).items()
        if isinstance(state, dict) and state.get("status") == status
    )


def prepare_checkpoint(
    checkpoint: dict[str, Any],
    physical: dict[str, Any],
    generation_id: str,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    result = copy.deepcopy(checkpoint)
    physical_codes = physical["codes"]
    completed_before = status_codes(result, "completed")
    if completed_before != physical_codes:
        raise RuntimeError("checkpoint completed set does not exactly match physical DB")
    reset = {
        "staged": status_codes(result, "staged"),
        "running": status_codes(result, "running"),
    }
    for previous_status, codes in reset.items():
        for code in codes:
            attempts = int(result["stocks"][code].get("attempts", 0))
            result["stocks"][code] = {
                "status": "pending",
                "rows": 0,
                "origin": None,
                "attempts": attempts,
                "remediation_reset_from": previous_status,
                "remediation_generation_id": generation_id,
                "remediation_reset_at": generated_at,
            }
    if status_codes(result, "staged") or status_codes(result, "running"):
        raise RuntimeError("staged/running checkpoint state remains after remediation")
    if status_codes(result, "completed") != physical_codes:
        raise RuntimeError("checkpoint completed set changed during remediation")
    result.update(
        {
            "generation_id": generation_id,
            "db_sha256": physical["db_sha256"],
            "code_domain_sha256": physical["code_domain_sha256"],
            "physical_rows": physical["rows"],
            "physical_stock_coverage": physical["stock_coverage"],
            "resume_allowed": False,
            "resume_blocked_reason": "awaiting read-only reaudit after atomic contract remediation",
            "updated_at": generated_at,
            "metadata_contract_version": 1,
        }
    )
    return result, reset


def prepare_manifest(
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    physical: dict[str, Any],
    generation_id: str,
    generated_at: str,
) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    asset = result["assets"]["financial_indicator"]
    pool_total = int(checkpoint["pool"]["total"])
    source_empty = len(status_codes(checkpoint, "source_empty"))
    failed = len(status_codes(checkpoint, "failed"))
    completed = len(status_codes(checkpoint, "completed"))
    pending = len(status_codes(checkpoint, "pending"))
    coverage = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "pool_total": pool_total,
        "processed_this_remediation": 0,
        "completed_with_data": completed,
        "source_empty": source_empty,
        "failed": failed,
        "pending": pending,
        "remaining": pending + failed + source_empty,
        "next_resume_point": None,
        "resume_allowed": False,
        "resume_blocked_reason": "awaiting read-only reaudit; no source fetch performed",
    }
    asset.update(
        {
            "rows": physical["rows"],
            "stock_coverage": physical["stock_coverage"],
            "min_date": physical["min_date"],
            "max_date": physical["max_date"],
            "duplicate_key_groups": physical["duplicate_key_groups"],
            "bj_rows": physical["bj_rows"],
            "natural_key_null_counts": physical["natural_key_null_counts"],
            "sha256": physical["db_sha256"],
            "generation_id": generation_id,
            "db_sha256": physical["db_sha256"],
            "code_domain_sha256": physical["code_domain_sha256"],
            "coverage": coverage,
            "resume_policy": "blocked until read-only reaudit; then resume only from reviewed pending domain",
            "status": "experimental/test_partial_contract_remediated_awaiting_reaudit_not_approved_for_production",
            "not_approved_for_production": True,
            "production_approved": False,
        }
    )
    result["financial_first_batch"] = coverage
    result["coverage_summary"]["financial_terminal_stock_coverage"] = {
        "covered_stocks": completed + source_empty,
        "pool_total": pool_total,
        "ratio": round((completed + source_empty) / pool_total, 6),
    }
    result["financial_contract_generation"] = {
        "generation_id": generation_id,
        "db_sha256": physical["db_sha256"],
        "code_domain_sha256": physical["code_domain_sha256"],
        "published_at": generated_at,
        "resume_allowed": False,
    }
    result["generated_at"] = generated_at
    result["production_assets_touched"] = False
    result["route_registry_touched"] = False
    result["l2_l8_triggered"] = False
    result["production_approved"] = False
    return result


def validate_contract(
    manifest_path: Path,
    checkpoint_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    checkpoint = load_json(checkpoint_path)
    physical = inspect_physical_db(db_path)
    asset = manifest["assets"]["financial_indicator"]
    completed = status_codes(checkpoint, "completed")
    checks = {
        "one_table_one_file": physical["tables"] == ["financial_indicator"],
        "generation_nonempty": bool(asset.get("generation_id")),
        "generation_match": asset.get("generation_id") == checkpoint.get("generation_id"),
        "db_hash_match": asset.get("db_sha256") == checkpoint.get("db_sha256") == physical["db_sha256"],
        "domain_hash_match": asset.get("code_domain_sha256") == checkpoint.get("code_domain_sha256") == physical["code_domain_sha256"],
        "row_count_match": int(asset.get("rows", -1)) == physical["rows"],
        "stock_coverage_match": int(asset.get("stock_coverage", -1)) == physical["stock_coverage"] == len(completed),
        "completed_codes_match": completed == physical["codes"],
        "no_staged": not status_codes(checkpoint, "staged"),
        "no_running": not status_codes(checkpoint, "running"),
        "duplicate_groups_zero": physical["duplicate_key_groups"] == 0,
        "bj_rows_zero": physical["bj_rows"] == 0,
        "natural_key_nulls_zero": all(value == 0 for value in physical["natural_key_null_counts"].values()),
        "experimental_only": str(asset.get("status", "")).startswith("experimental/test")
        and bool(asset.get("not_approved_for_production")),
    }
    public_physical = {key: value for key, value in physical.items() if key != "codes"}
    return {"valid": all(checks.values()), "checks": checks, "physical": public_physical}


def quarantine_legacy_tmp(path: Path, generation_id: str) -> Path | None:
    if not path.exists():
        return None
    quarantine_root = path.parent / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / f"{path.name}.{generation_id}.{file_sha256(path)[:12]}.quarantined"
    os.replace(path, target)
    return target


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    physical = report["after"]["physical"]
    lines = [
        "# AKShare experimental financial contract remediation",
        "",
        f"- status: {report['status']}",
        f"- generation_id: {report['generation_id']}",
        f"- rows / stocks: {physical['rows']} / {physical['stock_coverage']}",
        f"- date range: {physical['min_date']} - {physical['max_date']}",
        f"- duplicate groups / BJ rows: {physical['duplicate_key_groups']} / {physical['bj_rows']}",
        f"- reset staged / running: {len(report['reset']['staged'])} / {len(report['reset']['running'])}",
        f"- old tmp quarantined: {report['legacy_tmp']['quarantined']}",
        f"- DB unchanged: {report['db_unchanged']}",
        f"- completed code set unchanged: {report['completed_set_unchanged']}",
        "- source fetch performed: false",
        "- production assets touched: false",
        "- L2-L8 triggered: false",
        "- consumption status: awaiting read-only reaudit; not approved for production",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    generated_at = now_iso()
    production_before = tree_metadata_snapshot(PRODUCTION_ROOT)
    physical_before = inspect_physical_db(DB_PATH)
    checkpoint_before = load_json(CHECKPOINT_PATH)
    manifest_before = load_json(MANIFEST_PATH)
    completed_before = status_codes(checkpoint_before, "completed")
    if completed_before != physical_before["codes"]:
        raise RuntimeError("stop: baseline completed set differs from physical DB")
    if physical_before["stock_coverage"] != 3528:
        raise RuntimeError(f"stop: expected reviewed 3528-code baseline, got {physical_before['stock_coverage']}")

    tmp_before: dict[str, Any] | None = load_json(LEGACY_TMP_PATH) if LEGACY_TMP_PATH.exists() else None
    if tmp_before is not None and status_codes(tmp_before, "completed") != physical_before["codes"]:
        raise RuntimeError("stop: legacy tmp completed set differs from physical DB")
    generation_id = (
        "akshare-financial-"
        + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        + "-"
        + physical_before["code_domain_sha256"][:12]
    )
    checkpoint_candidate, reset = prepare_checkpoint(
        checkpoint_before, physical_before, generation_id, generated_at
    )
    manifest_candidate = prepare_manifest(
        manifest_before, checkpoint_candidate, physical_before, generation_id, generated_at
    )

    generation_dir = GENERATION_ROOT / generation_id
    generation_dir.mkdir(parents=True, exist_ok=False)
    version_checkpoint = generation_dir / CHECKPOINT_PATH.name
    version_manifest = generation_dir / MANIFEST_PATH.name
    version_validation = generation_dir / "validation.json"
    scaleout.atomic_json(version_checkpoint, checkpoint_candidate)
    scaleout.atomic_json(version_manifest, manifest_candidate)
    candidate_validation = validate_contract(version_manifest, version_checkpoint, DB_PATH)
    if not candidate_validation["valid"]:
        raise RuntimeError(f"stop: candidate contract validation failed: {candidate_validation['checks']}")
    scaleout.atomic_json(
        version_validation,
        {
            "task_id": TASK_ID,
            "generation_id": generation_id,
            "validated_at": now_iso(),
            "validation": candidate_validation,
            "source_fetch_performed": False,
            "production_assets_touched": False,
        },
    )

    quarantined = quarantine_legacy_tmp(LEGACY_TMP_PATH, generation_id)
    scaleout.atomic_json(CHECKPOINT_PATH, checkpoint_candidate)
    intermediate_fail_closed = not validate_contract(MANIFEST_PATH, CHECKPOINT_PATH, DB_PATH)["valid"]
    if not intermediate_fail_closed:
        raise RuntimeError("stop: intermediate checkpoint-only publish did not fail closed")
    scaleout.atomic_json(MANIFEST_PATH, manifest_candidate)
    active_validation = validate_contract(MANIFEST_PATH, CHECKPOINT_PATH, DB_PATH)
    if not active_validation["valid"]:
        raise RuntimeError(f"stop: active contract validation failed: {active_validation['checks']}")

    current_pointer = {
        "task_id": TASK_ID,
        "generation_id": generation_id,
        "published_at": generated_at,
        "manifest_path": str(version_manifest),
        "checkpoint_path": str(version_checkpoint),
        "validation_path": str(version_validation),
        "db_path": str(DB_PATH),
        "db_sha256": physical_before["db_sha256"],
        "code_domain_sha256": physical_before["code_domain_sha256"],
        "status": "experimental/test_awaiting_read_only_reaudit_not_approved_for_production",
    }
    scaleout.atomic_json(CURRENT_POINTER, current_pointer)

    physical_after = inspect_physical_db(DB_PATH)
    checkpoint_after = load_json(CHECKPOINT_PATH)
    db_unchanged = physical_after["db_sha256"] == physical_before["db_sha256"]
    completed_set_unchanged = status_codes(checkpoint_after, "completed") == completed_before
    if not db_unchanged or not completed_set_unchanged:
        raise RuntimeError("stop: DB content or checkpoint completed set changed")
    production_after = tree_metadata_snapshot(PRODUCTION_ROOT)
    production_tree_unchanged = production_before == production_after
    if not production_tree_unchanged:
        raise RuntimeError("stop: production asset metadata changed during remediation")

    report = {
        "task_id": TASK_ID,
        "status": "completed_awaiting_read_only_reaudit",
        "generated_at": generated_at,
        "generation_id": generation_id,
        "before": {
            "physical": {key: value for key, value in physical_before.items() if key != "codes"},
            "checkpoint": {
                "completed": len(completed_before),
                "pending": len(status_codes(checkpoint_before, "pending")),
                "staged": status_codes(checkpoint_before, "staged"),
                "running": status_codes(checkpoint_before, "running"),
            },
            "manifest": {
                "rows": manifest_before["assets"]["financial_indicator"].get("rows"),
                "stock_coverage": manifest_before["assets"]["financial_indicator"].get("stock_coverage"),
                "generation_id": manifest_before["assets"]["financial_indicator"].get("generation_id"),
            },
        },
        "reset": reset,
        "legacy_tmp": {
            "existed": tmp_before is not None,
            "completed": len(status_codes(tmp_before, "completed")) if tmp_before else 0,
            "staged": status_codes(tmp_before, "staged") if tmp_before else [],
            "quarantined": str(quarantined) if quarantined else None,
            "original_path_absent_after": not LEGACY_TMP_PATH.exists(),
        },
        "candidate_validation": candidate_validation,
        "intermediate_fail_closed": intermediate_fail_closed,
        "after": active_validation,
        "db_unchanged": db_unchanged,
        "completed_set_unchanged": completed_set_unchanged,
        "generation_paths": {
            "directory": str(generation_dir),
            "manifest": str(version_manifest),
            "checkpoint": str(version_checkpoint),
            "validation": str(version_validation),
            "current_pointer": str(CURRENT_POINTER),
        },
        "source_fetch_performed": False,
        "resume_allowed": False,
        "requires_read_only_reaudit": True,
        "test_site_consumption_approved": False,
        "production_assets_touched": False,
        "production_tree_unchanged": production_tree_unchanged,
        "production_tree_before": production_before,
        "production_tree_after": production_after,
        "production_route_or_registry_touched": False,
        "l2_l8_triggered": False,
        "legacy_odb_used": False,
        "evidence_paths": {
            "json": str(REPORT_JSON),
            "markdown": str(REPORT_MD),
            "hashes": str(HASH_JSON),
        },
    }
    scaleout.atomic_json(REPORT_JSON, report)
    write_markdown(REPORT_MD, report)
    hash_paths = [
        DB_PATH,
        CHECKPOINT_PATH,
        MANIFEST_PATH,
        version_checkpoint,
        version_manifest,
        version_validation,
        CURRENT_POINTER,
        REPORT_JSON,
        REPORT_MD,
    ]
    if quarantined:
        hash_paths.append(quarantined)
    hashes = {
        str(path): {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        for path in hash_paths
    }
    scaleout.atomic_json(
        HASH_JSON,
        {
            "task_id": TASK_ID,
            "generated_at": now_iso(),
            "algorithm": "SHA256",
            "files": hashes,
            "note": "hash evidence intentionally does not self-hash",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair experimental financial metadata contract without source calls")
    parser.parse_args()
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
