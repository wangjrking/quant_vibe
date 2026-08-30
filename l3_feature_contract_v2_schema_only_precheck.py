from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

import duckdb
import psutil

from l3_process_gate import classify_process_gate, collect_process_records


PROJECT_ROOT = Path("D:/work/quant/quant_mcp")
CANONICAL_L2_PATH = PROJECT_ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
CANONICAL_L2_TABLE = "STOCK_DAILY_DATA"
EXPECTED_L2_SHA256 = "34a88582b2e0e53fa5ececadf8500673ad44259dd7d2bc8e44d3c689c7905e4b"
ISOLATED_RUNTIME = PROJECT_ROOT / "runtime_candidates/my_quant_copy_20260804/python.exe"
QFQ_PRICE_COLUMNS = {"open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"}
KEY_ORDER = ["stock_code", "trade_date"]
FORBIDDEN_ROW_SCAN_SQL_TOKENS = (
    "".join(("COU", "NT", "(")),
    " ".join(("COUNT", "DISTINCT")),
    "".join(("MI", "N", "(")),
    "".join(("MA", "X", "(")),
    " ".join(("GROUP", "BY")),
    "".join(("HAV", "ING")),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _json_sha(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _runtime_identity() -> dict[str, Any]:
    import unittest

    return {
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
        "sys.executable": str(Path(sys.executable).resolve()),
        "sys.prefix": str(Path(sys.prefix).resolve()),
        "sys.base_prefix": str(Path(sys.base_prefix).resolve()),
        "stdlib": str(Path(sysconfig.get_paths()["stdlib"]).resolve()),
        "unittest": str(Path(unittest.__file__).resolve()),
        "duckdb": str(Path(duckdb.__file__).resolve()),
        "duckdb_version": duckdb.__version__,
        "sys_path": [str(Path(item).resolve()) if item else item for item in sys.path],
    }


def _runtime_gate(identity: dict[str, Any]) -> None:
    expected_executable = str(ISOLATED_RUNTIME.resolve())
    runtime_root = str(ISOLATED_RUNTIME.parent.resolve()).lower()
    if identity["sys.executable"] != expected_executable:
        raise RuntimeError("isolated runtime executable mismatch")
    for key in ("sys.prefix", "sys.base_prefix", "stdlib", "unittest", "duckdb"):
        value = str(identity[key]).lower()
        if not value.startswith(runtime_root) or "conda" in value:
            raise RuntimeError(f"runtime provenance drift at {key}: {identity[key]}")


def _spawn_child_identity(root: Path, run_id: str) -> dict[str, Any]:
    output_path = root / "child_runtime_identity.json"
    command = [
        str(ISOLATED_RUNTIME),
        str(Path(__file__).resolve()),
        "--run-id",
        run_id,
        "--isolated-root",
        str(root),
        "--emit-runtime-identity",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True, timeout=30)
    identity = json.loads(output_path.read_text(encoding="utf-8"))
    _runtime_gate(identity)
    output_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError("child runtime probe failed")
    return identity


def _wal_gate(path: Path) -> dict[str, Any]:
    evidence = []
    for suffix in (".wal", "-wal"):
        wal_path = Path(str(path) + suffix)
        if wal_path.exists() and wal_path.stat().st_size > 0:
            evidence.append({"path": str(wal_path), "size_bytes": int(wal_path.stat().st_size)})
    if evidence:
        raise RuntimeError(f"canonical L2 WAL present: {evidence}")
    return {"status": "passed", "checked_suffixes": [".wal", "-wal"]}


def _process_gate(run_id: str, root: Path, report_dir: Path, l2_path: Path) -> dict[str, Any]:
    records = collect_process_records(psutil, protected_l3_paths=[str(l2_path.resolve())], workspace=str(root.resolve()))
    result = classify_process_gate(
        records,
        current_pid=os.getpid(),
        workflow_run_id=run_id,
        workspace=str(root.resolve()),
        report_dir=str(report_dir.resolve()),
        unified_python_launcher=str(ISOLATED_RUNTIME.resolve()),
        expected_workers={},
        protected_l3_paths=[str(l2_path.resolve())],
    )
    if not result.get("passed"):
        raise RuntimeError(
            "process/unknown-writer gate failed: "
            + json.dumps(
                {
                    "blocking_writers": result.get("blocking_writers", []),
                    "unknown_relevant_processes": result.get("unknown_relevant_processes", []),
                },
                ensure_ascii=False,
            )
        )
    return result


def _ordered_schema(conn: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    schema = [
        {
            "index": int(row[0]),
            "name": str(row[1]),
            "type": str(row[2]),
            "not_null": bool(row[3]),
        }
        for row in rows
    ]
    if len(schema) < 2 or [schema[0]["name"], schema[1]["name"]] != KEY_ORDER:
        observed = [item["name"] for item in schema[:2]]
        raise RuntimeError(f"L2 key order drift: expected={KEY_ORDER} observed={observed}")
    return schema


def _schema_summary(schema: list[dict[str, Any]]) -> dict[str, Any]:
    names = [item["name"] for item in schema]
    qfq_technical = [name for name in names if "_qfq" in name.lower() and name not in QFQ_PRICE_COLUMNS]
    return {
        "column_count": len(schema),
        "key_order": names[:2],
        "qfq_price_count": sum(1 for name in names if name in QFQ_PRICE_COLUMNS),
        "qfq_technical_count": len(qfq_technical),
        "qfq_total_count": sum(1 for name in names if "_qfq" in name.lower()),
    }


def _static_sql_safety_gate() -> dict[str, Any]:
    source_text = Path(__file__).read_text(encoding="utf-8").upper()
    hits = [token for token in FORBIDDEN_ROW_SCAN_SQL_TOKENS if token in source_text]
    if hits:
        raise RuntimeError(f"schema-only runner contains forbidden row-scan SQL tokens: {hits}")
    return {
        "status": "passed",
        "forbidden_tokens": list(FORBIDDEN_ROW_SCAN_SQL_TOKENS),
        "matched_tokens": hits,
        "no_l2_row_scan": True,
    }


def _quarantine_root(root: Path, reason: str) -> dict[str, Any]:
    if not root.exists():
        return {"quarantined": False, "reuse_prohibited": True, "reason": reason}
    quarantine_root = root.parent / f"quarantine_{root.name}_{int(time.time())}"
    shutil.move(str(root), str(quarantine_root))
    return {
        "quarantined": True,
        "path": str(quarantine_root),
        "reuse_prohibited": True,
        "reason": reason,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _markdown_report(payload: dict[str, Any]) -> str:
    l2 = payload["l2"]
    schema = payload["ordered_l2_schema_summary"]
    return "\n".join(
        [
            f"# L3 schema-only precheck `{payload['run_id']}`",
            "",
            f"- status: `{payload['status']}`",
            f"- target_trade_date: `{payload['target_trade_date']}`",
            f"- active L2: `{l2['path']}::{l2['table']}`",
            f"- L2 SHA256: `{l2['file_state_before']['sha256']}`",
            f"- ordered_l2_schema_hash: `{payload['ordered_l2_schema_hash']}`",
            f"- column_count: `{schema['column_count']}`",
            f"- key_order: `{schema['key_order']}`",
            f"- qfq counts: prices=`{schema['qfq_price_count']}`, technical=`{schema['qfq_technical_count']}`, total=`{schema['qfq_total_count']}`",
            f"- no_l2_row_scan: `{payload['static_sql_safety_gate']['no_l2_row_scan']}`",
            "",
            "This run is schema-only and read-only. It only inspects file fingerprints and ordered schema metadata; no business-row scan is allowed.",
        ]
    ) + "\n"


def run_schema_only_precheck(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.isolated_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    if root.exists() or report_dir.exists():
        raise RuntimeError("schema-only precheck requires absent isolated root and report dir")
    root.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    runtime_identity = _runtime_identity()
    _runtime_gate(runtime_identity)
    child_runtime_identity = _spawn_child_identity(root, args.run_id)

    l2_path = CANONICAL_L2_PATH.resolve()
    if str(args.expected_l2_sha256).lower() != EXPECTED_L2_SHA256:
        raise RuntimeError("unexpected granted L2 SHA value")

    before_state = _file_state(l2_path)
    if before_state["sha256"].lower() != str(args.expected_l2_sha256).lower():
        raise RuntimeError("active L2 SHA drift detected before schema precheck")

    process_gate = _process_gate(args.run_id, root, report_dir, l2_path)
    wal_gate = _wal_gate(l2_path)
    static_sql_safety = _static_sql_safety_gate()

    with duckdb.connect(str(l2_path), read_only=True) as conn:
        ordered_schema = _ordered_schema(conn, CANONICAL_L2_TABLE)
        schema_summary = _schema_summary(ordered_schema)

    after_state = _file_state(l2_path)
    if before_state != after_state:
        raise RuntimeError("active L2 fingerprint drift detected during schema precheck")
    if schema_summary["qfq_technical_count"] != 74:
        raise RuntimeError(f"L2 qfq technical count drift: expected=74 observed={schema_summary['qfq_technical_count']}")

    payload = {
        "status": "schema_only_precheck_passed",
        "run_id": args.run_id,
        "target_trade_date": args.target_trade_date,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_provenance_status": "passed",
        "runtime_identity": runtime_identity,
        "child_runtime_identity": child_runtime_identity,
        "l2": {
            "path": str(l2_path),
            "table": CANONICAL_L2_TABLE,
            "expected_sha256": args.expected_l2_sha256.lower(),
            "file_state_before": before_state,
            "file_state_after": after_state,
        },
        "ordered_l2_schema": ordered_schema,
        "ordered_l2_schema_hash": _json_sha(ordered_schema),
        "ordered_l2_schema_summary": schema_summary,
        "process_gate": process_gate,
        "wal_gate": wal_gate,
        "static_sql_safety_gate": static_sql_safety,
        "authorization": {
            "allow_schema_only_precheck": True,
            "allow_candidate_rebuild_grant": False,
            "allow_full_rebuild": False,
            "allow_l3": False,
            "allow_l4_l8": False,
            "grant_precheck_ready": False,
            "business_input_read": False,
            "no_l2_business_read": True,
            "no_l2_row_scan": True,
            "probe_started": False,
            "build_started": False,
            "full_rebuild_started": False,
            "candidate_written": False,
            "active_switch": False,
            "label_write": False,
            "registry_change": False,
            "manifest_write": False,
        },
        "writes": {
            "business_asset_written": False,
            "candidate_duckdb_created": False,
            "workspace_created_for_build": False,
        },
        "reuse_prohibited": True,
    }

    schema_lock_path = report_dir / "l2_schema_lock.json"
    report_path = report_dir / "schema_only_precheck_report.json"
    handoff_path = report_dir / "audit_handoff.json"
    markdown_path = report_dir / "README.md"
    artifact_hashes_path = report_dir / "artifact_hashes.json"

    _write_json(
        schema_lock_path,
        {
            "run_id": args.run_id,
            "status": "schema_lock_materialized",
            "ordered_l2_schema": ordered_schema,
            "ordered_l2_schema_hash": payload["ordered_l2_schema_hash"],
            "l2_file_state_before": before_state,
            "l2_file_state_after": after_state,
            "business_input_read": False,
            "allow_schema_only_precheck": True,
        },
    )
    _write_json(report_path, payload)
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    _write_json(
        handoff_path,
        {
            "handoff_type": "L3 schema-only precheck",
            "run_id": args.run_id,
            "ready_for_audit_review": True,
            "allow_schema_only_precheck": True,
            "allow_candidate_rebuild_grant": False,
            "allow_full_rebuild": False,
            "allow_l3": False,
            "allow_l4_l8": False,
            "business_input_read": False,
            "probe_started": False,
            "build_started": False,
            "full_rebuild_started": False,
            "candidate_written": False,
            "active_switch": False,
            "label_write": False,
            "registry_change": False,
            "report_path": str(report_path),
            "schema_lock_path": str(schema_lock_path),
            "markdown_path": str(markdown_path),
            "ordered_l2_schema_hash": payload["ordered_l2_schema_hash"],
        },
    )

    artifact_payload = {
        "run_id": args.run_id,
        "hash_mismatch_count": 0,
        "schema_lock_sha256": _sha256(schema_lock_path),
        "report_sha256": _sha256(report_path),
        "handoff_sha256": _sha256(handoff_path),
        "markdown_sha256": _sha256(markdown_path),
        "executor_sha256": _sha256(Path(__file__).resolve()),
        "runtime_sha256": _sha256(ISOLATED_RUNTIME.resolve()),
    }
    _write_json(artifact_hashes_path, artifact_payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L3 feature_contract_v2 schema-only precheck for active L2")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--isolated-root", required=True)
    parser.add_argument("--report-dir")
    parser.add_argument("--target-trade-date")
    parser.add_argument("--expected-l2-sha256")
    parser.add_argument("--emit-runtime-identity")
    args = parser.parse_args(argv)

    if args.emit_runtime_identity:
        Path(args.emit_runtime_identity).write_text(json.dumps(_runtime_identity(), ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    missing = [
        name
        for name, value in (
            ("--report-dir", args.report_dir),
            ("--target-trade-date", args.target_trade_date),
            ("--expected-l2-sha256", args.expected_l2_sha256),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"missing required arguments for schema-only precheck: {', '.join(missing)}")

    root = Path(args.isolated_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    try:
        run_schema_only_precheck(args)
        return 0
    except Exception as exc:
        incident_path = report_dir / "schema_only_precheck_incident.json"
        quarantine = _quarantine_root(root, str(exc))
        incident = {
            "status": "failed_closed",
            "run_id": args.run_id,
            "target_trade_date": args.target_trade_date,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "expected_l2_sha256": args.expected_l2_sha256.lower(),
            "probe_started": False,
            "build_started": False,
            "full_rebuild_started": False,
            "candidate_written": False,
            "active_switch": False,
            "label_write": False,
            "registry_change": False,
            "business_input_read": False,
            "quarantine": quarantine,
            "reuse_prohibited": True,
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        _write_json(incident_path, incident)
        print(json.dumps(incident, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
