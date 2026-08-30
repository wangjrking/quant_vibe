"""Revision-2 synthetic-only hard-gate package for the L3 fast candidate.

This module never opens a business DuckDB table. It adds real DuckDB-shaped
synthetic subprocess evidence, WAL/unknown-writer gates, closed runtime
provenance, and read-only file-state sampling around the existing synthetic
topology benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow
import xgboost

import l3_memory_bounded_v2_fast_candidate as base


NEGATIVE_SHIFT_RE = re.compile(r"shift\s*\(\s*-\d+\s*\)", re.IGNORECASE)
FUTURE_COLUMN_RE = re.compile(r"(?:^|_)(?:future|label)(?:_|$)|_post\d+_", re.IGNORECASE)
EXPLICIT_FUTURE_COLUMNS = {"index_2000_post10_close"}
ACTIVE_STATUS_BASENAMES = {"l3_feature_current.duckdb", "l3_label_current.duckdb", "production_assets.json"}


def _within(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def runtime_provenance_closed(expected_executable: str) -> dict[str, Any]:
    identity = base.runtime_identity(expected_executable)
    runtime_root = Path(sys.prefix).resolve()
    # PYTHONPATH may intentionally expose the candidate source tree.  It is
    # recorded in evidence, but it is not the interpreter provenance.  The
    # executable, prefixes, stdlib, unittest, and imported third-party
    # packages must remain self-contained; any conda residue still fails closed.
    key_paths = [
        identity["sys_executable"],
        identity["sys_prefix"],
        identity["sys_base_prefix"],
        identity["stdlib_path"],
        identity["unittest_path"],
        str(Path(duckdb.__file__).resolve()),
        str(Path(pyarrow.__file__).resolve()),
        str(Path(xgboost.__file__).resolve()),
    ]
    residue = [p for p in key_paths if "conda" in p.lower() or not _within(p, runtime_root)]
    if residue:
        raise RuntimeError(f"runtime provenance failed closed: {residue}")
    return {
        **identity,
        "runtime_root": str(runtime_root),
        "duckdb_version": duckdb.__version__,
        "duckdb_file": str(Path(duckdb.__file__).resolve()),
        "pyarrow_version": pyarrow.__version__,
        "pyarrow_file": str(Path(pyarrow.__file__).resolve()),
        "xgboost_version": xgboost.__version__,
        "xgboost_file": str(Path(xgboost.__file__).resolve()),
        "runtime_provenance_status": "passed_closed",
    }


def validate_future_lineage(columns: list[str]) -> dict[str, Any]:
    rejected: list[dict[str, str]] = []
    for column in columns:
        lowered = column.lower()
        if column in EXPLICIT_FUTURE_COLUMNS:
            rejected.append({"column": column, "reason": "explicit_future_denylist"})
        elif NEGATIVE_SHIFT_RE.search(lowered):
            rejected.append({"column": column, "reason": "negative_shift"})
        elif FUTURE_COLUMN_RE.search(lowered):
            rejected.append({"column": column, "reason": "future_or_label_name"})
    return {"passed": not rejected, "rejected": rejected}


def assert_fresh_report_dir(path: str | Path) -> None:
    report = Path(path)
    if report.exists() and any(report.rglob("*")):
        raise RuntimeError(f"report directory must be new and empty: {report}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_active_file_state(paths: dict[str, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).resolve()
        if path.name not in ACTIVE_STATUS_BASENAMES:
            raise RuntimeError(f"active status path not allowlisted: {path}")
        if not path.is_file():
            raise RuntimeError(f"active status file missing: {path}")
        stat = path.stat()
        result[name] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
            "read_only_status_only": True,
        }
    return result


def assert_active_file_state_unchanged(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> None:
    for name in before:
        for key in ("path", "size_bytes", "mtime_ns", "sha256"):
            if before[name][key] != after[name][key]:
                raise RuntimeError(f"active file state drift: {name}.{key}")


def _duckdb_worker(db_path: str, temp_dir: str, expected_executable: str, conn: Any) -> None:
    try:
        runtime = runtime_provenance_closed(expected_executable)
        connection = duckdb.connect(db_path)
        connection.execute("PRAGMA threads=1")
        connection.execute("PRAGMA memory_limit='3GB'")
        connection.execute("SET temp_directory = ?", [temp_dir])
        connection.execute("CREATE TABLE synthetic_rows (id INTEGER, value DOUBLE)")
        connection.execute("BEGIN TRANSACTION")
        connection.execute("INSERT INTO synthetic_rows SELECT range, range::DOUBLE FROM range(10000)")
        settings = {
            "threads": connection.execute("SELECT current_setting('threads')").fetchone()[0],
            "memory_limit": connection.execute("SELECT current_setting('memory_limit')").fetchone()[0],
            "temp_directory": connection.execute("SELECT current_setting('temp_directory')").fetchone()[0],
        }
        conn.send(
            {
                "status": "transaction_open",
                "pid": os.getpid(),
                "runtime": runtime,
                "settings": settings,
                "db_path": db_path,
                "wal_path": db_path + ".wal",
            }
        )
        command = conn.recv()
        if command != "rollback":
            raise RuntimeError(f"unexpected synthetic command: {command}")
        connection.execute("ROLLBACK")
        connection.execute("CHECKPOINT")
        connection.close()
        conn.send({"status": "closed", "pid": os.getpid()})
    except Exception as exc:
        try:
            conn.send({"status": "failed", "error": repr(exc), "pid": os.getpid()})
        except Exception:
            pass
    finally:
        conn.close()


def writer_wal_gate(db_path: str | Path, unknown_writer_marker: str | Path) -> dict[str, Any]:
    wal_path = Path(str(db_path) + ".wal")
    marker = Path(unknown_writer_marker)
    reasons: list[str] = []
    if wal_path.is_file() and wal_path.stat().st_size > 0:
        reasons.append("wal_present")
    if marker.is_file():
        reasons.append("unknown_writer_marker_present")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "wal_path": str(wal_path),
        "wal_exists": wal_path.is_file(),
        "wal_size_bytes": wal_path.stat().st_size if wal_path.is_file() else 0,
        "unknown_writer_marker": str(marker),
        "unknown_writer_marker_exists": marker.is_file(),
    }


def run_duckdb_synthetic_gate(evidence_dir: Path, expected_executable: str) -> dict[str, Any]:
    runtime = runtime_provenance_closed(expected_executable)
    synthetic_dir = evidence_dir / "synthetic_duckdb"
    synthetic_dir.mkdir(parents=True, exist_ok=False)
    db_path = synthetic_dir / "synthetic_l3.duckdb"
    temp_dir = synthetic_dir / "duckdb_temp"
    temp_dir.mkdir()
    marker = synthetic_dir / "unknown_writer.marker"
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(target=_duckdb_worker, args=(str(db_path), str(temp_dir), expected_executable, child_conn))
    process.start()
    child_conn.close()
    opened = parent_conn.recv()
    if opened.get("status") != "transaction_open":
        process.join(timeout=10)
        raise RuntimeError(f"synthetic DuckDB worker failed: {opened}")
    wal_block = writer_wal_gate(db_path, marker)
    if wal_block["passed"] or "wal_present" not in wal_block["reasons"]:
        parent_conn.send("rollback")
        process.join(timeout=30)
        raise RuntimeError(f"WAL gate did not fail closed on real WAL: {wal_block}")
    marker.write_text("synthetic unknown writer", encoding="ascii")
    unknown_block = writer_wal_gate(db_path, marker)
    if unknown_block["passed"] or "unknown_writer_marker_present" not in unknown_block["reasons"]:
        parent_conn.send("rollback")
        process.join(timeout=30)
        raise RuntimeError(f"unknown writer gate did not fail closed: {unknown_block}")
    marker.unlink()
    parent_conn.send("rollback")
    closed = parent_conn.recv()
    process.join(timeout=30)
    parent_conn.close()
    if process.exitcode != 0 or closed.get("status") != "closed":
        raise RuntimeError(f"synthetic DuckDB worker did not close cleanly: {closed}")
    after_close = writer_wal_gate(db_path, marker)
    if not after_close["passed"]:
        raise RuntimeError(f"synthetic writer/WAL gate remained blocked: {after_close}")
    return {
        "runtime": runtime,
        "parent_pid": os.getpid(),
        "child_pid": opened["pid"],
        "child_exit_code": process.exitcode,
        "db_path": str(db_path),
        "temp_directory": str(temp_dir),
        "settings": opened["settings"],
        "wal_gate_fail_closed": wal_block,
        "unknown_writer_gate_fail_closed": unknown_block,
        "post_close_gate": after_close,
        "business_inputs_read": False,
        "production_paths_opened": False,
    }


def run_revision2(
    evidence_dir: str | Path,
    expected_executable: str,
    active_paths: dict[str, str],
) -> dict[str, Any]:
    evidence = Path(evidence_dir)
    assert_fresh_report_dir(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    runtime = runtime_provenance_closed(expected_executable)
    before = sample_active_file_state(active_paths)
    lineage = validate_future_lineage(
        ["open_qfq", "ema_qfq_10", "macdsignal_qfq", "index_2000_post10_close", "foo_shift(-1)"]
    )
    if lineage["passed"] or len(lineage["rejected"]) != 2:
        raise RuntimeError(f"future lineage gate fixture failed: {lineage}")
    duckdb_gate = run_duckdb_synthetic_gate(evidence, expected_executable)
    benchmark = base.run_benchmark(
        base.BenchmarkConfig(stock_count=12, rows_per_stock=32, expected_executable=expected_executable)
    )
    after = sample_active_file_state(active_paths)
    assert_active_file_state_unchanged(before, after)
    result = {
        "candidate_revision": "l3-memory-bounded-v2-fast-candidate-20260805-r2",
        "status": "passed_candidate_only",
        "runtime": runtime,
        "active_status_sampling": {"before": before, "after": after, "unchanged": True, "read_only": True},
        "duckdb_synthetic_gate": duckdb_gate,
        "future_lineage_gate": lineage,
        "topology_benchmark": benchmark,
        "business_inputs_read": False,
        "production_paths_opened": False,
        "business_asset_written": False,
        "candidate_written": False,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "full_rebuild_started": False,
        "probe_started": False,
        "allow_probe_only": False,
        "allow_full_rebuild": False,
        "allow_l3": False,
        "allow_next_layer_continue": False,
        "memory_profile": {
            "candidate_startup_gate_bytes": 34 * base.GIB,
            "original_production_startup_gate_bytes": 64 * base.GIB,
            "candidate_profile_is_not_production_profile": True,
        },
    }
    report = evidence / "revision2_synthetic_gate_report.json"
    report.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the candidate-only revision-2 synthetic gates.")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--expected-runtime", required=True)
    parser.add_argument("--active-feature-path", required=True)
    parser.add_argument("--active-label-path", required=True)
    parser.add_argument("--registry-path", required=True)
    args = parser.parse_args(argv)
    active_paths = {
        "feature": args.active_feature_path,
        "label": args.active_label_path,
        "registry": args.registry_path,
    }
    result = run_revision2(args.evidence_dir, args.expected_runtime, active_paths)
    print(json.dumps({"status": result["status"], "evidence_dir": args.evidence_dir}))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
