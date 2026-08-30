"""Candidate-only runner for a future read-only canonical L2 probe.

The default path is plan-only and never opens the canonical database.  Real
execution is available only behind an explicit ``--execute-probe`` switch and
must pass every pre-open gate before the read-only DuckDB connection is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb

try:
    import psutil
except ImportError as exc:  # pragma: no cover - runtime gate
    raise RuntimeError("psutil is required for controlled probe") from exc


CANONICAL_L2_PATH = Path(
    "D:/work/quant/quant_mcp/quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
)
CANONICAL_L2_TABLE = "STOCK_DAILY_DATA"
EXPECTED_L2_SHA256 = "34a88582b2e0e53fa5ececadf8500673ad44259dd7d2bc8e44d3c689c7905e4b"
TARGET_START = "20260803"
TARGET_END = "20260804"
HASH_BUCKET_COUNT = 16
HASH_BUCKET_ID = 3
QFQ_PRICE_COLUMNS = ("open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq")
FUTURE_DENYLIST = {"index_2000_post10_close"}
NEGATIVE_SHIFT_RE = re.compile(r"shift\s*\(\s*-\d+\s*\)", re.IGNORECASE)
FORBIDDEN_COLUMN_RE = re.compile(r"(?:^|_)(?:future|label)(?:_|$)|_post\d+_", re.IGNORECASE)
FORBIDDEN_PATH_TOKENS = ("quarantine", "legacy", "sqlite", "odb.db", ".parquet", "snapshot")
GIB = 1024**3
SYSTEM_HARD_STOP = 20 * GIB
PARENT_BUDGET = 4 * GIB
WORKER_BUDGET = 6 * GIB
RESERVE = 4 * GIB
CANDIDATE_STARTUP_GATE = 34 * GIB
RECOVERY_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ProbeConfig:
    run_id: str
    workspace: Path
    staging_dir: Path
    report_dir: Path
    execute_probe: bool = False
    l2_path: Path = CANONICAL_L2_PATH
    l2_table: str = CANONICAL_L2_TABLE
    expected_l2_sha256: str = EXPECTED_L2_SHA256
    target_start: str = TARGET_START
    target_end: str = TARGET_END
    hash_bucket_count: int = HASH_BUCKET_COUNT
    hash_bucket_id: int = HASH_BUCKET_ID


def runtime_identity() -> dict[str, Any]:
    import unittest

    return {
        "sys_executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "sys_base_prefix": str(Path(sys.base_prefix).resolve()),
        "stdlib_path": str(Path(os.__file__).resolve().parent),
        "unittest_path": str(Path(unittest.__file__).resolve()),
        "sys_path": [str(Path(item).resolve()) for item in sys.path if item],
    }


def runtime_gate(identity: dict[str, Any], expected_executable: str) -> dict[str, Any]:
    expected = str(Path(expected_executable).resolve())
    prefix = Path(identity["sys_prefix"]).resolve()
    key_paths = [identity["sys_executable"], identity["sys_prefix"], identity["sys_base_prefix"], identity["stdlib_path"], identity["unittest_path"]]
    errors = []
    if identity["sys_executable"] != expected:
        errors.append("runtime_executable_mismatch")
    if identity["sys_base_prefix"] != identity["sys_prefix"]:
        errors.append("runtime_base_prefix_mismatch")
    if any("conda" in value.lower() for value in key_paths):
        errors.append("conda_runtime_residue")
    for value in key_paths:
        try:
            Path(value).resolve().relative_to(prefix)
        except ValueError:
            errors.append("runtime_path_outside_prefix")
            break
    return {"passed": not errors, "errors": sorted(set(errors)), "identity": identity}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _sha256(path)}


def assert_fresh_output_paths(config: ProbeConfig) -> None:
    root = config.workspace.parent.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"isolated output root must be absent or empty: {root}")
    for path in (config.workspace, config.staging_dir, config.report_dir):
        if path.exists():
            raise RuntimeError(f"output path must be absent before probe: {path}")
        lowered = str(path).lower().replace("\\", "/")
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            raise RuntimeError(f"forbidden output path: {path}")


def validate_canonical_route(config: ProbeConfig) -> None:
    if config.l2_path.resolve() != CANONICAL_L2_PATH.resolve():
        raise RuntimeError("canonical L2 path mismatch")
    if config.l2_table != CANONICAL_L2_TABLE:
        raise RuntimeError("canonical L2 table mismatch")
    if config.expected_l2_sha256.lower() != EXPECTED_L2_SHA256:
        raise RuntimeError("canonical L2 SHA256 mismatch")
    if not 0 <= config.hash_bucket_id < config.hash_bucket_count:
        raise RuntimeError("invalid deterministic hash bucket")


def wal_writer_gate(l2_path: Path, unknown_writer_marker: Path) -> dict[str, Any]:
    candidates = [Path(str(l2_path) + ".wal"), Path(str(l2_path) + "-wal")]
    present = [str(path) for path in candidates if path.is_file() and path.stat().st_size > 0]
    errors = []
    if present:
        errors.append("l2_wal_present")
    if unknown_writer_marker.exists():
        errors.append("unknown_writer_present")
    return {"passed": not errors, "errors": errors, "wal_paths": present, "unknown_writer_marker": str(unknown_writer_marker)}


def lease_gate(lease: dict[str, Any] | None, run_id: str) -> dict[str, Any]:
    if not lease or lease.get("owner_run_id") != run_id or lease.get("active") is not True:
        return {"passed": False, "errors": ["exclusive_writer_lease_missing_or_invalid"]}
    return {"passed": True, "errors": [], "lease_nonce": lease.get("lease_nonce")}


def process_writer_gate(relevant_processes: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Reject any unknown or writable L3-related process before opening L2."""
    blockers = []
    for process in relevant_processes or []:
        role = str(process.get("role", "unknown")).lower()
        decision = str(process.get("decision", "block")).lower()
        if role in {"writer", "unknown", "l3_writer"} or decision != "allow":
            blockers.append(process)
    return {"passed": not blockers, "blockers": blockers}


def memory_gate(available_before: int, available_after: int | None, rss_peak: int, recovery_seconds: float | None) -> dict[str, Any]:
    required = max(CANDIDATE_STARTUP_GATE, SYSTEM_HARD_STOP + PARENT_BUDGET + max(GIB, int(rss_peak or 0)) + RESERVE)
    errors = []
    if available_before < required:
        errors.append("available_memory_before_below_required")
    if available_before < SYSTEM_HARD_STOP:
        errors.append("system_hard_stop_breached")
    if available_after is not None and available_after < required:
        errors.append("available_memory_after_below_required")
    if recovery_seconds is not None and recovery_seconds > RECOVERY_TIMEOUT_SECONDS:
        errors.append("memory_recovery_timeout")
    return {
        "passed": not errors,
        "errors": errors,
        "required_before_bytes": required,
        "available_before_bytes": available_before,
        "available_after_bytes": available_after,
        "rss_peak_bytes": rss_peak,
        "recovery_seconds": recovery_seconds,
        "candidate_startup_gate_bytes": CANDIDATE_STARTUP_GATE,
        "original_production_startup_gate_bytes": 64 * GIB,
    }


def wait_memory_recovery(required_bytes: int, timeout_seconds: int = RECOVERY_TIMEOUT_SECONDS) -> tuple[int, float]:
    started = time.monotonic()
    while True:
        available = int(psutil.virtual_memory().available)
        elapsed = time.monotonic() - started
        if available >= required_bytes:
            return available, elapsed
        if elapsed >= timeout_seconds:
            return available, elapsed
        time.sleep(0.5)


def quarantine_workspace(path: Path, reason: str) -> dict[str, Any]:
    """Move only isolated probe output to a sibling quarantine directory."""
    if not path.exists():
        return {"quarantined": False, "reason": "workspace_absent", "reuse_prohibited": True}
    target = path.parent / f"quarantine_{path.name}_{int(time.time())}"
    shutil.move(str(path), str(target))
    return {"quarantined": True, "path": str(target), "reason": reason, "reuse_prohibited": True}


def quarantine_isolated_outputs(config: ProbeConfig, reason: str) -> dict[str, Any]:
    roots = {config.workspace.parent.resolve(), config.staging_dir.parent.resolve(), config.report_dir.parent.resolve()}
    if len(roots) != 1:
        return {"quarantined": False, "reason": "output_paths_do_not_share_isolation_root", "reuse_prohibited": True}
    root = next(iter(roots))
    if not root.exists():
        return {"quarantined": False, "reason": "isolation_root_absent", "reuse_prohibited": True}
    target = root.parent / f"quarantine_{root.name}_{int(time.time())}"
    shutil.move(str(root), str(target))
    return {"quarantined": True, "path": str(target), "reason": reason, "reuse_prohibited": True}


def pre_open_gates(
    config: ProbeConfig,
    expected_executable: str,
    lease: dict[str, Any] | None,
    unknown_writer_marker: Path,
    relevant_processes: list[dict[str, Any]] | None,
    child_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    validate_canonical_route(config)
    assert_fresh_output_paths(config)
    runtime = runtime_gate(runtime_identity(), expected_executable)
    if child_identity is None:
        runtime["passed"] = False
        runtime.setdefault("errors", []).append("child_runtime_provenance_missing")
    elif child_identity != runtime["identity"]:
        runtime["passed"] = False
        runtime.setdefault("errors", []).append("parent_child_runtime_mismatch")
    process = process_writer_gate(relevant_processes)
    writer = wal_writer_gate(config.l2_path, unknown_writer_marker)
    lease_result = lease_gate(lease, config.run_id)
    available = int(psutil.virtual_memory().available)
    memory = memory_gate(available, None, 0, None)
    gates = {"runtime": runtime, "process": process, "writer_wal": writer, "lease": lease_result, "memory": memory}
    if not all(item["passed"] for item in gates.values()):
        raise RuntimeError(f"pre-open gate failed: {gates}")
    return gates


def explicit_projection(schema_columns: Iterable[str]) -> list[str]:
    columns = list(schema_columns)
    if "trade_date" not in columns or "stock_code" not in columns:
        raise RuntimeError("L2 schema missing key columns")
    rejected = [column for column in columns if column in FUTURE_DENYLIST or NEGATIVE_SHIFT_RE.search(column) or FORBIDDEN_COLUMN_RE.search(column)]
    if rejected:
        raise RuntimeError(f"future/label columns rejected before projection: {rejected}")
    qfq_technical = [column for column in columns if "_qfq" in column.lower() and column not in QFQ_PRICE_COLUMNS]
    if len(qfq_technical) != 74:
        raise RuntimeError(f"expected 74 L2 qfq technical columns, got {len(qfq_technical)}")
    ordered = ["trade_date", "stock_code"] + [column for column in columns if column not in {"trade_date", "stock_code"}]
    return ordered


def projection_sql(config: ProbeConfig, columns: list[str]) -> str:
    if not columns or any(column == "*" for column in columns):
        raise RuntimeError("explicit projection is required")
    quoted = ", ".join('"' + column.replace('"', '""') + '"' for column in columns)
    return (
        f"SELECT {quoted} FROM \"{config.l2_table}\" "
        f"WHERE trade_date BETWEEN DATE '{config.target_start[:4]}-{config.target_start[4:6]}-{config.target_start[6:]}' "
        f"AND DATE '{config.target_end[:4]}-{config.target_end[4:6]}-{config.target_end[6:]}' "
        f"AND MOD(ABS(HASH(stock_code)), {config.hash_bucket_count}) = {config.hash_bucket_id}"
    )


def build_plan(config: ProbeConfig, expected_executable: str) -> dict[str, Any]:
    validate_canonical_route(config)
    return {
        "status": "plan_only",
        "execution_started": False,
        "probe_started": False,
        "canonical_l2_opened": False,
        "explicit_execute_probe_required": True,
        "runtime_expected": expected_executable,
        "scope": asdict(config),
        "sql": "NOT_BUILT_UNTIL_READ_ONLY_SCHEMA_GATE",
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "allow_probe_only": False,
        "allow_full_rebuild": False,
        "allow_l3": False,
        "allow_next_layer_continue": False,
    }


def execute_probe(
    config: ProbeConfig,
    expected_executable: str,
    lease: dict[str, Any] | None,
    unknown_writer_marker: Path,
    relevant_processes: list[dict[str, Any]] | None = None,
    child_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Open canonical L2 only after explicit execute and all pre-open gates."""
    if config.execute_probe is not True:
        raise RuntimeError("probe execution requires explicit --execute-probe")
    gates = pre_open_gates(config, expected_executable, lease, unknown_writer_marker, relevant_processes, child_identity)
    if not config.l2_path.is_file():
        raise RuntimeError("canonical L2 file missing")
    before = file_fingerprint(config.l2_path)
    if before["sha256"].lower() != config.expected_l2_sha256.lower():
        raise RuntimeError("canonical L2 SHA drift before open")
    rss_start = psutil.Process().memory_info().rss
    started = time.monotonic()
    config.workspace.mkdir(parents=True)
    config.staging_dir.mkdir(parents=True)
    config.report_dir.mkdir(parents=True)
    connection = duckdb.connect(str(config.l2_path), read_only=True)
    try:
        connection.execute("PRAGMA threads=1")
        connection.execute("PRAGMA memory_limit='3GB'")
        temp_directory = config.staging_dir / "duckdb_temp"
        temp_directory.mkdir()
        connection.execute("SET temp_directory = ?", [str(temp_directory)])
        schema = [row[0] for row in connection.execute(f"DESCRIBE \"{config.l2_table}\"").fetchall()]
        columns = explicit_projection(schema)
        sql = projection_sql(config, columns)
        frame = connection.execute(sql).fetch_arrow_table()
        output_path = config.staging_dir / "l2_probe_output.parquet"
        frame.to_pandas().to_parquet(output_path)
    except Exception:
        quarantine_isolated_outputs(config, "probe_execution_failed")
        raise
    finally:
        connection.close()
    elapsed = time.monotonic() - started
    rss_end = psutil.Process().memory_info().rss
    available_after, recovery_seconds = wait_memory_recovery(gates["memory"]["required_before_bytes"])
    post_memory = memory_gate(gates["memory"]["available_before_bytes"], available_after, max(rss_start, rss_end), recovery_seconds)
    if not post_memory["passed"]:
        quarantine_isolated_outputs(config, "memory_or_recovery_gate_failed")
        raise RuntimeError(f"post-read memory gate failed: {post_memory}")
    after = file_fingerprint(config.l2_path)
    if before != after:
        quarantine_isolated_outputs(config, "canonical_l2_fingerprint_drift")
        raise RuntimeError("canonical L2 fingerprint drift after probe")
    return {"status": "probe_completed_read_only", "source_before": before, "source_after": after, "schema": schema, "projection": columns, "sql": sql, "output_path": str(output_path), "process_rss": {"start": rss_start, "end": rss_end, "available_after": available_after, "elapsed_seconds": elapsed, "recovery_seconds": recovery_seconds}, "gates": gates, "post_memory_gate": post_memory, "reuse_prohibited": True, "approved_for_candidate": False, "approved_for_active": False, "approved_for_downstream": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--expected-runtime", required=True)
    parser.add_argument("--execute-probe", action="store_true")
    parser.add_argument("--lease-nonce")
    parser.add_argument("--child-runtime-json")
    parser.add_argument("--process-state-json")
    parser.add_argument("--unknown-writer-marker")
    args = parser.parse_args(argv)
    config = ProbeConfig(args.run_id, Path(args.workspace), Path(args.staging_dir), Path(args.report_dir), args.execute_probe)
    if not args.execute_probe:
        print(json.dumps(build_plan(config, args.expected_runtime), indent=2, default=str))
        return 0
    if not args.lease_nonce or not args.child_runtime_json:
        raise SystemExit("--execute-probe requires --lease-nonce and --child-runtime-json")
    child_identity = json.loads(Path(args.child_runtime_json).read_text(encoding="utf-8"))
    process_state = []
    if args.process_state_json:
        process_state = json.loads(Path(args.process_state_json).read_text(encoding="utf-8"))
    marker = Path(args.unknown_writer_marker) if args.unknown_writer_marker else config.workspace.parent / f"{config.run_id}.unknown_writer"
    result = execute_probe(config, args.expected_runtime, {"owner_run_id": config.run_id, "lease_nonce": args.lease_nonce, "active": True}, marker, process_state, child_identity)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
