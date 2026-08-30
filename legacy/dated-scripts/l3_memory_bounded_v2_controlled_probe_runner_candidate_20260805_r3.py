"""Candidate-only canonical L2 probe runner (revision 3).

This module is intentionally conservative: the default CLI path is plan-only,
and the probe path owns its lease, child-runtime evidence, process scan and
isolated output root.  It never writes production assets.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import psutil

from l3_active_writer_lease import (
    acquire_active_l3_writer_lease,
    release_active_l3_writer_lease,
    validate_active_l3_writer_lease,
)


ROOT = Path("D:/work/quant/quant_mcp")
CANONICAL_L2 = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
CANONICAL_TABLE = "STOCK_DAILY_DATA"
EXPECTED_L2_SHA256 = "34a88582b2e0e53fa5ececadf8500673ad44259dd7d2bc8e44d3c689c7905e4b"
RUNTIME = ROOT / "runtime_candidates/my_quant_copy_20260804/python.exe"
LINEAGE_SOURCE = ROOT / "quant/main/data_process_module.py"
CONTRACT_SOURCE = ROOT / "quant/main/feature_contract_v2_candidate.py"
START = "20260803"
END = "20260804"
BUCKET_COUNT = 16
BUCKET_ID = 3
GIB = 1024**3
REQUIRED_AVAILABLE = 34 * GIB
RECOVERY_TIMEOUT = 120
FUTURE_DENYLIST = {"index_2000_post10_close"}
QFQ_PRICES = {"open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"}
PROBE_SCOPE = {"trade_date_start": START, "trade_date_end": END, "hash_bucket_count": BUCKET_COUNT, "hash_bucket_id": BUCKET_ID}
CONSUMED_GRANT_DIR = ROOT / "quant/data_file/runtime/agent_workspaces/factor-agent/consumed_probe_grants"
AUDIT_RELEASE_REGISTRY = ROOT / "quant/data_file/runtime/audit_release/l3_controlled_probe_release.json"
DEPENDENCY_SHA256 = {
    str(ROOT / "quant/main/l3_active_writer_lease.py"): "6abea9a0e7e2e270f484b27d254c10c88797637124b927ad503bcb31c794a50d",
    str(LINEAGE_SOURCE): "415493f9a5bbb11a65795f8f3e2fae75cc6d628b72d6e8e64877c3807eda1d87",
    str(CONTRACT_SOURCE): "d2a4b91084a1fbdf155e5ae8e1a5de1127cafefb677512c03c0307238b5515ab",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256(path)}


def runtime_identity() -> dict[str, Any]:
    import unittest
    return {
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "stdlib": str(Path(os.__file__).resolve()),
        "unittest": str(Path(unittest.__file__).resolve()),
        "duckdb": str(Path(duckdb.__file__).resolve()),
        "duckdb_version": duckdb.__version__,
    }


def runtime_gate(identity: dict[str, Any]) -> None:
    expected = str(RUNTIME.resolve())
    root = str(RUNTIME.parent.resolve()).lower()
    paths = [identity[k] for k in ("executable", "prefix", "base_prefix", "stdlib", "unittest", "duckdb")]
    if identity["executable"] != expected or identity["prefix"].lower() != root or identity["base_prefix"].lower() != root:
        raise RuntimeError("runtime provenance mismatch")
    if any("conda" in str(p).lower() or not str(p).lower().startswith(root) for p in paths):
        raise RuntimeError("runtime provenance outside isolated runtime")


def _child_identity() -> dict[str, Any]:
    return runtime_identity()


def spawn_and_verify_child(root: Path, run_id: str) -> dict[str, Any]:
    output = root / "child_runtime_identity.json"
    command = [
        str(RUNTIME),
        str(Path(__file__).resolve()),
        "--run-id", run_id,
        "--isolated-root", str(root.resolve()),
        "--emit-runtime-identity", str(output),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), check=True, capture_output=True, text=True, timeout=30)
    identity = json.loads(output.read_text(encoding="utf-8"))
    runtime_gate(identity)
    if identity["pid"] <= 0 or completed.returncode != 0:
        raise RuntimeError("child runtime evidence invalid")
    output.unlink(missing_ok=True)
    return identity


def acquire_lease(root: Path, run_id: str) -> dict[str, Any]:
    lease_path = root / "exclusive_probe.lease.json"
    nonce = uuid.uuid4().hex
    payload = {"run_id": run_id, "nonce": nonce, "pid": os.getpid(), "create_time": psutil.Process().create_time()}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(lease_path), flags)
    except FileExistsError as exc:
        raise RuntimeError("exclusive lease already exists") from exc
    try:
        os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
    finally:
        os.close(fd)
    return {"path": str(lease_path), **payload}


def release_lease(lease: dict[str, Any]) -> None:
    path = Path(lease["path"])
    if not path.is_file():
        raise RuntimeError("lease disappeared before release")
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("nonce") != lease["nonce"] or current.get("pid") != os.getpid():
        raise RuntimeError("lease ownership changed")
    path.unlink()


def ancestor_pids() -> set[int]:
    """Return only the current process and its real parent chain."""
    excluded = {os.getpid()}
    try:
        process = psutil.Process()
        while True:
            parent = process.parent()
            if parent is None:
                break
            excluded.add(int(parent.pid))
            process = parent
    except (psutil.Error, OSError):
        pass
    return excluded


def scan_processes() -> list[dict[str, Any]]:
    excluded_pids = ancestor_pids()
    blockers = []
    for proc in psutil.process_iter(["pid", "create_time", "cmdline"]):
        if proc.info["pid"] in excluded_pids:
            continue
        cmd = " ".join(proc.info.get("cmdline") or []).lower()
        if "l3_memory_bounded" in cmd or str(CANONICAL_L2).lower() in cmd or "l2_stock_daily_data.duckdb" in cmd:
            blockers.append({"pid": proc.info["pid"], "create_time": proc.info.get("create_time"), "cmdline": cmd})
    return blockers


def wal_gate() -> None:
    for suffix in (".wal", "-wal"):
        p = Path(str(CANONICAL_L2) + suffix)
        if p.is_file() and p.stat().st_size:
            raise RuntimeError(f"canonical L2 WAL present: {p}")


def dependency_gate() -> dict[str, str]:
    observed = {path: sha256(Path(path)) for path in DEPENDENCY_SHA256}
    expected = {path: str(value).lower() for path, value in DEPENDENCY_SHA256.items()}
    normalized = {path: value.lower() for path, value in observed.items()}
    if normalized != expected:
        raise RuntimeError(f"frozen dependency hash mismatch: {observed}")
    return observed


def memory_snapshot() -> dict[str, int]:
    """Capture the process/system values needed by the recovery gate."""
    process = psutil.Process()
    return {
        "rss_bytes": int(process.memory_info().rss),
        "available_bytes": int(psutil.virtual_memory().available),
        "captured_at_ns": time.time_ns(),
    }


def recovery_gate(before: dict[str, int], after: dict[str, int]) -> dict[str, Any]:
    available_recovered = after["available_bytes"] >= REQUIRED_AVAILABLE
    return {
        "status": "passed" if available_recovered else "failed_closed",
        "required_available_bytes": REQUIRED_AVAILABLE,
        "available_before_bytes": before["available_bytes"],
        "available_after_bytes": after["available_bytes"],
        "rss_before_bytes": before["rss_bytes"],
        "rss_after_bytes": after["rss_bytes"],
        "recovery_timeout_seconds": RECOVERY_TIMEOUT,
        "recovered": available_recovered,
    }


def lineage_gate(columns: list[str]) -> dict[str, Any]:
    source_text = LINEAGE_SOURCE.read_text(encoding="utf-8")
    contract_text = CONTRACT_SOURCE.read_text(encoding="utf-8")
    if "shift(-" not in source_text:
        raise RuntimeError("lineage source unexpectedly lacks negative-shift evidence")
    if "index_2000_post10_close" not in contract_text and "index_2000_post10_close" not in source_text:
        raise RuntimeError("future denylist evidence missing")
    tree = ast.parse(source_text)
    negative_shift_nodes = []
    mapped_future_columns = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "shift" or not call.args:
            continue
        arg = call.args[0]
        if not (isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)):
            continue
        negative_shift_nodes.append(node.lineno)
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                mapped_future_columns[target.slice.value] = f"shift({ast.unparse(arg)})"
    rejected = [c for c in columns if c in mapped_future_columns or c in FUTURE_DENYLIST]
    return {
        "source_files": [str(LINEAGE_SOURCE), str(CONTRACT_SOURCE)],
        "source_sha256": {str(LINEAGE_SOURCE): sha256(LINEAGE_SOURCE), str(CONTRACT_SOURCE): sha256(CONTRACT_SOURCE)},
        "negative_shift_nodes": len(negative_shift_nodes),
        "negative_shift_column_map": mapped_future_columns,
        "rejected_future_columns": rejected,
        "future_columns": [],
        "status": "passed" if not rejected else "failed_closed",
    }


def projection(columns: list[str]) -> list[str]:
    if "trade_date" not in columns or "stock_code" not in columns:
        raise RuntimeError("canonical L2 key columns missing")
    lineage = lineage_gate(columns)
    if lineage["rejected_future_columns"]:
        raise RuntimeError(f"future lineage rejected from projection: {lineage['rejected_future_columns']}")
    if any(c in FUTURE_DENYLIST or c.lower().startswith("label") for c in columns):
        raise RuntimeError("future/label column in canonical L2 projection")
    qfq_technical = [c for c in columns if "_qfq" in c.lower() and c not in QFQ_PRICES]
    if len(qfq_technical) != 74:
        raise RuntimeError(f"expected 74 source qfq technical columns, got {len(qfq_technical)}")
    return ["trade_date", "stock_code", *[c for c in columns if c not in {"trade_date", "stock_code"}]]


def sql(columns: list[str]) -> str:
    quoted = ", ".join('"' + c.replace('"', '""') + '"' for c in columns)
    return f"SELECT {quoted} FROM \"{CANONICAL_TABLE}\" WHERE trade_date BETWEEN '20260803' AND '20260804' AND MOD(ABS(HASH(stock_code)), 16) = 3"


def validate_output(frame: Any, columns: list[str], source_rows: int | None = None, source_key_count: int | None = None) -> dict[str, Any]:
    if list(frame.columns) != columns:
        raise RuntimeError("output schema/order mismatch")
    if frame[["trade_date", "stock_code"]].duplicated().any():
        raise RuntimeError("duplicate key in probe output")
    if any(str(code).startswith("BJ") for code in frame["stock_code"].astype(str)):
        raise RuntimeError("BJ code in probe output")
    dates = frame["trade_date"].astype("datetime64[ns]").dt.strftime("%Y%m%d")
    if dates.min() < START or dates.max() > END:
        raise RuntimeError("output date scope mismatch")
    qfq_technical = [c for c in columns if "_qfq" in c.lower() and c not in QFQ_PRICES]
    null_profile = {str(c): int(frame[c].isna().sum()) for c in columns}
    output_rows = int(len(frame))
    target_scope_rows = output_rows
    if source_rows is not None and output_rows != int(source_rows):
        raise RuntimeError(f"source/output row mismatch: {source_rows} != {output_rows}")
    output_key_count = int(frame["stock_code"].nunique())
    if source_key_count is not None and output_key_count != int(source_key_count):
        raise RuntimeError(f"source/output key-domain mismatch: {source_key_count} != {output_key_count}")
    return {
        "source_rows": int(source_rows) if source_rows is not None else target_scope_rows,
        "output_rows": output_rows,
        "target_scope_rows": target_scope_rows,
        "key_domain_closed": source_key_count is None or output_key_count == int(source_key_count),
        "source_key_count": int(source_key_count) if source_key_count is not None else output_key_count,
        "output_key_count": output_key_count,
        "columns": len(columns),
        "duplicate": 0,
        "BJ": 0,
        "qfq_source_technical": len(qfq_technical),
        "qfq_contract": {
            "l2_source_technical": 74,
            "production_feature_technical": 76,
            "derived_technical": 2,
            "gtja_qfq": 191,
            "status": "l2_input_contract_verified; downstream_feature_counts_require_l3_output_audit",
        },
        "future_columns": [],
        "label_columns": [],
        "source_limited_null_transmission": {"status": "verified_on_projected_frame", "null_profile": null_profile},
        "source_null_profile": null_profile,
        "output_null_profile": null_profile,
    }


def quarantine(root: Path, reason: str) -> dict[str, Any]:
    if not root.exists():
        return {"quarantined": False, "reuse_prohibited": True, "reason": reason}
    target = root.parent / f"quarantine_{root.name}_{int(time.time())}"
    shutil.move(str(root), str(target))
    return {"quarantined": True, "path": str(target), "reuse_prohibited": True, "reason": reason}


def plan(run_id: str) -> dict[str, Any]:
    return {"status": "plan_only", "run_id": run_id, "canonical_l2_opened": False, "execution_started": False, "allow_probe_only": False, "allow_full_rebuild": False, "allow_l3": False, "reuse_prohibited": True}


def consume_grant(grant_nonce: str) -> None:
    if not grant_nonce or any(ch not in "0123456789abcdefABCDEF-_" for ch in grant_nonce):
        raise RuntimeError("invalid grant nonce")
    CONSUMED_GRANT_DIR.mkdir(parents=True, exist_ok=True)
    target = CONSUMED_GRANT_DIR / f"{grant_nonce}.consumed"
    try:
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError("probe grant replay detected") from exc
    os.write(fd, str(time.time_ns()).encode("ascii")); os.close(fd)


def audit_release_gate(authorization: dict[str, Any], run_id: str, root: Path) -> dict[str, Any]:
    if not AUDIT_RELEASE_REGISTRY.is_file():
        raise RuntimeError("independent audit release registry missing")
    release = json.loads(AUDIT_RELEASE_REGISTRY.read_text(encoding="utf-8"))
    if release.get("allow_probe_only") is not True or release.get("audit_verdict") != "passed/P2":
        raise RuntimeError("independent audit release does not allow probe")
    if release.get("owner_approved") is not True or release.get("approval_source") != "commander-owner":
        raise RuntimeError("owner approval missing from audit release")
    if str(release.get("runner_sha256", "")).lower() != sha256(Path(__file__).resolve()).lower():
        raise RuntimeError("audit release runner hash mismatch")
    if release.get("run_id") != run_id or release.get("isolated_root") != str(root.resolve()):
        raise RuntimeError("audit release run/root binding mismatch")
    if release.get("grant_id") != authorization.get("grant_id") or release.get("grant_nonce") != authorization.get("grant_nonce"):
        raise RuntimeError("audit release grant binding mismatch")
    if float(release.get("expires_at_epoch", 0)) <= time.time():
        raise RuntimeError("audit release expired")
    return release


def validate_probe_authorization(authorization: dict[str, Any] | None, root: Path, run_id: str) -> dict[str, Any]:
    if not authorization or authorization.get("allow_probe_only") is not True:
        raise RuntimeError("probe authorization token required")
    if authorization.get("scope") != PROBE_SCOPE:
        raise RuntimeError("probe authorization scope mismatch")
    if str(authorization.get("runner_sha256", "")).lower() != sha256(Path(__file__).resolve()).lower():
        raise RuntimeError("probe authorization runner hash mismatch")
    if authorization.get("grant_id") == "" or not authorization.get("grant_id"):
        raise RuntimeError("probe authorization grant id missing")
    if authorization.get("owner_approved") is not True or authorization.get("approval_source") != "commander-owner":
        raise RuntimeError("explicit owner approval required")
    if authorization.get("run_id") != run_id or Path(authorization.get("isolated_root", "")).resolve() != root.resolve():
        raise RuntimeError("probe grant run/root binding mismatch")
    if float(authorization.get("expires_at_epoch", 0)) <= time.time():
        raise RuntimeError("probe grant expired")
    if not authorization.get("grant_nonce"):
        raise RuntimeError("probe grant nonce missing")
    release = audit_release_gate(authorization, run_id, root)
    consume_grant(str(authorization["grant_nonce"]))
    return {"grant_id": authorization["grant_id"], "approval_source": authorization["approval_source"], "grant_nonce": authorization["grant_nonce"], "audit_release_id": release.get("release_id")}


def execute(run_id: str, root: Path, authorization: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_probe_authorization(authorization, root, run_id)
    if root.exists():
        raise RuntimeError("isolated root must be absent")
    root.mkdir(parents=True)
    lease = None
    before = None
    try:
        runtime_gate(runtime_identity())
        dependencies = dependency_gate()
        child = spawn_and_verify_child(root, run_id)
        blockers = scan_processes()
        if blockers:
            raise RuntimeError(f"writer/process blockers: {blockers}")
        wal_gate()
        lease = acquire_active_l3_writer_lease(
            workflow_run_id=run_id,
            workspace=root / "workspace",
            report_dir=root / "report",
            process_role="controlled_read_only_probe",
        )
        before = fingerprint(CANONICAL_L2)
        if before["sha256"].lower() != EXPECTED_L2_SHA256:
            raise RuntimeError("canonical L2 SHA mismatch")
        memory_before = memory_snapshot()
        available_before = memory_before["available_bytes"]
        if available_before < REQUIRED_AVAILABLE:
            raise RuntimeError("candidate memory gate below 34GiB")
        workspace, staging, report = (root / n for n in ("workspace", "staging", "report"))
        workspace.mkdir(); staging.mkdir(); report.mkdir()
        con = duckdb.connect(str(CANONICAL_L2), read_only=True)
        try:
            con.execute("PRAGMA threads=1")
            con.execute("PRAGMA memory_limit='3GB'")
            con.execute("SET temp_directory = ?", [str(staging / "duckdb_temp")])
            schema = [r[0] for r in con.execute(f'DESCRIBE "{CANONICAL_TABLE}"').fetchall()]
            cols = projection(schema)
            source_rows, source_key_count = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT stock_code) FROM ({sql(cols)})"
            ).fetchone()
            frame = con.execute(sql(cols)).fetch_df()
        finally:
            con.close()
        quality = validate_output(frame, cols, int(source_rows), int(source_key_count))
        output = staging / "l2_probe_output.parquet"
        frame.to_parquet(output, index=False)
        readback = pd.read_parquet(output)
        independent_quality = validate_output(readback, cols, int(source_rows), int(source_key_count))
        if quality["source_null_profile"] != independent_quality["output_null_profile"]:
            raise RuntimeError("source/readback null profile mismatch")
        process_blockers_after = scan_processes()
        if process_blockers_after:
            raise RuntimeError(f"post-read writer/process blockers: {process_blockers_after}")
        wal_gate()
        after = fingerprint(CANONICAL_L2)
        if before != after:
            raise RuntimeError("canonical L2 fingerprint drift")
        memory_after = memory_snapshot()
        recovery = recovery_gate(memory_before, memory_after)
        if recovery["status"] != "passed":
            raise RuntimeError("memory recovery gate failed")
        result = {
            "status": "probe_completed_read_only",
            "run_id": run_id,
            "source_before": before,
            "source_after": after,
            "child_runtime": child,
            "dependency_sha256": dependencies,
            "lease": lease,
            "quality": quality,
            "independent_readback_quality": independent_quality,
            "lineage": lineage_gate(cols),
            "output": str(output),
            "output_format": "parquet",
            "memory": recovery,
            "process_gate": {"pre_read": "passed", "post_read": "passed", "residual_processes": 0},
            "wal_gate": "passed",
            "source_limited_null_transmission": True,
            "approved_for_candidate": False,
            "approved_for_active": False,
            "approved_for_downstream": False,
            "reuse_prohibited": True,
            "active_switch": False,
            "label_write": False,
            "registry_change": False,
            "allow_l3": False,
        }
        release_active_l3_writer_lease(lease); lease = None
        return result
    except Exception as exc:
        if lease is not None:
            try: release_active_l3_writer_lease(lease)
            except Exception: pass
        quarantine_info = quarantine(root, str(exc))
        raise RuntimeError(json.dumps({"error": str(exc), "quarantine": quarantine_info, "reuse_prohibited": True})) from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--isolated-root", required=True)
    p.add_argument("--execute-probe", action="store_true")
    p.add_argument("--grant-json")
    p.add_argument("--emit-runtime-identity")
    args = p.parse_args(argv)
    if args.emit_runtime_identity:
        Path(args.emit_runtime_identity).write_text(json.dumps(_child_identity(), sort_keys=True), encoding="utf-8")
        return 0
    if not args.execute_probe:
        print(json.dumps(plan(args.run_id), indent=2))
        return 0
    if not args.grant_json:
        raise SystemExit("--execute-probe requires --grant-json")
    authorization = json.loads(Path(args.grant_json).read_text(encoding="utf-8"))
    result = execute(args.run_id, Path(args.isolated_root), authorization)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
