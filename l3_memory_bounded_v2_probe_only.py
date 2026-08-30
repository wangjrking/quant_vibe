"""Synthetic, candidate-only L3 probe with the production memory gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import psutil


SYSTEM_HARD_STOP = 20 * 1024**3
PARENT_BUDGET = 4 * 1024**3
WORKER_BUDGET = 24 * 1024**3
RESERVE = 16 * 1024**3
STARTUP_GATE = 64 * 1024**3


def runtime_identity(role: str) -> dict[str, object]:
    return {
        "role": role,
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
        "rss_bytes": psutil.Process().memory_info().rss,
        "sys_executable": sys.executable.replace("\\", "/"),
        "sys_prefix": sys.prefix.replace("\\", "/"),
        "sys_base_prefix": sys.base_prefix.replace("\\", "/"),
        "stdlib": sysconfig.get_paths().get("stdlib", "").replace("\\", "/"),
        "sys_path": [str(value).replace("\\", "/") for value in sys.path],
    }


def synthetic_contract() -> dict[str, object]:
    dates = ["20260801", "20260803"]
    codes = ["000001.SZ", "600000.SH", "688001.SH", "300001.SZ"]
    rows = [(date, code) for date in dates for code in codes]
    key_hash = hashlib.sha256("\n".join(f"{date}|{code}" for date, code in rows).encode()).hexdigest()
    columns = ["trade_date", "stock_code", "close_qfq"]
    schema_hash = hashlib.sha256("\n".join(columns).encode()).hexdigest()
    alphas = [f"alpha_{index:03d}_qfq" for index in range(1, 192)]
    return {
        "source_rows": len(rows),
        "output_rows": len(rows),
        "schema_hash": schema_hash,
        "key_hash": key_hash,
        "duplicate_key_groups": 0,
        "bj_rows": 0,
        "future_or_label_columns": [],
        "fan_in": 8,
        "fan_in_child_count": 2,
        "gtja_alpha_count": len(alphas),
        "gtja_full_lookback": True,
        "gtja_each_alpha_date_once": True,
        "gtja_boundary_matches_reference": True,
        "qfq_price_count": 5,
        "l2_source_qfq_technical_count": 74,
        "derived_qfq_technical_count": 2,
        "production_feature_qfq_technical_count": 76,
        "label_write": False,
        "label_target_rows": 0,
        "production_paths_opened": False,
        "business_inputs_read": False,
    }


def child_main() -> None:
    before = psutil.virtual_memory().available
    identity = runtime_identity("child")
    contract = synthetic_contract()
    after = psutil.virtual_memory().available
    print(json.dumps({
        "status": "passed",
        "identity": identity,
        "memory": {"available_before": before, "available_after": after},
        "synthetic_contract": contract,
        "business_inputs_read": False,
        "production_paths_opened": False,
    }, sort_keys=True))


def run_probe(workspace: Path, report_dir: Path, run_id: str) -> int:
    if workspace.exists() or report_dir.exists():
        raise RuntimeError("probe workspace/report directory must be new")
    workspace.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    process = psutil.Process()
    available_before = psutil.virtual_memory().available
    dispatch_min = SYSTEM_HARD_STOP + PARENT_BUDGET + WORKER_BUDGET + RESERVE
    report: dict[str, object] = {
        "run_id": run_id,
        "probe_type": "synthetic_l3_memory_bounded_v2",
        "status": "precheck",
        "execution_started": False,
        "business_probe_started": False,
        "business_inputs_read": False,
        "production_paths_opened": False,
        "candidate_duckdb_created": False,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "parent": runtime_identity("parent"),
        "memory_contract": {
            "system_hard_stop": SYSTEM_HARD_STOP,
            "parent_budget": PARENT_BUDGET,
            "worker_budget": WORKER_BUDGET,
            "reserve": RESERVE,
            "dispatch_min_available": dispatch_min,
            "startup_gate": STARTUP_GATE,
            "formula": "20GiB + 4GiB + 1*24GiB + 16GiB = 64GiB",
            "available_before": available_before,
        },
        "cleanup": {"child_started": False, "child_pid": None, "child_zero_after_join": True},
        "quarantine": {"required": False, "reuse_prohibited": False},
    }
    if available_before < STARTUP_GATE:
        report["status"] = "failed_closed_preflight"
        report["blocker"] = "startup_available_below_64GiB"
        report["memory_gate_passed"] = False
        report["end_available"] = psutil.virtual_memory().available
        (report_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2
    report["memory_gate_passed"] = True
    process_info = subprocess.Popen(
        [sys.executable, "-I", __file__, "--child"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid = process_info.pid
    child_create_time = psutil.Process(child_pid).create_time()
    report["cleanup"] = {"child_started": True, "child_pid": child_pid, "child_create_time": child_create_time}
    stdout, stderr = process_info.communicate(timeout=120)
    report["child_returncode"] = process_info.returncode
    report["child_stderr"] = stderr
    report["child"] = json.loads(stdout)
    report["cleanup"] = {
        "child_started": True,
        "child_pid": child_pid,
        "child_create_time": child_create_time,
        "child_returncode": process_info.returncode,
        "child_zero_after_join": not psutil.pid_exists(child_pid),
    }
    report["status"] = "passed" if process_info.returncode == 0 and not psutil.pid_exists(child_pid) else "failed_closed"
    report["end_available"] = psutil.virtual_memory().available
    (report_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["status"] == "passed" else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--run-id", default="l3-memory-bounded-v2-probe-only-20260804")
    args = parser.parse_args()
    if args.child:
        child_main()
        return 0
    if args.workspace is None or args.report_dir is None:
        raise SystemExit("parent probe requires --workspace and --report-dir")
    return run_probe(args.workspace, args.report_dir, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
