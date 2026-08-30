from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "quant" / "data_file" / "reports" / "l3_memory_bounded_v2_resource_probe_20260804"
RUNTIME_ROOT = PROJECT_ROOT / "runtime_candidates" / "my_quant_copy_20260804"
STARTUP_GATE_BYTES = 34 * 1024**3
CHILD_CODE = r'''
import json, os, sys, tempfile
from pathlib import Path
import duckdb
sys.path.insert(0, r"D:/work/quant/quant_mcp/quant/main")
import rebuild_l3_memory_bounded_v2_candidate as candidate

root = Path(sys.argv[1])
workspace = root / "synthetic_workspace"
workspace.mkdir(parents=True, exist_ok=True)
for index in range(10):
    shard = workspace / f"shard_{index:02d}.duckdb"
    with duckdb.connect(str(shard)) as conn:
        candidate._configure_worker_duckdb(conn, "3GB", workspace / "spill" / f"shard_{index:02d}")
        conn.execute("CREATE TABLE raw_factor(trade_date VARCHAR, stock_code VARCHAR, value DOUBLE)")
        conn.executemany("INSERT INTO raw_factor VALUES (?, ?, ?)", [(f"2026080{index % 4 + 1}", f"00000{index}.SZ", float(index))])
result = candidate._merge_shards(
    sorted(workspace.glob("shard_*.duckdb")),
    workspace / "merged.duckdb",
    "raw_factor",
    "raw_factor",
    workspace=workspace,
    policy=candidate._streaming_policy(),
)
print(json.dumps({"result": result, "pid": os.getpid(), "sys_executable": sys.executable, "sys_prefix": sys.prefix, "sys_base_prefix": sys.base_prefix, "duckdb_version": duckdb.__version__}, sort_keys=True))
'''


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"l3-memory-bounded-v2-resource-probe-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    workspace = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=str(REPORT_DIR)))
    parent = psutil.Process(os.getpid())
    before = psutil.virtual_memory()
    report: dict = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "started",
        "probe_scope": "synthetic_resource_profile_only",
        "business_inputs_read": False,
        "production_paths_opened": False,
        "candidate_written": False,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "full_rebuild_started": False,
        "L3_authorized": False,
        "resource_contract": {
            "startup_gate_bytes": STARTUP_GATE_BYTES,
            "system_hard_stop_bytes": 20 * 1024**3,
            "parent_budget_bytes": 4 * 1024**3,
            "worker_budget_bytes": 6 * 1024**3,
            "reserve_bytes": 4 * 1024**3,
            "duckdb_memory_limit": "3GB",
            "workers": 1,
            "fan_in": 8,
        },
        "parent": {
            "pid": parent.pid,
            "create_time": parent.create_time(),
            "rss_before": parent.memory_info().rss,
            "available_before": int(before.available),
            "total_physical": int(before.total),
        },
        "workspace": str(workspace),
        "runtime": {
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "sys_base_prefix": sys.base_prefix,
            "runtime_root": str(RUNTIME_ROOT),
            "runtime_expected": str(RUNTIME_ROOT / "python.exe"),
        },
    }
    child: subprocess.Popen[str] | None = None
    try:
        if int(before.available) < STARTUP_GATE_BYTES:
            report.update({"status": "fail_closed", "reason": "candidate_resource_gate_below_34GiB", "child_started": False})
            write_json(REPORT_DIR / "probe_report.json", report)
            return 2
        child = subprocess.Popen(
            [sys.executable, "-c", CHILD_CODE, str(workspace)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT / "quant" / "main"),
        )
        child_process = psutil.Process(child.pid)
        child_create_time = child_process.create_time()
        child_rss_peak = 0
        while child.poll() is None:
            try:
                child_rss_peak = max(child_rss_peak, child_process.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.2)
        stdout, stderr = child.communicate()
        after = psutil.virtual_memory()
        child_payload = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
        report.update({
            "status": "passed" if child.returncode == 0 else "fail_closed",
            "reason": None if child.returncode == 0 else "synthetic_child_failed",
            "child_started": True,
            "child_returncode": child.returncode,
            "child": {"pid": child.pid, "create_time": child_create_time, "rss_peak": child_rss_peak, "payload": child_payload},
            "parent_after": {"rss_after": parent.memory_info().rss, "available_after": int(after.available)},
            "stderr": stderr[-4000:],
        })
        return 0 if child.returncode == 0 else 3
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        shutil.rmtree(workspace, ignore_errors=True)
        report["workspace_removed"] = not workspace.exists()
        report["process_zero_verified"] = child is None or child.poll() is not None
        report["completed_at_epoch"] = time.time()
        write_json(REPORT_DIR / "probe_report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
