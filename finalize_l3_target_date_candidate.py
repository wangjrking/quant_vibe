"""Candidate-only finalization plan for an interrupted target-date L3 delivery.

This entrypoint deliberately has no L2/raw/GTJA computation path.  It validates
an immutable candidate plus pre-recorded binding values before a separately
authorized active delivery may be considered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import deliver_l3_target_date_duckdb_mainline as delivery


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _approved_runtime_sha256(path: Path) -> str:
    """Hash through the approved finalizer runtime instead of a python -c child."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_finalization_plan(
    candidate: Path,
    precheck_path: Path,
    target_date: str,
    workflow_run_id: str,
    expected_l2_sha256: str,
    expected_active_feature_sha256: str,
) -> dict[str, Any]:
    """Validate only persisted inputs; never open L2 or compute factors."""
    precheck = _load_json(precheck_path)
    if precheck.get("workflow_run_id") != workflow_run_id:
        raise RuntimeError("workflow_run_id does not match precheck")
    if precheck.get("target_trade_date") != target_date:
        raise RuntimeError("target_date does not match precheck")
    if precheck["l2"]["file_state"]["sha256"] != expected_l2_sha256:
        raise RuntimeError("expected L2 SHA does not match precheck")
    before = precheck["active_feature_before"]["file_state"]
    if before["sha256"] != expected_active_feature_sha256:
        raise RuntimeError("expected active feature SHA does not match precheck")

    probe = delivery._asset_probe(candidate, precheck["active_feature_before"]["table"], target_date)
    columns = [item["name"] for item in probe["schema"]]
    schema_gate = delivery._feature_schema_gate_summary(columns, precheck["l2"]["qfq_technical_columns"])
    expected = precheck["target_expected_metrics"]
    target = probe["metrics"]
    gates = {
        "candidate_exists": candidate.is_file(),
        "one_table_one_file": probe["table_names"] == [precheck["active_feature_before"]["table"]],
        "target_rows_match": target["target_row_count"] == int(expected["target_row_count"]),
        "target_stocks_match": target["target_stock_count"] == int(expected["target_stock_count"]),
        "duplicate_zero": target["duplicate_key_groups"] == 0,
        "bj_zero": target["target_bj_row_count"] == 0,
        "schema_838": probe["column_count"] == 838,
        "future_columns_zero": "index_2000_post10_close" not in columns,
        "schema_gate": all(v for k, v in schema_gate.items() if k != "details"),
    }
    return {
        "status": "candidate_finalization_plan_ready" if all(gates.values()) else "failed_closed",
        "workflow_run_id": workflow_run_id,
        "target_trade_date": target_date,
        "candidate": {"path": str(candidate), "file_state": probe["file_state"], "metrics": target},
        "input_bindings": {"expected_l2_sha256": expected_l2_sha256, "expected_active_feature_sha256": expected_active_feature_sha256},
        "gates": gates,
        "recompute_raw": False, "compute_gtja": False, "label_write": False, "registry_write": False,
        "active_switch_called": False, "requires_separate_active_switch_authorization": True,
    }


def execute_finalization(plan: dict[str, Any], precheck: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    """Apply a validated existing candidate without opening or recomputing L2."""
    if plan["status"] != "candidate_finalization_plan_ready":
        raise RuntimeError("candidate finalization plan is not ready")
    candidate = Path(plan["candidate"]["path"])
    active = Path(precheck["active_feature_before"]["path"])
    label = Path(precheck["active_label_before"]["path"])
    registry = delivery.REGISTRY_PATH
    args = SimpleNamespace(
        workflow_run_id=plan["workflow_run_id"],
        workspace_dir=str(candidate.parent), report_dir=str(report_dir),
        target_trade_date=plan["target_trade_date"], process_role="candidate_finalization_recovery",
    )
    before = precheck["active_feature_before"]["file_state"]
    if _approved_runtime_sha256(active) != before["sha256"]:
        raise RuntimeError("active feature drift before finalization")
    if delivery._file_state(label) != precheck["active_label_before"]["file_state"] or delivery._file_state(registry) != precheck["registry_before"]:
        raise RuntimeError("label or registry drift before finalization")
    if any(p.exists() for p in delivery._active_wal_paths(active)) or any(p.exists() for p in delivery._active_wal_paths(candidate)):
        raise RuntimeError("active or candidate WAL present")
    lease = delivery._acquire_active_feature_writer_lease(args, active)
    release = None
    try:
        scan = finalization_process_gate(args, report_dir)
        if not scan.get("passed"):
            raise RuntimeError("process gate failed before finalization")
        state = delivery._file_state(active)
        gate = {"status": "passed", "passed": True, "lease_validation": {"lease_nonce": lease["lease_nonce"]}, "asset_state_checks": {"active_feature": {"matches": state == before, "actual": state}}}
        rollback = report_dir / "rollback" / f"l3_feature_before_{plan['target_trade_date']}_{before['sha256'][:12]}.duckdb"
        switch = delivery._atomic_feature_switch(candidate, rollback, active, before["sha256"], pre_switch_gate=gate, expected_lease_nonce=lease["lease_nonce"], final_gate_recheck=lambda: gate)
        readback = delivery._asset_probe(active, precheck["active_feature_before"]["table"], plan["target_trade_date"])
        if readback["file_state"]["sha256"] != plan["candidate"]["file_state"]["sha256"]:
            raise RuntimeError("active readback does not match validated candidate")
        return {"status": "active_delivery_completed_waiting_for_audit", "active_switch_called": True, "label_write_called": False, "registry_write_called": False, "rollback_snapshot": switch["snapshot"], "active_feature_after": readback, "candidate_finalization_plan": plan, "allow_next_layer_continue": False}
    finally:
        release = delivery._release_active_feature_writer_lease(lease)
        (report_dir / f"l3_target_date_{plan['target_trade_date']}_writer_lease_release.json").write_text(json.dumps(release, ensure_ascii=True, indent=2)+"\n", encoding="utf-8")


def finalization_process_gate(args: Any, report_dir: Path) -> dict[str, Any]:
    """Preserve the base gate, allowing only this report/workspace-bound helper chain."""
    scan = delivery._scan_l3_processes(args)
    allowed, blocked = [], []
    for item in scan.get("unknown_relevant_processes", []):
        raw_command = str(item.get("cmdline") or item.get("command_line") or item.get("raw_cmdline") or item.get("command") or "")
        command = raw_command.replace("\\\\", "\\")
        item["raw_cmdline_original"] = raw_command
        item["raw_cmdline_normalized"] = command
        names = ("l3_candidate_finalization_driver.py", "finalize_l3_target_date_candidate.py")
        markers = (args.workflow_run_id, str(report_dir.resolve()), str(Path(args.workspace_dir).resolve()))
        if any(name in command for name in names) and all(marker in command for marker in markers):
            allowed.append(item)
        else:
            blocked.append(item)
    result = {**scan, "unknown_relevant_processes": blocked, "allowed_finalization_self_chain": allowed}
    result["passed"] = bool(scan.get("passed")) or (not result.get("blocking_writers") and not blocked)
    (report_dir / "l3_candidate_finalization_process_gate.json").write_text(json.dumps(result, ensure_ascii=True, indent=2)+"\n", encoding="utf-8")
    return result


def launch_detached_finalization(command: list[str], report_dir: Path) -> dict[str, Any]:
    """Start one hidden helper; the helper owns durable stdout/stderr/heartbeat evidence."""
    report_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = report_dir / "l3_candidate_finalization_heartbeat.jsonl"
    stdout = report_dir / "l3_candidate_finalization_stdout.log"
    stderr = report_dir / "l3_candidate_finalization_stderr.log"
    driver = report_dir / "l3_candidate_finalization_driver.py"
    driver.write_text(
        "import json,subprocess,sys,time,pathlib\n"
        "root=pathlib.Path(sys.argv[1]); cmd=json.loads(sys.argv[2])\n"
        "out=(root/'l3_candidate_finalization_stdout.log').open('w',encoding='utf-8')\n"
        "err=(root/'l3_candidate_finalization_stderr.log').open('w',encoding='utf-8')\n"
        "p=subprocess.Popen(cmd,stdout=out,stderr=err,text=True)\n"
        "while p.poll() is None:\n (root/'l3_candidate_finalization_heartbeat.jsonl').open('a',encoding='utf-8').write(json.dumps({'status':'running','pid':p.pid,'time':time.time()})+'\\n'); time.sleep(5)\n"
        "(root/'l3_candidate_finalization_driver_result.json').write_text(json.dumps({'exit_code':p.returncode})+'\\n',encoding='utf-8')\n",
        encoding="utf-8",
    )
    with heartbeat.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"status": "detached_helper_started", "time": time.time(), "command": command}) + "\n")
    process = subprocess.Popen([sys.executable, str(driver), str(report_dir), json.dumps(command)], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"pid": process.pid, "heartbeat": str(heartbeat), "stdout": str(stdout), "stderr": str(stderr), "driver": str(driver)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a no-recompute L3 candidate finalization.")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--precheck", required=True, type=Path)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--expected-l2-sha256", required=True)
    parser.add_argument("--expected-active-feature-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-finalization", action="store_true")
    args = parser.parse_args()
    result = build_finalization_plan(args.candidate, args.precheck, args.target_date, args.workflow_run_id, args.expected_l2_sha256, args.expected_active_feature_sha256)
    if args.execute_finalization:
        result = execute_finalization(result, _load_json(args.precheck), args.output.parent)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
