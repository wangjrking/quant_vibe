# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/l3_full_processing_20260717_candidate_attempt5"
PRECHECK_PATH = REPORT_DIR / "l3_full_processing_20260717_precheck.json"
L2_BASELINE_PATH = ROOT / "quant/data_file/reports/l2_full_processing_20260717_validation.json"
ORIGINAL_WORKSPACE = ROOT / "quant/data_file/runtime/agent_workspaces/factor-agent/work/l3_full_processing_20260717_candidate_attempt5"
QUARANTINE = ROOT / "quant/data_file/runtime/agent_workspaces/factor-agent/quarantine/l3_full_processing_20260717_candidate_attempt5_merge_memory_gate_20260719_0022"
INVENTORY_PATH = REPORT_DIR / "attempt5_complete_quarantine_inventory_20260719.json"
INCIDENT_PATH = REPORT_DIR / "attempt5_merge_memory_incident_closeout_20260719.json"
INCIDENT_MD_PATH = REPORT_DIR / "attempt5_merge_memory_incident_closeout_20260719.md"
MEMORY_PATH = REPORT_DIR / "attempt5_merge_memory_timeline_20260719.json"
MANIFEST_PATH = QUARANTINE / "quarantine_manifest.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def compare_state(name: str, baseline: dict[str, Any]) -> dict[str, Any]:
    current = file_state(Path(baseline["path"]))
    checks = {
        "size_unchanged": current["size_bytes"] == int(baseline["size_bytes"]),
        "mtime_unchanged": current["mtime_ns"] == int(baseline["mtime_ns"]),
        "sha256_unchanged": current["sha256"] == str(baseline["sha256"]),
    }
    return {"name": name, "baseline": baseline, "current": current, "checks": checks, "unchanged": all(checks.values())}


def attempt_processes() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline", "create_time"]):
        if process.pid == os.getpid():
            continue
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if "l3_full_processing_20260717_candidate_attempt5" not in cmdline:
            continue
        if not str(process.info.get("name") or "").lower().startswith("python"):
            continue
        found.append({
            "pid": process.pid,
            "ppid": process.info.get("ppid"),
            "name": process.info.get("name"),
            "exe": process.info.get("exe"),
            "cmdline": cmdline,
            "create_time": process.info.get("create_time"),
        })
    return found


def completion_hashes() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    completion_dir = QUARANTINE / "logs/raw_buckets"
    for path in sorted(completion_dir.glob("raw_bucket_*.json")):
        if "failure" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        output_name = Path(payload["path"]).name
        result[output_name] = {
            "sha256": payload["output_sha256"],
            "size_bytes": int(payload["output_size_bytes"]),
            "completion_path": str(path),
            "task_id": payload["task_id"],
        }
    return result


def build_inventory() -> dict[str, Any]:
    embedded = completion_hashes()
    files: list[dict[str, Any]] = []
    total_size = 0
    embedded_count = 0
    computed_count = 0
    for path in sorted(item for item in QUARANTINE.rglob("*") if item.is_file() and item != MANIFEST_PATH):
        relative = path.relative_to(QUARANTINE)
        stat = path.stat()
        total_size += stat.st_size
        row: dict[str, Any] = {
            "relative_path": str(relative),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        embedded_row = embedded.get(path.name) if relative.parts and relative.parts[0] == "raw_shards" else None
        if embedded_row:
            if stat.st_size != embedded_row["size_bytes"]:
                raise RuntimeError(f"completion size mismatch for {path}")
            row.update({
                "sha256": embedded_row["sha256"],
                "sha256_source": "validated_completion_record",
                "completion_path": embedded_row["completion_path"],
                "task_id": embedded_row["task_id"],
            })
            embedded_count += 1
        else:
            row.update({"sha256": sha256_file(path), "sha256_source": "computed_after_quarantine"})
            computed_count += 1
        files.append(row)

    spot_checks = []
    for name in ("raw_bucket_0000.duckdb", "raw_bucket_0127.duckdb", "raw_bucket_0255.duckdb"):
        path = QUARANTINE / "raw_shards" / name
        expected = embedded[name]["sha256"]
        actual = sha256_file(path)
        spot_checks.append({"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "passed": expected == actual})
    if not all(item["passed"] for item in spot_checks):
        raise RuntimeError("raw shard quarantine spot hash check failed")

    inventory = {
        "schema_version": 1,
        "status": "complete_quarantine_inventory",
        "generated_at": now_iso(),
        "attempt": "attempt-5",
        "original_workspace": str(ORIGINAL_WORKSPACE),
        "original_workspace_exists": ORIGINAL_WORKSPACE.exists(),
        "quarantine_path": str(QUARANTINE),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "raw_shard_count": sum(1 for item in files if item["relative_path"].startswith("raw_shards\\")),
        "completion_count": sum(1 for item in files if item["relative_path"].startswith("logs\\raw_buckets\\") and "failure" not in item["relative_path"]),
        "failure_file_count": sum(1 for item in files if "failure" in item["relative_path"]),
        "wal_count": sum(1 for item in files if item["relative_path"].lower().endswith(".wal")),
        "embedded_completion_hash_count": embedded_count,
        "computed_hash_count": computed_count,
        "raw_shard_spot_hash_checks": spot_checks,
        "files": files,
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
        "reuse_prohibited": True,
        "in_place_resume_prohibited": True,
    }
    atomic_json(INVENTORY_PATH, inventory)
    return inventory


def main() -> None:
    if ORIGINAL_WORKSPACE.exists():
        raise RuntimeError(f"original attempt-5 workspace still exists: {ORIGINAL_WORKSPACE}")
    if not QUARANTINE.is_dir():
        raise RuntimeError(f"attempt-5 quarantine missing: {QUARANTINE}")
    processes = attempt_processes()
    if processes:
        raise RuntimeError(f"attempt-5 Python processes still alive: {processes}")

    precheck = json.loads(PRECHECK_PATH.read_text(encoding="utf-8"))
    l2_baseline_report = json.loads(L2_BASELINE_PATH.read_text(encoding="utf-8"))
    l2_baseline = l2_baseline_report["active_asset"]["file_state"]
    active_before = precheck["active_before"]
    states = {
        "l2": compare_state("L2", l2_baseline),
        "feature": compare_state("active_feature", active_before["feature"]["file_state"]),
        "label": compare_state("active_label", active_before["label"]["file_state"]),
        "registry": compare_state("production_registry", active_before["registry"]),
    }
    if not all(item["unchanged"] for item in states.values()):
        raise RuntimeError(f"active state drift detected: {states}")

    inventory = build_inventory()
    temp_entries = [item for item in inventory["files"] if item["relative_path"].startswith("staging_raw_factor.duckdb.tmp")]
    gtja_entries = [item for item in inventory["files"] if "gtja" in item["relative_path"].lower()]
    feature_entries = [item for item in inventory["files"] if "feature_candidate" in item["relative_path"].lower()]
    label_entries = [item for item in inventory["files"] if "label_candidate" in item["relative_path"].lower()]
    pair_plan_files = list(REPORT_DIR.glob("*pair*change*plan*.json"))

    memory_timeline = {
        "schema_version": 1,
        "status": "incident_memory_timeline",
        "generated_at": now_iso(),
        "policy": {
            "parent_rss_budget_bytes": 4 * 1024**3,
            "system_hard_stop_bytes": 20 * 1024**3,
        },
        "observations": [
            {
                "time": "2026-07-19T00:20:05.2660059+08:00",
                "phase": "raw_merge",
                "orchestrator_pid": 31880,
                "working_set_gib": 52.86,
                "private_gib": 55.02,
                "system_available_gib": 26.46,
                "temp_size_bytes": 53341335552,
                "parent_rss_budget_failed": True,
                "system_hard_stop_crossed": False,
            },
            {
                "time": "2026-07-19T00:22:18.7957701+08:00",
                "phase": "after_process_tree_stop",
                "attempt5_python_process_count": 0,
                "system_available_gib": 79.50,
            },
        ],
        "root_cause": "_merge_shards runs ATTACH/INSERT/CREATE INDEX in the long-lived orchestrator outside ShortLivedTaskScheduler memory sampling, so the 4 GiB parent RSS gate was not enforced during raw merge.",
    }
    atomic_json(MEMORY_PATH, memory_timeline)

    incident = {
        "schema_version": 1,
        "status": "failed_closed_quarantined",
        "generated_at": now_iso(),
        "task_id": "incremental-trading-signal-20260717-L3-full-processing-candidate-staging-attempt-5",
        "target_trade_date": "20260717",
        "failure_stage": "raw_merge",
        "authoritative_blocker": "parent_rss_budget_exceeded_during_raw_merge",
        "root_cause": memory_timeline["root_cause"],
        "process_closeout": {
            "stopped_process_ids": [31880, 20668, 32260],
            "attempt5_python_process_count_after_stop": len(processes),
            "memory_recovered_to_gib": 79.50,
        },
        "raw_stage": {
            "completed_buckets": 256,
            "failure_files": inventory["failure_file_count"],
            "raw_shard_count": inventory["raw_shard_count"],
            "raw_merge_completed": False,
            "temp_entries": temp_entries,
        },
        "downstream_stage_absence": {
            "gtja_generated": bool(gtja_entries),
            "feature_candidate_generated": bool(feature_entries),
            "label_candidate_generated": bool(label_entries),
            "pair_change_plan_generated": bool(pair_plan_files),
            "pair_change_called": False,
            "active_switch_called": False,
        },
        "quarantine": {
            "path": str(QUARANTINE),
            "inventory_path": str(INVENTORY_PATH),
            "file_count": inventory["file_count"],
            "total_size_bytes": inventory["total_size_bytes"],
            "wal_count": inventory["wal_count"],
            "approved_for_candidate": False,
            "approved_for_active": False,
            "approved_for_downstream": False,
            "reuse_prohibited": True,
            "in_place_resume_prohibited": True,
        },
        "active_state_validation": states,
        "minimum_remediation_proposal": [
            "Move raw shard merge into a short-lived dedicated spawn process with the same RSS/system-available sampling and fail-closed cleanup contract.",
            "Avoid one long ATTACH/INSERT/CREATE INDEX transaction; merge bounded shard batches into immutable intermediate DuckDB files and compact them with bounded memory.",
            "Set DuckDB memory_limit and threads=1 explicitly for merge and index phases, and emit per-batch rows, bytes, RSS, available-memory and recovery evidence.",
            "Add focused tests for parent RSS overflow during merge, system hard-stop, interrupted merge quarantine, and active asset immutability.",
            "Any attempt-6 must use a new run-id/workspace and requires architect plus audit approval; attempt-5 evidence is prohibited as execution or validation input.",
        ],
        "ready_for_incident_audit_review": True,
        "allow_next_layer_continue": False,
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
        "reuse_prohibited": True,
        "boundaries": {
            "attempt6_started": False,
            "scheduler_or_merge_modified": False,
            "pair_change_executed": False,
            "l4_l8_frozen": True,
        },
        "evidence_paths": [str(INVENTORY_PATH), str(MEMORY_PATH), str(PRECHECK_PATH), str(QUARANTINE)],
    }
    if any(incident["downstream_stage_absence"].values()):
        raise RuntimeError(f"unexpected downstream attempt-5 artifact found: {incident['downstream_stage_absence']}")
    atomic_json(INCIDENT_PATH, incident)

    manifest = {
        "schema_version": 1,
        "status": "quarantine_evidence_only",
        "generated_at": now_iso(),
        "source_workspace": str(ORIGINAL_WORKSPACE),
        "quarantine_path": str(QUARANTINE),
        "reason": incident["authoritative_blocker"],
        "inventory_path": str(INVENTORY_PATH),
        "incident_path": str(INCIDENT_PATH),
        "file_count_before_manifest": inventory["file_count"],
        "total_size_bytes_before_manifest": inventory["total_size_bytes"],
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
        "reuse_prohibited": True,
        "in_place_resume_prohibited": True,
    }
    atomic_json(MANIFEST_PATH, manifest)

    md = f"""# L3 attempt-5 raw 合并内存硬门事故收口\n\n## 结论\n\n- attempt-5 已按 fail-closed 停止并整体隔离。\n- 触发点为 raw shard 合并阶段：主进程私有内存约 55.02 GiB，超过 4 GiB parent RSS 预算。\n- 停止前系统可用内存约 26.46 GiB，尚未跌破 20 GiB 系统硬停线；停止后恢复到约 79.50 GiB。\n- `256/256` raw bucket 已完成，但合并未完成，全部 raw shard、completion、临时 DuckDB 与可能存在的 WAL 仅作事故证据，禁止复用。\n\n## 隔离状态\n\n- 原工作区：`{ORIGINAL_WORKSPACE}`，当前不存在。\n- 隔离目录：`{QUARANTINE}`。\n- 隔离文件数：{inventory['file_count']}（不含随后写入的 quarantine manifest）。\n- 隔离总字节：{inventory['total_size_bytes']}。\n- WAL 数量：{inventory['wal_count']}。\n- `approved_for_candidate=false`、`approved_for_active=false`、`approved_for_downstream=false`、`reuse_prohibited=true`。\n\n## 生产资产保护\n\n- attempt-5 launcher、orchestrator、worker、DuckDB writer 已归零。\n- L2、active feature、active label、production registry 的 hash、size、mtime 均与启动基线一致。\n- 未进入 GTJA；未生成 feature/label candidate；未生成 pair-change plan；未调用 pair-change；未切换 active。\n\n## 根因\n\n`_merge_shards` 在长生命周期 orchestrator 内执行 256 个 shard 的 ATTACH/INSERT，并创建全表索引。该阶段不经过 `ShortLivedTaskScheduler` 的内存采样与 parent RSS 门禁，导致 4 GiB parent 预算未在合并阶段执行。\n\n## 最小整改建议\n\n1. raw merge 改为独立短生命周期 spawn 进程，并复用相同 RSS、系统可用内存、退出恢复和残留清理门禁。\n2. 合并改为有界批次与不可变中间 DuckDB，避免一次长事务吸收全部 shard；merge/index 显式设置 DuckDB `memory_limit` 与 `threads=1`。\n3. 增加合并阶段 parent RSS 越线、系统硬停、强制中断隔离、正式资产不变测试。\n4. attempt-6 必须使用新 run-id/workspace，并经架构与审计放行；attempt-5 任何资产不得作为执行或校验输入。\n\n## 边界\n\n本次未修改调度器或 merge 实现，未启动 attempt-6，未训练、预测、生成信号、回测或触发交易。`ready_for_incident_audit_review=true`，`allow_next_layer_continue=false`。\n"""
    INCIDENT_MD_PATH.write_text(md, encoding="utf-8")

    print(json.dumps({
        "status": incident["status"],
        "inventory_file_count": inventory["file_count"],
        "inventory_total_size_bytes": inventory["total_size_bytes"],
        "active_states_unchanged": all(item["unchanged"] for item in states.values()),
        "ready_for_incident_audit_review": True,
        "allow_next_layer_continue": False,
        "incident_path": str(INCIDENT_PATH),
        "inventory_path": str(INVENTORY_PATH),
        "manifest_path": str(MANIFEST_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
