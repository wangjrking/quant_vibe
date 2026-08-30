from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit("psutil is required") from exc


ROOT = pathlib.Path(r"D:\work\quant\quant_mcp")
REPORT = ROOT / "quant/data_file/reports/l3_feature_contract_v2_candidate_build_preflight_20260804_r5_readonly"
L2 = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
TMP = pathlib.Path(str(L2) + ".tmp")
QUARANTINE_TMP = ROOT / "quant/data_file/runtime/agent_workspaces/factor-agent/quarantine/l2_stock_daily_data.duckdb.tmp_quarantine_20260805_r1"
EXPECTED_L2_SHA = "34a88582b2e0e53fa5ececadf8500673ad44259dd7d2bc8e44d3c689c7905e4b"
EXPECTED_SCHEMA_SHA = "f23a0a6398e2b7bddac09901a1722e475145c42234fb650708b0c1eb9fadee9d"
SCHEMA_LOCK = ROOT / "quant/data_file/reports/l3_feature_contract_v2_schema_only_precheck_20260804_r2/l2_schema_lock.json"
SCHEMA_REPORT = ROOT / "quant/data_file/reports/l3_feature_contract_v2_schema_only_precheck_20260804_r2/schema_only_precheck_report.json"
R7_ROOT = ROOT / "quant/data_file/reports/l3_feature_contract_v2_candidate_rebuild_grant_20260804_schema_lock_r7_authoritative/root_file_digests.json"
R7_ARTIFACT = ROOT / "quant/data_file/reports/l3_feature_contract_v2_candidate_rebuild_grant_20260804_schema_lock_r7_authoritative/artifact_hashes.json"
EXECUTOR = ROOT / "quant/main/rebuild_l3_memory_bounded_v2_candidate.py"
TESTS = ROOT / "quant/main/tests/test_rebuild_l3_memory_bounded_v2_candidate.py"
RUNTIME = ROOT / "runtime_candidates/my_quant_copy_20260804/python.exe"
RUN_ID = "incremental-trading-signal-20260804-L3-feature-contract-v2-candidate-build-preflight-20260804-r5-readonly"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta(path: pathlib.Path, include_sha: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "is_dir": path.is_dir(),
        "size": stat.st_size,
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "ctime": dt.datetime.fromtimestamp(stat.st_ctime, dt.timezone.utc).isoformat(),
    }
    if include_sha and path.is_file():
        result["sha256"] = sha(path)
    return result


def write_json(path: pathlib.Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def parser_result(path: pathlib.Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    try:
        json.loads(path.read_text(encoding="utf-8"))
        results["python_json"] = True
    except Exception as exc:
        results["python_json"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run([str(RUNTIME), "-m", "json.tool", str(path)], capture_output=True, text=True)
        results["python_json_tool"] = result.returncode == 0
    except Exception as exc:
        results["python_json_tool"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(["node", "-e", "JSON.parse(require('fs').readFileSync(process.argv[1],'utf8'))", str(path)], capture_output=True, text=True)
        results["node_json_parse"] = result.returncode == 0
    except Exception as exc:
        results["node_json_parse"] = f"{type(exc).__name__}: {exc}"
    try:
        command = f"Get-Content -Raw -Encoding UTF8 '{path}' | ConvertFrom-Json | Out-Null"
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
        results["powershell_convertfromjson"] = result.returncode == 0
    except Exception as exc:
        results["powershell_convertfromjson"] = f"{type(exc).__name__}: {exc}"
    return results


def main() -> int:
    if REPORT.exists():
        raise SystemExit(f"fresh report directory already exists: {REPORT}")
    REPORT.mkdir(parents=True, exist_ok=False)

    l2 = meta(L2, include_sha=True)
    schema_lock_payload = json.loads(SCHEMA_LOCK.read_text(encoding="utf-8"))
    ordered_schema = schema_lock_payload.get("ordered_l2_schema") or []
    ordered_schema_hash = schema_lock_payload.get("ordered_l2_schema_hash")
    tmp_entries = []
    if TMP.is_dir():
        for child in sorted(TMP.iterdir(), key=lambda item: item.name):
            child_meta = meta(child, include_sha=False)
            child_meta["age_hours_at_capture"] = round((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromtimestamp(child.stat().st_mtime, dt.timezone.utc)).total_seconds() / 3600, 2)
            tmp_entries.append(child_meta)
    quarantine_entries = []
    if QUARANTINE_TMP.is_dir():
        for child in sorted(QUARANTINE_TMP.iterdir(), key=lambda item: item.name):
            child_meta = meta(child, include_sha=False)
            child_meta["age_hours_at_capture"] = round((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromtimestamp(child.stat().st_mtime, dt.timezone.utc)).total_seconds() / 3600, 2)
            quarantine_entries.append(child_meta)

    self_lineage: set[int] = set()
    try:
        process = psutil.Process(os.getpid())
        while process and process.pid not in self_lineage:
            self_lineage.add(process.pid)
            process = process.parent()
    except psutil.Error:
        pass
    path_tokens = [str(L2).lower(), str(TMP).lower(), (str(L2) + ".wal").lower()]
    command_hits = []
    open_hits = []
    for process in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline", "create_time"]):
        try:
            if process.pid in self_lineage:
                continue
            info = process.info
            command_line = " ".join(info.get("cmdline") or [])
            if any(token in command_line.lower() for token in path_tokens):
                lineage = []
                current = process
                seen: set[int] = set()
                while current and current.pid not in seen:
                    seen.add(current.pid)
                    try:
                        lineage.append({
                            "pid": current.pid,
                            "ppid": current.ppid(),
                            "name": current.name(),
                            "exe": current.exe(),
                            "create_time": dt.datetime.fromtimestamp(current.create_time(), dt.timezone.utc).isoformat(),
                            "cmdline": current.cmdline(),
                        })
                        current = current.parent()
                    except psutil.Error:
                        break
                command_hits.append({
                    "pid": info.get("pid"),
                    "ppid": info.get("ppid"),
                    "name": info.get("name"),
                    "exe": info.get("exe"),
                    "create_time": dt.datetime.fromtimestamp(info["create_time"], dt.timezone.utc).isoformat() if info.get("create_time") else None,
                    "cmdline": info.get("cmdline"),
                    "command_role": "unknown_relevant_process",
                    "parent_lineage": lineage,
                })
            try:
                open_files = process.open_files() or []
            except psutil.Error:
                open_files = []
            matching_open_files = [entry.path for entry in open_files if any(token in (entry.path or "").lower() for token in path_tokens)]
            if matching_open_files:
                open_hits.append({"pid": process.pid, "ppid": process.ppid(), "name": process.name(), "open_files": matching_open_files, "command_role": "unknown_writer_or_reader"})
        except psutil.Error:
            continue

    future_roots = {
        "candidate_workspace": str(ROOT / "quant/data_file/runtime/agent_workspaces/factor-agent/work/incremental-trading-signal-20260804-L3-feature-contract-v2-candidate-build-r5-workspace"),
        "candidate_report": str(ROOT / "quant/data_file/reports/incremental-trading-signal-20260804-L3-feature-contract-v2-candidate-build-r5"),
        "candidate_quarantine": str(ROOT / "quant/data_file/runtime/agent_workspaces/factor-agent/quarantine/incremental-trading-signal-20260804-L3-feature-contract-v2-candidate-build-r5"),
    }
    future_root_state = {name: {"path": path, "exists": pathlib.Path(path).exists()} for name, path in future_roots.items()}
    resource = {
        "captured_at": now(),
        "virtual_memory": dict(psutil.virtual_memory()._asdict()),
        "disk_usage": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
        "candidate_profile_gib": 34,
        "production_profile_gib": 64,
        "hard_gate_lowered": False,
    }
    runtime = {
        "captured_at": now(),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "runtime_path": str(RUNTIME),
        "runtime_sha256": sha(RUNTIME),
        "runtime_path_matches_expected": sys.executable.lower() == str(RUNTIME).lower(),
    }
    source_hashes = {str(path): sha(path) for path in (EXECUTOR, TESTS, RUNTIME, SCHEMA_LOCK, SCHEMA_REPORT, R7_ROOT, R7_ARTIFACT)}
    tmp_present = TMP.exists()
    preflight = {
        "run_id": RUN_ID,
        "captured_at": now(),
        "status": "failed_closed_preflight_blocked",
        "scope": "readonly_candidate_build_execution_preflight_only",
        "business_rows_read": False,
        "l2_file_opened_for_hash_only": True,
        "workspace_created": False,
        "candidate_written": False,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "build_started": False,
        "allow_candidate_rebuild_grant": False,
        "allow_full_rebuild": False,
        "allow_l3": False,
        "allow_l4_l8": False,
        "can_apply_candidate_only_build_grant": False,
        "blockers": (["stale_l2_tmp_present"] if tmp_present else []) + ["quarantine_disposition_audit_required", "independent_audit_required"],
        "l2_binding": {
            "path": str(L2),
            "expected_sha256": EXPECTED_L2_SHA,
            "actual_sha256": l2.get("sha256"),
            "sha_match": l2.get("sha256", "").lower() == EXPECTED_L2_SHA,
            "ordered_schema_hash": ordered_schema_hash,
            "ordered_schema_hash_expected": EXPECTED_SCHEMA_SHA,
            "ordered_schema_hash_match": ordered_schema_hash == EXPECTED_SCHEMA_SHA,
            "column_count": len(ordered_schema),
            "column_count_match": len(ordered_schema) == 534,
            "key_order": [entry.get("name") for entry in ordered_schema[:2]],
            "key_order_match": [entry.get("name") for entry in ordered_schema[:2]] == ["stock_code", "trade_date"],
        },
        "l2_file": l2,
        "tmp_classification": {
            "exists": tmp_present,
            "type": "directory" if TMP.is_dir() else ("file" if TMP.is_file() else "absent"),
            "classified_source": "stale_duckdb_temp_storage_artifacts" if tmp_present else "absent",
            "entry_count": len(tmp_entries),
            "entries": tmp_entries,
            "open_file_hit_count": len(open_hits),
        },
        "quarantine_tmp_classification": {
            "path": str(QUARANTINE_TMP),
            "exists": QUARANTINE_TMP.exists(),
            "type": "directory" if QUARANTINE_TMP.is_dir() else ("file" if QUARANTINE_TMP.is_file() else "absent"),
            "disposition": "isolated_reversible_quarantine" if QUARANTINE_TMP.is_dir() else "missing",
            "entry_count": len(quarantine_entries),
            "entries": quarantine_entries,
            "source_original_path": str(TMP),
            "source_original_present": TMP.exists(),
        },
        "wal_present": pathlib.Path(str(L2) + ".wal").exists(),
        "dash_wal_present": pathlib.Path(str(L2) + "-wal").exists(),
        "lock_sidecars": [meta(pathlib.Path(str(L2) + suffix)) for suffix in (".lock", ".lck", ".lease")],
        "matching_process_count": len(command_hits),
        "open_file_hit_count": len(open_hits),
        "unknown_writer_present": bool(command_hits or open_hits),
        "future_roots": future_root_state,
        "runtime": runtime,
        "source_hashes": source_hashes,
    }
    write_json(REPORT / "preflight_report.json", preflight)
    write_json(REPORT / "l2_tmp_classification.json", {"run_id": RUN_ID, "captured_at": now(), "tmp": preflight["tmp_classification"]})
    write_json(REPORT / "process_writer_scan.json", {"run_id": RUN_ID, "captured_at": now(), "self_lineage_excluded": sorted(self_lineage), "matching_processes": command_hits, "open_file_hits": open_hits, "unknown_writer_present": bool(command_hits or open_hits)})
    write_json(REPORT / "runtime_identity.json", runtime)
    write_json(REPORT / "future_build_roots.json", future_root_state)
    write_json(REPORT / "resource_snapshot.json", resource)
    write_json(REPORT / "candidate_grant_request.json", {"run_id": RUN_ID, "status": "audit_architecture_review_required", "grant_precheck_ready": False, "allow_candidate_rebuild_grant": False, "allow_full_rebuild": False, "allow_l3": False, "allow_l4_l8": False, "business_rows_read": False, "workspace_created": False, "candidate_written": False, "stale_tmp_present": tmp_present, "quarantine_tmp_present": QUARANTINE_TMP.is_dir(), "quarantine_entry_count": len(quarantine_entries), "can_apply_candidate_only_build_grant": False})
    write_json(REPORT / "candidate_manifest.json", {"run_id": RUN_ID, "reuse_prohibited": True, "business_rows_read": False, "workspace_created": False, "candidate_written": False, "active_switch": False, "label_write": False, "registry_change": False})
    write_json(REPORT / "freeze_lock.json", {"run_id": RUN_ID, "reuse_prohibited": True, "source_hashes": source_hashes, "l2_sha256_expected": EXPECTED_L2_SHA, "schema_hash_expected": EXPECTED_SCHEMA_SHA, "allow_candidate_rebuild_grant": False, "allow_l3": False})
    write_json(REPORT / "rollback.json", {"run_id": RUN_ID, "rollback_required": False, "reason": "readonly_preflight_no_business_write"})
    write_json(REPORT / "audit_handoff.json", {"run_id": RUN_ID, "ready_for_audit_review": True, "ready_for_architecture_review": True, "allow_candidate_rebuild_grant": False, "allow_full_rebuild": False, "allow_l3": False, "allow_l4_l8": False, "stale_tmp_present": tmp_present, "quarantine_tmp_present": QUARANTINE_TMP.is_dir(), "quarantine_entry_count": len(quarantine_entries), "matching_process_count": len(command_hits), "open_file_hit_count": len(open_hits), "self_check_report_path": str(REPORT / "package_self_check.json")})
    readme = "\\u0023 0804 L3 \\u5019\\u9009\\u91cd\\u5efa\\u53ea\\u8bfb\\u6267\\u884c\\u524d\\u7f6e\\u68c0\\u67e5\\n\\n\\u672c\\u76ee\\u5f55\\u53ea\\u4fdd\\u5b58\\u6587\\u4ef6\\u6307\\u7eb9\\u3001\\u4e34\\u65f6\\u6587\\u4ef6\\u3001\\u8fdb\\u7a0b\\u94fe\\u8def\\u3001\\u8fd0\\u884c\\u65f6\\u3001\\u8d44\\u6e90\\u548c\\u7a7a\\u76ee\\u5f55\\u72b6\\u6001\\u3002\\u672a\\u8bfb\\u53d6 L2 \\u4e1a\\u52a1\\u884c\\u3001\\u672a\\u521b\\u5efa\\u5019\\u9009\\u5de5\\u4f5c\\u533a\\u3001\\u672a\\u4fee\\u6539 active\\u3001label \\u6216 registry\\u3002\\u539f L2 .tmp \\u5df2\\u79fb\\u5165\\u53ef\\u56de\\u9000\\u9694\\u79bb\\u76ee\\u5f55\\uff0c\\u672c\\u5305\\u4ec5\\u8bb0\\u5f55\\u5904\\u7f6e\\u8bc1\\u636e\\uff0c\\u5019\\u9009\\u6784\\u5efa\\u4ecd\\u4fdd\\u6301 fail-closed\\u3002\\n"
    with (REPORT / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(readme.encode("ascii").decode("unicode_escape"))

    core_json = sorted(REPORT.glob("*.json"))
    parser_summary = {path.name: parser_result(path) for path in core_json}
    write_json(REPORT / "parser_summary.json", {"run_id": RUN_ID, "files_checked": sorted(parser_summary), "results": parser_summary})
    artifact_files = [path for path in REPORT.iterdir() if path.name not in {"artifact_hashes.json", "root_file_digests.json", "package_self_check.json"}]
    artifact = {path.name: sha(path) for path in sorted(artifact_files, key=lambda item: item.name)}
    write_json(REPORT / "artifact_hashes.json", {"run_id": RUN_ID, "files": artifact, "hash_mismatch_count": 0, "self_hash_excluded": True})
    root_files = [path for path in REPORT.iterdir() if path.name not in {"root_file_digests.json", "package_self_check.json"}]
    root = {path.name: sha(path) for path in sorted(root_files, key=lambda item: item.name)}
    write_json(REPORT / "root_file_digests.json", {"run_id": RUN_ID, "files": root, "mismatch_count": 0, "self_hash_excluded": True})
    all_json = sorted(REPORT.glob("*.json"), key=lambda item: item.name)
    artifact_actual = {path.name: sha(path) for path in artifact_files}
    root_actual = {path.name: sha(path) for path in root_files}
    artifact_expected = json.loads((REPORT / "artifact_hashes.json").read_text(encoding="utf-8"))["files"]
    root_expected = json.loads((REPORT / "root_file_digests.json").read_text(encoding="utf-8"))["files"]
    check = {"run_id": RUN_ID, "json_files_checked": [path.name for path in all_json], "self_hash_excluded": True, "artifact_hash_mismatch_count": sum(artifact_actual.get(name) != value for name, value in artifact_expected.items()), "root_hash_mismatch_count": sum(root_actual.get(name) != value for name, value in root_expected.items()), "all_json_parser_results": {path.name: parser_result(path) for path in all_json}, "stale_tmp_present": tmp_present, "quarantine_tmp_present": QUARANTINE_TMP.is_dir(), "quarantine_entry_count": len(quarantine_entries), "readme_contains_ascii_qmark": 0x3F in (REPORT / "README.md").read_bytes(), "allow_candidate_rebuild_grant": False, "allow_l3": False}
    write_json(REPORT / "package_self_check.json", check)
    print(json.dumps({"report": str(REPORT), "l2_sha256": l2.get("sha256"), "tmp_present": tmp_present, "tmp_entries": len(tmp_entries), "matching_process_count": len(command_hits), "open_file_hit_count": len(open_hits), "allow_candidate_rebuild_grant": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
