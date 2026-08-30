from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


L3_MUTATION_ENTRY_REGISTRY: dict[str, dict[str, Any]] = {
    "rebuild_l3_full_duckdb_mainline.py": {
        "policy": "mode_switch",
        "writer_modes": {"build"},
        "readonly_modes": {"precheck"},
        "mutation_scope": "L3 full-history candidate staging",
        "active_write_policy": "candidate_only_no_active_write",
    },
    "refresh_l3_active_duckdb_full_delivery.py": {
        "policy": "always_writer",
        "mutation_scope": "active L3 feature/label DuckDB",
        "active_write_policy": "shared_writer_lease_required",
    },
    "deliver_l3_target_date_duckdb_mainline.py": {
        "policy": "mode_switch",
        "writer_modes": {"candidate", "execute"},
        "readonly_modes": {"precheck"},
        "mutation_scope": "active L3 feature target-date atomic delivery",
        "active_write_policy": "shared_writer_lease_required",
    },
    "l3_duckdb_sync.py": {
        "policy": "always_writer",
        "mutation_scope": "active L3 feature/label DuckDB",
        "active_write_policy": "shared_writer_lease_required_at_sink",
    },
    "build_production_factor_parts.py": {
        "policy": "always_writer",
        "mutation_scope": "production factor staging and active L3 sync",
        "active_write_policy": "shared_writer_lease_required_at_l3_duckdb_sync_sink",
    },
    "build_production_factor_raw_gtja_parts.py": {
        "policy": "always_writer",
        "mutation_scope": "production factor staging and active L3 sync",
        "active_write_policy": "shared_writer_lease_required_at_l3_duckdb_sync_sink",
    },
    "build_prediction_label_parts.py": {
        "policy": "always_writer",
        "mutation_scope": "label staging and active L3 sync",
        "active_write_policy": "shared_writer_lease_required_at_l3_duckdb_sync_sink",
    },
    "incremental_factor_update_target_date.py": {
        "policy": "always_writer",
        "mutation_scope": "factor staging and active L3 target-date sync",
        "active_write_policy": "shared_writer_lease_required_at_l3_duckdb_sync_sink",
    },
    "incremental_prediction_label_update_target_date.py": {
        "policy": "always_writer",
        "mutation_scope": "label staging and active L3 target-date sync",
        "active_write_policy": "shared_writer_lease_required_at_l3_duckdb_sync_sink",
    },
    "run_incremental_factor_update_chunked.py": {
        "policy": "always_writer",
        "mutation_scope": "factor staging and active L3 target-date sync",
        "active_write_policy": "shared_writer_lease_required_at_l3_duckdb_sync_sink",
    },
    "rebuild_raw_factor_by_stock.py": {
        "policy": "always_writer",
        "mutation_scope": "raw factor staging",
    },
    "apply_production_asset_pair_change.py": {
        "policy": "execute_flag",
        "writer_flags": {"--execute"},
        "mutation_scope": "L3 feature/label pair change and production registry",
        "active_write_policy": "shared_writer_lease_required",
    },
    "apply_production_asset_change.py": {
        "policy": "always_writer",
        "mutation_scope": "single production registry change",
        "active_write_policy": "shared_writer_lease_required",
    },
    "production_asset_gate.py": {
        "policy": "readonly_observer",
        "mutation_scope": "production registry validation",
    },
    "scan_l3_write_bypass.py": {
        "policy": "readonly_observer",
        "mutation_scope": "L3 write bypass scan",
    },
}
KNOWN_L3_SCRIPTS = set(L3_MUTATION_ENTRY_REGISTRY)
MUTATION_PATH_HINTS = (
    "l3_feature_current.duckdb",
    "l3_label_current.duckdb",
    "production_assets.json",
)
MUTATION_COMMAND_HINTS = (
    "--execute",
    "--write",
    "--apply",
    "--sync",
    "--replace",
    "--build",
    "os.replace",
    "write_text",
    "write_bytes",
    "create or replace",
)
PYTHON_NAMES = {"python", "python.exe", "pythonw", "pythonw.exe"}
SHELL_NAMES = {"powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe"}


def _registered_module_names() -> dict[str, str]:
    modules: dict[str, str] = {}
    tool_scripts = {
        "apply_production_asset_pair_change.py",
        "apply_production_asset_change.py",
    }
    for script in L3_MUTATION_ENTRY_REGISTRY:
        module = script.removesuffix(".py")
        modules[module] = script
        modules[f"quant.main.{module}"] = script
        if script in tool_scripts:
            modules[f"tools.{module}"] = script
            modules[f"quant.main.tools.{module}"] = script
    return modules


REGISTERED_L3_MODULES = _registered_module_names()


def _command(record: dict[str, Any]) -> str:
    value = record.get("cmdline") or []
    if isinstance(value, str):
        return value
    return " ".join(str(part) for part in value)


def _flag_value(command: str, flag: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(flag)}(?:\s+|=)(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))", command)
    if not match:
        return None
    return next((value for value in match.groups() if value is not None), None)


def _parsed_module(command: str) -> str | None:
    match = re.search(
        r"(?:^|\s)-m(?:\s+|=)(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))(?=\s|$)",
        command,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return next((value for value in match.groups() if value is not None), None)


def _parsed_script(command: str) -> str | None:
    lowered = command.lower().replace("\\", "/")
    module = (_parsed_module(command) or "").lower()
    if module:
        registered = REGISTERED_L3_MODULES.get(module)
        if registered is not None:
            return registered
    for script in KNOWN_L3_SCRIPTS:
        if re.search(rf"(?<![a-z0-9_]){re.escape(script)}(?![a-z0-9_])", lowered):
            return script
    return None


def _script_contract(identity: dict[str, Any]) -> tuple[str, str]:
    script = identity.get("parsed_script")
    spec = L3_MUTATION_ENTRY_REGISTRY.get(str(script))
    if spec is None:
        return "unknown", "script is not registered in L3 mutation entry registry"
    policy = spec["policy"]
    command = identity["cmdline"].lower()
    if policy == "always_writer":
        return "writer", str(spec["mutation_scope"])
    if policy == "readonly_observer":
        return "readonly", str(spec["mutation_scope"])
    if policy == "mode_switch":
        mode = str(identity.get("mode") or "").lower()
        if mode in spec["writer_modes"]:
            return "writer", f"registered writer mode: {mode}"
        if mode in spec["readonly_modes"]:
            return "readonly", f"registered readonly mode: {mode}"
        return "unknown", f"unregistered mode for mutation entry: {mode or '<missing>'}"
    if policy == "execute_flag":
        if any(flag in command for flag in spec["writer_flags"]):
            return "writer", "registered execute/apply flag present"
        if "--dry-run-output" in command:
            return "unknown", "pair-change dry-run writes an output path that requires separate path proof"
        return "readonly", "pair-change dry-run without --execute"
    return "unknown", f"unknown mutation entry policy: {policy}"


def _has_mutation_path_hint(command: str) -> bool:
    lowered = command.lower()
    return any(hint in lowered for hint in MUTATION_PATH_HINTS)


def _has_mutation_command_hint(command: str) -> bool:
    lowered = command.lower()
    return any(hint in lowered for hint in MUTATION_COMMAND_HINTS)


def _is_related_python_mutation_command(identity: dict[str, Any], mutation_command_hint: bool) -> bool:
    if not mutation_command_hint or not _is_python(identity):
        return False
    command = identity["cmdline"].lower().replace("\\", "/")
    module = str(identity.get("parsed_module") or "").lower()
    module_related = bool(
        module.startswith("quant.main.")
        or module.startswith("tools.") and any(
            token in module for token in ("l3", "factor", "label", "gtja", "production_asset")
        )
    )
    project_path_related = any(
        token in command
        for token in (
            "/quant/main/",
            "quant_mcp",
            "production_assets",
            "asset_registry",
            "l3_",
            "factor_",
            "prediction_label",
            "production_asset",
            "gtja",
        )
    )
    return module_related or project_path_related


def _norm_path(value: str | None) -> str | None:
    if not value:
        return None
    return os.path.normcase(os.path.abspath(value.strip('"\'')))


def parse_process_identity(record: dict[str, Any]) -> dict[str, Any]:
    command = _command(record)
    name = str(record.get("name") or "")
    exe = str(record.get("exe") or "")
    return {
        "pid": int(record["pid"]),
        "ppid": int(record.get("ppid") or 0),
        "name": name,
        "exe": exe,
        "cmdline": command,
        "create_time": record.get("create_time"),
        "parsed_module": _parsed_module(command),
        "parsed_script": _parsed_script(command),
        "mode": _flag_value(command, "--mode"),
        "workflow_run_id": _flag_value(command, "--workflow-run-id"),
        "workspace": _norm_path(_flag_value(command, "--workspace-dir")),
        "report_dir": _norm_path(_flag_value(command, "--report-dir")),
        "task_id": _flag_value(command, "--task-id"),
        "access_error": record.get("access_error"),
        "open_files_error": record.get("open_files_error"),
        "open_files": [
            {"path": str(item.get("path")), "mode": item.get("mode")}
            if isinstance(item, dict) else {"path": str(item), "mode": None}
            for item in record.get("open_files") or []
        ],
    }


def _process_graph(records: list[dict[str, Any]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    parents = {int(record["pid"]): int(record.get("ppid") or 0) for record in records}
    children: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)

    ancestors: dict[int, list[int]] = {}
    descendants: dict[int, list[int]] = {}
    for pid in parents:
        lineage = []
        seen = {pid}
        parent = parents.get(pid, 0)
        while parent and parent not in seen:
            lineage.append(parent)
            seen.add(parent)
            parent = parents.get(parent, 0)
        ancestors[pid] = lineage

        found = []
        pending = list(children.get(pid, []))
        seen_descendants = set()
        while pending:
            child = pending.pop(0)
            if child in seen_descendants:
                continue
            seen_descendants.add(child)
            found.append(child)
            pending.extend(children.get(child, []))
        descendants[pid] = found
    return ancestors, descendants


def _is_python(identity: dict[str, Any]) -> bool:
    return identity["name"].lower() in PYTHON_NAMES or Path(identity["exe"]).name.lower() in PYTHON_NAMES


def _is_shell(identity: dict[str, Any]) -> bool:
    return identity["name"].lower() in SHELL_NAMES or Path(identity["exe"]).name.lower() in SHELL_NAMES


def _is_spawn_worker(identity: dict[str, Any]) -> bool:
    lowered = identity["cmdline"].lower()
    return "multiprocessing.spawn" in lowered or "spawn_main" in lowered


def _paths_collide(identity: dict[str, Any], workspace: str, report_dir: str) -> bool:
    return identity["workspace"] in {workspace, report_dir} or identity["report_dir"] in {workspace, report_dir}


def _relevant_lock_paths(identity: dict[str, Any], protected_paths: set[str], workspace: str) -> list[str]:
    matches = []
    for opened in identity["open_files"]:
        path = _norm_path(opened["path"])
        mode = str(opened.get("mode") or "").lower()
        protected_match = any(
            path == protected
            or (not Path(protected).suffix and path is not None and path.startswith(protected + os.sep))
            for protected in protected_paths
        )
        if protected_match and (not mode or any(token in mode for token in ("w", "a", "+"))):
            matches.append(path)
        elif path and (path.endswith(".wal") or path.endswith("-wal")) and (
            path.startswith(workspace) or any(path.startswith(item) for item in protected_paths)
        ):
            matches.append(path)
    return matches


def _evidence_item(
    identity: dict[str, Any],
    *,
    ancestors: list[int],
    descendants: list[int],
    process_role: str,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    return {
        **identity,
        "ancestors": ancestors,
        "descendants": descendants,
        "process_role": process_role,
        "decision": decision,
        "reason": reason,
    }


def classify_process_gate(
    records: Iterable[dict[str, Any]],
    *,
    current_pid: int,
    workflow_run_id: str,
    workspace: str,
    report_dir: str,
    unified_python_launcher: str,
    expected_workers: dict[int, dict[str, Any]] | None = None,
    protected_l3_paths: Iterable[str] = (),
) -> dict[str, Any]:
    records_list = [dict(record) for record in records]
    identities = {int(record["pid"]): parse_process_identity(record) for record in records_list}
    ancestors_by_pid, descendants_by_pid = _process_graph(records_list)
    current_ancestors = set(ancestors_by_pid.get(current_pid, []))
    current_ppid = identities.get(current_pid, {}).get("ppid")
    workspace_norm = _norm_path(workspace) or ""
    report_norm = _norm_path(report_dir) or ""
    launcher_norm = _norm_path(unified_python_launcher)
    protected = {_norm_path(path) for path in protected_l3_paths if _norm_path(path)}
    expected_workers = expected_workers or {}
    groups: dict[str, list[dict[str, Any]]] = {
        "allowed_current_chain": [],
        "allowed_readonly_observers": [],
        "blocking_writers": [],
        "unknown_relevant_processes": [],
    }

    for pid, identity in identities.items():
        ancestors = ancestors_by_pid.get(pid, [])
        descendants = descendants_by_pid.get(pid, [])
        lock_paths = _relevant_lock_paths(identity, protected, workspace_norm)
        mutation_path_hint = _has_mutation_path_hint(identity["cmdline"])
        mutation_command_hint = _has_mutation_command_hint(identity["cmdline"])
        mutation_flag_relevant = _is_related_python_mutation_command(identity, mutation_command_hint)
        relevant_command = bool(
            identity["parsed_script"]
            or _is_spawn_worker(identity)
            or mutation_path_hint
            or mutation_flag_relevant
        )
        relevant_access_failure = bool(identity["access_error"])
        relevant_open_failure = bool(identity["open_files_error"] and relevant_command)
        if not (relevant_command or lock_paths or relevant_access_failure or relevant_open_failure):
            continue

        worker = expected_workers.get(pid)
        valid_worker = False
        if worker is not None:
            actual_create_time_ns = int(float(identity.get("create_time") or 0.0) * 1_000_000_000)
            valid_worker = (
                current_pid in ancestors
                and str(worker.get("workflow_run_id")) == workflow_run_id
                and _norm_path(str(worker.get("workspace") or "")) == workspace_norm
                and bool(worker.get("task_id"))
                and bool(worker.get("worker_nonce"))
                and bool(worker.get("run_nonce"))
                and int(worker.get("parent_pid") or 0) == current_pid
                and int(worker.get("create_time_ns") or 0) == actual_create_time_ns
            )

        active_lock_paths = [
            path for path in lock_paths
            if any(
                path == protected_path
                or (not Path(protected_path).suffix and path.startswith(protected_path + os.sep))
                for protected_path in protected
            )
        ]
        if lock_paths and (not valid_worker or active_lock_paths):
            item = _evidence_item(
                identity,
                ancestors=ancestors,
                descendants=descendants,
                process_role="l3_lock_holder",
                decision="block",
                reason=f"holds protected L3 DuckDB/WAL path: {lock_paths}",
            )
            item["relevant_lock_paths"] = lock_paths
            groups["blocking_writers"].append(item)
            continue

        if identity["access_error"] or relevant_open_failure:
            groups["unknown_relevant_processes"].append(_evidence_item(
                identity,
                ancestors=ancestors,
                descendants=descendants,
                process_role="uninspectable_relevant_process",
                decision="block_unknown",
                reason=f"process inspection incomplete: {identity['access_error'] or identity['open_files_error']}",
            ))
            continue

        if pid == current_pid:
            groups["allowed_current_chain"].append(_evidence_item(
                identity,
                ancestors=ancestors,
                descendants=descendants,
                process_role="current_orchestrator",
                decision="allow",
                reason="current orchestrator PID",
            ))
            continue

        if worker is not None:
            target = "allowed_current_chain" if valid_worker else "blocking_writers"
            groups[target].append(_evidence_item(
                {**identity, "task_id": worker.get("task_id")},
                ancestors=ancestors,
                descendants=descendants,
                process_role="registered_short_lived_worker" if valid_worker else "invalid_worker_registration",
                decision="allow" if valid_worker else "block",
                reason="scheduler registry and parent chain verified" if valid_worker else "worker registry identity mismatch",
            ))
            continue

        if pid in current_ancestors:
            same_identity = (
                identity["workflow_run_id"] == workflow_run_id
                and identity["workspace"] == workspace_norm
                and identity["report_dir"] == report_norm
                and current_pid in descendants
            )
            direct_unified_python_shim = (
                pid == current_ppid
                and _is_python(identity)
                and _norm_path(identity["exe"]) == launcher_norm
                and same_identity
            )
            # Windows PowerShell/CMD normally launches the Python command through
            # a wrapper string, so the shell itself may not carry the Python
            # workflow flags. Trust only the verified current chain in that case:
            # the current Python identity is authoritative, while the shell must
            # be a real ancestor with no mutation or protected-path hints.
            shell_chain_identity = (
                _is_shell(identity)
                and current_pid in descendants
                and not mutation_path_hint
                and not mutation_command_hint
            )
            legal_shell_ancestor = _is_shell(identity) and (same_identity or shell_chain_identity)
            if direct_unified_python_shim or legal_shell_ancestor:
                groups["allowed_current_chain"].append(_evidence_item(
                    identity,
                    ancestors=ancestors,
                    descendants=descendants,
                    process_role="unified_python_launcher_shim" if direct_unified_python_shim else "legal_shell_launcher_ancestor",
                    decision="allow",
                    reason="verified ancestor relation and exact run/workspace/report identity",
                ))
            else:
                groups["unknown_relevant_processes"].append(_evidence_item(
                    identity,
                    ancestors=ancestors,
                    descendants=descendants,
                    process_role="unverified_launcher_ancestor",
                    decision="block_unknown",
                    reason="ancestor is not the exact unified Python shim or a verified shell launcher",
                ))
            continue

        if _is_spawn_worker(identity):
            groups["blocking_writers"].append(_evidence_item(
                identity,
                ancestors=ancestors,
                descendants=descendants,
                process_role="unregistered_or_foreign_worker",
                decision="block",
                reason="multiprocessing worker is not present in the scheduler registry",
            ))
            continue

        script_decision, script_reason = _script_contract(identity)
        if script_decision == "readonly":
            if not identity["workspace"] or not identity["report_dir"]:
                pair_change_dry_run = identity["parsed_script"] == "apply_production_asset_pair_change.py"
                if pair_change_dry_run and identity["report_dir"] is None and identity["workspace"] is None:
                    groups["allowed_readonly_observers"].append(_evidence_item(
                        identity,
                        ancestors=ancestors,
                        descendants=descendants,
                        process_role="allowlisted_pair_change_dry_run",
                        decision="allow_readonly",
                        reason=script_reason,
                    ))
                    continue
                groups["unknown_relevant_processes"].append(_evidence_item(
                    identity,
                    ancestors=ancestors,
                    descendants=descendants,
                    process_role="unproven_readonly_observer",
                    decision="block_unknown",
                    reason="readonly observer lacks explicit workspace/report-dir identity",
                ))
            elif _paths_collide(identity, workspace_norm, report_norm):
                groups["blocking_writers"].append(_evidence_item(
                    identity,
                    ancestors=ancestors,
                    descendants=descendants,
                    process_role="readonly_observer_path_collision",
                    decision="block",
                    reason="readonly observer collides with current workspace or report-dir",
                ))
            else:
                groups["allowed_readonly_observers"].append(_evidence_item(
                    identity,
                    ancestors=ancestors,
                    descendants=descendants,
                    process_role="allowlisted_readonly_observer",
                    decision="allow_readonly",
                    reason="explicit precheck allowlist with distinct workspace and report-dir",
                ))
            continue

        if script_decision == "writer":
            role = "independent_l3_writer"
            if current_pid in ancestors:
                role = "unexpected_orchestrator_child"
            elif _is_shell(identity):
                role = "foreign_shell_launcher"
            groups["blocking_writers"].append(_evidence_item(
                identity,
                ancestors=ancestors,
                descendants=descendants,
                process_role=role,
                decision="block",
                reason=f"independent registered L3 writer: {script_reason}",
            ))
            continue

        if identity["parsed_script"] or mutation_path_hint or mutation_flag_relevant:
            groups["unknown_relevant_processes"].append(_evidence_item(
                identity,
                ancestors=ancestors,
                descendants=descendants,
                process_role="unknown_l3_mutation_entry",
                decision="block_unknown",
                reason=(
                    script_reason if identity["parsed_script"]
                    else (
                        "unregistered process references protected L3 mutation path"
                        if mutation_path_hint
                        else "unregistered related Python process carries a mutation flag"
                    )
                    + (" with mutation command hint" if mutation_command_hint and mutation_path_hint else "")
                ),
            ))
            continue

        groups["unknown_relevant_processes"].append(_evidence_item(
            identity,
            ancestors=ancestors,
            descendants=descendants,
            process_role="unknown_relevant_process",
            decision="block_unknown",
            reason="related Python or launcher process could not be reliably parsed",
        ))

    passed = not groups["blocking_writers"] and not groups["unknown_relevant_processes"]
    return {
        "status": "process_gate_passed" if passed else "process_gate_blocked",
        "current_pid": current_pid,
        "workflow_run_id": workflow_run_id,
        "workspace": workspace_norm,
        "report_dir": report_norm,
        "allowed_current_chain": groups["allowed_current_chain"],
        "allowed_readonly_observers": groups["allowed_readonly_observers"],
        "blocking_writers": groups["blocking_writers"],
        "unknown_relevant_processes": groups["unknown_relevant_processes"],
        "passed": passed,
    }


def collect_process_records(psutil_module, *, protected_l3_paths: Iterable[str], workspace: str) -> list[dict[str, Any]]:
    protected = {_norm_path(path) for path in protected_l3_paths if _norm_path(path)}
    workspace_norm = _norm_path(workspace) or ""
    records = []
    for process in psutil_module.process_iter(["pid", "ppid", "name", "exe", "cmdline", "create_time"]):
        try:
            record = dict(process.info)
            command = _command(record)
            name = str(record.get("name") or "").lower()
            inspect_files = (
                bool(_parsed_script(command))
                or _has_mutation_path_hint(command)
                or "multiprocessing.spawn" in command.lower()
                or name in PYTHON_NAMES
                or name in SHELL_NAMES
            )
            if inspect_files:
                try:
                    record["open_files"] = [
                        {"path": item.path, "mode": getattr(item, "mode", None)}
                        for item in process.open_files()
                    ]
                except (psutil_module.NoSuchProcess, psutil_module.AccessDenied) as error:
                    record["open_files_error"] = type(error).__name__
            records.append(record)
        except psutil_module.NoSuchProcess:
            continue
        except psutil_module.AccessDenied as error:
            records.append({
                "pid": int(getattr(process, "pid", -1)),
                "ppid": 0,
                "name": "unknown",
                "exe": "",
                "cmdline": [],
                "create_time": None,
                "access_error": type(error).__name__,
            })

    for protected_path in protected:
        for wal in (f"{protected_path}.wal", f"{protected_path}-wal"):
            if os.path.exists(wal):
                records.append({
                    "pid": -abs(hash(wal)) % 2_000_000_000 - 1,
                    "ppid": 0,
                    "name": "duckdb_wal_evidence",
                    "exe": "",
                    "cmdline": ["duckdb_wal", wal],
                    "create_time": os.path.getmtime(wal),
                    "open_files": [{"path": wal, "mode": "w"}],
                })
    if workspace_norm and os.path.isdir(workspace_norm):
        for wal_path in Path(workspace_norm).rglob("*.wal"):
            records.append({
                "pid": -abs(hash(str(wal_path))) % 2_000_000_000 - 1,
                "ppid": 0,
                "name": "duckdb_wal_evidence",
                "exe": "",
                "cmdline": ["duckdb_wal", str(wal_path)],
                "create_time": wal_path.stat().st_mtime,
                "open_files": [{"path": str(wal_path), "mode": "w"}],
            })
    return records


def load_expected_workers(workspace: str) -> dict[int, dict[str, Any]]:
    path = Path(workspace) / "logs" / "short_lived_worker_registry.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    workers = [
        *(payload.get("launching_workers") or []),
        *(payload.get("active_workers") or []),
    ]
    return {
        int(item["pid"]): item
        for item in workers
        if item.get("pid") is not None
    }
