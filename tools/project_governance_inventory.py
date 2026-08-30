"""Build a read-only inventory for whole-project governance.

The inventory separates regenerable files, migration candidates, and protected
assets. It never moves, deletes, hashes, or opens business databases.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATE_TOKEN = re.compile(r"(?:^|_)(20\d{6})(?:_|\.|$)")
SCRIPT_PREFIXES = (
    "research_",
    "run_",
    "build_",
    "validate_",
    "export_",
    "analyze_",
    "evaluate_",
    "diagnose_",
    "archive_",
)
LARGE_FILE_BYTES = 1 << 30

PROTECTED_PATHS = (
    "quant/data_file/production_assets",
    "quant/data_file/asset_registry",
    "quant/data_file/runtime/agent_memory",
    "quant/data_file/runtime/orchestrator_reports/workflow_monitor",
    "quant/data_file/runtime/trading_agent",
    "quant/main/config/prediction_manifests",
    "quant/main/strategy_library/production",
)

GENERATED_CANDIDATES = (
    "site/node_modules",
    "site/dist.previous",
    "site/.filter-frames",
    "site/coverage",
    ".tmp",
    ".codex-tmp",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_bytes(value: int) -> str:
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.3f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def is_dated_script(name: str) -> bool:
    return name.endswith(".py") and DATE_TOKEN.search(name) is not None


def classify_root_python(name: str) -> str:
    for prefix in SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if not Path(current, name).is_symlink()]
        for name in files:
            path = Path(current, name)
            if not path.is_symlink():
                yield path


def tree_stats(root: Path) -> dict[str, Any]:
    files = 0
    size = 0
    if root.exists():
        for path in iter_files(root):
            try:
                size += path.stat().st_size
                files += 1
            except OSError:
                continue
    return {"path": str(root), "files": files, "bytes": size, "human": format_bytes(size)}


def git_status(root: Path) -> dict[str, Any]:
    marker = root / ".git"
    marker_children = len(list(marker.iterdir())) if marker.is_dir() else None
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line]
    return {
        "path": str(root),
        "is_git_repository": result.returncode == 0,
        "git_marker_exists": marker.exists(),
        "git_marker_children": marker_children,
        "tracked_changes": sum(not line.startswith("??") for line in lines),
        "untracked_entries": sum(line.startswith("??") for line in lines),
        "status_entries": len(lines),
        "error": result.stderr.strip() if result.returncode else None,
    }


def large_file_stats(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in iter_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < LARGE_FILE_BYTES:
            continue
        total += size
        rows.append(
            {
                "path": str(path),
                "bytes": size,
                "human": format_bytes(size),
                "suffix": path.suffix.lower(),
            }
        )
    rows.sort(key=lambda item: item["bytes"], reverse=True)
    return {
        "root": str(root),
        "threshold_bytes": LARGE_FILE_BYTES,
        "count": len(rows),
        "bytes": total,
        "human": format_bytes(total),
        "files": rows,
    }


def root_python_stats(main_dir: Path) -> dict[str, Any]:
    paths = sorted(main_dir.glob("*.py"), key=lambda path: path.name)
    names = [path.name for path in paths]
    prefixes = Counter(classify_root_python(name) for name in names)
    by_month = Counter()
    dated_registry = []
    for path in paths:
        match = DATE_TOKEN.search(path.name)
        if not match:
            continue
        stat = path.stat()
        dated_registry.append(
            {
                "path": path.name,
                "date_token": match.group(1),
                "category": classify_root_python(path.name),
                "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "lifecycle_state": "reference_review_required",
                "deletion_allowed": False,
            }
        )
        by_month[match.group(1)[:6]] += 1
    return {
        "root_python_files": len(names),
        "dated_python_files": len(dated_registry),
        "prefix_counts": dict(sorted(prefixes.items())),
        "dated_month_counts": dict(sorted(by_month.items())),
        "dated_python_registry": dated_registry,
    }


def workspace_shell_stats(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted((item for item in root.iterdir() if item.is_file()), key=lambda item: item.name.lower()):
        stat = path.stat()
        files.append(
            {
                "path": path.name,
                "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "lifecycle_state": "reference_review_required",
                "deletion_allowed": False,
            }
        )
    directories = sorted(item.name for item in root.iterdir() if item.is_dir())
    return {
        "role": "workspace_shell_not_git_repository",
        "files": files,
        "directories": directories,
        "python_files": sum(item["path"].endswith(".py") for item in files),
    }


def agent_stats(root: Path) -> dict[str, Any]:
    skills = list((root / ".codex" / "skills").rglob("*"))
    packages = list((root / ".codex" / "agent_packages").rglob("*"))
    role_cards = list((root / ".codex" / "agents").glob("*-agent.md"))
    role_lines = {}
    for path in role_cards:
        try:
            role_lines[path.name] = len(path.read_text(encoding="utf-8").splitlines())
        except OSError:
            role_lines[path.name] = None
    return {
        "skill_files": sum(path.is_file() for path in skills),
        "package_files": sum(path.is_file() for path in packages),
        "compatibility_role_cards": len(role_cards),
        "compatibility_role_card_lines": role_lines,
        "canonical_rule_source": ".codex/agent_packages/<agent-id>/",
        "platform_entry_source": ".codex/skills/<agent-id>/SKILL.md",
    }


def immediate_candidates(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    empty_git = root / ".git"
    if empty_git.is_dir() and not any(empty_git.iterdir()):
        rows.append({"path": str(empty_git), "reason": "empty misleading git marker", "action": "remove_empty_dir"})

    for name in ("1)", "tmp-devtools-screen.png", "tmp_l2_buyday_probe.py", "tmp_l5_l6_readback.py"):
        path = root / name
        if path.exists():
            rows.append({"path": str(path), "reason": "root temporary artifact", "action": "move_to_recycle_bin"})

    for relative in GENERATED_CANDIDATES:
        path = root / relative
        if path.exists():
            rows.append({"path": str(path), "reason": "regenerable or temporary tree", "action": "review_then_cleanup"})
    return rows


def build_inventory(root: Path) -> dict[str, Any]:
    data_root = root / "quant" / "data_file"
    site_root = root / "site"
    data_children = [tree_stats(path) for path in sorted(data_root.iterdir()) if path.is_dir()]
    site_children = [tree_stats(path) for path in sorted(site_root.iterdir()) if path.is_dir()]
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "project_root": str(root),
        "boundary": "read_only_inventory_no_move_no_delete_no_business_database_open",
        "git": [git_status(path) for path in (root, root / "quant" / "main", site_root, root / "wjr-quant-strategy-mcp")],
        "workspace_shell": workspace_shell_stats(root),
        "agents": agent_stats(root),
        "quant_main": root_python_stats(root / "quant" / "main"),
        "data_file_children": data_children,
        "site_children": site_children,
        "large_reports": large_file_stats(data_root / "reports"),
        "large_runtime": large_file_stats(data_root / "runtime"),
        "immediate_candidates": immediate_candidates(root),
        "protected_paths": [str(root / path) for path in PROTECTED_PATHS],
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# 全项目治理库存",
        "",
        f"生成时间：`{data['generated_at']}`",
        "",
        "本报告只读生成，不移动、不删除文件，也不打开业务数据库。",
        "",
        "## 摘要",
        "",
        f"- `quant/main` 顶层 Python：{data['quant_main']['root_python_files']}，其中带日期：{data['quant_main']['dated_python_files']}。",
        f"- workspace shell 顶层文件：{len(data['workspace_shell']['files'])}，其中 Python：{data['workspace_shell']['python_files']}。",
        f"- `reports` 大于等于 1 GiB 文件：{data['large_reports']['count']}，合计 {data['large_reports']['human']}。",
        f"- `runtime` 大于等于 1 GiB 文件：{data['large_runtime']['count']}，合计 {data['large_runtime']['human']}。",
        f"- AGENT 权威文档包文件：{data['agents']['package_files']}；兼容角色卡：{data['agents']['compatibility_role_cards']}。",
        "",
        "## Git 边界",
        "",
        "| 路径 | Git仓库 | tracked变化 | untracked |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in data["git"]:
        lines.append(
            f"| `{item['path']}` | `{str(item['is_git_repository']).lower()}` | {item['tracked_changes']} | {item['untracked_entries']} |"
        )
    lines.extend(["", "## 立即治理候选", "", "| 路径 | 建议 | 原因 |", "| --- | --- | --- |"]) 
    for item in data["immediate_candidates"]:
        lines.append(f"| `{item['path']}` | `{item['action']}` | {item['reason']} |")
    lines.extend(["", "## 保护路径", ""])
    lines.extend(f"- `{path}`" for path in data["protected_paths"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a read-only whole-project governance inventory.")
    parser.add_argument("--project-root", default=str(project_root()))
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    data = build_inventory(root)

    if args.json_output:
        output = Path(args.json_output)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        output = Path(args.markdown_output)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(data), encoding="utf-8")

    print(
        json.dumps(
            {
                "boundary": data["boundary"],
                "root_python_files": data["quant_main"]["root_python_files"],
                "dated_python_files": data["quant_main"]["dated_python_files"],
                "large_reports": data["large_reports"]["count"],
                "large_runtime": data["large_runtime"]["count"],
                "immediate_candidates": len(data["immediate_candidates"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
