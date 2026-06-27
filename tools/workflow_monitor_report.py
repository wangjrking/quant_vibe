"""Summarize workflow monitor JSON files.

This tool is read-only. It scans workflow monitor records and reports task
status, blockers, audit closure status, and production asset change state.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def monitor_dir(root: Path) -> Path:
    return (
        root
        / "quant"
        / "data_file"
        / "runtime"
        / "orchestrator_reports"
        / "workflow_monitor"
    )


def load_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_dir():
        errors.append(f"missing monitor dir: {path}")
        return records, errors

    for file_path in sorted(path.glob("*.json")):
        if file_path.name.endswith(".template.json") or file_path.name == "workflow_monitor.template.json":
            continue
        try:
            records.append(json.loads(file_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid json: {file_path}: {exc}")
    return records, errors


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    workflow_status = Counter(record.get("status", "unknown") for record in records)
    agent_status: Counter[str] = Counter()
    blocker_count = 0
    audit_closure_status: Counter[str] = Counter()
    production_change_status: Counter[str] = Counter()

    for record in records:
        for agent in record.get("agents", []):
            agent_status[agent.get("status", "unknown")] += 1
            if agent.get("blocked_reason"):
                blocker_count += 1

        for change in record.get("production_asset_changes", []):
            production_change_status[change.get("status", "unknown")] += 1
            audit_closure_status[change.get("audit_thread_closure_status", "unknown")] += 1

    return {
        "workflow_count": len(records),
        "workflow_status": dict(sorted(workflow_status.items())),
        "agent_status": dict(sorted(agent_status.items())),
        "blocker_count": blocker_count,
        "production_change_status": dict(sorted(production_change_status.items())),
        "audit_thread_closure_status": dict(sorted(audit_closure_status.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize workflow monitor records.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    records, errors = load_records(monitor_dir(root))
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1

    summary = summarize(records)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"workflow_count={summary['workflow_count']}")
    print(f"workflow_status={summary['workflow_status']}")
    print(f"agent_status={summary['agent_status']}")
    print(f"blocker_count={summary['blocker_count']}")
    print(f"production_change_status={summary['production_change_status']}")
    print(f"audit_thread_closure_status={summary['audit_thread_closure_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
