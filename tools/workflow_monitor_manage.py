"""Create and update workflow monitor records.

This is a governance utility. It writes monitor JSON only and does not execute
business workflows.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


VALID_STATUS = {
    "pending",
    "in_progress",
    "waiting_input",
    "blocked",
    "reviewing",
    "completed",
    "failed",
    "cancelled",
}

DEFAULT_AGENT_TASK = {
    "task": "task_description",
    "status": "pending",
    "assigned_at": None,
    "started_at": None,
    "eta_minutes": None,
    "last_update_at": None,
    "completed_at": None,
    "blocked_reason": None,
    "blocker_type": None,
    "improvement_required": False,
    "improvement_docs": [],
    "improvement_scripts": [],
    "evidence": [],
    "verification": None,
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


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


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "workflow"


def record_path(root: Path, workflow_id: str) -> Path:
    date_part = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    return monitor_dir(root) / f"workflow_monitor_{date_part}_{workflow_id}.json"


def load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def make_agent(agent_id: str, task: str, eta_minutes: int | None) -> dict[str, Any]:
    row = {"agent_id": agent_id, **DEFAULT_AGENT_TASK}
    row["task"] = task
    row["assigned_at"] = now_iso()
    row["eta_minutes"] = eta_minutes
    return row


def create_record(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    workflow_id = args.workflow_id or f"workflow-{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')}-{slug(args.workflow_name)}"
    path = record_path(root, workflow_id)
    if path.exists() and not args.overwrite:
        print(f"FAIL workflow monitor already exists: {path}")
        return 1

    agents = [
        make_agent(agent_id, args.agent_task or args.workflow_name, args.eta_minutes)
        for agent_id in split_csv(args.agents)
    ]
    record = {
        "workflow_id": workflow_id,
        "workflow_name": args.workflow_name,
        "created_at": now_iso(),
        "created_by": "commander-agent",
        "status": "pending",
        "user_goal": args.user_goal,
        "matched_workflow": args.matched_workflow,
        "requires_user_approval": args.requires_user_approval,
        "agents": agents,
        "open_risks": [],
        "approval_requests": [],
        "production_asset_changes": [],
        "improvement_items": [],
        "final_summary": None,
    }
    write_record(path, record)
    print(path)
    return 0


def find_agent(record: dict[str, Any], agent_id: str) -> dict[str, Any]:
    for agent in record.get("agents", []):
        if agent.get("agent_id") == agent_id:
            return agent
    agent = make_agent(agent_id, "task_description", None)
    record.setdefault("agents", []).append(agent)
    return agent


def update_agent(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    path = Path(args.file)
    if not path.is_absolute():
        path = monitor_dir(root) / path
    if not path.is_file():
        print(f"FAIL missing workflow monitor: {path}")
        return 1
    if args.status not in VALID_STATUS:
        print(f"FAIL invalid status: {args.status}")
        return 1

    record = load_record(path)
    agent = find_agent(record, args.agent_id)
    previous_status = agent.get("status")
    agent["status"] = args.status
    agent["last_update_at"] = now_iso()
    if args.task:
        agent["task"] = args.task
    if args.eta_minutes is not None:
        agent["eta_minutes"] = args.eta_minutes
    if args.blocked_reason:
        agent["blocked_reason"] = args.blocked_reason
    if args.blocker_type:
        agent["blocker_type"] = args.blocker_type
    if args.evidence:
        agent.setdefault("evidence", []).extend(split_csv(args.evidence))
    if args.verification:
        agent["verification"] = args.verification
    if args.status == "in_progress" and not agent.get("started_at"):
        agent["started_at"] = agent["last_update_at"]
    if args.status in {"completed", "failed", "cancelled"}:
        agent["completed_at"] = agent["last_update_at"]

    if record.get("status") == "pending" and args.status == "in_progress":
        record["status"] = "in_progress"
    if previous_status != args.status:
        record.setdefault("open_risks", [])
    write_record(path, record)
    print(path)
    return 0


def complete_record(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    path = Path(args.file)
    if not path.is_absolute():
        path = monitor_dir(root) / path
    if not path.is_file():
        print(f"FAIL missing workflow monitor: {path}")
        return 1
    if args.status not in {"completed", "failed", "cancelled"}:
        print(f"FAIL invalid final status: {args.status}")
        return 1
    record = load_record(path)
    record["status"] = args.status
    record["final_summary"] = args.summary
    record["completed_at"] = now_iso()
    write_record(path, record)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or update workflow monitor records.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a workflow monitor JSON.")
    create.add_argument("--workflow-id", default=None)
    create.add_argument("--workflow-name", required=True)
    create.add_argument("--user-goal", required=True)
    create.add_argument("--matched-workflow", required=True)
    create.add_argument("--agents", required=True, help="Comma-separated agent ids.")
    create.add_argument("--agent-task", default=None)
    create.add_argument("--eta-minutes", type=int, default=None)
    create.add_argument("--requires-user-approval", action="store_true")
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(func=create_record)

    update = subparsers.add_parser("update-agent", help="Update one agent row.")
    update.add_argument("--file", required=True)
    update.add_argument("--agent-id", required=True)
    update.add_argument("--status", required=True)
    update.add_argument("--task", default=None)
    update.add_argument("--eta-minutes", type=int, default=None)
    update.add_argument("--blocked-reason", default=None)
    update.add_argument("--blocker-type", default=None)
    update.add_argument("--evidence", default=None, help="Comma-separated evidence paths.")
    update.add_argument("--verification", default=None)
    update.set_defaults(func=update_agent)

    complete = subparsers.add_parser("complete", help="Set final workflow state.")
    complete.add_argument("--file", required=True)
    complete.add_argument("--status", default="completed")
    complete.add_argument("--summary", required=True)
    complete.set_defaults(func=complete_record)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
