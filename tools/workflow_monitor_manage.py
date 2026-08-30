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

try:
    from workflow_route_guard import validate_template_execution_contract
    from workflow_autopilot import decide_next_action
except ModuleNotFoundError:  # direct script execution from quant/main/tools
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from workflow_route_guard import validate_template_execution_contract
    from workflow_autopilot import decide_next_action


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


def workflow_template_dir(root: Path) -> Path:
    return root / "quant" / "main" / "workflows"


def workflow_template_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    primary = workflow_template_dir(root)
    fallback = workflow_template_dir(project_root())
    for item in (primary, fallback):
        if item not in dirs:
            dirs.append(item)
    return dirs


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "workflow"


def record_path(root: Path, workflow_id: str) -> Path:
    date_part = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    return monitor_dir(root) / f"workflow_monitor_{date_part}_{workflow_id}.json"


def load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8-sig",
    )


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def make_agent(
    agent_id: str,
    task: str,
    eta_minutes: int | None,
    *,
    layer: str | None = None,
    owner_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {"agent_id": agent_id, **DEFAULT_AGENT_TASK}
    row["task"] = task
    row["assigned_at"] = now_iso()
    row["eta_minutes"] = eta_minutes
    if layer is not None:
        row["layer"] = layer
    if owner_contract is not None:
        row["owner_contract"] = owner_contract
    return row


def list_workflow_templates(root: Path) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in workflow_template_dirs(root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            template_id = str(payload.get("template_id") or path.stem)
            if template_id in seen:
                continue
            seen.add(template_id)
            templates.append(
                {
                    "template_id": template_id,
                    "workflow_name": str(payload.get("workflow_name") or path.stem),
                    "path": str(path),
                }
            )
    return templates


def load_workflow_template(root: Path, template_id: str) -> dict[str, Any]:
    for directory in workflow_template_dirs(root):
        candidates = [
            directory / f"{template_id}.json",
        ]
        if directory.is_dir():
            for path in directory.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                if str(payload.get("template_id") or "") == template_id:
                    return payload
        for path in candidates:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8-sig"))
    available = ", ".join(item["template_id"] for item in list_workflow_templates(root))
    raise FileNotFoundError(f"unknown workflow template: {template_id}. available: {available}")


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


def create_record_from_template(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    template = load_workflow_template(root, args.template_id)
    target_trade_date = getattr(args, "target_trade_date", None)
    if not target_trade_date and args.workflow_id:
        match = re.search(r"(?:^|-)((?:19|20)\d{6})(?:-|$)", args.workflow_id)
        target_trade_date = match.group(1) if match else None
    if not target_trade_date:
        print("FAIL target_trade_date is required for incremental workflow templates")
        return 1
    route = json.loads(json.dumps(template.get("execution_route"), ensure_ascii=False))
    route_text = json.dumps(route, ensure_ascii=False).replace("{target_trade_date}", str(target_trade_date))
    template["execution_route"] = json.loads(route_text)
    route_errors = validate_template_execution_contract(template)
    if route_errors:
        print("FAIL invalid workflow execution route:")
        for error in route_errors:
            print(f"- {error}")
        return 1
    workflow_name = args.workflow_name or str(template["workflow_name"])
    workflow_id = args.workflow_id or f"workflow-{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')}-{slug(workflow_name)}"
    path = record_path(root, workflow_id)
    if path.exists() and not args.overwrite:
        print(f"FAIL workflow monitor already exists: {path}")
        return 1

    agents = []
    for item in template.get("agents", []):
        agents.append(
            make_agent(
                str(item["agent_id"]),
                str(item["task"]),
                item.get("eta_minutes"),
                layer=str(item.get("layer")) if item.get("layer") is not None else None,
                owner_contract={
                    "layer": item.get("layer"),
                    "template_id": template.get("template_id"),
                },
            )
        )

    record = {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "created_at": now_iso(),
        "created_by": "commander-agent",
        "status": "pending",
        "user_goal": args.user_goal,
        "matched_workflow": str(template["matched_workflow"]),
        "requires_user_approval": bool(template.get("requires_user_approval", False)),
        "workflow_template": str(template.get("template_id") or args.template_id),
        "workflow_contract": {
            "description": template.get("description"),
            "hard_rules": template.get("hard_rules", []),
            "gates": template.get("gates", []),
            "execution_scope": template.get("execution_scope"),
            "execution_profile": template.get("execution_profile"),
            "full_history_rebuild": template.get("full_history_rebuild"),
            "execution_route": template.get("execution_route"),
            "authorization_policy": (template.get("execution_route") or {}).get("authorization_policy"),
        },
        "autopilot": {
            "enabled": True,
            "current_layer": "L1",
            "state": "ready_to_dispatch",
            "business_retries_used": {},
            "last_decision": None,
        },
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


def list_templates_command(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    templates = list_workflow_templates(root)
    print(json.dumps({"templates": templates}, ensure_ascii=False, indent=2))
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


def advance_record(args: argparse.Namespace) -> int:
    """Apply one verified event to the routine workflow state machine."""

    root = Path(args.project_root).resolve()
    path = Path(args.file)
    if not path.is_absolute():
        path = monitor_dir(root) / path
    if not path.is_file():
        print(f"FAIL missing workflow monitor: {path}")
        return 1

    record = load_record(path)
    autopilot = record.get("autopilot")
    if not isinstance(autopilot, dict) or not autopilot.get("enabled"):
        print("FAIL workflow autopilot is not enabled")
        return 1

    retries = autopilot.setdefault("business_retries_used", {})
    retries_used = int(retries.get(args.layer, 0))
    decision = decide_next_action(
        layer=args.layer,
        event=args.event,
        contract_valid=args.contract_valid,
        audit_passed=args.audit_passed,
        failure_class=args.failure_class,
        retries_used=retries_used,
        retry_limit=1,
    )
    if decision.consume_business_retry:
        retries[args.layer] = retries_used + 1

    decision_payload = decision.as_dict()
    decision_payload["decided_at"] = now_iso()
    autopilot["last_decision"] = decision_payload
    autopilot["state"] = decision.workflow_status
    autopilot["current_layer"] = decision.next_layer or args.layer

    if decision.action == "dispatch_next_layer" and decision.next_layer:
        layer_agent = next(
            (item for item in record.get("agents", []) if item.get("layer") == decision.next_layer),
            None,
        )
        if layer_agent is not None:
            layer_agent["status"] = "in_progress"
            layer_agent["last_update_at"] = decision_payload["decided_at"]
            if not layer_agent.get("started_at"):
                layer_agent["started_at"] = decision_payload["decided_at"]
        record["status"] = "in_progress"
    elif decision.action == "request_audit":
        record["status"] = "reviewing"
    elif decision.action == "redispatch_control_message":
        record["status"] = "in_progress"
    elif decision.action == "remediate_and_retry_same_layer":
        record["status"] = "in_progress"
    elif decision.action == "hard_stop":
        record["status"] = "failed"
        record.setdefault("open_risks", []).append(
            {
                "layer": args.layer,
                "type": args.failure_class or decision.reason,
                "reason": decision.reason,
                "recorded_at": decision_payload["decided_at"],
            }
        )
    elif decision.action == "complete_workflow":
        record["status"] = "completed"
        record["completed_at"] = decision_payload["decided_at"]

    write_record(path, record)
    print(json.dumps(decision_payload, ensure_ascii=False))
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

    create_from_template = subparsers.add_parser(
        "create-from-template",
        help="Create a workflow monitor JSON from a standard workflow template.",
    )
    create_from_template.add_argument("--template-id", required=True)
    create_from_template.add_argument("--workflow-id", default=None)
    create_from_template.add_argument("--workflow-name", default=None)
    create_from_template.add_argument("--user-goal", required=True)
    create_from_template.add_argument("--target-trade-date", default=None)
    create_from_template.add_argument("--overwrite", action="store_true")
    create_from_template.set_defaults(func=create_record_from_template)

    list_templates = subparsers.add_parser("list-templates", help="List available workflow templates.")
    list_templates.set_defaults(func=list_templates_command)

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

    advance = subparsers.add_parser(
        "advance",
        help="Apply a verified layer/audit/control event and update autopilot state.",
    )
    advance.add_argument("--file", required=True)
    advance.add_argument("--layer", required=True)
    advance.add_argument(
        "--event",
        required=True,
        choices=("layer_completed", "audit_completed", "control_thread_timeout", "stage_failed"),
    )
    advance.add_argument("--contract-valid", action="store_true")
    advance.add_argument("--audit-passed", action="store_true")
    advance.add_argument("--failure-class", default=None)
    advance.set_defaults(func=advance_record)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
