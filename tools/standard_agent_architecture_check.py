"""Read-only checks for the standard multi-agent architecture.

This script validates governance scaffolding only. It does not run business
workflows, touch market data, compute factors, train models, predict, generate
signals, backtest, or trade.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AGENTS = [
    "commander-agent",
    "architect-agent",
    "audit-agent",
    "data-ingestion-agent",
    "data-integration-agent",
    "data-warehouse-agent",
    "factor-agent",
    "model-agent",
    "strategy-agent",
    "trading-agent",
    "research-agent",
]

REQUIRED_MANIFEST_FIELDS = {
    "asset_id",
    "track",
    "layer",
    "asset_type",
    "owner_agent",
    "status",
    "allowed_for_main_workflow",
    "asset_path",
    "created_at",
    "audit_record",
}

WORKFLOW_MONITOR_REQUIRED_FIELDS = {
    "workflow_id",
    "status",
    "matched_workflow",
    "agents",
    "production_asset_changes",
    "improvement_items",
}

ASSET_CHANGE_REQUIRED_FIELDS = {
    "change_id",
    "change_type",
    "layer",
    "owner_agent",
    "audit_required",
    "audit_status",
    "audit_record",
    "audit_thread_closure_status",
    "audit_reply_required",
    "status",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing json: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid json: {path}: {exc}")
        return None


def check_agent_packages(root: Path, errors: list[str]) -> None:
    for agent in AGENTS:
        package_dir = root / ".codex" / "agent_packages" / agent
        skill_dir = root / ".codex" / "skills" / agent
        if not (package_dir / "capabilities.md").is_file():
            errors.append(f"missing capabilities.md: {package_dir}")
        if not (package_dir / "contract.md").is_file():
            errors.append(f"missing contract.md: {package_dir}")
        if not (package_dir / "permissions.md").is_file():
            errors.append(f"missing permissions.md: {package_dir}")
        if not (package_dir / "handoff.md").is_file():
            errors.append(f"missing handoff.md: {package_dir}")
        if not (skill_dir / "SKILL.md").is_file():
            errors.append(f"missing platform entry SKILL.md: {skill_dir}")


def check_asset_registry(root: Path, errors: list[str]) -> None:
    registry_dir = root / "quant" / "data_file" / "asset_registry"
    production = load_json(registry_dir / "production_assets.json", errors)
    experimental = load_json(registry_dir / "experimental_assets.json", errors)
    schema = load_json(registry_dir / "manifest.schema.json", errors)

    if schema is not None:
        required = set(schema.get("required_fields", []))
        missing = REQUIRED_MANIFEST_FIELDS - required
        if missing:
            errors.append(
                "manifest.schema.json missing required fields: "
                + ", ".join(sorted(missing))
            )

    if production is not None:
        if production.get("audit_required_before_change") is not True:
            errors.append("production_assets.json must require audit before change")
        for asset in production.get("assets", []):
            check_asset_manifest(asset, "production", errors)
            if asset.get("allowed_for_main_workflow") is not True:
                errors.append(
                    f"production asset not allowed for main workflow: {asset.get('asset_id')}"
                )
            if not asset.get("audit_record"):
                errors.append(
                    f"production asset missing audit_record: {asset.get('asset_id')}"
                )

    if experimental is not None:
        if experimental.get("audit_required_before_promotion") is not True:
            errors.append("experimental_assets.json must require audit before promotion")
        for asset in experimental.get("assets", []):
            check_asset_manifest(asset, "experiment", errors)
            if asset.get("allowed_for_main_workflow") is not False:
                errors.append(
                    f"experimental asset allowed for main workflow: {asset.get('asset_id')}"
                )


def check_asset_manifest(asset: dict[str, Any], expected_track: str, errors: list[str]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(asset)
    asset_id = asset.get("asset_id", "<unknown>")
    if missing:
        errors.append(f"asset {asset_id} missing fields: {', '.join(sorted(missing))}")
    if asset.get("track") != expected_track:
        errors.append(
            f"asset {asset_id} has track={asset.get('track')}, expected {expected_track}"
        )


def check_workflow_monitor(root: Path, errors: list[str]) -> None:
    monitor_path = (
        root
        / "quant"
        / "data_file"
        / "runtime"
        / "orchestrator_reports"
        / "workflow_monitor"
        / "workflow_monitor.template.json"
    )
    monitor = load_json(monitor_path, errors)
    if monitor is None:
        return

    missing = WORKFLOW_MONITOR_REQUIRED_FIELDS - set(monitor)
    if missing:
        errors.append(
            "workflow_monitor.template.json missing fields: "
            + ", ".join(sorted(missing))
        )

    changes = monitor.get("production_asset_changes", [])
    if not changes:
        errors.append("workflow monitor template must include production_asset_changes")
        return

    sample = changes[0]
    missing_change_fields = ASSET_CHANGE_REQUIRED_FIELDS - set(sample)
    if missing_change_fields:
        errors.append(
            "production asset change template missing fields: "
            + ", ".join(sorted(missing_change_fields))
        )

    allowed_closure = {"not_required", "pending", "sent", "confirmed"}
    closure = sample.get("audit_thread_closure_status")
    if isinstance(closure, str):
        template_values = {part.strip() for part in closure.split("/")}
        if not allowed_closure.issubset(template_values):
            errors.append(
                "audit_thread_closure_status template must mention "
                + "/".join(sorted(allowed_closure))
            )


def check_governance_docs(root: Path, errors: list[str]) -> None:
    docs = [
        root / "quant" / "main" / "docs" / "governance" / "standard-agent-architecture.md",
        root / "quant" / "main" / "docs" / "governance" / "project-doc-map.md",
        root / "quant" / "main" / "docs" / "governance" / "skill-system.md",
        root / "quant" / "main" / "docs" / "governance" / "markdown-document-registry.md",
        root / "quant" / "main" / "tools" / "production_asset_gate.py",
        root / "quant" / "main" / "core" / "README.md",
        root / "quant" / "main" / "workflows" / "README.md",
        root / "quant" / "main" / "agent_tools" / "README.md",
        root / "quant" / "main" / "research" / "README.md",
        root / "quant" / "main" / "legacy" / "README.md",
    ]
    for path in docs:
        if not path.is_file():
            errors.append(f"missing governance doc: {path}")


def run_check(root: Path) -> list[str]:
    errors: list[str] = []
    check_governance_docs(root, errors)
    check_agent_packages(root, errors)
    check_asset_registry(root, errors)
    check_workflow_monitor(root, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate standard multi-agent architecture governance."
    )
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    errors = run_check(root)
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1

    print("PASS standard_agent_architecture_check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
