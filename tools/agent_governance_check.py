"""Validate multi-agent governance scaffolding.

This tool is read-only. It checks agent packages, runtime workspace
documentation, asset registry JSON files, platform AGENT entries, and
workflow monitor templates. It does not touch business data or execute
quant workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


AGENTS = [
    "commander-agent",
    "architect-agent",
    "audit-agent",
    "data-ingestion-agent",
    "data-integration-agent",
    "factor-agent",
    "model-agent",
    "strategy-agent",
    "trading-agent",
    "research-agent",
]

BASE_PACKAGE_FILES = [
    "README.md",
    "capabilities.md",
    "skills.md",
    "contract.md",
    "codex-automations.md",
    "tools.md",
    "permissions.md",
    "knowledge.md",
    "procedures.md",
    "handoff.md",
    "logging.md",
    "audit-checklist.md",
]

SPECIAL_FILES = [
    ".codex/agent_packages/README.md",
    ".codex/agent_packages/registry.md",
    ".codex/agent_packages/common-rules.md",
    ".codex/agent_packages/workflow-skill-improvement-loop.md",
    ".codex/agent_packages/commander-agent/workflows.md",
    ".codex/agent_packages/commander-agent/dispatch-rules.md",
    "quant/main/docs/governance/standard-agent-architecture.md",
    "quant/main/docs/governance/thread-based-agent-management.md",
    "quant/main/docs/governance/skill-system.md",
    "quant/main/tools/standard_agent_architecture_check.py",
    "quant/main/tools/workflow_monitor_report.py",
    "quant/main/tools/workflow_monitor_manage.py",
    "quant/main/tools/repo_noise_report.py",
    "quant/main/tools/runtime_noise_report.py",
    "quant/main/tools/runtime_loose_files_report.py",
    "quant/main/tools/structure_governance_dispatch.py",
    "quant/main/tools/prepare_structure_manifest_drafts.py",
    "quant/main/tools/production_asset_gate.py",
    "quant/main/core/README.md",
    "quant/main/workflows/README.md",
    "quant/main/agent_tools/README.md",
    "quant/main/research/README.md",
    "quant/main/research/archive/README.md",
    "quant/main/legacy/README.md",
    "quant/data_file/runtime/README.md",
    "quant/data_file/runtime/archive/README.md",
    "quant/data_file/runtime/agent_workspaces/README.md",
    "quant/main/labs/README.md",
    "quant/data_file/runtime/agent_memory/agent_thread_registry.json",
    "quant/data_file/production_assets/README.md",
    "quant/data_file/experimental_assets/README.md",
    "quant/data_file/asset_registry/README.md",
    "quant/data_file/runtime/orchestrator_reports/workflow_monitor/README.md",
    "quant/data_file/runtime/orchestrator_reports/workflow_monitor/workflow_monitor.template.md",
]

JSON_FILES = [
    "quant/data_file/asset_registry/production_assets.json",
    "quant/data_file/asset_registry/experimental_assets.json",
    "quant/data_file/asset_registry/manifest.schema.json",
    "quant/data_file/runtime/orchestrator_reports/workflow_monitor/workflow_monitor.template.json",
    "quant/data_file/runtime/agent_memory/agent_thread_registry.json",
]

KEYWORD_CHECKS = {
    ".codex/agent_packages/common-rules.md": [
        "工作流与技能改进闭环",
        "脚本改进闭环",
        "显式指定 manifest",
        "每个 Codex 会话线程必须绑定一个智能体身份",
        "agent_thread_registry.json",
        "审计智能体只在审计绑定线程输出结论",
        "正式资产发生任何变动之前",
    ],
    ".codex/agents/communication-layer.md": [
        "会话来自：本体",
        "任务分派模板",
        "权限申请模板",
        "每个 Codex 会话线程必须绑定一个智能体身份",
        "agent_thread_registry.json",
    ],
    ".codex/agent_packages/commander-agent/workflows.md": [
        "工作流与技能改进闭环",
        "多智能体初始化治理工作流",
        "会话线程档案是初始化前置资产",
        "低风险辅助脚本",
        "正式资产变动审计门禁",
    ],
    ".codex/agent_packages/commander-agent/procedures.md": [
        "初始化前置检查",
        "agent_thread_registry.json",
        "agent_governance_check.py",
    ],
    "WORKFLOW.md": [
        "初始化前置治理检查",
        "校验会话线程档案",
        "agent_governance_check.py",
    ],
    "quant/data_file/runtime/agent_memory/README.md": [
        "初始化要求",
        "agent_thread_registry.json",
        "每个正式智能体必须有唯一 active 会话线程",
    ],
    "quant/data_file/asset_registry/README.md": [
        "正式资产变动审计要求",
        "没有审计记录的变动不得进入主线工作流",
    ],
    "quant/main/docs/governance/skill-system.md": [
        "本项目不保留泛项目级业务 SKILL",
        "AGENT 入口",
        "智能体内部技能",
        "放置规则",
        "执行步骤",
        "权限与停止条件",
    ],
    ".codex/agent_packages/README.md": [
        "具体可复用能力必须归入对应智能体文档包的内部技能",
        "规则放置",
        "详细执行流程",
        "权限、禁止事项、停止条件",
    ],
    ".codex/agent_packages/factor-agent/skills.md": [
        "因子治理审计",
        "因子增量更新",
        "GTJA 公式审计",
        "详细流程见 `procedures.md`",
        "工具和脚本入口见 `tools.md`",
    ],
}

FORBIDDEN_FACTOR_SKILL_MARKERS = [
    "标准入口：",
    "统一 Python：",
    "停止条件：",
    "优先核查文件：",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def rel(root: Path, path: str) -> Path:
    return root / Path(path)


def check_exists(root: Path, paths: Iterable[str], errors: list[str]) -> None:
    for path in paths:
        if not rel(root, path).exists():
            errors.append(f"missing: {path}")


def check_agent_packages(root: Path, errors: list[str]) -> None:
    for agent in AGENTS:
        for filename in BASE_PACKAGE_FILES:
            path = f".codex/agent_packages/{agent}/{filename}"
            if not rel(root, path).is_file():
                errors.append(f"missing agent package file: {path}")

        workspace_dir = f"quant/data_file/runtime/agent_workspaces/{agent}"
        if not rel(root, workspace_dir).is_dir():
            errors.append(f"missing workspace dir: {workspace_dir}")

        labs_dir = f"quant/main/labs/{agent}"
        if not rel(root, labs_dir).is_dir():
            errors.append(f"missing labs dir: {labs_dir}")

        note = f"quant/data_file/runtime/agent_memory/agent_notes/{agent}.md"
        if not rel(root, note).is_file():
            errors.append(f"missing agent note: {note}")


def check_platform_agent_entries(root: Path, errors: list[str]) -> None:
    skills_dir = rel(root, ".codex/skills")
    if not skills_dir.is_dir():
        errors.append("missing platform AGENT entry dir: .codex/skills")
        return

    entry_names = sorted(path.name for path in skills_dir.iterdir() if path.is_dir())
    expected = sorted(AGENTS)
    if entry_names != expected:
        extra = sorted(set(entry_names) - set(expected))
        missing = sorted(set(expected) - set(entry_names))
        if extra:
            errors.append(f"platform .codex/skills has non-agent entries: {extra}")
        if missing:
            errors.append(f"platform .codex/skills missing agent entries: {missing}")

    for agent in AGENTS:
        skill_file = skills_dir / agent / "SKILL.md"
        if not skill_file.is_file():
            errors.append(f"missing platform AGENT SKILL.md: .codex/skills/{agent}/SKILL.md")
            continue
        text = skill_file.read_text(encoding="utf-8")
        if "只负责平台识别和路由" not in text:
            errors.append(f"platform AGENT entry must declare route-only scope: {agent}")


def check_json(root: Path, errors: list[str]) -> None:
    for path in JSON_FILES:
        full_path = rel(root, path)
        if not full_path.is_file():
            errors.append(f"missing json: {path}")
            continue
        try:
            json.loads(full_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid json: {path}: {exc}")


def check_thread_registry(root: Path, errors: list[str]) -> None:
    path = "quant/data_file/runtime/agent_memory/agent_thread_registry.json"
    full_path = rel(root, path)
    if not full_path.is_file():
        errors.append(f"missing thread registry: {path}")
        return

    try:
        data = json.loads(full_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid thread registry json: {path}: {exc}")
        return

    policy = data.get("thread_policy", {})
    if policy.get("single_owner_per_thread") is not True:
        errors.append("thread registry policy must set single_owner_per_thread=true")
    if policy.get("cross_agent_dialog_in_agent_thread_allowed") is not False:
        errors.append(
            "thread registry policy must set "
            "cross_agent_dialog_in_agent_thread_allowed=false"
        )

    threads = data.get("threads")
    if not isinstance(threads, list):
        errors.append("thread registry must contain a threads list")
        return

    agent_ids: list[str] = []
    thread_ids: list[str] = []
    for index, item in enumerate(threads):
        if not isinstance(item, dict):
            errors.append(f"thread registry item #{index} must be an object")
            continue

        agent_id = item.get("agent_id")
        thread_id = item.get("thread_id")
        status = item.get("thread_status", item.get("status"))
        thread_title = item.get("thread_title")
        if agent_id not in AGENTS:
            errors.append(f"thread registry contains unknown agent_id: {agent_id}")
        if not isinstance(thread_id, str) or not thread_id.strip():
            errors.append(f"thread registry missing thread_id for agent: {agent_id}")
        if not isinstance(thread_title, str) or not thread_title.strip():
            errors.append(f"thread registry missing thread_title for agent: {agent_id}")
        if status != "active":
            errors.append(f"thread registry agent is not active: {agent_id}")

        if isinstance(agent_id, str):
            agent_ids.append(agent_id)
        if isinstance(thread_id, str):
            thread_ids.append(thread_id)

    missing_agents = sorted(set(AGENTS) - set(agent_ids))
    extra_agents = sorted(set(agent_ids) - set(AGENTS))
    duplicate_agents = sorted(
        agent_id for agent_id in set(agent_ids) if agent_ids.count(agent_id) > 1
    )
    duplicate_threads = sorted(
        thread_id for thread_id in set(thread_ids) if thread_ids.count(thread_id) > 1
    )

    for agent_id in missing_agents:
        errors.append(f"thread registry missing agent: {agent_id}")
    for agent_id in extra_agents:
        errors.append(f"thread registry has extra agent: {agent_id}")
    for agent_id in duplicate_agents:
        errors.append(f"thread registry duplicate agent: {agent_id}")
    for thread_id in duplicate_threads:
        errors.append(f"thread registry duplicate thread_id: {thread_id}")


def check_keywords(root: Path, errors: list[str]) -> None:
    for path, keywords in KEYWORD_CHECKS.items():
        full_path = rel(root, path)
        if not full_path.is_file():
            errors.append(f"missing keyword target: {path}")
            continue
        text = full_path.read_text(encoding="utf-8")
        for keyword in keywords:
            if keyword not in text:
                errors.append(f"missing keyword in {path}: {keyword}")


def check_internal_skill_placement(root: Path, errors: list[str]) -> None:
    factor_skills = rel(root, ".codex/agent_packages/factor-agent/skills.md")
    if factor_skills.is_file():
        text = factor_skills.read_text(encoding="utf-8")
        for marker in FORBIDDEN_FACTOR_SKILL_MARKERS:
            if marker in text:
                errors.append(
                    "factor-agent skills.md contains procedural marker "
                    f"that belongs in procedures/tools/permissions: {marker}"
                )


def run_check(root: Path) -> list[str]:
    errors: list[str] = []
    check_exists(root, SPECIAL_FILES, errors)
    check_agent_packages(root, errors)
    check_platform_agent_entries(root, errors)
    check_json(root, errors)
    check_thread_registry(root, errors)
    check_keywords(root, errors)
    check_internal_skill_placement(root, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate agent governance scaffolding.")
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

    print(
        "PASS "
        f"agents={len(AGENTS)} "
        f"package_files_each={len(BASE_PACKAGE_FILES)} "
        f"json_files={len(JSON_FILES)} "
        "thread_registry=checked "
        "platform_agent_entries=checked "
        "internal_skill_placement=checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
