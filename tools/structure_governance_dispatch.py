"""Build per-agent structure-governance review packages.

This tool is read-only with respect to source files. It reads previously
generated governance reports and writes review packages under reports/.
It does not move, delete, archive, or execute business workflows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AGENT_NAMES = {
    "architect-agent": "架构师智能体",
    "audit-agent": "审计智能体",
    "data-warehouse-agent": "数仓智能体",
    "research-agent": "投研智能体",
    "strategy-agent": "策略智能体",
    "commander-agent": "指挥官智能体",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_for_agent(agent_id: str, candidates: list[dict[str, Any]]) -> str:
    name = AGENT_NAMES.get(agent_id, agent_id)
    counts = Counter(item["category"] for item in candidates)
    lines = [
        f"# {name}（{agent_id}）文件治理认领包",
        "",
        "## 边界",
        "",
        "本认领包仅用于只读核验和处置建议。未获指挥官和审计确认前，不得移动、归档或回收候选文件。",
        "",
        "不得触发业务脚本，不得补数据、算因子、训练模型、生成预测、生成信号或跑回测。",
        "",
        "## 统计",
        "",
        f"- 候选数量：{len(candidates)}",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`：{count}")

    lines.extend(
        [
            "",
            "## 回执格式",
            "",
            "逐项或分批回复：",
            "",
            "```text",
            "会话来自：<中文智能体名称>（<agent-id>）",
            "",
            "文件治理回执：",
            "- 文件路径：<path>",
            "- 判定：Keep / Move / Archive / Move to recycle bin",
            "- 判定依据：<说明是否仍被标准链路、测试、manifest、文档、报告、策略归档引用>",
            "- 建议目标路径：<如适用>",
            "- 风险：P0/P1/P2/P3 或无",
            "- 是否需要指挥官/用户审批：是/否",
            "```",
            "",
            "## 候选清单",
            "",
            "| 序号 | 文件 | 类型 | 建议动作 |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for index, item in enumerate(candidates, start=1):
        lines.append(
            f"| {index} | `quant/main/{item['path']}` | `{item['category']}` | `{item['suggested_action']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def markdown_for_runtime(candidates: list[dict[str, Any]]) -> str:
    counts = Counter(item["category"] for item in candidates)
    lines = [
        "# runtime 目录治理复核包",
        "",
        "## 边界",
        "",
        "本复核包用于确认 `quant/data_file/runtime` 根目录下各目录的保留、归档或回收建议。",
        "",
        "不得清理 `agent_memory`、`agent_observability`、`agent_workspaces`、`orchestrator_reports`、`recycle_bin`、`trading_agent` 等长期保留目录。",
        "",
        "## 统计",
        "",
        f"- 候选数量：{len(candidates)}",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`：{count}")

    lines.extend(
        [
            "",
            "## 候选清单",
            "",
            "| 序号 | 目录 | 类型 | 理由 | 建议动作 |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for index, item in enumerate(candidates, start=1):
        lines.append(
            f"| {index} | `quant/data_file/runtime/{item['path']}` | `{item['category']}` | {item['reason']} | `{item['suggested_action']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structure-governance dispatch packages.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    parser.add_argument(
        "--report-dir",
        default="quant/data_file/reports/project_structure_governance_20260626",
        help="Report directory containing quant_main_noise_report.json and runtime_noise_report.json.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = root / report_dir

    main_report = load_json(report_dir / "quant_main_noise_report.json")
    runtime_report = load_json(report_dir / "runtime_noise_report.json")

    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in main_report.get("candidates", []):
        by_agent[item["owner_agent"]].append(item)

    packages_dir = report_dir / "agent_review_packages"
    package_index: dict[str, Any] = {
        "boundary": "review_packages_only_no_file_moves",
        "packages": [],
    }

    for agent_id, items in sorted(by_agent.items()):
        json_path = packages_dir / f"{agent_id}_quant_main_candidates.json"
        md_path = packages_dir / f"{agent_id}_quant_main_candidates.md"
        payload = {
            "agent_id": agent_id,
            "agent_name": AGENT_NAMES.get(agent_id, agent_id),
            "candidate_count": len(items),
            "category_counts": dict(sorted(Counter(item["category"] for item in items).items())),
            "candidates": items,
            "boundary": "review_only_no_file_moves",
        }
        write_json(json_path, payload)
        write_markdown(md_path, markdown_for_agent(agent_id, items))
        package_index["packages"].append(
            {
                "agent_id": agent_id,
                "candidate_count": len(items),
                "json": str(json_path.relative_to(root)).replace("\\", "/"),
                "markdown": str(md_path.relative_to(root)).replace("\\", "/"),
            }
        )

    runtime_json = packages_dir / "runtime_candidates.json"
    runtime_md = packages_dir / "runtime_candidates.md"
    write_json(runtime_json, runtime_report)
    write_markdown(runtime_md, markdown_for_runtime(runtime_report.get("candidates", [])))
    package_index["runtime_package"] = {
        "json": str(runtime_json.relative_to(root)).replace("\\", "/"),
        "markdown": str(runtime_md.relative_to(root)).replace("\\", "/"),
    }

    write_json(packages_dir / "package_index.json", package_index)
    print(json.dumps(package_index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
