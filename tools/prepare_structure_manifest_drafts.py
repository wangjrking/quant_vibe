"""Prepare governance manifest drafts for project structure cleanup.

This tool is read-only with respect to governed source assets. It writes draft
manifest JSON files under the reports directory to support later reviewed moves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_files(root: Path, predicates: Iterable[tuple[str, callable]]) -> dict[str, list[str]]:
    main_dir = root / "quant" / "main"
    file_names = sorted(path.name for path in main_dir.glob("research_*.py"))
    result: dict[str, list[str]] = {}
    for group_name, predicate in predicates:
        result[group_name] = [name for name in file_names if predicate(name)]
    return result


def diagnostics_predicate(name: str) -> bool:
    if "_20260620.py" in name:
        return True
    if "20260621" in name and ("confirm" in name or "probe" in name or "new5d" in name or "core10d" in name):
        return True
    if name in {
        "research_core10d_risk_sharpe_refine_20260622.py",
        "research_extend_10d_no_star_candidate_20260624.py",
        "research_extend_1d3d_candidates_20260624.py",
        "research_latest5d10d_manifest_grid_20260621.py",
    }:
        return True
    return False


def recycle_20260623_predicate(name: str) -> bool:
    if "20260623" not in name:
        return False
    excluded = {
        "research_candidate_stability_scan_20260623.py",
        "research_formal_l4_goal_bucket_scale_20260623.py",
        "research_formal_l4_high_annual_goal_20260623.py",
        "research_formal_l4_latest_5d10d_allow_st_probe_20260623.py",
        "research_formal_l4_latest_5d10d_stclean_refill_20260623.py",
    }
    if name in excluded:
        return False
    keywords = ("gate", "proxy", "topgate", "local_reblend", "daymix", "dategate", "reblend", "guard")
    return any(keyword in name for keyword in keywords)


def recycle_20260626_predicate(name: str) -> bool:
    return "20260626" in name


def prod_micro_predicate(name: str) -> bool:
    if name.startswith("research_prod_") and "20260625" in name:
        return True
    return name in {
        "research_amt8w_execution_rescue_20260625.py",
        "research_candidate_stability_scan_20260623.py",
        "research_5d10d_robustness_matrix_20260621.py",
    }


def build_runtime_archive_manifest(root: Path) -> dict:
    members = [
        "quant/data_file/runtime/snapshot_repro_sc1p00011",
        "quant/data_file/runtime/strategy_repro",
        "quant/data_file/runtime/strategy_snapshot_repro",
    ]
    return {
        "actor": "commander-agent",
        "approval_scope": "审计复核后执行",
        "batch_id": "draft_runtime_archive_20260627",
        "group_name": "runtime_archive_pending",
        "reason": "runtime 根目录 repro/snapshot 目录不再长期停留在根目录，但仍有复现与审计价值",
        "family_rule": "runtime/{snapshot_repro_*,strategy_repro,strategy_snapshot_repro}",
        "family_members": members,
        "moves": [
            {
                "original_path": member,
                "archived_path": member.replace("quant/data_file/runtime/", "quant/data_file/runtime/archive/"),
                "evidence_links": [
                    "quant/data_file/reports/project_structure_governance_20260626/second_stage_thread_status_20260627.md",
                    "quant/main/docs/governance/runtime-governance.md",
                ],
                "still_referenced_by": [],
            }
            for member in members
        ],
    }


def build_runtime_loose_files_manifest() -> dict:
    moves = [
        {
            "original_path": "quant/data_file/runtime/qmt_screen.png",
            "archived_path": "quant/data_file/runtime/archive/qmt_screen.png",
            "reason": "历史验证截图，交易侧证据价值偏弱，建议进入 runtime/archive",
            "evidence_links": [
                "quant/data_file/reports/project_structure_governance_20260626/runtime_loose_files_report_20260627.json",
                "quant/data_file/reports/project_structure_governance_20260626/second_stage_thread_status_20260627.md",
            ],
            "still_referenced_by": [],
        },
        {
            "original_path": "quant/data_file/runtime/qmt_screen_2.png",
            "archived_path": "quant/data_file/runtime/trading_agent/manual_evidence/qmt_screen_2.png",
            "reason": "交易侧历史人工核验截图，建议并入 trading_agent 证据目录",
            "evidence_links": [
                "quant/data_file/reports/project_structure_governance_20260626/runtime_loose_files_report_20260627.json",
                "quant/data_file/reports/project_structure_governance_20260626/second_stage_thread_status_20260627.md",
            ],
            "still_referenced_by": [],
        },
        {
            "original_path": "quant/data_file/runtime/orchestrator_task_assignments_20260616.json",
            "archived_path": "quant/data_file/runtime/archive/orchestrator_task_assignments_20260616.json",
            "reason": "历史指挥官任务分派记录，保留治理追溯价值",
            "evidence_links": [
                "quant/data_file/reports/project_structure_governance_20260626/runtime_loose_files_report_20260627.json",
                "quant/data_file/reports/project_structure_governance_20260626/second_stage_thread_status_20260627.md",
            ],
            "still_referenced_by": [],
        },
    ]
    return {
        "actor": "commander-agent",
        "approval_scope": "待交易智能体和审计智能体最终复核后执行",
        "batch_id": "draft_runtime_loose_files_20260627",
        "group_name": "runtime_loose_files_archive_review",
        "reason": "runtime 根目录散落文件治理草稿",
        "family_rule": "explicit_runtime_loose_files",
        "family_members": [move["original_path"] for move in moves],
        "moves": moves,
    }


def build_research_group_manifest(root: Path, group_name: str, files: list[str], reason: str) -> dict:
    target_dir = {
        "diagnostics_20260620_20260621": "quant/main/research/archive/diagnostics_20260620_20260621",
        "20260623_gate_proxy_variants": "quant/data_file/runtime/recycle_bin",
        "20260626_fixed4y_scan_sweeps": "quant/data_file/runtime/recycle_bin",
        "prod_micro_and_isolated_oneoff": "quant/data_file/runtime/recycle_bin",
    }[group_name]

    moves = []
    for name in files:
        original = f"quant/main/{name}"
        if "archive" in target_dir:
            moves.append(
                {
                    "original_path": original,
                    "action": "archive",
                    "target_path": f"{target_dir}/{name}",
                    "evidence_links": [
                        "quant/data_file/reports/project_structure_governance_20260626/second_stage_thread_status_20260627.md",
                        "quant/data_file/reports/project_structure_governance_20260626/next_execution_plan_20260627.md",
                    ],
                    "still_referenced_by": [],
                }
            )
            continue

        moves.append(
            {
                "original_path": original,
                "action": "move_to_recycle_bin",
                "recycle_bin_root": target_dir,
                "proposed_relative_path": original,
                "observation_window_days": 30,
                "evidence_links": [
                    "quant/data_file/reports/project_structure_governance_20260626/second_stage_thread_status_20260627.md",
                    "quant/data_file/reports/project_structure_governance_20260626/next_execution_plan_20260627.md",
                ],
                "still_referenced_by": [],
            }
        )
    return {
        "actor": "commander-agent",
        "approval_scope": "待审计最终复核后执行",
        "batch_id": f"draft_{group_name}_20260627",
        "group_name": group_name,
        "reason": reason,
        "family_rule": group_name,
        "family_members": [f"quant/main/{name}" for name in files],
        "moves": moves,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare structure-governance manifest drafts.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    parser.add_argument(
        "--report-dir",
        default="quant/data_file/reports/project_structure_governance_20260626",
        help="Base report directory.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = root / report_dir
    drafts_dir = report_dir / "manifest_drafts"

    groups = find_files(
        root,
        [
            ("diagnostics_20260620_20260621", diagnostics_predicate),
            ("20260623_gate_proxy_variants", recycle_20260623_predicate),
            ("20260626_fixed4y_scan_sweeps", recycle_20260626_predicate),
            ("prod_micro_and_isolated_oneoff", prod_micro_predicate),
        ],
    )

    write_json(drafts_dir / "runtime_archive_pending.manifest.json", build_runtime_archive_manifest(root))
    write_json(drafts_dir / "runtime_loose_files_archive_review.manifest.json", build_runtime_loose_files_manifest())
    write_json(
        drafts_dir / "diagnostics_20260620_20260621.manifest.json",
        build_research_group_manifest(
            root,
            "diagnostics_20260620_20260621",
            groups["diagnostics_20260620_20260621"],
            "20260620-20260621 诊断、确认、探针、锚点研究脚本，适合归档保留研究溯源",
        ),
    )
    write_json(
        drafts_dir / "20260623_gate_proxy_variants.manifest.json",
        build_research_group_manifest(
            root,
            "20260623_gate_proxy_variants",
            groups["20260623_gate_proxy_variants"],
            "历史 gate/proxy/local-reblend/daymix 变体脚本，建议进入项目回收站 30 天观察",
        ),
    )
    write_json(
        drafts_dir / "20260626_fixed4y_scan_sweeps.manifest.json",
        build_research_group_manifest(
            root,
            "20260626_fixed4y_scan_sweeps",
            groups["20260626_fixed4y_scan_sweeps"],
            "历史 fixed4y/featurecount/structure/selfeatures/fullwindow/mainline sweep 脚本，建议整组进入项目回收站 30 天观察",
        ),
    )
    write_json(
        drafts_dir / "prod_micro_and_isolated_oneoff.manifest.json",
        build_research_group_manifest(
            root,
            "prod_micro_and_isolated_oneoff",
            groups["prod_micro_and_isolated_oneoff"],
            "历史 prod micro 与孤立 one-off 研究脚本，建议进入项目回收站 30 天观察",
        ),
    )

    index = {
        "drafts": sorted(rel(path, root) for path in drafts_dir.glob("*.json")),
        "group_counts": {name: len(files) for name, files in groups.items()},
        "boundary": "draft_manifests_only_no_file_moves",
    }
    write_json(drafts_dir / "index.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
