"""Report loose files in quant/data_file/runtime.

This tool is read-only. It does not move, delete, archive, or execute business
workflows. It classifies runtime root-level files so they can be reviewed
separately from runtime directories.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


KEEP_FILES = {
    "README.md": "runtime 根目录说明文件",
    "production_factor_industry_encode_mapping.json": "L3 production factor industry_encode 治理映射资产，被因子链路和审计报告引用",
}

ARCHIVE_REVIEW_FILES = {
    "orchestrator_task_assignments_20260616.json": "历史指挥官任务分派记录，可能有治理追溯价值",
    "qmt_screen.png": "QMT 界面截图，可能属于交易侧历史证据",
    "qmt_screen_2.png": "QMT 界面截图，可能属于交易侧历史证据",
}

REVIEW_FILES = {
    "agent_state.example.json": "示例状态文件，需确认是否仍作为模板使用",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def classify_file(name: str) -> tuple[str, str, str]:
    if name in KEEP_FILES:
        return "keep", KEEP_FILES[name], "keep_at_original_path"
    if name in ARCHIVE_REVIEW_FILES:
        return "archive_review", ARCHIVE_REVIEW_FILES[name], "review_then_archive_or_recycle"
    if name in REVIEW_FILES:
        return "review", REVIEW_FILES[name], "manual_review"
    return "review", "未匹配固定规则，需要人工确认", "manual_review"


def scan(runtime_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in runtime_dir.iterdir() if item.is_file()):
        category, reason, suggested_action = classify_file(path.name)
        rows.append(
            {
                "path": path.name,
                "category": category,
                "reason": reason,
                "suggested_action": suggested_action,
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Report runtime loose-file governance candidates.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--write-report", default=None, help="Write full JSON report to a path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    runtime_dir = root / "quant" / "data_file" / "runtime"
    if not runtime_dir.is_dir():
        print(f"FAIL missing runtime dir: {runtime_dir}")
        return 1

    rows = scan(runtime_dir)
    counts = Counter(row["category"] for row in rows)
    result = {
        "candidate_count": len(rows),
        "category_counts": dict(sorted(counts.items())),
        "candidates": rows,
        "boundary": "read_only_report_no_file_moves",
    }

    if args.write_report:
        output = Path(args.write_report)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"candidate_count={result['candidate_count']}")
    print(f"category_counts={result['category_counts']}")
    print("boundary=read_only_report_no_file_moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
