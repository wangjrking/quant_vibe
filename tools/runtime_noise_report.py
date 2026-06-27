"""Report runtime directory governance candidates.

This tool is read-only. It does not move, delete, archive, or execute business
workflows. The output is a first-pass queue for commander/audit review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LONG_TERM_KEEP = {
    "agent_memory",
    "agent_observability",
    "agent_workspaces",
    "orchestrator_reports",
    "recycle_bin",
    "trading_agent",
    "archive",
}

RECYCLE_PREFIXES = (
    "tmp_",
)

ARCHIVE_PREFIXES = (
    "snapshot_",
    "strategy_repro",
    "strategy_snapshot_repro",
)

REVIEW_NAMES = {
    "slippage_tmp_logs",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def classify_dir(name: str) -> tuple[str, str]:
    if name in LONG_TERM_KEEP:
        return "keep", "长期运行态目录"
    if name in REVIEW_NAMES:
        return "review", "名称含临时含义，但可能仍有交易或滑点证据价值"
    if name.startswith(RECYCLE_PREFIXES):
        return "recycle_review", "临时目录候选，需确认无审计或复现引用后进入项目回收站"
    if name.startswith(ARCHIVE_PREFIXES):
        return "archive_review", "复现或 snapshot 候选，优先进入 runtime/archive 而不是回收站"
    return "review", "未匹配固定规则，需要人工确认"


def scan(runtime_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in runtime_dir.iterdir() if item.is_dir()):
        category, reason = classify_dir(path.name)
        rows.append(
            {
                "path": str(path.relative_to(runtime_dir)),
                "category": category,
                "reason": reason,
                "suggested_action": suggested_action(category),
            }
        )
    return rows


def suggested_action(category: str) -> str:
    if category == "keep":
        return "keep_at_original_path"
    if category == "archive_review":
        return "review_then_archive_under_runtime_archive"
    if category == "recycle_review":
        return "review_then_move_to_project_recycle_bin"
    return "manual_review"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report runtime governance candidates.")
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
