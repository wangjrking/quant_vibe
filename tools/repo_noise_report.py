"""Report likely non-mainline scripts in quant/main.

This tool is read-only. It does not move, delete, archive, or execute candidate
files. The output is only a first-pass governance queue for agent review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PATTERNS = {
    "research": ("research_",),
    "tuning": ("tune_", "optimize_", "search_"),
    "validation": ("validate_", "check_", "probe_", "stress_"),
    "archive_helper": ("archive_",),
    "diagnostic": ("diagnose_", "analyze_", "compare_"),
    "build_helper": ("build_", "prepare_", "refresh_", "report_"),
}

OWNER_BY_CATEGORY = {
    "research": "research-agent",
    "tuning": "strategy-agent",
    "validation": "audit-agent",
    "archive_helper": "strategy-agent",
    "diagnostic": "architect-agent",
    "build_helper": "architect-agent",
}

DEFAULT_KEEP = {
    "main.py",
    "run_production_tasks.py",
    "daily_strategy.py",
    "export_gm_signals.py",
    "stock_daily_data_route.py",
    "model_asset_route.py",
    "prediction_manifest.py",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def classify(file_path: Path) -> str | None:
    name = file_path.name
    if name in DEFAULT_KEEP:
        return None
    for category, prefixes in PATTERNS.items():
        if name.startswith(prefixes):
            return category
    return None


def scan(main_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in sorted(main_dir.glob("*.py")):
        category = classify(file_path)
        if category is None:
            continue
        rows.append(
            {
                "path": str(file_path.relative_to(main_dir)),
                "category": category,
                "owner_agent": OWNER_BY_CATEGORY.get(category, "architect-agent"),
                "suggested_action": "review_then_move_to_recycle_bin_or_research_area",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Report likely repo noise candidates.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--write-report", default=None, help="Write full JSON report to a path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    main_dir = root / "quant" / "main"
    if not main_dir.is_dir():
        print(f"FAIL missing main dir: {main_dir}")
        return 1

    rows = scan(main_dir)
    counts = Counter(row["category"] for row in rows)
    owner_counts = Counter(row["owner_agent"] for row in rows)
    result = {
        "candidate_count": len(rows),
        "category_counts": dict(sorted(counts.items())),
        "owner_counts": dict(sorted(owner_counts.items())),
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
    print(f"owner_counts={result['owner_counts']}")
    print("boundary=read_only_report_no_file_moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
