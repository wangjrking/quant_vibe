"""Build the large-asset retention and rollback registry.

The tool reads metadata and bounded head/tail samples only. It does not open
business databases, move files, or authorize deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAMPLE_BYTES = 1 << 20
EVIDENCE_SUFFIXES = {".json", ".md", ".txt"}
MAX_EVIDENCE_BYTES = 8 << 20


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sample_fingerprint(path: Path, sample_bytes: int = SAMPLE_BYTES) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def owner_for(path: Path) -> str:
    value = str(path).replace("\\", "/").lower()
    rules = (
        (("l1_", "raw_ingest", "data-ingestion"), "data-ingestion-agent"),
        (("l2_", "stock_daily_data"), "data-integration-agent"),
        (("l3_", "factor", "feature"), "factor-agent"),
        (("l4_", "model_agent", "prediction"), "model-agent"),
        (("strategy", "l5_", "l6_"), "strategy-agent"),
        (("trading", "l7_"), "trading-agent"),
        (("mcp_agent", "l8_"), "mcp-agent"),
    )
    for tokens, owner in rules:
        if any(token in value for token in tokens):
            return owner
    return "audit-agent"


def lifecycle_for(path: Path) -> str:
    value = str(path).replace("\\", "/").lower()
    if "/rollback/" in value or "apply_guard" in path.name.lower() or "pre_write_snapshot" in value:
        return "rollback_review"
    if "/runtime/" in value or value.startswith("runtime/"):
        return "runtime_candidate_review"
    if "/reports/" in value or value.startswith("reports/"):
        return "frozen_evidence_review"
    return "manual_review"


def local_evidence(path: Path, project: Path) -> list[str]:
    rows: list[str] = []
    for candidate in sorted(path.parent.iterdir()):
        if not candidate.is_file() or candidate == path or candidate.suffix.lower() not in EVIDENCE_SUFFIXES:
            continue
        try:
            if candidate.stat().st_size > MAX_EVIDENCE_BYTES:
                continue
        except OSError:
            continue
        try:
            rows.append(str(candidate.relative_to(project)).replace("\\", "/"))
        except ValueError:
            rows.append(str(candidate))
    return rows


def normalize_asset(path_text: str, project: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = project / path
    return path.resolve()


def build_registry(inventory: dict[str, Any], project: Path) -> dict[str, Any]:
    entries = inventory["large_reports"]["files"] + inventory["large_runtime"]["files"]
    assets = []
    for source in sorted(entries, key=lambda item: item["path"].lower()):
        path = normalize_asset(source["path"], project)
        stat = path.stat()
        assets.append(
            {
                "path": str(path.relative_to(project)).replace("\\", "/"),
                "bytes": stat.st_size,
                "mtime_utc": utc_iso(stat.st_mtime),
                "sample_fingerprint": sample_fingerprint(path),
                "full_sha256": None,
                "owner_agent": owner_for(path),
                "lifecycle_state": lifecycle_for(path),
                "local_evidence": local_evidence(path, project),
                "deletion_allowed": False,
                "required_before_disposition": [
                    "owner_confirmation",
                    "active_and_open_audit_reference_scan",
                    "rollback_dependency_check",
                    "fixed_audit_path_approval",
                    "content_addressed_archive_or_explicit_archive_waiver",
                ],
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "boundary": "registry_only_no_move_no_delete",
        "large_asset_threshold_bytes": inventory["large_reports"]["threshold_bytes"],
        "protected_roots": inventory["protected_paths"],
        "asset_count": len(assets),
        "asset_bytes": sum(item["bytes"] for item in assets),
        "assets": assets,
    }


def render_markdown(registry: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for asset in registry["assets"]:
        state = asset["lifecycle_state"]
        counts[state] = counts.get(state, 0) + 1
    lines = [
        "# 大资产 Retention / Rollback Registry",
        "",
        f"- 资产数：{registry['asset_count']}",
        f"- 总字节：{registry['asset_bytes']}",
        "- 删除授权：全部为 false",
        "",
        "## 生命周期统计",
        "",
    ]
    lines.extend(f"- `{name}`：{count}" for name, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## 处置规则",
            "",
            "登记表建立后仍需 owner、引用、rollback 和 fixed audit 四重闭合。未闭合记录不得移动或删除。",
            "",
            "## 资产",
            "",
            "| 路径 | owner | 状态 | 本地证据数 |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for asset in registry["assets"]:
        lines.append(
            f"| `{asset['path']}` | `{asset['owner_agent']}` | `{asset['lifecycle_state']}` | {len(asset['local_evidence'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the large-asset retention registry.")
    parser.add_argument("--project-root", default=str(project_root()))
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    inventory_path = normalize_asset(args.inventory, project)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    registry = build_registry(inventory, project)

    output = normalize_asset(args.output, project)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown = normalize_asset(args.markdown_output, project)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(registry), encoding="utf-8")

    print(json.dumps({"asset_count": registry["asset_count"], "asset_bytes": registry["asset_bytes"], "boundary": registry["boundary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
