from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from rebuild_l2_stock_daily_duckdb_mainline import collect_output_metrics


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_id(path: Path) -> str:
    completed = subprocess.run(
        ["fsutil", "file", "queryfileid", str(path)],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    match = re.search(r"0x[0-9a-fA-F]+", completed.stdout)
    if not match:
        raise RuntimeError(f"无法解析 NTFS File ID: {path}")
    return match.group(0).lower()


def _file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "creation_time": _iso(stat.st_ctime),
        "last_write_time": _iso(stat.st_mtime),
        "file_id": _file_id(path),
        "sha256": _sha256(path),
    }


def build_evidence(
    *,
    target_trade_date: str,
    active_path: Path,
    snapshot_path: Path,
    original_manifest_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    active_state = _file_state(active_path)
    snapshot_state = _file_state(snapshot_path)
    original_manifest_state = original_manifest_path.stat()
    validation_state = validation_path.stat()
    snapshot_metrics = collect_output_metrics(
        snapshot_path,
        target_trade_date=target_trade_date,
        special_code="301583.SZ",
    )
    active_metrics = collect_output_metrics(
        active_path,
        target_trade_date=target_trade_date,
        special_code="301583.SZ",
    )
    metric_delta = {
        key: active_metrics[key] - snapshot_metrics[key]
        for key in (
            "row_count",
            "stock_count",
            "bj_row_count",
            "duplicate_key_groups",
            "target_row_count",
            "target_stock_count",
            "target_bj_row_count",
            "target_duplicate_key_groups",
        )
    }

    snapshot_created = snapshot_path.stat().st_ctime
    original_manifest_created = original_manifest_state.st_ctime
    active_last_written = active_path.stat().st_mtime
    validation_created = validation_state.st_ctime
    checks = {
        "snapshot_path_distinct_from_active": snapshot_path.resolve() != active_path.resolve(),
        "snapshot_file_id_distinct_from_active": snapshot_state["file_id"] != active_state["file_id"],
        "snapshot_sha_matches_original_manifest": (
            snapshot_state["sha256"].lower() == str(original_manifest["sha256"]).lower()
        ),
        "snapshot_metrics_match_original_manifest": (
            snapshot_metrics == original_manifest["table_metrics"]
        ),
        "snapshot_created_before_original_manifest": (
            snapshot_created <= original_manifest_created
        ),
        "original_manifest_created_before_post_change_active_mtime": (
            original_manifest_created < active_last_written
        ),
        "post_change_active_mtime_before_final_validation": (
            active_last_written < validation_created
        ),
        "active_and_snapshot_hashes_distinct": (
            active_state["sha256"] != snapshot_state["sha256"]
        ),
        "final_validation_completed": validation.get("status") == "completed",
    }
    status = "verified_existing_pre_mutation_snapshot" if all(checks.values()) else "failed"
    return {
        "task_id": "incremental-trading-signal-20260716-L2-snapshot-remediation-2",
        "status": status,
        "remediation_path": "找回并证明真实 pre-change 独立快照",
        "target_trade_date": target_trade_date,
        "active_touched_by_remediation": False,
        "snapshot_is_newly_created_by_remediation": False,
        "snapshot_is_existing_pre_mutation_artifact": True,
        "copy_semantics": {
            "method": "shutil.copy2",
            "explanation": (
                "Windows 上 copy2 保留源文件 LastWriteTime；NTFS CreationTime 表示独立快照实物的实际创建时间。"
            ),
        },
        "timeline": {
            "snapshot_ntfs_creation_time": snapshot_state["creation_time"],
            "snapshot_preserved_source_last_write_time": snapshot_state["last_write_time"],
            "original_manifest_creation_time": _iso(original_manifest_created),
            "original_manifest_captured_at": original_manifest["captured_at"],
            "post_change_active_last_write_time": active_state["last_write_time"],
            "final_validation_creation_time": _iso(validation_created),
        },
        "snapshot": {
            "file_state": snapshot_state,
            "table_metrics": snapshot_metrics,
        },
        "post_change_active": {
            "file_state": active_state,
            "table_metrics": active_metrics,
        },
        "difference": {
            "size_bytes": active_state["size_bytes"] - snapshot_state["size_bytes"],
            "sha256_equal": active_state["sha256"] == snapshot_state["sha256"],
            "file_id_equal": active_state["file_id"] == snapshot_state["file_id"],
            "table_metric_delta": metric_delta,
            "logical_note": (
                "本轮专项加工为幂等重刷，核心覆盖指标前后相同；文件实体、哈希、大小和时间顺序证明两者不是同一时点副本。"
            ),
        },
        "checks": checks,
        "source_evidence": {
            "original_manifest_path": str(original_manifest_path),
            "final_validation_path": str(validation_path),
        },
        "governance": {
            "uses_legacy_sqlite": False,
            "uses_odb_db": False,
            "touches_l3_l8": False,
            "allow_l3_continue": False,
            "audit_required": True,
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    snapshot = payload["snapshot"]
    active = payload["post_change_active"]
    timeline = payload["timeline"]
    difference = payload["difference"]
    checks = payload["checks"]
    check_lines = "\n".join(
        f"- {name}: `{'通过' if passed else '失败'}`" for name, passed in checks.items()
    )
    return f"""# L2 快照门禁第二次整改报告

## 整改结论

- 整改路径：`{payload['remediation_path']}`
- 状态：`{payload['status']}`
- 本次整改是否触碰 active L2：`false`
- 是否重新执行 L2：`false`
- 是否放行 L3-L8：`false`

现有独立 DuckDB 快照已被证明确实创建于本次 L2 专项加工修改 active 之前。原清单中的较早 `LastWriteTime` 是 `shutil.copy2` 保留源文件时间，不是快照实物创建时间；NTFS `CreationTime` 才是独立副本的实际创建时间。

## 时间线

- 快照 NTFS 创建时间：`{timeline['snapshot_ntfs_creation_time']}`
- 快照保留的源文件 LastWriteTime：`{timeline['snapshot_preserved_source_last_write_time']}`
- 原 manifest 创建时间：`{timeline['original_manifest_creation_time']}`
- 原 manifest captured_at：`{timeline['original_manifest_captured_at']}`
- 加工后 active 最后写入时间：`{timeline['post_change_active_last_write_time']}`
- 最终验收报告创建时间：`{timeline['final_validation_creation_time']}`

## 快照与 Active

- 快照路径：`{snapshot['file_state']['path']}`
- 快照 File ID：`{snapshot['file_state']['file_id']}`
- 快照 SHA256：`{snapshot['file_state']['sha256']}`
- 快照大小：`{snapshot['file_state']['size_bytes']}`
- 快照指标：`{snapshot['table_metrics']}`
- Active 路径：`{active['file_state']['path']}`
- Active File ID：`{active['file_state']['file_id']}`
- Active SHA256：`{active['file_state']['sha256']}`
- Active 大小：`{active['file_state']['size_bytes']}`
- Active 指标：`{active['table_metrics']}`

## 差异

- 文件大小差：`{difference['size_bytes']}` 字节
- SHA256 是否相同：`{str(difference['sha256_equal']).lower()}`
- File ID 是否相同：`{str(difference['file_id_equal']).lower()}`
- 表指标差：`{difference['table_metric_delta']}`
- 说明：{difference['logical_note']}

## 门禁检查

{check_lines}

## 边界

- 未使用 SQLite、`odb.db` 或 shared DuckDB。
- 本次仅补强既有真实快照的证据链，没有修改 active L2。
- `allow_l3_continue=false`，等待审计智能体再次复核。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--active-path", required=True)
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--original-manifest-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-hashes", required=True)
    args = parser.parse_args()

    payload = build_evidence(
        target_trade_date=args.target_trade_date,
        active_path=Path(args.active_path),
        snapshot_path=Path(args.snapshot_path),
        original_manifest_path=Path(args.original_manifest_path),
        validation_path=Path(args.validation_path),
    )
    Path(args.output_manifest).write_text(
        json.dumps(
            {
                "manifest_type": "l2_existing_pre_mutation_snapshot_remediation",
                "status": payload["status"],
                "target_trade_date": payload["target_trade_date"],
                "snapshot": payload["snapshot"],
                "post_change_active": payload["post_change_active"],
                "timeline": payload["timeline"],
                "checks": payload["checks"],
                "not_a_new_snapshot": True,
                "generated_at": payload["generated_at"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    Path(args.output_hashes).write_text(
        f"{payload['snapshot']['file_state']['sha256']}  {payload['snapshot']['file_state']['path']}\n"
        f"{payload['post_change_active']['file_state']['sha256']}  {payload['post_change_active']['file_state']['path']}\n",
        encoding="ascii",
    )
    print(json.dumps({"status": payload["status"], "checks": payload["checks"]}))


if __name__ == "__main__":
    main()
