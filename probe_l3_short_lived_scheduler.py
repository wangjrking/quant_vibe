from __future__ import annotations

import argparse
import json
import os
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

import duckdb

from adjustment_semantics import FRONT_ADJUSTED_MARKET_PRICE_COLUMNS
from rebuild_l3_full_duckdb_mainline import (
    EXPECTED_L2_ASSET_ID,
    EXPECTED_L2_PATH,
    EXPECTED_L2_TABLE,
    EXPECTED_L2_SHA256_20260717,
    QFQ_TECHNICAL_EXPECTED_COUNT,
    _active_snapshot,
    _active_unchanged,
    _build_raw_bucket,
    _file_sha256,
    _file_state,
    _stable_bucket,
    _validate_raw_completion,
    _active_registry_asset,
    REGISTRY_PATH,
)
from short_lived_task_scheduler import MemoryPolicy, ShortLivedTaskScheduler, atomic_json, now_iso, require_psutil


DEFAULT_BUCKETS = (0, 2, 4, 6)
DEFAULT_RUN_ID = "incremental-trading-signal-20260717-L3-short-lived-scheduler-memory-probe"


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _logger(path: Path):
    def write(stage: str, **details: Any) -> None:
        payload = {"time": now_iso(), "stage": stage, **details}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)

    return write


def _probe_shard_metrics(path: Path) -> dict[str, Any]:
    with closing(duckdb.connect(str(path), read_only=True)) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT stock_code),
                   MIN(CAST(trade_date AS VARCHAR)), MAX(CAST(trade_date AS VARCHAR)),
                   COALESCE(SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END), 0)
            FROM raw_factor
            """
        ).fetchone()
        duplicate = int(conn.execute(
            "SELECT COUNT(*) FROM (SELECT trade_date, stock_code, COUNT(*) n FROM raw_factor GROUP BY 1,2 HAVING COUNT(*)>1)"
        ).fetchone()[0])
        columns = [str(item[1]) for item in conn.execute("PRAGMA table_info('raw_factor')").fetchall()]
    qfq_columns = [column for column in columns if "_qfq" in column]
    qfq_technical = [column for column in qfq_columns if column not in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS]
    return {
        "row_count": int(row[0]),
        "stock_count": int(row[1]),
        "min_trade_date": str(row[2]),
        "max_trade_date": str(row[3]),
        "bj_row_count": int(row[4]),
        "duplicate_key_groups": duplicate,
        "column_count": len(columns),
        "qfq_price_count": sum(column in columns for column in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS),
        "qfq_technical_count": len(qfq_technical),
    }


def _markdown(payload: dict[str, Any]) -> str:
    return "\n".join([
        "# L3 短生命周期调度器内存探针",
        "",
        f"- 状态：`{payload['status']}`",
        f"- Run ID：`{payload['run_id']}`",
        f"- Bucket：`{payload['bucket_indices']}`",
        f"- Worker：`{payload['scheduler']['policy']['workers']}`",
        f"- Worker 生命周期：`{payload['scheduler']['worker_lifecycle_count']}`",
        f"- 观察到的唯一 PID：`{payload['scheduler']['unique_worker_pid_count']}`",
        f"- 最大并发：`{payload['scheduler']['max_active_observed']}`",
        f"- 总行数：`{payload['metrics']['row_count']}`",
        f"- BJ 行数：`{payload['metrics']['bj_row_count']}`",
        f"- 重复键：`{payload['metrics']['duplicate_key_groups']}`",
        f"- 单 worker 最大 RSS：`{payload['metrics']['max_worker_rss_bytes']}` 字节",
        f"- 系统最低 available：`{payload['metrics']['system_available_min_bytes']}` 字节",
        f"- 退出后最低 available：`{payload['metrics']['system_available_after_exit_min_bytes']}` 字节",
        f"- Active feature 未变：`{payload['active_unchanged']['feature']}`",
        f"- Active label 未变：`{payload['active_unchanged']['label']}`",
        f"- Registry 未变：`{payload['active_unchanged']['registry']}`",
        "",
        "本探针仅用于非生产 PID 生命周期与内存治理验证，禁止作为 candidate、active 或下游输入。正式 attempt-5 仍需独立审计和指挥官放行。",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four non-production raw buckets through the L3 short-lived scheduler.")
    parser.add_argument("--target-trade-date", default="20260717")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--raw-buckets", type=int, default=256)
    parser.add_argument("--bucket-indices", default=",".join(str(value) for value in DEFAULT_BUCKETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers != 2:
        raise RuntimeError("the approved memory probe requires workers=2")
    bucket_indices = tuple(int(value) for value in args.bucket_indices.split(",") if value)
    if len(bucket_indices) != 4 or len(set(bucket_indices)) != 4:
        raise RuntimeError("the approved memory probe requires four unique bucket indices")
    workspace = Path(args.workspace_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    if workspace.exists() and any(workspace.rglob("*")):
        raise RuntimeError(f"existing non-empty probe workspace prohibits resume: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    progress_path = report_dir / "l3_short_lived_memory_probe_progress.jsonl"
    logger = _logger(progress_path)
    psutil = require_psutil()
    logger("probe_precheck_start", run_id=args.run_id)

    l2_asset, l2_path, l2_table = _active_registry_asset("L2")
    if l2_asset.get("asset_id") != EXPECTED_L2_ASSET_ID or l2_path != EXPECTED_L2_PATH.resolve() or l2_table != EXPECTED_L2_TABLE:
        raise RuntimeError(f"active L2 route drift: {l2_asset.get('asset_id')} {l2_path}::{l2_table}")
    l2_hash_before = _file_sha256(l2_path)
    if l2_hash_before != EXPECTED_L2_SHA256_20260717:
        raise RuntimeError(f"active L2 hash drift: {l2_hash_before}")
    feature_before = _active_snapshot("L3_features", args.target_trade_date)
    label_before = _active_snapshot("L3_labels", args.target_trade_date)
    registry_before = _file_state(REGISTRY_PATH)

    with closing(duckdb.connect(str(l2_path), read_only=True)) as conn:
        codes = [str(row[0]) for row in conn.execute(
            f"SELECT DISTINCT stock_code FROM {_quote_ident(l2_table)} WHERE stock_code NOT LIKE '%.BJ' ORDER BY stock_code"
        ).fetchall()]
    buckets = {index: [] for index in bucket_indices}
    for code in codes:
        index = _stable_bucket(code, args.raw_buckets)
        if index in buckets:
            buckets[index].append(code)

    policy = MemoryPolicy(workers=2)
    tasks = []
    for index in bucket_indices:
        task_id = f"raw_bucket_{index:04d}"
        tasks.append({
            "task_id": task_id,
            "stage": "raw_probe",
            "bucket_index": index,
            "codes": buckets[index],
            "l2_path": str(l2_path),
            "l2_table": l2_table,
            "output_path": str(workspace / "raw_shards" / f"{task_id}.duckdb"),
            "completion_path": str(workspace / "logs" / f"{task_id}.json"),
            "failure_path": str(workspace / "logs" / f"{task_id}_failure.json"),
            "duckdb_memory_limit": policy.duckdb_memory_limit,
        })

    scheduler = ShortLivedTaskScheduler(
        stage="raw_probe",
        worker_fn=_build_raw_bucket,
        workspace=workspace,
        quarantine_root=workspace.parent.parent / "quarantine",
        logger=logger,
        workflow_run_id=args.run_id,
        run_nonce=uuid.uuid4().hex,
        policy=policy,
        result_validator=_validate_raw_completion,
    )
    scheduler_summary = scheduler.run(tasks)
    shard_metrics = []
    for result in scheduler_summary["results"]:
        metrics = _probe_shard_metrics(Path(result["path"]))
        shard_metrics.append({**result, "table_metrics": metrics})
        if metrics["bj_row_count"] or metrics["duplicate_key_groups"]:
            raise RuntimeError(f"probe shard integrity failure: {result['task_id']}")
        if metrics["qfq_price_count"] != 5 or metrics["qfq_technical_count"] < QFQ_TECHNICAL_EXPECTED_COUNT:
            raise RuntimeError(f"probe qfq schema failure: {result['task_id']} {metrics}")

    l2_hash_after = _file_sha256(l2_path)
    feature_after = _active_snapshot("L3_features", args.target_trade_date)
    label_after = _active_snapshot("L3_labels", args.target_trade_date)
    registry_after = _file_state(REGISTRY_PATH)
    active_unchanged = {
        "l2": l2_hash_before == l2_hash_after,
        "feature": _active_unchanged(feature_before, feature_after),
        "label": _active_unchanged(label_before, label_after),
        "registry": registry_before == registry_after,
    }
    if not all(active_unchanged.values()):
        raise RuntimeError(f"probe active drift: {active_unchanged}")
    residual_worker_identities = []
    for identity in scheduler_summary["worker_lifecycles"]:
        pid = int(identity["pid"])
        try:
            process = psutil.Process(pid)
            create_time_ns = int(float(process.create_time()) * 1_000_000_000)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError(f"unable to verify probe worker exit: pid={pid}") from error
        if create_time_ns == int(identity["create_time_ns"]):
            residual_worker_identities.append(identity)
    if residual_worker_identities:
        raise RuntimeError(f"probe residual child processes: {residual_worker_identities}")

    results = scheduler_summary["results"]
    metrics = {
        "row_count": sum(int(item["output_rows"]) for item in results),
        "source_row_count": sum(int(item["source_rows"]) for item in results),
        "stock_assignments": sum(int(item["code_count"]) for item in results),
        "duplicate_key_groups": sum(int(item["duplicate_key_groups"]) for item in results),
        "bj_row_count": sum(int(item["table_metrics"]["bj_row_count"]) for item in shard_metrics),
        "max_worker_rss_bytes": max(int(item["rss_peak_bytes"]) for item in results),
        "system_available_min_bytes": min(int(item["system_available_min_bytes"]) for item in results),
        "system_available_after_exit_min_bytes": min(int(item["system_available_after_exit_bytes"]) for item in results),
        "output_size_bytes": sum(int(item["output_size_bytes"]) for item in results),
    }
    payload = {
        "status": "probe_passed_waiting_for_audit",
        "generated_at": now_iso(),
        "target_trade_date": args.target_trade_date,
        "run_id": args.run_id,
        "bucket_indices": list(bucket_indices),
        "input": {"path": str(l2_path), "table": l2_table, "sha256_before": l2_hash_before, "sha256_after": l2_hash_after},
        "workspace": str(workspace),
        "probe_only": True,
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
        "reuse_prohibited": True,
        "scheduler": {key: value for key, value in scheduler_summary.items() if key != "results"},
        "results": shard_metrics,
        "metrics": metrics,
        "active_before": {"feature": feature_before, "label": label_before, "registry": registry_before},
        "active_after": {"feature": feature_after, "label": label_after, "registry": registry_after},
        "active_unchanged": active_unchanged,
        "residual_worker_identities": residual_worker_identities,
        "pair_change_called": False,
        "feature_candidate_created": False,
        "label_candidate_created": False,
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    report_json = report_dir / "l3_short_lived_memory_probe_20260718.json"
    report_md = report_dir / "l3_short_lived_memory_probe_20260718.md"
    manifest_path = workspace / "probe_manifest.json"
    atomic_json(report_json, payload)
    atomic_json(manifest_path, {
        "status": "probe_evidence_only",
        "run_id": args.run_id,
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
        "reuse_prohibited": True,
        "report": str(report_json),
    })
    report_md.write_text(_markdown(payload), encoding="utf-8")
    logger("probe_report_written", report_json=str(report_json), report_md=str(report_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
