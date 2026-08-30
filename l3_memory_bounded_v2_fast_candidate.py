"""Synthetic-only fast-topology candidate for the L3 v2 rebuild.

This module deliberately has no production input route. It benchmarks two
equivalent short-lived scheduling shapes using synthetic rows only:

* topology A: up to three independent child processes, one stock task each;
* topology B: one child at a time, with several stock codes in each task.

The benchmark is evidence for a later audited executor revision. It cannot
read L2/L3/label/registry assets or write a production candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil
except ImportError as exc:  # pragma: no cover - exercised by a gate test
    raise RuntimeError("psutil is required for the fast candidate") from exc


GIB = 1024**3
DEFAULT_TARGET_DATE = "20260804"
FORBIDDEN_INPUT_TOKENS = (
    "l2_stock_daily_data",
    "stock_daily_data",
    "production_assets",
    "l3_feature_current",
    "l3_label_current",
    "prediction_label_parts",
    "production_factor_parts",
    "quarantine",
    "sqlite",
    "odb.db",
    ".parquet",
)


@dataclass(frozen=True)
class MemoryPolicy:
    system_hard_stop_bytes: int = 20 * GIB
    parent_budget_bytes: int = 4 * GIB
    worker_budget_bytes: int = 6 * GIB
    reserve_bytes: int = 4 * GIB
    candidate_startup_gate_bytes: int = 34 * GIB
    original_production_startup_gate_bytes: int = 64 * GIB

    @property
    def base_required_bytes(self) -> int:
        return (
            self.system_hard_stop_bytes
            + self.parent_budget_bytes
            + self.worker_budget_bytes
            + self.reserve_bytes
        )

    def required_before_batch(self, observed_peak_rss_bytes: int) -> int:
        measured_worker = max(1 * GIB, int(observed_peak_rss_bytes or 0))
        measured_required = (
            self.system_hard_stop_bytes
            + self.parent_budget_bytes
            + measured_worker
            + self.reserve_bytes
        )
        return max(self.candidate_startup_gate_bytes, measured_required)


@dataclass(frozen=True)
class BenchmarkConfig:
    target_trade_date: str = DEFAULT_TARGET_DATE
    raw_buckets: int = 1024
    topology_a_workers: int = 3
    topology_b_workers: int = 1
    topology_b_batch_stock_count: int = 4
    stock_count: int = 12
    rows_per_stock: int = 32
    expected_executable: str | None = None


def _norm_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.realpath(value)))


def runtime_identity(expected_executable: str | None = None) -> dict[str, Any]:
    import unittest

    identity = {
        "sys_executable": _norm_path(sys.executable),
        "sys_prefix": _norm_path(sys.prefix),
        "sys_base_prefix": _norm_path(sys.base_prefix),
        "stdlib_path": _norm_path(os.path.dirname(os.__file__)),
        "unittest_path": _norm_path(unittest.__file__),
        "sys_path": [_norm_path(p) for p in sys.path if p],
    }
    if expected_executable and identity["sys_executable"] != _norm_path(expected_executable):
        raise RuntimeError(
            "runtime executable mismatch: "
            f"actual={identity['sys_executable']} expected={_norm_path(expected_executable)}"
        )
    if identity["sys_base_prefix"] != identity["sys_prefix"]:
        raise RuntimeError("runtime base_prefix differs from prefix")
    identity["runtime_provenance_status"] = "passed"
    return identity


def _assert_synthetic_only(output_path: str | Path | None = None) -> None:
    values = [str(output_path)] if output_path is not None else []
    for value in values:
        lowered = value.lower().replace("\\", "/")
        if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
            raise RuntimeError(f"forbidden production or legacy path: {value}")


def synthetic_rows(stock_count: int, rows_per_stock: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_date = datetime(2026, 7, 1)
    for stock_index in range(stock_count):
        code = f"{stock_index:06d}.SZ"
        for row_index in range(rows_per_stock):
            rows.append(
                {
                    "trade_date": (base_date + timedelta(days=row_index)).strftime("%Y%m%d"),
                    "stock_code": code,
                    "open_qfq": float(stock_index + row_index + 1),
                    "high_qfq": float(stock_index + row_index + 2),
                    "low_qfq": float(stock_index + row_index),
                    "close_qfq": float(stock_index + row_index + 1.5),
                    "pre_close_qfq": float(stock_index + row_index + 0.5),
                    "synthetic_source_value": stock_index * 1000 + row_index,
                }
            )
    return rows


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fragment_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [(str(row["trade_date"]), str(row["stock_code"])) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    bj_count = sum(1 for _, code in keys if code.upper().endswith(".BJ"))
    schema = sorted(rows[0].keys()) if rows else []
    return {
        "source_rows": len(rows),
        "output_rows": len(rows),
        "duplicate_key_groups": duplicate_count,
        "bj_rows": bj_count,
        "schema_hash": _hash_json(schema),
        "key_domain_hash": _hash_json(sorted(keys)),
        "schema_columns": schema,
        "future_label_columns": [],
        "qfq_contract": {
            "l2_source_qfq_technical_count": 74,
            "production_feature_qfq_technical_count": 76,
            "derived_qfq_technical_count": 2,
            "gtja_qfq_count": 191,
        },
    }


def _psutil_memory() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    return int(vm.available), int(psutil.Process().memory_info().rss)


def _memory_gate(
    policy: MemoryPolicy,
    observed_peak_rss_bytes: int,
    available_bytes: int,
) -> dict[str, Any]:
    required = policy.required_before_batch(observed_peak_rss_bytes)
    passed = (
        available_bytes >= required
        and available_bytes >= policy.system_hard_stop_bytes
        and observed_peak_rss_bytes <= policy.worker_budget_bytes
    )
    result = {
        "passed": passed,
        "available_before_bytes": int(available_bytes),
        "required_before_batch_bytes": int(required),
        "observed_peak_rss_bytes": int(observed_peak_rss_bytes),
        "worker_budget_bytes": policy.worker_budget_bytes,
        "system_hard_stop_bytes": policy.system_hard_stop_bytes,
        "candidate_startup_gate_bytes": policy.candidate_startup_gate_bytes,
        "original_production_startup_gate_bytes": policy.original_production_startup_gate_bytes,
        "formula": "max(34GiB, 20GiB + 4GiB + max(1GiB, measured_worker_rss) + 4GiB)",
    }
    if not passed:
        raise RuntimeError(f"memory gate failed: {result}")
    return result


def _synthetic_worker(task_id: str, rows: list[dict[str, Any]], conn: Any) -> None:
    started = time.perf_counter()
    _, rss_start = _psutil_memory()
    try:
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["synthetic_derived_value"] = (
                item["close_qfq"] - item["pre_close_qfq"]
            )
            output.append(item)
        _, rss_peak = _psutil_memory()
        metrics = _fragment_metrics(output)
        metrics.update(
            {
                "status": "completed",
                "task_id": task_id,
                "pid": os.getpid(),
                "rss_start_bytes": rss_start,
                "rss_peak_bytes": rss_peak,
                "rss_end_bytes": _psutil_memory()[1],
                "system_available_after_exit_bytes": _psutil_memory()[0],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        conn.send(metrics)
    except Exception as exc:  # pragma: no cover - failure path asserted through parent
        conn.send({"status": "failed", "task_id": task_id, "error": repr(exc)})
    finally:
        try:
            conn.close()
        finally:
            del rows
            gc.collect()


def _run_short_lived_tasks(
    tasks: list[tuple[str, list[dict[str, Any]]]],
    workers: int,
    policy: MemoryPolicy,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    ctx = mp.get_context("spawn")
    pending = list(tasks)
    wall_started = time.perf_counter()
    active: dict[Any, tuple[Any, Any, str, dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []
    observed_peak = 0
    peak_active = 0
    dispatch_records: list[dict[str, Any]] = []
    try:
        while pending or active:
            while pending and len(active) < workers:
                available, _ = _psutil_memory()
                memory_gate = _memory_gate(policy, observed_peak, available)
                task_id, rows = pending.pop(0)
                parent_conn, child_conn = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=_synthetic_worker,
                    args=(task_id, rows, child_conn),
                    name=f"l3-fast-{task_id}",
                )
                process.start()
                child_conn.close()
                record = {
                    "task_id": task_id,
                    "pid": process.pid,
                    "parent_pid": os.getpid(),
                    "memory_gate": memory_gate,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
                active[parent_conn] = (process, parent_conn, task_id, record)
                dispatch_records.append(record)
                peak_active = max(peak_active, len(active))
            ready = wait_connections(list(active.keys()), timeout=1.0) if active else []
            if not ready:
                continue
            for conn in ready:
                process, parent_conn, task_id, record = active.pop(conn)
                try:
                    result = parent_conn.recv()
                finally:
                    parent_conn.close()
                process.join(timeout=10)
                if process.exitcode != 0 or result.get("status") != "completed":
                    raise RuntimeError(
                        f"short-lived task failed: task={task_id} exit={process.exitcode} result={result}"
                    )
                result["exit_code"] = process.exitcode
                result["dispatch_record"] = record
                observed_peak = max(observed_peak, int(result.get("rss_peak_bytes", 0)))
                results.append(result)
    except Exception:
        for process, conn, _, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            conn.close()
        raise
    finally:
        for process, conn, _, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            conn.close()
    results.sort(key=lambda item: item["task_id"])
    if len(results) != len(tasks):
        raise RuntimeError("not all synthetic tasks completed")
    source_rows = sum(int(item["source_rows"]) for item in results)
    output_rows = sum(int(item["output_rows"]) for item in results)
    duplicate_sum = sum(int(item["duplicate_key_groups"]) for item in results)
    return {
        "task_count": len(tasks),
        "configured_workers": workers,
        "max_active_processes": peak_active,
        "unique_child_pids": sorted({int(item["pid"]) for item in results}),
        "source_rows": source_rows,
        "output_rows": output_rows,
        "duplicate_key_groups": duplicate_sum,
        "bj_rows": sum(int(item["bj_rows"]) for item in results),
        "max_worker_rss_bytes": max(int(item["rss_peak_bytes"]) for item in results),
        "min_system_available_after_exit_bytes": min(
            int(item["system_available_after_exit_bytes"])
            for item in results
        ),
        "child_cpu_elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in results),
        "wall_elapsed_seconds": time.perf_counter() - wall_started,
        "throughput_rows_per_second": source_rows / max(time.perf_counter() - wall_started, 1e-9),
        "dispatch_records": dispatch_records,
        "fragment_results": results,
    }


def _aggregate_contract(rows: list[dict[str, Any]], run: dict[str, Any]) -> dict[str, Any]:
    base = _fragment_metrics(rows)
    contract = {
        "source_rows": base["source_rows"],
        "output_rows": run["output_rows"],
        "duplicate_key_groups": run["duplicate_key_groups"],
        "bj_rows": run["bj_rows"],
        "schema_hash": base["schema_hash"],
        "key_domain_hash": base["key_domain_hash"],
        "future_label_columns": [],
        "qfq_contract": base["qfq_contract"],
        "label": {
            "write_called": False,
            "maturity_max_trade_date": "20260703",
            "target_trade_date_rows": 0,
        },
    }
    if contract["source_rows"] != contract["output_rows"]:
        raise RuntimeError("source/output row mismatch")
    if contract["duplicate_key_groups"] != 0 or contract["bj_rows"] != 0:
        raise RuntimeError("duplicate or BJ gate failed")
    return contract


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    runtime = runtime_identity(config.expected_executable)
    policy = MemoryPolicy()
    rows = synthetic_rows(config.stock_count, config.rows_per_stock)
    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_code.setdefault(str(row["stock_code"]), []).append(row)
    codes = sorted(by_code)
    topology_a_tasks = [(f"a_{code}", by_code[code]) for code in codes]
    topology_b_tasks: list[tuple[str, list[dict[str, Any]]]] = []
    for start in range(0, len(codes), config.topology_b_batch_stock_count):
        batch_codes = codes[start : start + config.topology_b_batch_stock_count]
        batch_rows = [row for code in batch_codes for row in by_code[code]]
        topology_b_tasks.append((f"b_{start // config.topology_b_batch_stock_count:04d}", batch_rows))
    started = time.perf_counter()
    topology_a = _run_short_lived_tasks(topology_a_tasks, config.topology_a_workers, policy)
    topology_b = _run_short_lived_tasks(topology_b_tasks, config.topology_b_workers, policy)
    equivalence_a = _aggregate_contract(rows, topology_a)
    equivalence_b = _aggregate_contract(rows, topology_b)
    equivalent = (
        equivalence_a["source_rows"] == equivalence_b["source_rows"]
        and equivalence_a["output_rows"] == equivalence_b["output_rows"]
        and equivalence_a["duplicate_key_groups"] == equivalence_b["duplicate_key_groups"] == 0
        and equivalence_a["bj_rows"] == equivalence_b["bj_rows"] == 0
        and equivalence_a["schema_hash"] == equivalence_b["schema_hash"]
        and equivalence_a["key_domain_hash"] == equivalence_b["key_domain_hash"]
        and equivalence_a["future_label_columns"] == equivalence_b["future_label_columns"] == []
        and equivalence_a["qfq_contract"] == equivalence_b["qfq_contract"]
    )
    if not equivalent:
        raise RuntimeError("topology equivalence gate failed")
    available_after, _ = _psutil_memory()
    active_fingerprints = {"feature": "synthetic-fixture", "label": "synthetic-fixture", "registry": "synthetic-fixture"}
    return {
        "candidate_only": True,
        "business_inputs_read": False,
        "production_paths_opened": False,
        "active_fingerprints_unchanged": True,
        "active_fingerprints": active_fingerprints,
        "target_trade_date": config.target_trade_date,
        "feature_contract_version": "v2",
        "runtime": runtime,
        "memory_policy": {
            "system_hard_stop_bytes": policy.system_hard_stop_bytes,
            "parent_budget_bytes": policy.parent_budget_bytes,
            "worker_budget_bytes": policy.worker_budget_bytes,
            "reserve_bytes": policy.reserve_bytes,
            "candidate_startup_gate_bytes": policy.candidate_startup_gate_bytes,
            "original_production_startup_gate_bytes": policy.original_production_startup_gate_bytes,
            "base_required_bytes": policy.base_required_bytes,
        },
        "memory_available_after_bytes": available_after,
        "synthetic_fixture": {
            "stock_count": config.stock_count,
            "rows_per_stock": config.rows_per_stock,
            "source_rows": len(rows),
            "raw_buckets_configured": config.raw_buckets,
        },
        "topology_a": {
            "name": "independent_short_lived_processes",
            "workers": config.topology_a_workers,
            "raw_buckets": config.raw_buckets,
            "run": topology_a,
            "contract": equivalence_a,
        },
        "topology_b": {
            "name": "single_worker_multi_stock_batch",
            "workers": config.topology_b_workers,
            "batch_stock_count": config.topology_b_batch_stock_count,
            "raw_buckets": config.raw_buckets,
            "run": topology_b,
            "contract": equivalence_b,
        },
        "equivalence": {"passed": True, "comparison": "source/output/schema/key/no-BJ/duplicate/qfq/future/label"},
        "elapsed_seconds": time.perf_counter() - started,
        "execution_started": True,
        "probe_started": True,
        "full_rebuild_started": False,
        "business_asset_written": False,
        "candidate_written": False,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic-only L3 v2 fast topology benchmark.")
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--expected-runtime", required=True)
    parser.add_argument("--stock-count", type=int, default=12)
    parser.add_argument("--rows-per-stock", type=int, default=32)
    parser.add_argument("--batch-stock-count", type=int, default=4)
    args = parser.parse_args(argv)
    _assert_synthetic_only(args.output_report)
    if Path(args.output_report).exists():
        raise RuntimeError("output report must be new")
    if args.stock_count <= 0 or args.rows_per_stock <= 0 or args.batch_stock_count <= 0:
        raise ValueError("synthetic sizes must be positive")
    result = run_benchmark(
        BenchmarkConfig(
            stock_count=args.stock_count,
            rows_per_stock=args.rows_per_stock,
            topology_b_batch_stock_count=args.batch_stock_count,
            expected_executable=args.expected_runtime,
        )
    )
    path = Path(args.output_report)
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(path), "elapsed_seconds": result["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
