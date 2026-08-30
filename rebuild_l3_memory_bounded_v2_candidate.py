from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import sysconfig
import time
import traceback
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import unittest

import duckdb
import pandas as pd

from adjustment_semantics import (
    FRONT_ADJUSTED_MARKET_PRICE_COLUMNS,
    NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS,
    NAKED_MARKET_PRICE_COLUMNS,
)
from build_prediction_label_parts import (
    DEFAULT_COMPUTED_LABELS,
    DEFAULT_EXISTING_LABELS,
    EXECUTABLE_LABEL_SPECS,
    LABEL_NULL_DEPENDENCIES,
    TAG_LABEL_DEPENDENCIES,
    BUY_COMMISSION_RATE,
    SELL_COMMISSION_RATE,
    SELL_TAX_RATE,
    SLIPPAGE_RATE,
    available_labels,
)
from build_production_factor_parts import (
    GTJA_TO_PRODUCTION_COLUMN_MAP,
    is_future_or_label_column,
    production_raw_columns,
)
from feature_contract_v2_candidate import (
    audit_candidate_feature_schema,
    candidate_production_raw_columns,
    validate_future_lineage,
)
from data_process_module import group_factor_eng
from gtja_alpha_workflow import (
    GTJA_ALPHA_COLUMNS,
    compute_gtja_alpha_from_raw_factor,
    required_gtja_raw_columns,
)
from l3_process_gate import (
    classify_process_gate,
    collect_process_records,
    load_expected_workers,
)
from production_asset_registry import active_main_workflow_asset, split_asset_path
from rebuild_factor_data_batched import _normalize_types
from short_lived_task_scheduler import (
    MemoryPolicy,
    ShortLivedSchedulerError,
    ShortLivedTaskScheduler,
    cleanup_descendants_for_run,
    require_psutil,
)
from workflow_contract import build_layer_handoff_contract, validate_layer_handoff_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP_FACTOR_ENGINE_SOURCE = inspect.getsource(group_factor_eng)
DATA_DIR = PROJECT_ROOT / "quant" / "data_file"
REGISTRY_PATH = DATA_DIR / "asset_registry" / "production_assets.json"
EXPECTED_L2_PATH = DATA_DIR / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
EXPECTED_L2_TABLE = "STOCK_DAILY_DATA"
EXPECTED_L2_ASSET_ID = "prod_l2_duckdb_stock_daily_data_split_20260701"
EXPECTED_FEATURE_ASSET_ID = "prod_l3_feature_duckdb_split_20260701"
EXPECTED_LABEL_ASSET_ID = "prod_l3_label_duckdb_split_20260701"
EXPECTED_L2_SHA256_20260717 = "7836534c8339128e51c3080daf4ec511857802ce7f5ba2250da822047e616aa6"
# The L2 contract owns the key order. Every raw/GTJA/feature/label table must
# preserve this order; relying on natural relation or dataframe insertion order is a gate failure.
KEY_COLUMNS = ["stock_code", "trade_date"]
MAX_LABEL_HORIZON = 22
GTJA_LOOKBACK_DATES = 300
MIN_FREE_BYTES = 250 * 1024**3
FORBIDDEN_PATH_TOKENS = (
    "20260716",
    "cancelled",
    "quarantine",
    "stock_factor_data.parquet",
    "production_factor_parts",
    "prediction_label_parts",
    "odb.db",
    ".db::",
)
QFQ_TECHNICAL_EXPECTED_COUNT = 74
UNIFIED_PYTHON_LAUNCHER = PROJECT_ROOT / "runtime_candidates" / "my_quant_copy_20260804" / "python.exe"

# This candidate uses a separately audited streaming profile.  The original
# full-history entrypoint keeps its 64 GiB gate; this profile is valid only for
# the single-worker, short-lived, layered-merge topology implemented below.
STREAMING_SYSTEM_HARD_STOP_BYTES = 20 * 1024**3
STREAMING_PARENT_BUDGET_BYTES = 4 * 1024**3
STREAMING_WORKER_BUDGET_BYTES = 6 * 1024**3
STREAMING_RESERVE_BYTES = 4 * 1024**3
STREAMING_STARTUP_AVAILABLE_BYTES = (
    STREAMING_SYSTEM_HARD_STOP_BYTES
    + STREAMING_PARENT_BUDGET_BYTES
    + STREAMING_WORKER_BUDGET_BYTES
    + STREAMING_RESERVE_BYTES
)
STREAMING_FAN_IN = 8
STREAMING_DUCKDB_MEMORY_LIMIT = "3GB"
STREAMING_TEMP_DIR_NAME = "duckdb_temp"
RUNTIME_ROOT = (PROJECT_ROOT / "runtime_candidates" / "my_quant_copy_20260804").resolve()
NEGATIVE_SHIFT_LINEAGE_RE = re.compile(
    r"new_columns\[['\"](?P<field>[^'\"]+)['\"]\]\s*=.*?\.shift\(\s*-\s*(?P<horizon>\d+)\s*\)"
)


LABEL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "tag": ("post_close",),
    "5d_tag": ("post5_close",),
    "2d_tag": ("post2_close",),
    "yield_rate": ("post_close",),
    "close_yield_rate": ("post_close", "post2_close"),
    "open_yield_rate": ("post_open",),
    "open2_yield_rate": ("post_open", "post2_open"),
    "open3_yield_rate": ("post_open", "post3_open"),
    "open4_yield_rate": ("post_open", "post4_open"),
    "open5_yield_rate": ("post_open", "post5_open"),
    "open6_yield_rate": ("post_open", "post6_open"),
    "open12_yield_rate": ("post_open", "post12_open"),
    "open22_yield_rate": ("post_open", "post22_open"),
    "5d_yield_rate": ("post5_close",),
    "6d_yield_rate": ("post6_close",),
    "10d_yield_rate": ("post10_close",),
    "15d_yield_rate": ("post15_close",),
    "22d_yield_rate": ("post22_close",),
    "2d_yield_rate": ("post2_close",),
    "log_yield_rate": ("post_close",),
    "log_2d_yield_rate": ("post2_close",),
    "next_open_yield_rate": ("post_open",),
    "close_open_yield_rate": ("post_open", "post_close"),
    "close_low_yield_rate": ("post_low", "post_close"),
    "low_close_yield_rate": ("post_low", "post_close"),
    "post5_most_high_yield_rate": ("post_open", "post_high", "post2_high", "post3_high", "post4_high", "post5_high"),
    "index_2000_post10_yield_rate": ("index_2000_post10_close",),
    "adjust_10d_yield_rate": ("post10_close", "index_2000_post10_close"),
    "executable_1d_open_return": ("post_open", "post2_open"),
    "executable_2d_open_return": ("post_open", "post2_open"),
    "executable_3d_open_return": ("post_open", "post4_open"),
    "executable_5d_open_return": ("post_open", "post6_open"),
    "executable_10d_open_return": ("post_open", "post12_open"),
}


def _streaming_policy() -> MemoryPolicy:
    """Return the resource contract for this candidate executor only."""

    return MemoryPolicy(
        workers=1,
        worker_rss_hard_bytes=STREAMING_WORKER_BUDGET_BYTES,
        parent_rss_budget_bytes=STREAMING_PARENT_BUDGET_BYTES,
        startup_available_bytes=STREAMING_STARTUP_AVAILABLE_BYTES,
        dispatch_available_bytes=STREAMING_STARTUP_AVAILABLE_BYTES,
        system_hard_stop_bytes=STREAMING_SYSTEM_HARD_STOP_BYTES,
        recovery_timeout_seconds=120,
        recovery_tolerance_bytes=1 * 1024**3,
        baseline_drop_limit_bytes=2 * 1024**3,
        baseline_completion_count=2,
        sample_interval_seconds=1.0,
        duckdb_memory_limit=STREAMING_DUCKDB_MEMORY_LIMIT,
    )


def _streaming_resource_contract(policy: MemoryPolicy) -> dict[str, Any]:
    return {
        "profile_id": "l3-memory-bounded-v2-streaming-candidate",
        "topology": {
            "workers": policy.workers,
            "short_lived_processes": True,
            "raw_stock_buckets": 1024,
            "gtja_date_chunk_days": 5,
            "gtja_alpha_batch_size": 1,
            "merge_fan_in": STREAMING_FAN_IN,
            "merge_levels": "fixed_fan_in_until_single_output",
        },
        "memory": {
            "system_hard_stop_bytes": policy.system_hard_stop_bytes,
            "parent_budget_bytes": policy.parent_rss_budget_bytes,
            "worker_budget_bytes": policy.worker_rss_hard_bytes,
            "reserve_bytes": STREAMING_RESERVE_BYTES,
            "startup_available_bytes": policy.startup_available_bytes,
            "dispatch_available_bytes": policy.dispatch_available_bytes,
            "duckdb_memory_limit": policy.duckdb_memory_limit,
            "formula": "20GiB + 4GiB + 6GiB + 4GiB = 34GiB",
        },
        "safety": {
            "original_64_gib_profile_unchanged": True,
            "unknown_writer_fail_closed": True,
            "wal_fail_closed": True,
            "candidate_only": True,
            "active_switch": False,
            "label_write": False,
            "registry_change": False,
        },
    }


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _runtime_provenance_gate() -> dict[str, str]:
    """Fail closed unless parent/worker Python provenance is the audited copy."""

    executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    unittest_path = Path(unittest.__file__).resolve()
    expected_executable = (RUNTIME_ROOT / "python.exe").resolve()
    paths = {
        "sys.executable": executable,
        "sys.prefix": prefix,
        "sys.base_prefix": base_prefix,
        "stdlib": stdlib,
        "unittest": unittest_path,
    }
    if executable != expected_executable or any(not _path_is_under(path, RUNTIME_ROOT) for path in paths.values()):
        raise RuntimeError(
            "candidate runtime provenance failed closed: "
            + json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, sort_keys=True)
        )
    return {key: str(value) for key, value in paths.items()}


def _negative_shift_lineage_from_source() -> dict[str, str]:
    """Extract every negative-shift output from the source lineage and audit it."""

    source_path = PROJECT_ROOT / "quant" / "main" / "data_process_module.py"
    source = source_path.read_text(encoding="utf-8")
    lineage = {
        match.group("field"): f"source.shift(-{match.group('horizon')})"
        for match in NEGATIVE_SHIFT_LINEAGE_RE.finditer(source)
    }
    if not lineage:
        raise RuntimeError(f"negative-shift lineage source scan returned no fields: {source_path}")
    validate_future_lineage(lineage)
    return lineage


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _file_sha256(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_state(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(path) if include_hash else None,
    }


def _schema(conn: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, Any]]:
    return [
        {"column_index": int(row[0]), "name": str(row[1]), "type": str(row[2]), "not_null": bool(row[3])}
        for row in conn.execute(f"PRAGMA table_info({_quote_literal(table)})").fetchall()
    ]


def _schema_hash(schema: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "index": int(item.get("index", item.get("column_index", index))),
            "name": str(item["name"]),
            "type": str(item["type"]),
            "not_null": bool(item["not_null"]),
        }
        for index, item in enumerate(schema)
    ]
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ordered_columns(columns: Iterable[str], *, context: str) -> list[str]:
    names = [str(column) for column in columns]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{context} schema has duplicate columns")
    missing = [column for column in KEY_COLUMNS if column not in names]
    if missing:
        raise RuntimeError(f"{context} schema missing key columns: {missing}")
    if names[: len(KEY_COLUMNS)] != KEY_COLUMNS:
        raise RuntimeError(
            f"{context} schema key order mismatch: expected={KEY_COLUMNS} observed={names[:len(KEY_COLUMNS)]}"
        )
    return [*KEY_COLUMNS, *[column for column in names if column not in KEY_COLUMNS]]


def _ordered_projection(columns: Iterable[str], *, context: str, relation: str | None = None) -> str:
    ordered = _ordered_columns(columns, context=context)
    prefix = f"{relation}." if relation else ""
    return ", ".join(f"{prefix}{_quote_ident(column)}" for column in ordered)


def _relation_columns(conn: duckdb.DuckDBPyConnection, relation: str) -> list[str]:
    return [str(row[0]) for row in conn.execute(f"DESCRIBE {relation}").fetchall()]


def _relation_schema(conn: duckdb.DuckDBPyConnection, relation: str) -> list[dict[str, Any]]:
    """Return an ordered schema for a local or attached DuckDB relation."""
    rows = conn.execute(f"DESCRIBE {relation}").fetchall()
    return [
        {
            "column_index": index,
            "name": str(row[0]),
            "type": str(row[1]),
            "not_null": str(row[2]).upper() == "NO",
        }
        for index, row in enumerate(rows)
    ]


def _assert_schema_contract(actual: list[dict[str, Any]], expected: list[dict[str, Any]], *, context: str) -> None:
    actual_signature = [(item["name"], item["type"], bool(item["not_null"])) for item in actual]
    expected_signature = [(item["name"], item["type"], bool(item["not_null"])) for item in expected]
    if actual_signature != expected_signature:
        raise RuntimeError(
            f"{context} schema contract mismatch: expected={expected_signature} observed={actual_signature}"
        )


def _schema_with_lowercase_names(schema: list[dict[str, Any]], *, context: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in schema:
        lowered = str(item["name"]).lower()
        if lowered in seen:
            raise RuntimeError(f"{context} lower-case schema name collision: {lowered}")
        seen.add(lowered)
        copied = dict(item)
        copied["name"] = lowered
        normalized.append(copied)
    return normalized


def _literal_iter_values(node: ast.AST) -> list[Any] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[Any] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, (str, int)):
            return None
        values.append(element.value)
    return values


def _render_new_column_key(node: ast.AST, scope: dict[str, Any]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
            if value.value.id not in scope:
                return None
            parts.append(str(scope[value.value.id]))
        else:
            return None
    return "".join(parts)


def _collect_group_factor_derived_columns_from_ast(source: str) -> set[str]:
    tree = ast.parse(source)
    derived: set[str] = set()

    def visit(node: ast.AST, scope: dict[str, Any]) -> None:
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            values = _literal_iter_values(node.iter)
            if values is not None:
                for item in values:
                    scoped = dict(scope)
                    scoped[node.target.id] = item
                    for child in node.body:
                        visit(child, scoped)
                for child in node.orelse:
                    visit(child, scope)
                return
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "new_columns"
                ):
                    rendered = _render_new_column_key(target.slice, scope)
                    if rendered is not None:
                        derived.add(rendered)
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, {})
    return derived


def _approved_group_factor_derived_columns() -> set[str]:
    derived = _collect_group_factor_derived_columns_from_ast(GROUP_FACTOR_ENGINE_SOURCE)
    if not derived:
        derived = {str(name) for name in re.findall(r"new_columns\[['\"]([^'\"]+)['\"]\]", GROUP_FACTOR_ENGINE_SOURCE)}
    return derived


def _group_factor_source_output_order(source_names: list[str]) -> list[str]:
    ordered = [*KEY_COLUMNS, *[column for column in source_names if column not in KEY_COLUMNS]]
    if "pre_close" in ordered:
        ordered = [column for column in ordered if column != "pre_close"]
        ordered.append("pre_close")
    return ordered


def _raw_factor_output_contract(
    source_schema: list[dict[str, Any]], output_columns: Iterable[str], *, context: str
) -> dict[str, Any]:
    source_names = _ordered_columns([item["name"] for item in source_schema], context=f"{context} L2 input")
    output_names = _ordered_columns(output_columns, context=f"{context} output")
    missing_source = [column for column in source_names if column not in output_names]
    if missing_source:
        raise RuntimeError(f"{context} raw output dropped L2 input columns: {missing_source}")
    derived = [column for column in output_names if column not in source_names]
    approved = _approved_group_factor_derived_columns()
    unauthorized = [column for column in derived if column not in approved]
    if unauthorized:
        raise RuntimeError(f"{context} raw output contains unapproved derived columns: {unauthorized}")
    canonical_order = [*_group_factor_source_output_order(source_names), *derived]
    if output_names != canonical_order:
        raise RuntimeError(
            f"{context} raw output order mismatch: expected={canonical_order} observed={output_names}"
        )
    return {
        "input_schema": source_schema,
        "input_schema_hash": _schema_hash(source_schema),
        "output_columns": output_names,
        "derived_columns": derived,
        "derived_lineage": "data_process_module.group_factor_eng:new_columns",
    }


def _load_expected_schema(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        schema = payload.get("ordered_l2_schema") or payload.get("ordered_schema")
    else:
        schema = payload
    if not isinstance(schema, list) or not schema:
        raise RuntimeError(f"expected ordered L2 schema file is empty or invalid: {path}")
    return [dict(item) for item in schema]


def _key_metrics(conn: duckdb.DuckDBPyConnection, table: str, target_date: str) -> dict[str, Any]:
    row = conn.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT stock_code),
               MIN(CAST(trade_date AS VARCHAR)), MAX(CAST(trade_date AS VARCHAR)),
               COALESCE(SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END), 0),
               BIT_XOR(HASH(CAST(trade_date AS VARCHAR), stock_code)),
               SUM(CAST(HASH(CAST(trade_date AS VARCHAR), stock_code) AS HUGEINT))
        FROM {_quote_ident(table)}
        """
    ).fetchone()
    target = conn.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT stock_code),
               COALESCE(SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END), 0)
        FROM {_quote_ident(table)} WHERE CAST(trade_date AS VARCHAR) = ?
        """,
        [target_date],
    ).fetchone()
    duplicate = conn.execute(
        f"""
        SELECT COUNT(*) FROM (
          SELECT trade_date, stock_code, COUNT(*) AS n
          FROM {_quote_ident(table)} GROUP BY 1, 2 HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    logical = f"{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[5]}|{row[6]}"
    return {
        "row_count": int(row[0]),
        "stock_count": int(row[1]),
        "min_trade_date": str(row[2]),
        "max_trade_date": str(row[3]),
        "bj_row_count": int(row[4]),
        "duplicate_key_groups": int(duplicate),
        "logical_key_hash": hashlib.sha256(logical.encode("utf-8")).hexdigest(),
        "logical_key_hash_components": {
            "bit_xor": str(row[5]),
            "hash_sum": str(row[6]),
        },
        "target_trade_date": target_date,
        "target_row_count": int(target[0]),
        "target_stock_count": int(target[1]),
        "target_bj_row_count": int(target[2]),
    }


def _active_registry_asset(layer: str) -> tuple[dict[str, Any], Path, str]:
    asset = active_main_workflow_asset(layer, data_dir=DATA_DIR)
    if not asset:
        raise RuntimeError(f"active registry asset missing for {layer}")
    path, table = split_asset_path(asset.get("asset_path"))
    if path is None or not table:
        raise RuntimeError(f"active registry asset is not path::table for {layer}: {asset.get('asset_path')}")
    return asset, path.resolve(), table


def _active_snapshot(layer: str, target_date: str) -> dict[str, Any]:
    asset, path, table = _active_registry_asset(layer)
    with closing(duckdb.connect(str(path), read_only=True)) as conn:
        metrics = _key_metrics(conn, table, target_date)
        schema = _schema(conn, table)
    return {
        "asset_id": asset.get("asset_id"),
        "asset_path": asset.get("asset_path"),
        "path": str(path),
        "table": table,
        "file_state": _file_state(path),
        "metrics": metrics,
        "schema_hash": _schema_hash(schema),
        "column_count": len(schema),
    }


def _scan_l3_processes(args: argparse.Namespace) -> dict[str, Any]:
    psutil = require_psutil()
    workspace = str(Path(args.workspace_dir).resolve())
    report_dir = str(Path(args.report_dir).resolve())
    active_feature = _active_registry_asset("L3_features")[1]
    active_label = _active_registry_asset("L3_labels")[1]
    protected_paths = [
        str(active_feature),
        str(active_label),
        str(REGISTRY_PATH.resolve()),
    ]
    records = collect_process_records(psutil, protected_l3_paths=protected_paths, workspace=workspace)
    return classify_process_gate(
        records,
        current_pid=os.getpid(),
        workflow_run_id=args.workflow_run_id,
        workspace=workspace,
        report_dir=report_dir,
        unified_python_launcher=str(UNIFIED_PYTHON_LAUNCHER),
        expected_workers=load_expected_workers(workspace),
        protected_l3_paths=protected_paths,
    )


def _load_expected_l2_hash(target_date: str) -> str:
    report = DATA_DIR / "reports" / f"l2_handoff_contract_{target_date}.json"
    if report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        value = (((payload.get("layer_payload") or {}).get("active_asset") or {}).get("file_state") or {}).get("sha256")
        if value:
            return str(value).lower()
    if target_date == "20260717":
        return EXPECTED_L2_SHA256_20260717
    raise RuntimeError(f"expected L2 SHA256 is unavailable for {target_date}")


def _precheck(args: argparse.Namespace) -> dict[str, Any]:
    process_gate_path = Path(args.report_dir).resolve() / f"l3_full_processing_{args.target_trade_date}_process_gate.json"
    try:
        process_scan = _scan_l3_processes(args)
    except BaseException as error:
        process_scan = {
            "status": "process_gate_scan_failed",
            "current_pid": os.getpid(),
            "workflow_run_id": args.workflow_run_id,
            "workspace": str(Path(args.workspace_dir).resolve()),
            "report_dir": str(Path(args.report_dir).resolve()),
            "allowed_current_chain": [],
            "allowed_readonly_observers": [],
            "blocking_writers": [],
            "unknown_relevant_processes": [{
                "pid": os.getpid(),
                "process_role": "process_gate_scan_failure",
                "decision": "block_unknown",
                "reason": f"{type(error).__name__}: {error}",
            }],
            "passed": False,
        }
        _write_json(process_gate_path, process_scan)
        raise
    _write_json(process_gate_path, process_scan)
    psutil = require_psutil()
    available_memory = int(psutil.virtual_memory().available)
    policy = _streaming_policy()
    startup_memory_limit = policy.startup_available_bytes
    if available_memory < startup_memory_limit:
        raise RuntimeError(
            f"L3 startup available memory gate failed: available={available_memory} limit={startup_memory_limit}"
        )
    free_bytes = int(shutil.disk_usage(DATA_DIR).free)
    processes = process_scan["blocking_writers"]
    unknown_processes = process_scan["unknown_relevant_processes"]
    l2_asset, l2_path, l2_table = _active_registry_asset("L2")
    if l2_asset.get("asset_id") != EXPECTED_L2_ASSET_ID:
        raise RuntimeError(f"active L2 asset-id drift: {l2_asset.get('asset_id')}")
    if l2_path != EXPECTED_L2_PATH.resolve() or l2_table != EXPECTED_L2_TABLE:
        raise RuntimeError(f"active L2 route drift: {l2_path}::{l2_table}")
    expected_hash = str(args.expected_l2_sha256 or _load_expected_l2_hash(args.target_trade_date)).lower()
    actual_hash = _file_sha256(l2_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"active L2 SHA256 drift: expected={expected_hash} actual={actual_hash}")
    with closing(duckdb.connect(str(l2_path), read_only=True)) as conn:
        metrics = _key_metrics(conn, l2_table, args.target_trade_date)
        schema = _schema(conn, l2_table)
    columns = [item["name"] for item in schema]
    ordered_l2_schema = _ordered_columns(columns, context="audited L2")
    ordered_l2_schema_hash = _schema_hash(schema)
    expected_schema = _load_expected_schema(args.expected_l2_schema_json)
    expected_schema_hash = str(args.expected_l2_schema_hash or "").lower()
    if not expected_schema_hash or expected_schema is None:
        raise RuntimeError("exact expected ordered L2 schema and schema hash are required before precheck")
    _assert_schema_contract(schema, expected_schema, context="audited L2 vs expected")
    if _schema_hash(expected_schema) != expected_schema_hash or ordered_l2_schema_hash != expected_schema_hash:
        raise RuntimeError(
            f"audited L2 schema hash drift: expected={expected_schema_hash} observed={ordered_l2_schema_hash}"
        )
    qfq_columns = [column for column in columns if "_qfq" in column]
    qfq_technical = [column for column in qfq_columns if column not in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS]
    gates = {
        "free_disk_at_least_250gb": free_bytes >= MIN_FREE_BYTES,
        "startup_available_at_least_streaming_profile": available_memory >= startup_memory_limit,
        "no_residual_l3_write_process": process_scan["passed"],
        "l2_asset_id_locked": l2_asset.get("asset_id") == EXPECTED_L2_ASSET_ID,
        "l2_route_locked": l2_path == EXPECTED_L2_PATH.resolve() and l2_table == EXPECTED_L2_TABLE,
        "l2_sha256_locked": actual_hash == expected_hash,
        "l2_full_history": metrics["min_trade_date"] == "20100104" and metrics["max_trade_date"] == args.target_trade_date,
        "l2_target_coverage": metrics["target_row_count"] == metrics["target_stock_count"] and metrics["target_row_count"] > 0,
        "l2_no_bj": metrics["bj_row_count"] == 0 and metrics["target_bj_row_count"] == 0,
        "l2_duplicate_zero": metrics["duplicate_key_groups"] == 0,
        "l2_explicit_qfq_price_5": all(column in columns for column in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS),
        "l2_qfq_technical_74": len(qfq_technical) == QFQ_TECHNICAL_EXPECTED_COUNT,
        "l2_schema_key_order": columns[: len(KEY_COLUMNS)] == KEY_COLUMNS,
        "l2_schema_ordered_contract": columns == ordered_l2_schema,
        "l2_schema_hash_locked": ordered_l2_schema_hash == expected_schema_hash,
    }
    if not all(gates.values()):
        failed_gates = [key for key, value in gates.items() if not value]
        raise RuntimeError(
            f"L3 precheck gates failed: {failed_gates}; "
            f"residual_l3_write_processes={processes}; unknown_relevant_processes={unknown_processes}; "
            f"process_gate_path={process_gate_path}"
        )
    return {
        "status": "precheck_passed",
        "generated_at": _now(),
        "target_trade_date": args.target_trade_date,
        "workflow_run_id": args.workflow_run_id,
        "processing_mode": "full_history_rebuild",
        "resource_contract": _streaming_resource_contract(policy),
        "disk": {"free_bytes": free_bytes, "required_free_bytes": MIN_FREE_BYTES},
        "memory": {"available_bytes": available_memory, "required_startup_available_bytes": startup_memory_limit},
        "running_processes": processes,
        "process_scan": process_scan,
        "process_gate_path": str(process_gate_path),
        "l2": {
            "asset_id": l2_asset.get("asset_id"),
            "path": str(l2_path),
            "table": l2_table,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "schema_hash": ordered_l2_schema_hash,
            "ordered_schema": schema,
            "ordered_schema_hash": ordered_l2_schema_hash,
            "expected_ordered_schema_hash": expected_schema_hash,
            "key_order": KEY_COLUMNS,
            "column_count": len(columns),
            "qfq_column_count": len(qfq_columns),
            "qfq_technical_column_count": len(qfq_technical),
            "qfq_technical_columns": qfq_technical,
            "metrics": metrics,
        },
        "gates": gates,
    }


def _validate_no_forbidden_runtime_paths(paths: Iterable[Path], target_date: str) -> None:
    for path in paths:
        lowered = str(path).replace("\\", "/").lower()
        for token in FORBIDDEN_PATH_TOKENS:
            if token.lower() in lowered and token != target_date:
                raise RuntimeError(f"forbidden legacy/cancelled/quarantine path: {path} token={token}")


def _require_no_shrink(source_rows: int, output_rows: int, *, stage: str) -> None:
    if int(source_rows) != int(output_rows):
        raise RuntimeError(f"{stage} full-history shrink: source={source_rows} output={output_rows}")


def _compute_maturity_date(trade_dates: list[str], horizon: int = MAX_LABEL_HORIZON) -> str:
    ordered = sorted({str(value) for value in trade_dates})
    if len(ordered) <= horizon:
        raise RuntimeError("not enough trade dates to compute label maturity")
    return ordered[-(horizon + 1)]


def _feature_schema_gate_summary(columns: list[str], l2_qfq_technical: list[str]) -> dict[str, Any]:
    qfq_technical = [column for column in l2_qfq_technical if column in columns]
    gtja_qfq = [column for column in columns if column.startswith("gtja_alpha") and column.endswith("_qfq")]
    gtja_naked = [column for column in columns if column.startswith("gtja_alpha") and not column.endswith("_qfq")]
    naked_aliases = [column for column in [*NAKED_MARKET_PRICE_COLUMNS, *NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS] if column in columns]
    future_columns = [column for column in columns if is_future_or_label_column(column)]
    return {
        "qfq_price_5": all(column in columns for column in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS),
        "qfq_technical_74": len(qfq_technical) == QFQ_TECHNICAL_EXPECTED_COUNT,
        "gtja_qfq_191": len(gtja_qfq) == len(GTJA_ALPHA_COLUMNS),
        "gtja_naked_zero": len(gtja_naked) == 0,
        "naked_qfq_alias_zero": len(naked_aliases) == 0,
        "future_label_leakage_zero": len(future_columns) == 0,
        "details": {
            "qfq_technical_columns": qfq_technical,
            "gtja_qfq_columns": gtja_qfq,
            "gtja_naked_columns": gtja_naked,
            "naked_aliases": naked_aliases,
            "future_columns": future_columns,
        },
    }


def _stable_bucket(code: str, bucket_count: int) -> int:
    return int(hashlib.sha256(code.encode("utf-8")).hexdigest()[:16], 16) % bucket_count


def _deduplicate_columns_preserve_source(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.duplicated().any():
        return frame
    return frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()


def _configure_worker_duckdb(
    conn: duckdb.DuckDBPyConnection,
    memory_limit: str,
    temp_directory: Path | None = None,
) -> None:
    safe_limit = str(memory_limit).replace("'", "")
    conn.execute("SET threads=1")
    conn.execute(f"SET memory_limit='{safe_limit}'")
    if temp_directory is not None:
        temp_directory.mkdir(parents=True, exist_ok=True)
        conn.execute(f"SET temp_directory='{str(temp_directory).replace(chr(39), chr(39) * 2)}'")


def _build_raw_bucket(task: dict[str, Any]) -> dict[str, Any]:
    _runtime_provenance_gate()
    started = time.perf_counter()
    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    source_path = Path(task["l2_path"])
    source_table = str(task["l2_table"])
    codes = list(task["codes"])
    source_rows = 0
    output_rows = 0
    duplicate_columns_removed: set[str] = set()
    with closing(duckdb.connect(str(source_path), read_only=True)) as source, closing(duckdb.connect(str(output_path))) as target:
        temp_directory = Path(task["temp_directory"])
        _configure_worker_duckdb(source, task["duckdb_memory_limit"], temp_directory)
        _configure_worker_duckdb(target, task["duckdb_memory_limit"], temp_directory)
        source_schema = _relation_schema(source, _quote_ident(source_table))
        raw_source_schema = _schema_with_lowercase_names(source_schema, context="L2 raw pandas input")
        ordered_source_columns = _ordered_columns([item["name"] for item in source_schema], context="L2 source")
        source_projection = ", ".join(_quote_ident(column) for column in ordered_source_columns)
        created = False
        expected_raw_schema: list[dict[str, Any]] | None = None
        raw_contract: dict[str, Any] | None = None
        for code in codes:
            frame = source.execute(
                f"SELECT {source_projection} FROM {_quote_ident(source_table)} WHERE stock_code = ? ORDER BY trade_date",
                [code],
            ).fetchdf()
            if frame.empty:
                continue
            source_rows += int(frame.shape[0])
            frame.columns = frame.columns.str.lower()
            factor = group_factor_eng(frame, include_future_labels=True)
            duplicates = factor.columns[factor.columns.duplicated()].tolist()
            duplicate_columns_removed.update(str(item) for item in duplicates)
            factor = _deduplicate_columns_preserve_source(factor)
            factor = _normalize_types(factor)
            factor = factor[~factor["stock_code"].astype(str).str.endswith(".BJ")].copy()
            raw_contract = _raw_factor_output_contract(
                raw_source_schema,
                factor.columns,
                context=f"raw factor {code}",
            )
            factor = factor[raw_contract["output_columns"]]
            if factor.shape[0] != frame.shape[0]:
                raise RuntimeError(f"raw factor row mismatch for {code}: source={frame.shape[0]} output={factor.shape[0]}")
            target.register("factor_frame", factor)
            try:
                factor_frame_schema = _relation_schema(target, "factor_frame")
                if expected_raw_schema is None:
                    expected_raw_schema = factor_frame_schema
                else:
                    _assert_schema_contract(
                        factor_frame_schema,
                        expected_raw_schema,
                        context=f"raw factor bucket {task['bucket_index']} code {code}",
                    )
                if not created:
                    target.execute(
                        f"CREATE TABLE raw_factor AS SELECT {_ordered_projection(factor.columns, context='raw factor', relation='factor_frame')} FROM factor_frame"
                    )
                    created = True
                else:
                    target.execute(
                        f"INSERT INTO raw_factor BY NAME SELECT {_ordered_projection(factor.columns, context='raw factor', relation='factor_frame')} FROM factor_frame"
                    )
            finally:
                target.unregister("factor_frame")
            output_rows += int(factor.shape[0])
            del factor
            del frame
            gc.collect()
        if not created:
            raise RuntimeError(f"raw bucket produced no rows: {task['bucket_index']}")
        duplicate_groups = target.execute(
            "SELECT COUNT(*) FROM (SELECT trade_date, stock_code, COUNT(*) n FROM raw_factor GROUP BY 1,2 HAVING COUNT(*)>1)"
        ).fetchone()[0]
        actual_columns = _relation_columns(target, "raw_factor")
        actual_schema = _relation_schema(target, "raw_factor")
        if expected_raw_schema is None:
            raise RuntimeError("raw factor output schema contract was not materialized")
        _assert_schema_contract(actual_schema, expected_raw_schema, context="raw factor materialization")
        if actual_columns != [item["name"] for item in expected_raw_schema]:
            raise RuntimeError("raw factor materialization column order drift")
    if source_rows != output_rows or duplicate_groups:
        raise RuntimeError(
            f"raw bucket integrity failure bucket={task['bucket_index']} source={source_rows} output={output_rows} duplicate={duplicate_groups}"
        )
    result = {
        "status": "completed",
        "bucket_index": int(task["bucket_index"]),
        "path": str(output_path),
        "code_count": len(codes),
        "source_rows": source_rows,
        "output_rows": output_rows,
        "duplicate_key_groups": int(duplicate_groups),
        "raw_output_schema_hash": _schema_hash(expected_raw_schema),
        "raw_input_schema_hash": _schema_hash(source_schema),
        "raw_output_derived_columns": (raw_contract or {}).get("derived_columns", []),
        "duplicate_columns_removed_preserving_l2_source": sorted(duplicate_columns_removed),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return result


def _validate_raw_completion(task: dict[str, Any], result: dict[str, Any]) -> None:
    if int(result.get("source_rows", -1)) != int(result.get("output_rows", -2)):
        raise RuntimeError(f"raw completion row mismatch: task={task.get('task_id')}")
    if int(result.get("duplicate_key_groups", -1)) != 0:
        raise RuntimeError(f"raw completion duplicate keys: task={task.get('task_id')}")
    if int(result.get("code_count", -1)) != len(task.get("codes") or []):
        raise RuntimeError(f"raw completion code count mismatch: task={task.get('task_id')}")


def _run_short_lived_tasks(
    *,
    stage: str,
    tasks: list[dict[str, Any]],
    workers: int,
    workspace: Path,
    logger,
    worker_fn,
    result_validator,
    policy: MemoryPolicy | None = None,
    workflow_run_id: str | None = None,
    run_nonce: str | None = None,
) -> dict[str, Any]:
    quarantine_root = workspace.parent.parent / "quarantine"
    scheduler = ShortLivedTaskScheduler(
        stage=stage,
        worker_fn=worker_fn,
        workspace=workspace,
        quarantine_root=quarantine_root,
        logger=logger,
        workflow_run_id=workflow_run_id,
        run_nonce=run_nonce,
        policy=policy or MemoryPolicy(workers=workers),
        result_validator=result_validator,
    )
    return scheduler.run(tasks)


def _merge_shards(
    shards: Iterable[Path],
    output_path: Path,
    table: str,
    source_table: str,
    *,
    workspace: Path,
    policy: MemoryPolicy,
) -> dict[str, Any]:
    """Merge shards through immutable, fixed fan-in stages.

    Each group is materialized by a short-lived stage file.  The final index is
    built only after row/key equivalence has been checked, and no active asset
    is opened by this candidate-only function.
    """
    started = time.perf_counter()
    current = sorted(Path(item) for item in shards)
    if not current:
        raise RuntimeError(f"no shards available for {table}")
    level = 0
    merge_root = workspace / "merge_levels" / table
    while len(current) > 1:
        next_level: list[Path] = []
        level_dir = merge_root / f"level_{level:02d}"
        level_dir.mkdir(parents=True, exist_ok=True)
        for group_index in range(0, len(current), STREAMING_FAN_IN):
            group = current[group_index : group_index + STREAMING_FAN_IN]
            staged = level_dir / f"merge_{group_index // STREAMING_FAN_IN:05d}.duckdb"
            temp_path = staged.with_suffix(staged.suffix + ".tmp")
            temp_path.unlink(missing_ok=True)
            total = 0
            with closing(duckdb.connect(str(temp_path))) as target:
                _configure_worker_duckdb(
                    target,
                    policy.duckdb_memory_limit,
                    workspace / STREAMING_TEMP_DIR_NAME / f"merge_{table}_{level:02d}_{group_index // STREAMING_FAN_IN:05d}",
                )
                created = False
                expected_schema: list[dict[str, Any]] | None = None
                for index, shard in enumerate(group):
                    alias = f"s{index}"
                    target.execute(f"ATTACH {_quote_literal(str(shard))} AS {_quote_ident(alias)} (READ_ONLY)")
                    count = int(target.execute(f"SELECT COUNT(*) FROM {_quote_ident(alias)}.{_quote_ident(source_table)}").fetchone()[0])
                    source_relation = f"{_quote_ident(alias)}.{_quote_ident(source_table)}"
                    source_schema = _relation_schema(target, source_relation)
                    source_columns = [item["name"] for item in source_schema]
                    ordered_columns = _ordered_columns(source_columns, context=f"merge {table} shard {shard.name}")
                    if expected_schema is None:
                        expected_schema = source_schema
                    else:
                        _assert_schema_contract(source_schema, expected_schema, context=f"merge {table} shard {shard.name}")
                    if ordered_columns != [item["name"] for item in expected_schema]:
                        raise RuntimeError(f"merge schema order drift table={table} shard={shard.name}")
                    projection = _ordered_projection(ordered_columns, context=f"merge {table}", relation="src")
                    if not created:
                        target.execute(f"CREATE TABLE {_quote_ident(table)} AS SELECT {projection} FROM {source_relation} AS src")
                        created = True
                    else:
                        target.execute(f"INSERT INTO {_quote_ident(table)} BY NAME SELECT {projection} FROM {source_relation} AS src")
                    total += count
                    target.execute(f"DETACH {_quote_ident(alias)}")
                if not created:
                    raise RuntimeError(f"empty merge group for {table} level={level} group={group_index}")
                target_rows = int(target.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0])
                duplicate = int(target.execute(
                    f"SELECT COUNT(*) FROM (SELECT trade_date, stock_code, COUNT(*) n FROM {_quote_ident(table)} GROUP BY 1,2 HAVING COUNT(*)>1)"
                ).fetchone()[0])
            if target_rows != total or duplicate:
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"shard merge integrity failure table={table} level={level} group={group_index}: "
                    f"source={total} target={target_rows} duplicate={duplicate}"
                )
            os.replace(temp_path, staged)
            next_level.append(staged)
        current = next_level
        level += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_final = current[0]
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_output.unlink(missing_ok=True)
    shutil.copy2(staged_final, temp_output)
    with closing(duckdb.connect(str(temp_output))) as target:
        _configure_worker_duckdb(target, policy.duckdb_memory_limit, workspace / STREAMING_TEMP_DIR_NAME / f"index_{table}")
        final_columns = _relation_columns(target, _quote_ident(table))
        _ordered_columns(final_columns, context=f"merged {table}")
        target.execute(f"CREATE INDEX idx_{table}_key ON {_quote_ident(table)}(stock_code, trade_date)")
        target_rows = int(target.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0])
        duplicate = int(target.execute(
            f"SELECT COUNT(*) FROM (SELECT trade_date, stock_code, COUNT(*) n FROM {_quote_ident(table)} GROUP BY 1,2 HAVING COUNT(*)>1)"
        ).fetchone()[0])
    if duplicate:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"final shard merge duplicate keys table={table}: {duplicate}")
    os.replace(temp_output, output_path)
    return {
        "path": str(output_path),
        "table": table,
        "row_count": target_rows,
        "duplicate_key_groups": duplicate,
        "fan_in": STREAMING_FAN_IN,
        "merge_levels": level,
        "intermediate_files": len(list(merge_root.rglob("*.duckdb"))),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _build_raw_stage(precheck: dict[str, Any], args: argparse.Namespace, workspace: Path, logger) -> dict[str, Any]:
    policy = _streaming_policy()
    shards_dir = workspace / "raw_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    with closing(duckdb.connect(precheck["l2"]["path"], read_only=True)) as conn:
        codes = [str(row[0]) for row in conn.execute(
            f"SELECT DISTINCT stock_code FROM {_quote_ident(precheck['l2']['table'])} WHERE stock_code NOT LIKE '%.BJ' ORDER BY stock_code"
        ).fetchall()]
    buckets: list[list[str]] = [[] for _ in range(args.raw_buckets)]
    for code in codes:
        buckets[_stable_bucket(code, args.raw_buckets)].append(code)
    tasks = []
    completion_dir = workspace / "logs" / "raw_buckets"
    for index, bucket_codes in enumerate(buckets):
        if not bucket_codes:
            continue
        tasks.append({
            "task_id": f"raw_bucket_{index:04d}",
            "stage": "raw",
            "bucket_index": index,
            "codes": bucket_codes,
            "l2_path": precheck["l2"]["path"],
            "l2_table": precheck["l2"]["table"],
            "output_path": str(shards_dir / f"raw_bucket_{index:04d}.duckdb"),
            "completion_path": str(completion_dir / f"raw_bucket_{index:04d}.json"),
            "failure_path": str(completion_dir / f"raw_bucket_{index:04d}_failure.json"),
            "duckdb_memory_limit": policy.duckdb_memory_limit,
            "temp_directory": str(workspace / STREAMING_TEMP_DIR_NAME / f"raw_{index:04d}"),
        })
    logger("raw_bucket_plan", bucket_count=len(tasks), stock_count=len(codes), workers=args.workers)
    scheduler_summary = _run_short_lived_tasks(
        stage="raw",
        tasks=tasks,
        workers=args.workers,
        workspace=workspace,
        logger=logger,
        worker_fn=_build_raw_bucket,
        result_validator=_validate_raw_completion,
        policy=policy,
        workflow_run_id=args.workflow_run_id,
        run_nonce=args.run_nonce,
    )
    results = scheduler_summary["results"]
    output_path = workspace / "staging_raw_factor.duckdb"
    merged = _merge_shards(
        [Path(item["path"]) for item in results], output_path, "raw_factor", "raw_factor", workspace=workspace, policy=policy
    )
    _require_no_shrink(precheck["l2"]["metrics"]["row_count"], merged["row_count"], stage="raw")
    logger("raw_merge_done", **merged)
    return {
        "bucket_results": sorted(results, key=lambda item: item["bucket_index"]),
        "scheduler": {key: value for key, value in scheduler_summary.items() if key != "results"},
        "merged": merged,
    }


def _date_chunks(dates: list[str], size: int) -> list[dict[str, Any]]:
    chunks = []
    for start in range(0, len(dates), size):
        output_dates = dates[start : start + size]
        first_pos = start
        read_start = dates[max(0, first_pos - GTJA_LOOKBACK_DATES)]
        chunks.append({
            "chunk_index": len(chunks),
            "output_dates": output_dates,
            "read_start": read_start,
            "read_end": output_dates[-1],
        })
    return chunks


def _build_gtja_chunk(task: dict[str, Any]) -> dict[str, Any]:
    _runtime_provenance_gate()
    started = time.perf_counter()
    raw_path = Path(task["raw_path"])
    output_path = Path(task["output_path"])
    log_path = Path(task["log_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    columns = required_gtja_raw_columns()
    select_columns = ", ".join(_quote_ident(column) for column in _ordered_columns(columns, context="GTJA raw input"))
    with closing(duckdb.connect(str(raw_path), read_only=True)) as conn:
        _configure_worker_duckdb(conn, task["duckdb_memory_limit"], Path(task["temp_directory"]))
        frame = conn.execute(
            f"SELECT {select_columns} FROM raw_factor WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date, stock_code",
            [task["read_start"], task["read_end"]],
        ).fetchdf()
    events: list[dict[str, Any]] = []

    def progress(event: dict[str, Any]) -> None:
        payload = {"time": _now(), "chunk_index": task["chunk_index"], **event}
        events.append(payload)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    computed = compute_gtja_alpha_from_raw_factor(
        frame,
        encode=False,
        drop_ts=True,
        cross_sectional_rank_mode="rank",
        output_dates=list(task["output_dates"]),
        alpha_batch_size=int(task["alpha_batch_size"]),
        progress_callback=progress,
    )
    keep = [*KEY_COLUMNS, *GTJA_ALPHA_COLUMNS]
    missing = [column for column in keep if column not in computed.columns]
    if missing:
        raise RuntimeError(f"GTJA chunk missing columns: {missing}")
    computed = computed[keep].copy()
    computed["trade_date"] = computed["trade_date"].astype(str)
    computed = computed[computed["trade_date"].isin(task["output_dates"])].copy()
    with closing(duckdb.connect(str(output_path))) as target:
        _configure_worker_duckdb(target, task["duckdb_memory_limit"], Path(task["temp_directory"]))
        target.register("gtja_frame", computed)
        try:
            target.execute(
                f"CREATE TABLE gtja_factor AS SELECT {_ordered_projection(keep, context='GTJA output', relation='gtja_frame')} FROM gtja_frame"
            )
            duplicate = int(target.execute(
                "SELECT COUNT(*) FROM (SELECT trade_date, stock_code, COUNT(*) n FROM gtja_factor GROUP BY 1,2 HAVING COUNT(*)>1)"
            ).fetchone()[0])
        finally:
            target.unregister("gtja_frame")
    if duplicate:
        raise RuntimeError(f"GTJA chunk duplicate keys: chunk={task['chunk_index']} duplicate={duplicate}")
    result = {
        "chunk_index": int(task["chunk_index"]),
        "path": str(output_path),
        "log_path": str(log_path),
        "read_start": task["read_start"],
        "read_end": task["read_end"],
        "output_start": task["output_dates"][0],
        "output_end": task["output_dates"][-1],
        "output_date_count": len(task["output_dates"]),
        "input_rows": int(frame.shape[0]),
        "output_rows": int(computed.shape[0]),
        "duplicate_key_groups": duplicate,
        "alpha_batch_count": len(events),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    del computed
    del frame
    gc.collect()
    return result


def _validate_gtja_completion(task: dict[str, Any], result: dict[str, Any]) -> None:
    if int(result.get("duplicate_key_groups", -1)) != 0:
        raise RuntimeError(f"GTJA completion duplicate keys: task={task.get('task_id')}")
    if int(result.get("output_rows", 0)) <= 0:
        raise RuntimeError(f"GTJA completion has no output rows: task={task.get('task_id')}")
    if int(result.get("output_date_count", -1)) != len(task.get("output_dates") or []):
        raise RuntimeError(f"GTJA completion date count mismatch: task={task.get('task_id')}")


def _build_gtja_stage(raw_stage: dict[str, Any], args: argparse.Namespace, workspace: Path, logger) -> dict[str, Any]:
    policy = _streaming_policy()
    raw_path = Path(raw_stage["merged"]["path"])
    with closing(duckdb.connect(str(raw_path), read_only=True)) as conn:
        dates = [str(row[0]) for row in conn.execute("SELECT DISTINCT CAST(trade_date AS VARCHAR) FROM raw_factor ORDER BY 1").fetchall()]
    chunks = _date_chunks(dates, args.gtja_date_chunk_days)
    shards_dir = workspace / "gtja_shards"
    logs_dir = workspace / "logs" / "gtja_chunks"
    tasks = []
    for chunk in chunks:
        tasks.append({
            **chunk,
            "task_id": f"gtja_chunk_{chunk['chunk_index']:04d}",
            "stage": "gtja",
            "raw_path": str(raw_path),
            "alpha_batch_size": args.gtja_alpha_batch_size,
            "output_path": str(shards_dir / f"gtja_chunk_{chunk['chunk_index']:04d}.duckdb"),
            "log_path": str(logs_dir / f"gtja_chunk_{chunk['chunk_index']:04d}.jsonl"),
            "completion_path": str(logs_dir / f"gtja_chunk_{chunk['chunk_index']:04d}.json"),
            "failure_path": str(logs_dir / f"gtja_chunk_{chunk['chunk_index']:04d}_failure.json"),
            "duckdb_memory_limit": policy.duckdb_memory_limit,
            "temp_directory": str(workspace / STREAMING_TEMP_DIR_NAME / f"gtja_{chunk['chunk_index']:04d}"),
        })
    logger("gtja_chunk_plan", chunk_count=len(tasks), date_count=len(dates), workers=args.workers, alpha_batch_size=args.gtja_alpha_batch_size)
    scheduler_summary = _run_short_lived_tasks(
        stage="gtja",
        tasks=tasks,
        workers=args.workers,
        workspace=workspace,
        logger=logger,
        worker_fn=_build_gtja_chunk,
        result_validator=_validate_gtja_completion,
        policy=policy,
        workflow_run_id=args.workflow_run_id,
        run_nonce=args.run_nonce,
    )
    results = scheduler_summary["results"]
    output_path = workspace / "staging_gtja_factor.duckdb"
    merged = _merge_shards(
        [Path(item["path"]) for item in results], output_path, "gtja_factor", "gtja_factor", workspace=workspace, policy=policy
    )
    _require_no_shrink(raw_stage["merged"]["row_count"], merged["row_count"], stage="GTJA")
    logger("gtja_merge_done", **merged)
    return {
        "chunk_results": sorted(results, key=lambda item: item["chunk_index"]),
        "scheduler": {key: value for key, value in scheduler_summary.items() if key != "results"},
        "merged": merged,
    }


def _candidate_table_metrics(path: Path, table: str, target_date: str) -> dict[str, Any]:
    with closing(duckdb.connect(str(path), read_only=True)) as conn:
        metrics = _key_metrics(conn, table, target_date)
        schema = _schema(conn, table)
    columns = [item["name"] for item in schema]
    _ordered_columns(columns, context=f"candidate {table}")
    return {
        **metrics,
        "column_count": len(schema),
        "schema_hash": _schema_hash(schema),
        "schema_order": columns,
        "columns": columns,
    }


def _build_feature_candidate(
    precheck: dict[str, Any], raw_stage: dict[str, Any], gtja_stage: dict[str, Any], args: argparse.Namespace, workspace: Path, logger
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_path = Path(raw_stage["merged"]["path"])
    gtja_path = Path(gtja_stage["merged"]["path"])
    lineage_audit = validate_future_lineage(_negative_shift_lineage_from_source())
    candidate_dir = workspace / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"l3_feature_candidate_{args.target_trade_date}.duckdb"
    temp_path = candidate_path.with_suffix(".duckdb.tmp")
    temp_path.unlink(missing_ok=True)
    candidate_path.unlink(missing_ok=True)
    with closing(duckdb.connect(str(raw_path), read_only=True)) as conn:
        raw_columns = [str(row[1]) for row in conn.execute("PRAGMA table_info('raw_factor')").fetchall()]
    if args.feature_contract_version == "v2":
        selected = candidate_production_raw_columns(raw_columns)
    else:
        selected = production_raw_columns(raw_columns)
    selected = _ordered_columns(selected, context="feature raw projection")
    raw_exprs = [f"r.{_quote_ident(column)}" for column in selected]
    gtja_exprs = [
        f"g.{_quote_ident(column)} AS {_quote_ident(GTJA_TO_PRODUCTION_COLUMN_MAP[column])}"
        for column in GTJA_ALPHA_COLUMNS
    ]
    active_feature = _active_registry_asset("L3_features")
    table = active_feature[2]
    with closing(duckdb.connect(str(temp_path))) as conn:
        _configure_worker_duckdb(conn, STREAMING_DUCKDB_MEMORY_LIMIT, workspace / STREAMING_TEMP_DIR_NAME / "feature")
        conn.execute(f"ATTACH {_quote_literal(str(raw_path))} AS rawsrc (READ_ONLY)")
        conn.execute(f"ATTACH {_quote_literal(str(gtja_path))} AS gtjasrc (READ_ONLY)")
        conn.execute(
            f"""
            CREATE TABLE {_quote_ident(table)} AS
            SELECT {', '.join([*raw_exprs, *gtja_exprs])}
            FROM rawsrc.raw_factor r
            LEFT JOIN gtjasrc.gtja_factor g
              ON CAST(r.trade_date AS VARCHAR) = CAST(g.trade_date AS VARCHAR)
             AND r.stock_code = g.stock_code
            WHERE r.stock_code NOT LIKE '%.BJ'
            """
        )
        conn.execute(f"CREATE INDEX idx_feature_key ON {_quote_ident(table)}(stock_code, trade_date)")
        conn.execute("DETACH rawsrc")
        conn.execute("DETACH gtjasrc")
    os.replace(temp_path, candidate_path)
    metrics = _candidate_table_metrics(candidate_path, table, args.target_trade_date)
    columns = metrics.pop("columns")
    qfq_columns = [column for column in columns if "_qfq" in column]
    schema_gates = _feature_schema_gate_summary(columns, precheck["l2"]["qfq_technical_columns"])
    v2_schema_audit = None
    if args.feature_contract_version == "v2":
        v2_schema_audit = audit_candidate_feature_schema(columns)
        schema_gates.update(
            {
                "feature_contract_v2_schema": True,
                "qfq_technical_74_source": v2_schema_audit["l2_source_qfq_technical_count"] == 74,
                "qfq_technical_76_output": v2_schema_audit["production_feature_qfq_technical_count"] == 76,
                "derived_qfq_technical_2": v2_schema_audit["production_feature_qfq_technical_count"] - v2_schema_audit["l2_source_qfq_technical_count"] == 2,
                "future_label_leakage_zero": not v2_schema_audit["future_or_label_columns"],
            }
        )
    details = schema_gates.pop("details")
    qfq_technical = details["qfq_technical_columns"]
    gtja_qfq = details["gtja_qfq_columns"]
    gtja_naked = details["gtja_naked_columns"]
    naked_aliases = details["naked_aliases"]
    future_columns = details["future_columns"]
    source_mismatch_expr = " + ".join(
        f"CASE WHEN f.{_quote_ident(column)} IS DISTINCT FROM l.{_quote_ident(column)} THEN 1 ELSE 0 END"
        for column in precheck["l2"]["qfq_technical_columns"]
    )
    with closing(duckdb.connect(str(candidate_path), read_only=True)) as conn:
        conn.execute(f"ATTACH {_quote_literal(precheck['l2']['path'])} AS l2src (READ_ONLY)")
        l2_missing = int(conn.execute(
            f"""
            SELECT COUNT(*) FROM l2src.{_quote_ident(precheck['l2']['table'])} l
            LEFT JOIN {_quote_ident(table)} f
              ON CAST(l.trade_date AS VARCHAR)=CAST(f.trade_date AS VARCHAR) AND l.stock_code=f.stock_code
            WHERE f.stock_code IS NULL
            """
        ).fetchone()[0])
        feature_extra = int(conn.execute(
            f"""
            SELECT COUNT(*) FROM {_quote_ident(table)} f
            LEFT JOIN l2src.{_quote_ident(precheck['l2']['table'])} l
              ON CAST(l.trade_date AS VARCHAR)=CAST(f.trade_date AS VARCHAR) AND l.stock_code=f.stock_code
            WHERE l.stock_code IS NULL
            """
        ).fetchone()[0])
        source_mismatch = int(conn.execute(
            f"""
            SELECT COALESCE(SUM({source_mismatch_expr}),0)
            FROM {_quote_ident(table)} f
            JOIN l2src.{_quote_ident(precheck['l2']['table'])} l
              ON CAST(l.trade_date AS VARCHAR)=CAST(f.trade_date AS VARCHAR) AND l.stock_code=f.stock_code
            """
        ).fetchone()[0])
    gates = {
        "row_count_equals_l2": metrics["row_count"] == precheck["l2"]["metrics"]["row_count"],
        "target_row_count_equals_l2": metrics["target_row_count"] == precheck["l2"]["metrics"]["target_row_count"],
        "key_domain_l2_missing_zero": l2_missing == 0,
        "key_domain_feature_extra_zero": feature_extra == 0,
        "duplicate_zero": metrics["duplicate_key_groups"] == 0,
        "no_bj": metrics["bj_row_count"] == 0,
        **schema_gates,
        "source_limited_null_exact": source_mismatch == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"feature candidate gates failed: {[key for key,value in gates.items() if not value]}")
    result = {
        "path": str(candidate_path),
        "table": table,
        "sha256": _file_sha256(candidate_path),
        "metrics": metrics,
        "qfq_column_count": len(qfq_columns),
        "qfq_technical_column_count": len(qfq_technical),
        "production_feature_qfq_technical_count": len(qfq_technical),
        "l2_source_qfq_technical_count": 74 if args.feature_contract_version == "v2" else len(qfq_technical),
        "derived_qfq_technical_count": 2 if args.feature_contract_version == "v2" else None,
        "feature_contract_version": args.feature_contract_version,
        "v2_schema_audit": v2_schema_audit,
        "lineage_audit": lineage_audit,
        "gtja_qfq_count": len(gtja_qfq),
        "gtja_naked_columns": gtja_naked,
        "naked_qfq_aliases": naked_aliases,
        "future_label_columns": future_columns,
        "l2_missing_keys": l2_missing,
        "feature_extra_keys": feature_extra,
        "source_limited_qfq_mismatch_cells": source_mismatch,
        "gates": gates,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    logger("feature_candidate_done", **result)
    return result


def _label_expr(label: str, raw_columns: set[str]) -> str:
    dependencies = LABEL_DEPENDENCIES.get(label, ())
    missing = [column for column in dependencies if column not in raw_columns]
    if missing:
        raise RuntimeError(f"label {label} missing raw dependencies: {missing}")
    guard = " AND ".join(f"{_quote_ident(column)} IS NOT NULL" for column in dependencies) or "TRUE"
    if label in EXECUTABLE_LABEL_SPECS:
        sell = EXECUTABLE_LABEL_SPECS[label]
        expression = (
            f"({_quote_ident(sell)} * {1.0 - SELL_COMMISSION_RATE - SELL_TAX_RATE - SLIPPAGE_RATE}) / "
            f"({_quote_ident('post_open')} * {1.0 + BUY_COMMISSION_RATE + SLIPPAGE_RATE}) - 1.0"
        )
    elif label in TAG_LABEL_DEPENDENCIES:
        dependency = TAG_LABEL_DEPENDENCIES[label]
        expression = f"CAST({_quote_ident(dependency)} > 0 AS DOUBLE)"
    else:
        expression = _quote_ident(label)
    return f"CASE WHEN {guard} THEN {expression} ELSE NULL END AS {_quote_ident(label)}"


def _build_label_candidate(precheck: dict[str, Any], raw_stage: dict[str, Any], args: argparse.Namespace, workspace: Path, logger) -> dict[str, Any]:
    started = time.perf_counter()
    raw_path = Path(raw_stage["merged"]["path"])
    with closing(duckdb.connect(str(raw_path), read_only=True)) as conn:
        raw_columns_list = [str(row[1]) for row in conn.execute("PRAGMA table_info('raw_factor')").fetchall()]
        trade_dates = [str(row[0]) for row in conn.execute("SELECT DISTINCT CAST(trade_date AS VARCHAR) FROM raw_factor ORDER BY 1").fetchall()]
    maturity_date = _compute_maturity_date(trade_dates)
    labels = available_labels(raw_columns_list, [*DEFAULT_EXISTING_LABELS, *DEFAULT_COMPUTED_LABELS])
    if len(labels) != len(DEFAULT_EXISTING_LABELS) + len(DEFAULT_COMPUTED_LABELS):
        missing = sorted(set([*DEFAULT_EXISTING_LABELS, *DEFAULT_COMPUTED_LABELS]) - set(labels))
        raise RuntimeError(f"full label schema unavailable: {missing}")
    raw_columns = set(raw_columns_list)
    projections = ["stock_code", "CAST(trade_date AS VARCHAR) AS trade_date", *[_label_expr(label, raw_columns) for label in labels]]
    candidate_dir = workspace / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"l3_label_candidate_{args.target_trade_date}.duckdb"
    temp_path = candidate_path.with_suffix(".duckdb.tmp")
    temp_path.unlink(missing_ok=True)
    candidate_path.unlink(missing_ok=True)
    table = _active_registry_asset("L3_labels")[2]
    with closing(duckdb.connect(str(temp_path))) as conn:
        _configure_worker_duckdb(conn, STREAMING_DUCKDB_MEMORY_LIMIT, workspace / STREAMING_TEMP_DIR_NAME / "label")
        conn.execute(f"ATTACH {_quote_literal(str(raw_path))} AS rawsrc (READ_ONLY)")
        conn.execute(
            f"""
            CREATE TABLE {_quote_ident(table)} AS
            SELECT {', '.join(projections)}
            FROM rawsrc.raw_factor
            WHERE CAST(trade_date AS VARCHAR) <= {_quote_literal(maturity_date)}
              AND stock_code NOT LIKE '%.BJ'
            """
        )
        conn.execute(f"CREATE INDEX idx_label_key ON {_quote_ident(table)}(stock_code, trade_date)")
        conn.execute("DETACH rawsrc")
    os.replace(temp_path, candidate_path)
    metrics = _candidate_table_metrics(candidate_path, table, args.target_trade_date)
    columns = metrics.pop("columns")
    dependency_violations = 0
    with closing(duckdb.connect(str(candidate_path), read_only=True)) as conn:
        conn.execute(f"ATTACH {_quote_literal(str(raw_path))} AS rawsrc (READ_ONLY)")
        for label in labels:
            deps = LABEL_DEPENDENCIES.get(label, ())
            missing_condition = " OR ".join(f"r.{_quote_ident(dep)} IS NULL" for dep in deps)
            if not missing_condition:
                continue
            count = int(conn.execute(
                f"""
                SELECT COUNT(*) FROM {_quote_ident(table)} l
                JOIN rawsrc.raw_factor r
                  ON l.trade_date=CAST(r.trade_date AS VARCHAR) AND l.stock_code=r.stock_code
                WHERE l.{_quote_ident(label)} IS NOT NULL AND ({missing_condition})
                """
            ).fetchone()[0])
            dependency_violations += count
    gates = {
        "label_max_date_equals_maturity": metrics["max_trade_date"] == maturity_date,
        "label_target_rows_zero": metrics["target_row_count"] == 0,
        "duplicate_zero": metrics["duplicate_key_groups"] == 0,
        "no_bj": metrics["bj_row_count"] == 0,
        "label_dependency_violations_zero": dependency_violations == 0,
        "label_schema_complete": len(columns) == len(labels) + 2,
    }
    if not all(gates.values()):
        raise RuntimeError(f"label candidate gates failed: {[key for key,value in gates.items() if not value]}")
    result = {
        "path": str(candidate_path),
        "table": table,
        "sha256": _file_sha256(candidate_path),
        "metrics": metrics,
        "maturity_horizon_trade_dates": MAX_LABEL_HORIZON,
        "maturity_max_trade_date": maturity_date,
        "labels": labels,
        "dependency_violation_count": dependency_violations,
        "gates": gates,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    logger("label_candidate_done", **result)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _pair_change_plan(
    feature: dict[str, Any],
    label: dict[str, Any],
    active_before: dict[str, dict[str, Any]],
    registry_before: dict[str, Any],
    args: argparse.Namespace,
    report_dir: Path,
) -> dict[str, Any]:
    transaction_id = f"l3-pair-{args.target_trade_date}-{feature['sha256'][:12]}-{label['sha256'][:12]}"
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {str(item.get("asset_id")): item for item in registry.get("assets", [])}
    feature_before = by_id[EXPECTED_FEATURE_ASSET_ID]
    label_before = by_id[EXPECTED_LABEL_ASSET_ID]
    feature_final = f"quant/data_file/production_assets/duckdb/l3_feature_{args.target_trade_date}_{feature['sha256'][:12]}.duckdb::{feature['table']}"
    label_final = f"quant/data_file/production_assets/duckdb/l3_label_{args.target_trade_date}_{label['sha256'][:12]}.duckdb::{label['table']}"
    plan = {
        "schema_version": 1,
        "pair_transaction_id": transaction_id,
        "status": "prepared_not_applied",
        "prepared_at": _now(),
        "expected_registry_sha256": registry_before["sha256"],
        "candidate_sources": {"feature": feature["path"], "label": label["path"]},
        "candidate_sha256": {"feature": feature["sha256"], "label": label["sha256"]},
        "changes": [
            {
                "asset_before_id": EXPECTED_FEATURE_ASSET_ID,
                "asset_before_expected": {
                    "asset_path": active_before["feature"]["asset_path"],
                    "path": active_before["feature"]["path"],
                    "table": active_before["feature"]["table"],
                    "sha256": active_before["feature"]["file_state"]["sha256"],
                },
                "candidate": {"path": feature["path"], "table": feature["table"], "sha256": feature["sha256"]},
                "asset_after": {
                    **feature_before,
                    "asset_id": f"prod_l3_feature_full_{args.target_trade_date}_{feature['sha256'][:12]}",
                    "asset_path": feature_final,
                    "input_assets": [EXPECTED_L2_ASSET_ID],
                    "audit_report": str(report_dir / f"l3_full_processing_{args.target_trade_date}_audit.json"),
                    "audit_record": str(report_dir / f"l3_full_processing_{args.target_trade_date}_audit.md"),
                    "pair_transaction_id": transaction_id,
                    "created_at": _now(),
                },
            },
            {
                "asset_before_id": EXPECTED_LABEL_ASSET_ID,
                "asset_before_expected": {
                    "asset_path": active_before["label"]["asset_path"],
                    "path": active_before["label"]["path"],
                    "table": active_before["label"]["table"],
                    "sha256": active_before["label"]["file_state"]["sha256"],
                },
                "candidate": {"path": label["path"], "table": label["table"], "sha256": label["sha256"]},
                "asset_after": {
                    **label_before,
                    "asset_id": f"prod_l3_label_full_{args.target_trade_date}_{label['sha256'][:12]}",
                    "asset_path": label_final,
                    "input_assets": [EXPECTED_L2_ASSET_ID],
                    "audit_report": str(report_dir / f"l3_full_processing_{args.target_trade_date}_audit.json"),
                    "audit_record": str(report_dir / f"l3_full_processing_{args.target_trade_date}_audit.md"),
                    "pair_transaction_id": transaction_id,
                    "created_at": _now(),
                },
            },
        ],
    }
    return plan


def _active_unchanged(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return (
        before["asset_id"] == after["asset_id"]
        and before["asset_path"] == after["asset_path"]
        and before["file_state"] == after["file_state"]
        and before["metrics"] == after["metrics"]
        and before["schema_hash"] == after["schema_hash"]
    )


def _build_contract(report: dict[str, Any], args: argparse.Namespace, evidence_paths: list[str]) -> dict[str, Any]:
    if report.get("processing_mode") != "full_history_rebuild":
        raise RuntimeError("target-date-only delivery cannot produce a full_history_rebuild contract")
    feature = report["candidate_feature"]
    label = report["candidate_label"]
    l2_rows = int(report["precheck"]["l2"]["metrics"]["row_count"])
    if int(feature["metrics"]["row_count"]) != l2_rows:
        raise RuntimeError("full_history_rebuild contract requires candidate feature row_count to equal active L2")
    gates = []
    for section in (report["precheck"]["gates"], feature["gates"], label["gates"], report["active_unchanged_gates"]):
        gates.extend({"name": name, "passed": bool(value), "severity": "blocker"} for name, value in section.items())
    payload = build_layer_handoff_contract(
        workflow_run_id=args.workflow_run_id,
        layer="L3",
        target_trade_date=args.target_trade_date,
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=[f"{report['precheck']['l2']['path']}::{report['precheck']['l2']['table']}"],
        active_output_assets=[f"{feature['path']}::{feature['table']}", f"{label['path']}::{label['table']}"],
        gate_checks=gates,
        handoff_constraints=[
            "candidate-only stage1; active L3 and registry are unchanged",
            "stage2 requires independent audit and one atomic feature/label pair registry switch",
            "L4 remains blocked until stage2 audit and commander approval",
        ],
        evidence_paths=evidence_paths,
        residual_risk=[{"severity": "P2", "item": "candidate_not_active", "detail": "stage1 produces audited candidates only"}],
        boundaries={
            "processing_mode": "full_history_rebuild",
            "candidate_only": True,
            "active_switch_called": False,
            "pair_transaction_id": report["pair_change_plan"]["pair_transaction_id"],
            "no_training": True,
            "no_prediction": True,
            "no_signal": True,
            "no_backtest": True,
        },
        layer_payload=report,
        hard_rules=["no-BJ", "DuckDB-only", "one-table-one-file", "explicit-qfq", "fail-closed", "source-limited/null", "full-history-rebuild"],
    )
    errors = validate_layer_handoff_contract(payload, expected_layer="L3", expected_owner_agent="factor-agent")
    if errors:
        raise RuntimeError(f"workflow contract validation failed: {errors}")
    return payload


def _quarantine(workspace: Path, error: BaseException, report_dir: Path, target_date: str) -> dict[str, Any]:
    if isinstance(error, ShortLivedSchedulerError) and error.quarantine_path:
        payload = {
            "status": "failed_quarantined_by_short_lived_scheduler",
            "target_trade_date": target_date,
            "generated_at": _now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "quarantine_path": error.quarantine_path,
            "active_switch_called": False,
            "approved_for_candidate": False,
            "approved_for_active": False,
            "approved_for_downstream": False,
        }
        _write_json(report_dir / f"l3_full_processing_{target_date}_incident.json", payload)
        return payload
    quarantine_dir = workspace / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in (workspace / "candidates").glob("*.duckdb") if (workspace / "candidates").exists() else []:
        target = quarantine_dir / path.name
        os.replace(path, target)
        moved.append(str(target))
    payload = {
        "status": "failed_quarantined",
        "target_trade_date": target_date,
        "generated_at": _now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "quarantined_candidate_paths": moved,
        "active_switch_called": False,
    }
    _write_json(report_dir / f"l3_full_processing_{target_date}_incident.json", payload)
    return payload


def _allow_same_run_precheck_continuation(report_dir: Path, args: argparse.Namespace) -> bool:
    """Allow only the immutable precheck artifacts from this exact run to remain."""
    precheck_path = report_dir / f"l3_full_processing_{args.target_trade_date}_precheck.json"
    if not precheck_path.is_file():
        return False
    try:
        payload = json.loads(precheck_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "precheck_passed_active_unchanged":
        return False
    if payload.get("workflow_run_id") != args.workflow_run_id:
        return False
    if payload.get("target_trade_date") != args.target_trade_date:
        return False
    if payload.get("l2", {}).get("sha256") != args.expected_l2_sha256:
        return False
    if payload.get("l2", {}).get("expected_sha256") != args.expected_l2_sha256:
        return False
    if payload.get("feature_contract_version") not in (None, args.feature_contract_version):
        return False
    allowed_names = {
        precheck_path.name,
        f"l3_full_processing_{args.target_trade_date}_process_gate.json",
        f"l3_full_processing_{args.target_trade_date}_progress.jsonl",
        f"l3_full_processing_{args.target_trade_date}_outer_cleanup.json",
    }
    return all(path.name in allowed_names for path in report_dir.iterdir())


def _raise_startup_incident(report_dir: Path, args: argparse.Namespace, message: str) -> None:
    incident = {
        "status": "failed_closed_startup_gate",
        "target_trade_date": args.target_trade_date,
        "workflow_run_id": args.workflow_run_id,
        "generated_at": _now(),
        "error_type": "RuntimeError",
        "error": message,
        "stage": "startup_gate_before_orchestrator_try",
        "active_switch_called": False,
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
        "reuse_prohibited": True,
    }
    _write_json(report_dir / f"l3_full_processing_{args.target_trade_date}_incident.json", incident)
    raise RuntimeError(message)


def _markdown_report(payload: dict[str, Any]) -> str:
    feature = payload.get("candidate_feature") or {}
    label = payload.get("candidate_label") or {}
    return "\n".join(
        [
            f"# {payload['target_trade_date']} L3 全量加工候选交付报告",
            "",
            f"- 状态：`{payload['status']}`",
            f"- 加工模式：`{payload['processing_mode']}`",
            f"- Active 切换：未执行",
            f"- L2 SHA256：`{payload['precheck']['l2']['sha256']}`",
            f"- Feature candidate：`{feature.get('path')}`",
            f"- Feature candidate SHA256：`{feature.get('sha256')}`",
            f"- Feature 全表：`{(feature.get('metrics') or {}).get('row_count')}` 行，目标日 `{(feature.get('metrics') or {}).get('target_row_count')}` 行",
            f"- Label candidate：`{label.get('path')}`",
            f"- Label candidate SHA256：`{label.get('sha256')}`",
            f"- Label 成熟上限：`{label.get('maturity_max_trade_date')}`，目标日行数 `{(label.get('metrics') or {}).get('target_row_count')}`",
            f"- Pair transaction：`{(payload.get('pair_change_plan') or {}).get('pair_transaction_id')}`（仅预备，未应用）",
            f"- Active feature 未变：`{payload['active_unchanged_gates']['feature_active_unchanged']}`",
            f"- Active label 未变：`{payload['active_unchanged_gates']['label_active_unchanged']}`",
            f"- Registry 未变：`{payload['active_unchanged_gates']['registry_unchanged']}`",
            "",
            "本报告仅用于 stage1 候选审计。审计和指挥官未放行前，不得切换 active，也不得推进 L4-L8。",
        ]
    ) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild complete L3 feature and mature-label candidates from the active L2 DuckDB mainline.")
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--mode", choices=("precheck", "build"), default="precheck")
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--raw-buckets", type=int, default=1024)
    parser.add_argument("--gtja-date-chunk-days", type=int, default=5)
    parser.add_argument("--gtja-alpha-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--workspace-dir")
    parser.add_argument("--expected-l2-sha256")
    parser.add_argument("--expected-l2-schema-hash")
    parser.add_argument("--expected-l2-schema-json")
    parser.add_argument("--feature-contract-version", choices=("v1", "v2"), default="v2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_provenance = _runtime_provenance_gate()
    if args.raw_buckets <= 0 or args.gtja_date_chunk_days <= 0 or args.gtja_alpha_batch_size <= 0 or args.workers <= 0:
        raise ValueError("raw-buckets, gtja-date-chunk-days, gtja-alpha-batch-size, and workers must be positive")
    if args.workers != 1:
        raise ValueError("memory-bounded v2 candidate requires exactly one worker")
    if args.raw_buckets != 1024 or args.gtja_date_chunk_days != 5 or args.gtja_alpha_batch_size != 1:
        raise ValueError("memory-bounded v2 candidate requires raw_buckets=1024, gtja_date_chunk_days=5, alpha_batch_size=1")
    if args.feature_contract_version != "v2":
        raise ValueError("memory-bounded v2 candidate requires feature_contract_version=v2")
    report_dir = Path(args.report_dir).resolve()
    workspace = Path(args.workspace_dir).resolve() if args.workspace_dir else report_dir / "workspace"
    _validate_no_forbidden_runtime_paths([report_dir, workspace], args.target_trade_date)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "build" and workspace.exists() and any(workspace.rglob("*")):
        _raise_startup_incident(report_dir, args, f"existing non-empty workspace prohibits resume: {workspace}")
    if args.mode == "build" and report_dir.exists() and any(report_dir.rglob("*")):
        if not _allow_same_run_precheck_continuation(report_dir, args):
            _raise_startup_incident(report_dir, args, f"existing non-empty report directory prohibits reuse: {report_dir}")
    workspace.mkdir(parents=True, exist_ok=True)
    progress_path = report_dir / f"l3_full_processing_{args.target_trade_date}_progress.jsonl"
    psutil = require_psutil()
    args.run_nonce = uuid.uuid4().hex
    orchestrator_pid = os.getpid()
    orchestrator_create_time_ns = int(float(psutil.Process(orchestrator_pid).create_time()) * 1_000_000_000)
    orchestrator_started_create_time_ns = int(time.time() * 1_000_000_000)
    outer_cleanup_path = report_dir / f"l3_full_processing_{args.target_trade_date}_outer_cleanup.json"
    outer_cleanup: dict[str, Any] | None = None

    def cleanup_outer_descendants(reason: str) -> dict[str, Any]:
        nonlocal outer_cleanup
        outer_cleanup = cleanup_descendants_for_run(
            run_nonce=args.run_nonce,
            parent_pid=orchestrator_pid,
            parent_create_time_ns=orchestrator_create_time_ns,
            started_create_time_ns=orchestrator_started_create_time_ns,
            evidence_path=outer_cleanup_path,
            reason=reason,
        )
        return outer_cleanup

    def logger(stage: str, **details: Any) -> None:
        payload = {"time": _now(), "stage": stage, **details}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)

    try:
        logger("precheck_start", mode=args.mode)
        precheck = _precheck(args)
        active_feature_before = _active_snapshot("L3_features", args.target_trade_date)
        active_label_before = _active_snapshot("L3_labels", args.target_trade_date)
        registry_before = _file_state(REGISTRY_PATH)
        precheck_path = report_dir / f"l3_full_processing_{args.target_trade_date}_precheck.json"
        if args.mode == "precheck":
            active_feature_after = _active_snapshot("L3_features", args.target_trade_date)
            active_label_after = _active_snapshot("L3_labels", args.target_trade_date)
            registry_after = _file_state(REGISTRY_PATH)
            unchanged = {
                "feature_active_unchanged": _active_unchanged(active_feature_before, active_feature_after),
                "label_active_unchanged": _active_unchanged(active_label_before, active_label_after),
                "registry_unchanged": registry_before == registry_after,
            }
            if not all(unchanged.values()):
                raise RuntimeError(f"precheck touched active state: {unchanged}")
            precheck_payload = {
                **precheck,
                "run_nonce": args.run_nonce,
                "active_before": {
                    "feature": active_feature_before,
                    "label": active_label_before,
                    "registry": registry_before,
                },
                "active_switch_called": False,
            }
            output = {**precheck_payload, "status": "precheck_passed_active_unchanged", "active_after": {"feature": active_feature_after, "label": active_label_after, "registry": registry_after}, "active_unchanged_gates": unchanged, "outer_cleanup": outer_cleanup}
            _write_json(precheck_path, output)
            logger("precheck_done", report_path=str(precheck_path), status="precheck_passed_active_unchanged")
            cleanup_outer_descendants("precheck_success")
            output["outer_cleanup"] = outer_cleanup
            _write_json(precheck_path, output)
            return 0

        precheck_payload = {
            **precheck,
            "run_nonce": args.run_nonce,
            "active_before": {
                "feature": active_feature_before,
                "label": active_label_before,
                "registry": registry_before,
            },
            "active_switch_called": False,
        }
        _write_json(precheck_path, precheck_payload)
        logger("precheck_done", report_path=str(precheck_path))

        raw_stage = _build_raw_stage(precheck, args, workspace, logger)
        gtja_stage = _build_gtja_stage(raw_stage, args, workspace, logger)
        feature = _build_feature_candidate(precheck, raw_stage, gtja_stage, args, workspace, logger)
        label = _build_label_candidate(precheck, raw_stage, args, workspace, logger)
        cleanup_outer_descendants("candidate_build_stages_completed")
        pair_plan = _pair_change_plan(
            feature,
            label,
            {"feature": active_feature_before, "label": active_label_before},
            registry_before,
            args,
            report_dir,
        )
        pair_path = report_dir / f"l3_pair_registry_change_{args.target_trade_date}_prepared.json"
        _write_json(pair_path, pair_plan)

        active_feature_after = _active_snapshot("L3_features", args.target_trade_date)
        active_label_after = _active_snapshot("L3_labels", args.target_trade_date)
        registry_after = _file_state(REGISTRY_PATH)
        unchanged = {
            "feature_active_unchanged": _active_unchanged(active_feature_before, active_feature_after),
            "label_active_unchanged": _active_unchanged(active_label_before, active_label_after),
            "registry_unchanged": registry_before == registry_after,
        }
        if not all(unchanged.values()):
            raise RuntimeError(f"stage1 touched active state: {unchanged}")

        report = {
            "schema_version": 1,
            "status": "candidate_ready_for_audit_review",
            "generated_at": _now(),
            "target_trade_date": args.target_trade_date,
            "workflow_run_id": args.workflow_run_id,
            "run_nonce": args.run_nonce,
            "processing_mode": "full_history_rebuild",
            "resource_contract": _streaming_resource_contract(_streaming_policy()),
            "runtime_provenance": runtime_provenance,
            "precheck": precheck,
            "raw_stage": raw_stage,
            "gtja_stage": gtja_stage,
            "candidate_feature": feature,
            "candidate_label": label,
            "pair_change_plan": pair_plan,
            "active_before": {"feature": active_feature_before, "label": active_label_before, "registry": registry_before},
            "active_after": {"feature": active_feature_after, "label": active_label_after, "registry": registry_after},
            "active_unchanged_gates": unchanged,
            "outer_cleanup": outer_cleanup,
            "allow_next_layer_continue": False,
            "active_switch_called": False,
        }
        report_json = report_dir / f"l3_full_processing_{args.target_trade_date}_candidate.json"
        report_md = report_dir / f"l3_full_processing_{args.target_trade_date}_candidate.md"
        evidence_paths = [str(precheck_path), str(report_json), str(report_md), str(progress_path), str(pair_path), feature["path"], label["path"]]
        contract = _build_contract(report, args, evidence_paths)
        contract_path = report_dir / f"l3_handoff_contract_{args.target_trade_date}_candidate.json"
        _write_json(report_json, report)
        report_md.write_text(_markdown_report(report), encoding="utf-8")
        _write_json(contract_path, contract)
        logger("candidate_report_done", report_json=str(report_json), contract_path=str(contract_path))
        return 0
    except Exception as error:
        try:
            cleanup_outer_descendants(f"orchestrator_failure:{type(error).__name__}")
        except Exception as cleanup_error:
            incident = {
                "status": "failed_closed_cleanup_incomplete",
                "target_trade_date": args.target_trade_date,
                "workflow_run_id": args.workflow_run_id,
                "run_nonce": args.run_nonce,
                "generated_at": _now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "cleanup_error_type": type(cleanup_error).__name__,
                "cleanup_error": str(cleanup_error),
                "outer_cleanup_path": str(outer_cleanup_path),
                "workspace_quarantine_completed": False,
                "active_switch_called": False,
                "approved_for_candidate": False,
                "approved_for_active": False,
                "approved_for_downstream": False,
                "reuse_prohibited": True,
            }
            _write_json(report_dir / f"l3_full_processing_{args.target_trade_date}_incident.json", incident)
            print(json.dumps(incident, ensure_ascii=False), file=sys.stderr, flush=True)
            return 1
        incident = _quarantine(workspace, error, report_dir, args.target_trade_date)
        incident["run_nonce"] = args.run_nonce
        incident["outer_cleanup"] = outer_cleanup
        _write_json(report_dir / f"l3_full_processing_{args.target_trade_date}_incident.json", incident)
        process_gate_path = report_dir / f"l3_full_processing_{args.target_trade_date}_process_gate.json"
        if process_gate_path.is_file():
            process_gate = json.loads(process_gate_path.read_text(encoding="utf-8"))
            incident.update({
                "process_gate_path": str(process_gate_path),
                "process_gate_status": process_gate.get("status"),
                "blocking_writers": process_gate.get("blocking_writers") or [],
                "unknown_relevant_processes": process_gate.get("unknown_relevant_processes") or [],
            })
            _write_json(report_dir / f"l3_full_processing_{args.target_trade_date}_incident.json", incident)
        print(json.dumps(incident, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
