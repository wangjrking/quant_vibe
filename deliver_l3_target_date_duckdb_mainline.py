# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import gc
import json
import multiprocessing
import os
import shutil
import stat
import sys
import time
import traceback
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd

from adjustment_semantics import validate_strategy_output_field_names
from build_production_factor_parts import (
    GTJA_TO_PRODUCTION_COLUMN_MAP,
    KEY_COLUMNS,
    apply_industry_encode,
    default_industry_encode_mapping_path,
    is_future_or_label_column,
    load_industry_encode_mapping,
    normalize_industry_values,
    production_raw_columns,
)
from data_process_module import group_factor_eng
from feature_contract_v2_candidate import (
    SEMANTIC_FUTURE_DERIVED_COLUMNS,
    candidate_production_raw_columns,
)
from gtja_alpha_workflow import GTJA_ALPHA_COLUMNS, compute_gtja_alpha_from_raw_factor, required_gtja_raw_columns
from l3_active_writer_lease import (
    ACTIVE_L3_WRITER_LEASE_PATH,
    acquire_active_l3_writer_lease,
    release_active_l3_writer_lease,
    validate_active_l3_writer_lease,
)
from rebuild_factor_data_batched import _normalize_types
from rebuild_l3_full_duckdb_mainline import (
    DATA_DIR,
    EXPECTED_L2_ASSET_ID,
    EXPECTED_L2_PATH,
    EXPECTED_L2_TABLE,
    QFQ_TECHNICAL_EXPECTED_COUNT,
    REGISTRY_PATH,
    _active_registry_asset,
    _feature_schema_gate_summary,
    _file_sha256,
    _file_state,
    _key_metrics,
    _scan_l3_processes,
    _schema,
    _schema_hash,
)
from workflow_contract import build_layer_handoff_contract, validate_layer_handoff_contract


PROCESSING_MODE = "target_date_incremental_delivery"
FEATURE_CONTRACT_ACTIVE = "active"
FEATURE_CONTRACT_V2 = "v2"
FEATURE_CONTRACT_CHOICES = (FEATURE_CONTRACT_ACTIVE, FEATURE_CONTRACT_V2)
FEATURE_CONTRACT_COLUMN_COUNTS = {
    FEATURE_CONTRACT_ACTIVE: 839,
    FEATURE_CONTRACT_V2: 838,
}
MIN_FREE_BYTES = 250 * 1024**3
LEGACY_MIN_AVAILABLE_BYTES = 40 * 1024**3
TARGET_DATE_V2_MIN_AVAILABLE_BYTES = 32 * 1024**3
FORBIDDEN_INPUT_TOKENS = (
    "quarantine",
    "candidate_attempt",
    "memory_probe",
    "20260716",
    "production_factor_parts",
    "prediction_label_parts",
    "stock_factor_data.parquet",
    "odb.db",
    "stock_daily_data.db",
)
RAW_DERIVED_COMPATIBILITY_ALLOWLIST = {
    "cci": {
        "producer": "data_process_module.group_factor_eng",
        "source_location": "quant/main/data_process_module.py",
        "source_columns": ["high", "low", "close"],
        "source_semantics": "raw_market_price",
        "formula": "cci(high, low, close, length=14)",
        "reason": "active 839-column schema retains raw-price CCI alongside cci_qfq",
    }
}
FEATURE_CONTRACT_V2_FORBIDDEN_COLUMNS = tuple(sorted(SEMANTIC_FUTURE_DERIVED_COLUMNS))
VERIFIED_COPY_CHUNK_BYTES = 16 * 1024 * 1024
ACTIVE_FEATURE_LEASE_DIR = ACTIVE_L3_WRITER_LEASE_PATH.parent
ACTIVE_STABILITY_SAMPLES = 2
ACTIVE_STABILITY_SAMPLE_DELAY_SECONDS = 1.0
CODE_PROVENANCE_SOURCES = (
    Path(__file__).resolve(),
    (Path(__file__).resolve().parent / "l3_active_writer_lease.py").resolve(),
    (Path(__file__).resolve().parent / "l3_duckdb_sync.py").resolve(),
    (Path(__file__).resolve().parent / "l3_process_gate.py").resolve(),
    (Path(__file__).resolve().parent / "scan_l3_write_bypass.py").resolve(),
    (Path(__file__).resolve().parent / "rebuild_l3_full_duckdb_mainline.py").resolve(),
    (Path(__file__).resolve().parent / "refresh_l3_active_duckdb_full_delivery.py").resolve(),
    (Path(__file__).resolve().parent / "tools" / "apply_production_asset_pair_change.py").resolve(),
    (Path(__file__).resolve().parent / "tools" / "apply_production_asset_change.py").resolve(),
    (Path(__file__).resolve().parent / "tests" / "test_deliver_l3_target_date_duckdb_mainline.py").resolve(),
    (Path(__file__).resolve().parent / "tests" / "test_l3_active_writer_lease.py").resolve(),
)


def _memory_resource_contract(feature_contract_version: str) -> dict[str, Any]:
    if feature_contract_version == FEATURE_CONTRACT_V2:
        return {
            "profile_id": "target_date_incremental_v2_single_worker_32gib",
            "min_available_bytes": TARGET_DATE_V2_MIN_AVAILABLE_BYTES,
            "gate_name": "available_memory_at_least_32gib_target_date_v2",
            "duckdb_worker_memory_limit": "16GB",
            "workers": "1_or_2_cli_limited",
            "scope": "target_date_incremental_only",
            "full_history_rebuild": False,
            "rationale": "target-date v2 delivery snapshots active L3 and computes only the requested trade date; it is not the full-history candidate rebuild profile",
        }
    return {
        "profile_id": "legacy_target_date_incremental_40gib",
        "min_available_bytes": LEGACY_MIN_AVAILABLE_BYTES,
        "gate_name": "available_memory_at_least_40gib",
        "duckdb_worker_memory_limit": "16GB",
        "scope": "legacy_target_date_incremental",
        "full_history_rebuild": False,
    }


def _configure_multiprocessing_executable() -> str:
    executable = str(Path(sys.executable).resolve())
    if hasattr(sys, "_base_executable"):
        sys._base_executable = executable
    multiprocessing.set_executable(executable)
    return executable


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _append_progress(path: Path, stage: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": _now(), "stage": stage, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _workspace_entries(workspace: Path) -> list[str]:
    if not workspace.exists():
        return []
    return sorted(item.name for item in workspace.iterdir())


def _workspace_gate(workspace: Path, policy: str) -> tuple[dict[str, Any], dict[str, bool]]:
    entries = _workspace_entries(workspace)
    exists = workspace.exists()
    state = {
        "policy": policy,
        "path": str(workspace),
        "exists": exists,
        "entries": entries,
        "entry_count": len(entries),
    }
    if policy == "absent":
        return state, {"workspace_absent_before_execute": not exists}
    if policy == "authoritative_empty":
        return state, {
            "workspace_authoritative_root_exists": exists,
            "workspace_authoritative_root_empty": exists and not entries,
        }
    raise RuntimeError(f"unsupported workspace gate policy: {policy}")


def _assert_authoritative_execute_workspace(workspace: Path) -> dict[str, Any]:
    state, gates = _workspace_gate(workspace, "authoritative_empty")
    if not all(gates.values()):
        raise RuntimeError(
            "workspace is not the empty authoritative execute root: "
            f"path={workspace} exists={state['exists']} entries={state['entries']}"
        )
    return state


def _semantic_future_columns(columns: list[str]) -> list[str]:
    return sorted(set(columns) & set(FEATURE_CONTRACT_V2_FORBIDDEN_COLUMNS))


def _feature_schema_gate_for_contract(
    columns: list[str],
    l2_qfq_technical: list[str],
    feature_contract_version: str,
) -> dict[str, Any]:
    active_gate = _feature_schema_gate_summary(columns, l2_qfq_technical)
    if feature_contract_version == FEATURE_CONTRACT_ACTIVE:
        return active_gate
    if feature_contract_version != FEATURE_CONTRACT_V2:
        raise RuntimeError(f"unsupported feature contract version: {feature_contract_version}")

    active_future = list(active_gate.get("details", {}).get("future_columns", []))
    allowed_legacy = set(FEATURE_CONTRACT_V2_FORBIDDEN_COLUMNS)
    unexpected_future = sorted(set(active_future) - allowed_legacy)
    candidate_columns = [column for column in columns if column not in allowed_legacy]
    candidate_gate = _feature_schema_gate_summary(candidate_columns, l2_qfq_technical)
    return {
        **candidate_gate,
        "semantic_future_columns_known": not unexpected_future,
        "v2_candidate_schema_838": len(candidate_columns) == 838,
        "details": {
            **candidate_gate["details"],
            "active_feature_columns": list(columns),
            "active_feature_future_columns": active_future,
            "semantic_future_columns_to_remove": sorted(set(active_future) & allowed_legacy),
            "unexpected_future_columns": unexpected_future,
            "v2_candidate_columns": candidate_columns,
        },
    }


def _feature_schema_version_gate(
    feature_probe: dict[str, Any],
    feature_schema_gate: dict[str, Any],
    feature_contract_version: str,
) -> dict[str, Any]:
    """Bind the precheck count to the selected feature contract."""

    if feature_contract_version not in FEATURE_CONTRACT_COLUMN_COUNTS:
        raise RuntimeError(f"unsupported feature contract version: {feature_contract_version}")

    observed_count = int(feature_probe["column_count"])
    details = feature_schema_gate.get("details", {})
    if feature_contract_version == FEATURE_CONTRACT_ACTIVE:
        expected_count = FEATURE_CONTRACT_COLUMN_COUNTS[FEATURE_CONTRACT_ACTIVE]
        return {
            "contract_version": feature_contract_version,
            "expected_column_count": expected_count,
            "observed_active_column_count": observed_count,
            "candidate_column_count": observed_count,
            "passed": observed_count == expected_count,
            "legacy_v1_schema_rejected_when_not_839": observed_count != expected_count,
        }

    candidate_columns = list(details.get("v2_candidate_columns", []))
    expected_count = FEATURE_CONTRACT_COLUMN_COUNTS[FEATURE_CONTRACT_V2]
    return {
        "contract_version": feature_contract_version,
        "expected_column_count": expected_count,
        "observed_active_column_count": observed_count,
        "candidate_column_count": len(candidate_columns),
        "passed": bool(feature_schema_gate.get("v2_candidate_schema_838")),
        "index_2000_post10_close_absent": "index_2000_post10_close" not in candidate_columns,
        "future_lineage_zero": bool(feature_schema_gate.get("future_label_leakage_zero")),
        "legacy_v1_schema_rejected_when_not_839": False,
    }


def _feature_contract_columns(active_columns: list[str], feature_contract_version: str) -> list[str]:
    if feature_contract_version == FEATURE_CONTRACT_ACTIVE:
        return list(active_columns)
    if feature_contract_version == FEATURE_CONTRACT_V2:
        return [column for column in active_columns if column not in FEATURE_CONTRACT_V2_FORBIDDEN_COLUMNS]
    raise RuntimeError(f"unsupported feature contract version: {feature_contract_version}")


def _feature_contract_schema(schema: list[dict[str, Any]], feature_contract_version: str) -> list[dict[str, Any]]:
    if feature_contract_version == FEATURE_CONTRACT_ACTIVE:
        return list(schema)
    filtered = [
        {key: value for key, value in item.items() if key != "column_index"}
        for item in schema
        if item["name"] not in FEATURE_CONTRACT_V2_FORBIDDEN_COLUMNS
    ]
    return [{"column_index": index, **item} for index, item in enumerate(filtered)]


def _expected_feature_contract(precheck: dict[str, Any], feature_contract_version: str) -> dict[str, Any]:
    before = precheck["active_feature_before"]
    expected_schema = _feature_contract_schema(before["schema"], feature_contract_version)
    return {
        "version": feature_contract_version,
        "schema": expected_schema,
        "schema_hash": _schema_hash(expected_schema),
        "column_count": len(expected_schema),
        "excluded_columns": [
            item["name"]
            for item in before["schema"]
            if item["name"] in FEATURE_CONTRACT_V2_FORBIDDEN_COLUMNS
        ],
    }


def _apply_feature_contract_schema(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    feature_contract: dict[str, Any],
) -> dict[str, Any]:
    dropped: list[str] = []
    for column in feature_contract["excluded_columns"]:
        conn.execute(f"ALTER TABLE {_quote_ident(table)} DROP COLUMN {_quote_ident(column)}")
        dropped.append(column)
    return {"dropped_columns": dropped, "contract_version": feature_contract["version"]}


def _copy_file_verified(
    source: Path,
    destination: Path,
    expected_sha256: str,
    *,
    chunk_bytes: int = VERIFIED_COPY_CHUNK_BYTES,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise RuntimeError(f"verified copy destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f"{destination.name}.copying-{time.time_ns()}.tmp")
    if temp.exists():
        raise RuntimeError(f"verified copy temporary path already exists: {temp}")

    copy_started_at = _now()
    copy_started = time.perf_counter()
    source_before_stat = source.stat()
    source_before_hash = _file_sha256(source)
    source_before = {
        "path": str(source),
        "size_bytes": int(source_before_stat.st_size),
        "mtime_ns": int(source_before_stat.st_mtime_ns),
        "sha256": source_before_hash,
    }
    if source_before_hash != expected_sha256:
        raise RuntimeError(
            f"verified copy source hash drifted before copy: expected={expected_sha256}, actual={source_before_hash}"
        )

    copied_bytes = 0
    with source.open("rb") as reader, temp.open("xb") as writer:
        while True:
            block = reader.read(chunk_bytes)
            if not block:
                break
            writer.write(block)
            copied_bytes += len(block)
        writer.flush()
        os.fsync(writer.fileno())

    source_size = source_before_stat.st_size
    if copied_bytes != source_size or temp.stat().st_size != source_size:
        raise RuntimeError(
            "verified copy size mismatch: "
            f"source={source_size}, copied={copied_bytes}, destination={temp.stat().st_size}"
        )
    source_after_stat = source.stat()
    source_after_hash = _file_sha256(source)
    source_after = {
        "path": str(source),
        "size_bytes": int(source_after_stat.st_size),
        "mtime_ns": int(source_after_stat.st_mtime_ns),
        "sha256": source_after_hash,
    }
    if source_after_hash != expected_sha256:
        raise RuntimeError(
            f"verified copy source hash drifted during copy: expected={expected_sha256}, actual={source_after_hash}"
        )
    if not _same_file_state(source_before, source_after):
        raise RuntimeError(
            "verified copy source metadata drifted during copy: "
            f"before={source_before}, after={source_after}"
        )
    destination_hash = _file_sha256(temp)
    if destination_hash != expected_sha256:
        raise RuntimeError(
            f"verified copy destination hash mismatch: expected={expected_sha256}, actual={destination_hash}"
        )

    shutil.copystat(source, temp)
    os.replace(temp, destination)
    destination_stat = destination.stat()
    destination_state = {
        "path": str(destination),
        "size_bytes": int(destination_stat.st_size),
        "mtime_ns": int(destination_stat.st_mtime_ns),
        "sha256": destination_hash,
    }
    if not _same_file_state(source_after, destination_state):
        raise RuntimeError(
            "verified copy promoted destination metadata mismatch: "
            f"source={source_after}, destination={destination_state}"
        )
    return {
        "path": str(destination),
        "sha256": destination_hash,
        "size": source_size,
        "source_sha256_before": source_before_hash,
        "source_sha256_after": source_after_hash,
        "copied_bytes": copied_bytes,
        "copy_started_at": copy_started_at,
        "copy_completed_at": _now(),
        "copy_duration_seconds": round(time.perf_counter() - copy_started, 6),
        "source_before": source_before,
        "source_after": source_after,
        "destination": destination_state,
        "copy_protocol": "chunked_flush_fsync_source_before_after_hash_destination_hash_atomic_promote",
    }


def _same_file_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare immutable file identity fields while allowing a different path."""
    return all(
        left.get(key) == right.get(key)
        for key in ("size_bytes", "mtime_ns", "sha256")
    )


def _active_wal_paths(path: Path) -> list[Path]:
    resolved = path.resolve()
    return [
        Path(f"{resolved}.wal"),
        Path(f"{resolved}-wal"),
    ]


def _assert_no_duckdb_wal(path: Path, *, asset_role: str) -> None:
    found = [str(candidate) for candidate in _active_wal_paths(path) if candidate.exists()]
    if found:
        raise RuntimeError(f"{asset_role} WAL is present; fail closed: {found}")


def _assert_no_active_wal(path: Path) -> None:
    _assert_no_duckdb_wal(path, asset_role="active feature")


def _sample_stable_active_feature(
    source: Path,
    expected_state: dict[str, Any],
    *,
    samples: int = ACTIVE_STABILITY_SAMPLES,
    delay_seconds: float = ACTIVE_STABILITY_SAMPLE_DELAY_SECONDS,
) -> dict[str, Any]:
    if samples < 2:
        raise RuntimeError("active feature stability sampling requires at least two samples")
    observations: list[dict[str, Any]] = []
    for index in range(samples):
        _assert_no_active_wal(source)
        state = _file_state(source)
        observation = {"sample_index": index + 1, "sampled_at": _now(), **state}
        observations.append(observation)
        if not _same_file_state(state, expected_state):
            raise RuntimeError(
                "active feature state drifted before candidate snapshot: "
                f"expected={expected_state.get('sha256')} actual={state.get('sha256')}"
            )
        if index + 1 < samples and delay_seconds > 0:
            time.sleep(delay_seconds)
    if any(not _same_file_state(observations[0], item) for item in observations[1:]):
        raise RuntimeError("active feature changed during stability sampling")
    return {
        "status": "stable",
        "samples": observations,
        "sample_count": samples,
        "sample_delay_seconds": delay_seconds,
    }


def _active_feature_lease_path() -> Path:
    if ACTIVE_FEATURE_LEASE_DIR != ACTIVE_L3_WRITER_LEASE_PATH.parent:
        return ACTIVE_FEATURE_LEASE_DIR / ACTIVE_L3_WRITER_LEASE_PATH.name
    return ACTIVE_L3_WRITER_LEASE_PATH


def _acquire_active_feature_writer_lease(args: argparse.Namespace, feature_path: Path) -> dict[str, Any]:
    """Create the shared, fail-closed active L3 writer lease."""
    return acquire_active_l3_writer_lease(
        workflow_run_id=args.workflow_run_id,
        workspace=args.workspace_dir,
        report_dir=args.report_dir,
        process_role="target_date_feature_delivery",
        lease_path=_active_feature_lease_path(),
    )


def _release_active_feature_writer_lease(lease: dict[str, Any]) -> dict[str, Any]:
    return release_active_l3_writer_lease(lease)


def _validate_active_feature_writer_lease(
    lease: dict[str, Any],
    args: argparse.Namespace,
    feature_path: Path,
) -> dict[str, Any]:
    return validate_active_l3_writer_lease(
        lease,
        workflow_run_id=args.workflow_run_id,
        workspace=args.workspace_dir,
        report_dir=args.report_dir,
        process_role="target_date_feature_delivery",
        protected_path=feature_path,
    )


def _set_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def _capture_code_provenance(report_dir: Path, run_id: str) -> dict[str, Any]:
    """Freeze all delivery/governance sources without relying on Git state."""
    snapshot_dir = report_dir.resolve() / "code_snapshot" / run_id
    if snapshot_dir.exists():
        raise RuntimeError(f"code provenance snapshot already exists; overwrite is prohibited: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    captured_at = _now()
    files: list[dict[str, Any]] = []
    diffs: list[str] = []
    project_root = Path(__file__).resolve().parents[2]
    for source in CODE_PROVENANCE_SOURCES:
        source = source.resolve()
        if not source.is_file():
            raise RuntimeError(f"code provenance source is missing: {source}")
        try:
            relative = source.relative_to(project_root)
        except ValueError:
            relative = Path(source.name)
        destination = snapshot_dir / relative
        source_before = _file_state(source)
        copy = _copy_file_verified(source, destination, source_before["sha256"])
        source_after = _file_state(source)
        snapshot_state = _file_state(destination)
        if not _same_file_state(source_before, source_after):
            raise RuntimeError(f"code provenance source drifted during capture: {source}")
        if not _same_file_state(source_after, snapshot_state):
            raise RuntimeError(f"code provenance snapshot mismatch: {source} -> {destination}")
        source_text = source.read_text(encoding="utf-8").splitlines(keepends=True)
        snapshot_text = destination.read_text(encoding="utf-8").splitlines(keepends=True)
        file_diff = list(
            difflib.unified_diff(
                source_text,
                snapshot_text,
                fromfile=str(source),
                tofile=str(destination),
            )
        )
        diffs.extend(file_diff)
        files.append({
            "relative_path": str(relative).replace("\\", "/"),
            "source_before": source_before,
            "source_after": source_after,
            "snapshot": snapshot_state,
            "copy": copy,
            "diff_line_count": len(file_diff),
        })
        _set_read_only(destination)
    diff_path = snapshot_dir / "source_snapshot.diff"
    diff_path.write_text("".join(diffs), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "immutable_code_snapshot_ready",
        "workflow_run_id": run_id,
        "captured_at": captured_at,
        "snapshot_dir": str(snapshot_dir),
        "file_count": len(files),
        "files": files,
        "diff_path": str(diff_path),
        "diff_line_count": len(diffs),
        "git_required": False,
        "overwrite_prohibited": True,
    }
    manifest_path = snapshot_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_hash = _file_sha256(manifest_path)
    seal_path = snapshot_dir / "manifest.sha256"
    seal_path.write_text(f"{manifest_hash}  manifest.json\n", encoding="ascii")
    for path in (diff_path, manifest_path, seal_path):
        _set_read_only(path)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "seal_path": str(seal_path),
        "snapshot_read_only": True,
    }


def _snapshot_active_feature_candidate_base(
    source: Path,
    destination: Path,
    expected_state: dict[str, Any],
) -> dict[str, Any]:
    """Freeze a verified candidate base before any long-running factor calculation."""
    _assert_no_duckdb_wal(destination, asset_role="candidate base")
    stability = _sample_stable_active_feature(source, expected_state)
    copy = _copy_file_verified(source, destination, str(expected_state["sha256"]))
    _assert_no_active_wal(source)
    _assert_no_duckdb_wal(destination, asset_role="candidate base")
    source_after = _file_state(source)
    candidate_state = _file_state(destination)
    if not _same_file_state(source_after, expected_state):
        raise RuntimeError(
            "active feature state drifted after candidate snapshot: "
            f"expected={expected_state.get('sha256')} actual={source_after.get('sha256')}"
        )
    if not _same_file_state(candidate_state, expected_state):
        raise RuntimeError(
            "candidate base state does not match locked active feature: "
            f"expected={expected_state.get('sha256')} actual={candidate_state.get('sha256')}"
        )
    return {
        "status": "verified_candidate_base_ready",
        "source_before": stability["samples"][0],
        "source_after": source_after,
        "candidate": candidate_state,
        "stability": stability,
        "copy": copy,
        "copy_duration_seconds": copy["copy_duration_seconds"],
        "protocol": "writer_lease_process_gate_wal_absent_stability_samples_verified_copy_before_long_compute",
    }


def _target_expected_metrics(l2_probe: dict[str, Any]) -> dict[str, int]:
    metrics = l2_probe["metrics"]
    return {
        "target_row_count": int(metrics["target_row_count"]),
        "target_stock_count": int(metrics["target_stock_count"]),
    }


def _assert_input_path(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    matches = [token for token in FORBIDDEN_INPUT_TOKENS if token in normalized]
    if matches:
        raise RuntimeError(f"forbidden target-date input path: {path}; tokens={matches}")


def _historical_key_metrics(conn: duckdb.DuckDBPyConnection, table: str, target_date: str) -> dict[str, Any]:
    row = conn.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT stock_code),
               BIT_XOR(HASH(CAST(trade_date AS VARCHAR), stock_code)),
               SUM(CAST(HASH(CAST(trade_date AS VARCHAR), stock_code) AS HUGEINT))
        FROM {_quote_ident(table)}
        WHERE CAST(trade_date AS VARCHAR) <> ?
        """,
        [target_date],
    ).fetchone()
    return {
        "row_count": int(row[0]),
        "stock_count": int(row[1]),
        "bit_xor": str(row[2]),
        "hash_sum": str(row[3]),
    }


def _table_names(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _asset_probe(path: Path, table: str, target_date: str) -> dict[str, Any]:
    with closing(duckdb.connect(str(path), read_only=True)) as conn:
        schema = _schema(conn, table)
        metrics = _key_metrics(conn, table, target_date)
        historical = _historical_key_metrics(conn, table, target_date)
        tables = _table_names(conn)
    return {
        "path": str(path),
        "table": table,
        "file_state": _file_state(path),
        "metrics": metrics,
        "historical_key_metrics": historical,
        "schema": schema,
        "schema_hash": _schema_hash(schema),
        "column_count": len(schema),
        "table_names": tables,
    }


def _precheck(args: argparse.Namespace, *, suffix: str = "", workspace_policy: str = "absent") -> dict[str, Any]:
    import psutil

    report_dir = Path(args.report_dir).resolve()
    workspace = Path(args.workspace_dir).resolve()
    process_gate_path = report_dir / f"l3_target_date_{args.target_trade_date}_process_gate{suffix}.json"
    process_scan = _scan_l3_processes(args)
    _write_json(process_gate_path, process_scan)
    workspace_state, workspace_gates = _workspace_gate(workspace, workspace_policy)

    l2_asset, l2_path, l2_table = _active_registry_asset("L2")
    feature_asset, feature_path, feature_table = _active_registry_asset("L3_features")
    label_asset, label_path, label_table = _active_registry_asset("L3_labels")
    for path in (l2_path, feature_path, label_path):
        _assert_input_path(path)

    actual_l2_hash = _file_sha256(l2_path)
    expected_hash = str(args.expected_l2_sha256).lower()
    if actual_l2_hash != expected_hash:
        raise RuntimeError(f"active L2 SHA256 drift: expected={expected_hash} actual={actual_l2_hash}")

    l2_probe = _asset_probe(l2_path, l2_table, args.target_trade_date)
    feature_probe = _asset_probe(feature_path, feature_table, args.target_trade_date)
    label_probe = _asset_probe(label_path, label_table, args.target_trade_date)
    registry_state = _file_state(REGISTRY_PATH)
    l2_columns = [item["name"] for item in l2_probe["schema"]]
    feature_columns = [item["name"] for item in feature_probe["schema"]]
    qfq_columns = [column for column in l2_columns if "_qfq" in column]
    qfq_technical = [column for column in qfq_columns if column not in {"open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"}]
    feature_schema_gate = _feature_schema_gate_for_contract(
        feature_columns,
        qfq_technical,
        args.feature_contract_version,
    )
    feature_schema_version_gate = _feature_schema_version_gate(
        feature_probe,
        feature_schema_gate,
        args.feature_contract_version,
    )
    available = int(psutil.virtual_memory().available)
    free_disk = int(shutil.disk_usage(DATA_DIR).free)
    target_expected = _target_expected_metrics(l2_probe)
    memory_contract = _memory_resource_contract(args.feature_contract_version)
    memory_gate_name = str(memory_contract["gate_name"])

    gates = {
        "process_gate_passed": bool(process_scan.get("passed")),
        "l2_asset_locked": l2_asset.get("asset_id") == EXPECTED_L2_ASSET_ID,
        "l2_route_locked": l2_path == EXPECTED_L2_PATH.resolve() and l2_table == EXPECTED_L2_TABLE,
        "l2_hash_locked": actual_l2_hash == expected_hash,
        "l2_target_rows_positive": target_expected["target_row_count"] > 0,
        "l2_target_stocks_positive": target_expected["target_stock_count"] > 0,
        "l2_target_rows_equal_stocks": target_expected["target_row_count"] == target_expected["target_stock_count"],
        "l2_no_bj": l2_probe["metrics"]["bj_row_count"] == 0 and l2_probe["metrics"]["target_bj_row_count"] == 0,
        "l2_duplicate_zero": l2_probe["metrics"]["duplicate_key_groups"] == 0,
        "l2_qfq_price_5": all(column in l2_columns for column in ["open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"]),
        "l2_qfq_technical_74": len(qfq_technical) == QFQ_TECHNICAL_EXPECTED_COUNT,
        "feature_one_table_one_file": feature_probe["table_names"] == [feature_table],
        "feature_column_count_contract": bool(feature_schema_version_gate["passed"]),
        "feature_v2_index_2000_post10_close_absent": bool(
            feature_schema_version_gate.get("index_2000_post10_close_absent", True)
        ),
        "feature_future_lineage_zero": bool(feature_schema_version_gate.get("future_lineage_zero", True)),
        "feature_schema_contract": all(value for key, value in feature_schema_gate.items() if key != "details"),
        "feature_duplicate_zero": feature_probe["metrics"]["duplicate_key_groups"] == 0,
        "feature_no_bj": feature_probe["metrics"]["bj_row_count"] == 0,
        "label_one_table_one_file": label_probe["table_names"] == [label_table],
        "label_target_not_mature": label_probe["metrics"]["target_row_count"] == 0,
        "label_max_before_target": label_probe["metrics"]["max_trade_date"] < args.target_trade_date,
        "free_disk_at_least_250gib": free_disk >= MIN_FREE_BYTES,
    }
    gates.update(workspace_gates)
    gates[memory_gate_name] = available >= int(memory_contract["min_available_bytes"])
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        failed_report = {
            "schema_version": 1,
            "status": "failed_closed_precheck",
            "generated_at": _now(),
            "workflow_run_id": args.workflow_run_id,
            "target_trade_date": args.target_trade_date,
            "processing_mode": PROCESSING_MODE,
            "feature_contract_version": args.feature_contract_version,
            "process_gate_path": str(process_gate_path),
            "process_scan": process_scan,
            "workspace": str(workspace),
            "workspace_gate": workspace_state,
            "report_dir": str(report_dir),
            "target_expected_metrics": target_expected,
            "l2": {"asset": l2_asset, **l2_probe, "expected_sha256": expected_hash, "qfq_technical_columns": qfq_technical},
            "active_feature_before": {"asset": feature_asset, **feature_probe},
            "active_label_before": {"asset": label_asset, **label_probe},
            "registry_before": registry_state,
            "feature_schema_gate": feature_schema_gate,
            "feature_schema_version_gate": feature_schema_version_gate,
            "resources": {
                "available_memory_bytes": available,
                "free_disk_bytes": free_disk,
                "memory_contract": memory_contract,
            },
            "gates": gates,
            "failed_gates": failed,
            "active_switch_called": False,
            "label_write_called": False,
            "ready_for_audit_review": False,
            "allow_next_layer_continue": False,
        }
        failure_path = report_dir / f"l3_target_date_{args.target_trade_date}_precheck_failed{suffix}.json"
        _write_json(failure_path, failed_report)
        raise RuntimeError(f"target-date L3 precheck failed: {failed}")
    return {
        "schema_version": 1,
        "status": "precheck_passed",
        "generated_at": _now(),
        "workflow_run_id": args.workflow_run_id,
        "target_trade_date": args.target_trade_date,
        "processing_mode": PROCESSING_MODE,
        "feature_contract_version": args.feature_contract_version,
        "process_gate_path": str(process_gate_path),
        "process_scan": process_scan,
        "workspace": str(workspace),
        "workspace_gate": workspace_state,
        "report_dir": str(report_dir),
        "target_expected_metrics": target_expected,
        "l2": {"asset": l2_asset, **l2_probe, "expected_sha256": expected_hash, "qfq_technical_columns": qfq_technical},
        "active_feature_before": {"asset": feature_asset, **feature_probe},
        "active_label_before": {"asset": label_asset, **label_probe},
        "registry_before": registry_state,
        "feature_schema_gate": feature_schema_gate,
        "feature_schema_version_gate": feature_schema_version_gate,
        "resources": {"available_memory_bytes": available, "free_disk_bytes": free_disk, "memory_contract": memory_contract},
        "gates": gates,
        "active_switch_called": False,
        "label_write_called": False,
    }


def _split(items: list[str], bucket_count: int) -> list[list[str]]:
    bucket_count = max(1, min(int(bucket_count), len(items)))
    size = (len(items) + bucket_count - 1) // bucket_count
    return [items[index : index + size] for index in range(0, len(items), size)]


def _resolve_lookback_start(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    target_date: str,
    lookback_trading_days: int,
) -> str:
    """Resolve the smallest bounded trading-date window for target-date factors."""
    if lookback_trading_days < 1:
        raise ValueError("lookback_trading_days must be positive")
    dates = [
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT DISTINCT CAST(trade_date AS VARCHAR)
            FROM {_quote_ident(table)}
            WHERE CAST(trade_date AS VARCHAR) <= ?
            ORDER BY 1 DESC
            LIMIT ?
            """,
            [target_date, int(lookback_trading_days)],
        ).fetchall()
    ]
    if not dates:
        raise RuntimeError(f"no L2 trading dates available through {target_date}")
    return dates[-1]


def _compute_bucket(task: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    started = time.perf_counter()
    codes = list(task["codes"])
    placeholders = ",".join("?" for _ in codes)
    with closing(duckdb.connect(task["l2_path"], read_only=True)) as conn:
        conn.execute("SET threads=1")
        conn.execute("SET memory_limit='16GB'")
        source = conn.execute(
            f"""
            SELECT * FROM {_quote_ident(task['l2_table'])}
            WHERE stock_code IN ({placeholders})
              AND CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
              AND stock_code NOT LIKE '%.BJ'
            ORDER BY stock_code, trade_date
            """,
            [*codes, task["read_start"], task["target_date"]],
        ).fetchdf()
    source.columns = source.columns.str.lower()
    gtja_needed = required_gtja_raw_columns()
    target_frames: list[pd.DataFrame] = []
    gtja_frames: list[pd.DataFrame] = []
    duplicate_columns: set[str] = set()
    for _, group in source.groupby("stock_code", sort=False):
        factor = group_factor_eng(group.sort_values("trade_date").copy(), include_future_labels=True)
        duplicates = factor.columns[factor.columns.duplicated()].tolist()
        duplicate_columns.update(str(item) for item in duplicates)
        factor = factor.loc[:, ~factor.columns.duplicated(keep="first")].copy()
        factor = _normalize_types(factor)
        factor["trade_date"] = factor["trade_date"].astype(str)
        missing_gtja = [column for column in gtja_needed if column not in factor.columns]
        if missing_gtja:
            raise RuntimeError(f"GTJA input columns missing in active L2-derived raw factor: {missing_gtja}")
        gtja_frames.append(factor[gtja_needed].copy())
        target = factor[factor["trade_date"].eq(task["target_date"])].copy()
        if not target.empty:
            target_frames.append(target)
        del factor
        gc.collect()
    raw_target = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    gtja_input = pd.concat(gtja_frames, ignore_index=True) if gtja_frames else pd.DataFrame()
    result = {
        "bucket_index": int(task["bucket_index"]),
        "code_count": len(codes),
        "source_rows": int(source.shape[0]),
        "raw_target_rows": int(raw_target.shape[0]),
        "gtja_rows": int(gtja_input.shape[0]),
        "duplicate_columns_removed_preserving_l2_source": sorted(duplicate_columns),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_target": raw_target,
        "gtja_input": gtja_input,
    }
    del source
    gc.collect()
    return result


def _compute_l2_only_frames(args: argparse.Namespace, precheck: dict[str, Any], progress: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    l2_path = Path(precheck["l2"]["path"])
    l2_table = precheck["l2"]["table"]
    with closing(duckdb.connect(str(l2_path), read_only=True)) as conn:
        codes = [str(row[0]) for row in conn.execute(
            f"SELECT stock_code FROM {_quote_ident(l2_table)} WHERE CAST(trade_date AS VARCHAR)=? AND stock_code NOT LIKE '%.BJ' ORDER BY stock_code",
            [args.target_trade_date],
        ).fetchall()]
        bounded_read_start = _resolve_lookback_start(
            conn,
            l2_table,
            args.target_trade_date,
            args.lookback_trading_days,
        )
    buckets = _split(codes, args.raw_buckets)
    tasks = [
        {
            "bucket_index": index,
            "codes": bucket,
            "l2_path": str(l2_path),
            "l2_table": l2_table,
            "read_start": bounded_read_start,
            "target_date": args.target_trade_date,
        }
        for index, bucket in enumerate(buckets)
    ]
    _append_progress(
        progress,
        "raw_bucket_plan",
        buckets=len(tasks),
        workers=args.workers,
        target_codes=len(codes),
        lookback_trading_days=args.lookback_trading_days,
        read_start=bounded_read_start,
        target_date=args.target_trade_date,
    )
    raw_frames: list[pd.DataFrame] = []
    gtja_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            result = _compute_bucket(task)
            raw_frames.append(result.pop("raw_target"))
            gtja_frames.append(result.pop("gtja_input"))
            metrics.append(result)
            _append_progress(progress, "raw_bucket_done", execution_mode="inline_single_worker", **result)
    else:
        context = multiprocessing.get_context("spawn")
        for wave_start in range(0, len(tasks), args.workers):
            wave = tasks[wave_start : wave_start + args.workers]
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(wave), mp_context=context) as executor:
                futures = [executor.submit(_compute_bucket, task) for task in wave]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    raw_frames.append(result.pop("raw_target"))
                    gtja_frames.append(result.pop("gtja_input"))
                    metrics.append(result)
                    _append_progress(progress, "raw_bucket_done", execution_mode="spawn_pool", **result)
    raw_target = _normalize_types(pd.concat(raw_frames, ignore_index=True))
    raw_target["trade_date"] = raw_target["trade_date"].astype(str)
    raw_target.drop_duplicates(KEY_COLUMNS, keep="last", inplace=True)
    gtja_input = _normalize_types(pd.concat(gtja_frames, ignore_index=True))
    gtja_input["trade_date"] = gtja_input["trade_date"].astype(str)
    gtja_input.drop_duplicates(KEY_COLUMNS, keep="last", inplace=True)
    _append_progress(
        progress,
        "raw_target_done",
        rows=int(raw_target.shape[0]),
        stocks=int(raw_target["stock_code"].nunique()),
        gtja_input_rows=int(gtja_input.shape[0]),
    )
    return raw_target, gtja_input, sorted(metrics, key=lambda item: item["bucket_index"])


def _compute_gtja_target(gtja_input: pd.DataFrame, target_date: str, progress: Path) -> pd.DataFrame:
    def callback(event: dict[str, Any]) -> None:
        _append_progress(progress, "gtja_alpha_batch_done", **event)

    _append_progress(progress, "gtja_compute_start", rows=int(gtja_input.shape[0]))
    gtja = compute_gtja_alpha_from_raw_factor(
        gtja_input,
        encode=False,
        drop_ts=True,
        cross_sectional_rank_mode="rank",
        output_dates=[target_date],
        alpha_batch_size=4,
        progress_callback=callback,
    )
    gtja["trade_date"] = gtja["trade_date"].astype(str)
    target = gtja[gtja["trade_date"].eq(target_date)].copy()
    keep = [*KEY_COLUMNS, *[column for column in GTJA_ALPHA_COLUMNS if column in target.columns]]
    target = target[keep]
    _append_progress(progress, "gtja_compute_done", rows=int(target.shape[0]), stocks=int(target["stock_code"].nunique()))
    return target


def _restore_active_raw_derived_columns(
    feature: pd.DataFrame,
    raw_target: pd.DataFrame,
    active_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    restored: dict[str, dict[str, Any]] = {}
    for column in active_columns:
        if column in feature.columns or column not in raw_target.columns:
            continue
        lineage = RAW_DERIVED_COMPATIBILITY_ALLOWLIST.get(column)
        if lineage is None:
            continue
        if is_future_or_label_column(column):
            raise RuntimeError(f"future/label column cannot be in raw-derived compatibility allowlist: {column}")
        validate_strategy_output_field_names([column], context="target-date active raw-derived compatibility")
        feature[column] = raw_target[column].to_numpy(copy=False)
        restored[column] = dict(lineage)
    return feature, restored


def _build_feature_target(
    raw_target: pd.DataFrame,
    gtja_target: pd.DataFrame,
    active_columns: list[str],
    *,
    feature_contract_version: str = FEATURE_CONTRACT_ACTIVE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feature_contract_version == FEATURE_CONTRACT_V2:
        raw_columns = candidate_production_raw_columns(list(raw_target.columns))
    else:
        raw_columns = production_raw_columns(list(raw_target.columns))
    target_columns = _feature_contract_columns(active_columns, feature_contract_version)
    feature = raw_target[raw_columns].copy()
    feature, restored_raw_lineage = _restore_active_raw_derived_columns(feature, raw_target, target_columns)
    gtja_qfq = gtja_target.rename(columns={column: GTJA_TO_PRODUCTION_COLUMN_MAP[column] for column in GTJA_ALPHA_COLUMNS if column in gtja_target.columns})
    feature = feature.merge(gtja_qfq, on=KEY_COLUMNS, how="left", validate="one_to_one")
    mapping_path = default_industry_encode_mapping_path()
    mapping = load_industry_encode_mapping(mapping_path)
    industries = normalize_industry_values(feature["industry"] if "industry" in feature.columns else [])
    unknown = sorted({value for value in industries if value not in mapping})
    if unknown:
        raise RuntimeError(f"industry mapping update requires separate governance approval: {unknown}")
    feature = apply_industry_encode(feature, mapping)
    feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing = [column for column in target_columns if column not in feature.columns]
    extra = [column for column in feature.columns if column not in target_columns]
    if missing:
        raise RuntimeError(f"target feature missing active schema columns: {missing}")
    feature = feature[target_columns].copy()
    feature.sort_values(KEY_COLUMNS, inplace=True)
    return feature, {
        "mapping_path": str(mapping_path),
        "unknown_industries": unknown,
        "restored_active_raw_derived_lineage": restored_raw_lineage,
        "ignored_extra_columns": extra,
        "feature_contract_version": feature_contract_version,
        "excluded_semantic_future_columns": _semantic_future_columns(active_columns),
    }


def _target_frame_gates(
    feature: pd.DataFrame,
    precheck: dict[str, Any],
    *,
    feature_contract_version: str = FEATURE_CONTRACT_ACTIVE,
) -> dict[str, Any]:
    columns = list(feature.columns)
    technical = precheck["l2"]["qfq_technical_columns"]
    schema_gate = _feature_schema_gate_summary(columns, technical)
    duplicate = int(feature.duplicated(KEY_COLUMNS).sum())
    bj_rows = int(feature["stock_code"].astype(str).str.endswith(".BJ").sum())
    future = [
        column
        for column in columns
        if is_future_or_label_column(column) or column in FEATURE_CONTRACT_V2_FORBIDDEN_COLUMNS
    ]
    expected = precheck["target_expected_metrics"]
    if "active_feature_before" in precheck and "schema" in precheck["active_feature_before"]:
        feature_contract = _expected_feature_contract(precheck, feature_contract_version)
    else:
        feature_contract = {
            "version": feature_contract_version,
            "schema": [],
            "schema_hash": None,
            "column_count": len(columns),
            "excluded_columns": [],
        }
    gates = {
        "rows_match_l2_target": int(feature.shape[0]) == int(expected["target_row_count"]),
        "stocks_match_l2_target": int(feature["stock_code"].nunique()) == int(expected["target_stock_count"]),
        "duplicate_zero": duplicate == 0,
        "bj_zero": bj_rows == 0,
        "columns_expected": len(columns) == int(feature_contract["column_count"]),
        "future_label_zero": not future,
        "qfq_schema": all(value for key, value in schema_gate.items() if key != "details"),
    }
    return {
        "gates": gates,
        "schema_gate": schema_gate,
        "duplicate_rows": duplicate,
        "bj_rows": bj_rows,
        "future_columns": future,
        "feature_contract": feature_contract,
    }


def _source_qfq_audit(candidate_path: Path, feature_table: str, precheck: dict[str, Any], target_date: str) -> dict[str, Any]:
    qfq_columns = ["open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq", *precheck["l2"]["qfq_technical_columns"]]
    l2_path = precheck["l2"]["path"]
    l2_table = precheck["l2"]["table"]
    with closing(duckdb.connect(str(candidate_path))) as conn:
        conn.execute(f"ATTACH {_quote_literal(l2_path)} AS l2 (READ_ONLY)")
        try:
            mismatch_expr = ",".join(
                f"SUM(CASE WHEN f.{_quote_ident(column)} IS DISTINCT FROM l.{_quote_ident(column)} THEN 1 ELSE 0 END) AS {_quote_ident(column)}"
                for column in qfq_columns
            )
            null_feature_expr = ",".join(
                f"SUM(CASE WHEN f.{_quote_ident(column)} IS NULL THEN 1 ELSE 0 END) AS {_quote_ident(column)}"
                for column in precheck["l2"]["qfq_technical_columns"]
            )
            null_l2_expr = ",".join(
                f"SUM(CASE WHEN l.{_quote_ident(column)} IS NULL THEN 1 ELSE 0 END) AS {_quote_ident(column)}"
                for column in precheck["l2"]["qfq_technical_columns"]
            )
            join_sql = (
                f" FROM {_quote_ident(feature_table)} f JOIN l2.{_quote_ident(l2_table)} l "
                "ON CAST(f.trade_date AS VARCHAR)=CAST(l.trade_date AS VARCHAR) AND f.stock_code=l.stock_code "
                "WHERE CAST(f.trade_date AS VARCHAR)=?"
            )
            mismatches = conn.execute("SELECT " + mismatch_expr + join_sql, [target_date]).fetchone()
            feature_nulls = conn.execute("SELECT " + null_feature_expr + join_sql, [target_date]).fetchone()
            l2_nulls = conn.execute("SELECT " + null_l2_expr + join_sql, [target_date]).fetchone()
            joined = int(conn.execute("SELECT COUNT(*)" + join_sql, [target_date]).fetchone()[0])
        finally:
            conn.execute("DETACH l2")
    mismatch_by_column = dict(zip(qfq_columns, [int(value or 0) for value in mismatches]))
    feature_null_by_column = dict(zip(precheck["l2"]["qfq_technical_columns"], [int(value or 0) for value in feature_nulls]))
    l2_null_by_column = dict(zip(precheck["l2"]["qfq_technical_columns"], [int(value or 0) for value in l2_nulls]))
    return {
        "joined_rows": joined,
        "mismatch_by_column": mismatch_by_column,
        "mismatch_total": sum(mismatch_by_column.values()),
        "feature_null_by_column": feature_null_by_column,
        "l2_null_by_column": l2_null_by_column,
        "feature_null_total": sum(feature_null_by_column.values()),
        "l2_null_total": sum(l2_null_by_column.values()),
        "null_pattern_preserved": feature_null_by_column == l2_null_by_column,
    }


def _validate_candidate(
    path: Path,
    table: str,
    target_date: str,
    precheck: dict[str, Any],
    *,
    feature_contract_version: str = FEATURE_CONTRACT_ACTIVE,
) -> dict[str, Any]:
    probe = _asset_probe(path, table, target_date)
    columns = [item["name"] for item in probe["schema"]]
    schema_gate = _feature_schema_gate_summary(columns, precheck["l2"]["qfq_technical_columns"])
    qfq_audit = _source_qfq_audit(path, table, precheck, target_date)
    before = precheck["active_feature_before"]
    expected = precheck["target_expected_metrics"]
    if "schema" in before:
        feature_contract = _expected_feature_contract(precheck, feature_contract_version)
    else:
        feature_contract = {
            "version": feature_contract_version,
            "schema": [],
            "schema_hash": before.get("schema_hash"),
            "column_count": probe["column_count"],
            "excluded_columns": [],
        }
    expected_rows = before["metrics"]["row_count"] - before["metrics"]["target_row_count"] + int(expected["target_row_count"])
    gates = {
        "one_table_one_file": probe["table_names"] == [table],
        "rows_expected": probe["metrics"]["row_count"] == expected_rows,
        "target_rows_match_l2_target": probe["metrics"]["target_row_count"] == int(expected["target_row_count"]) and probe["metrics"]["target_stock_count"] == int(expected["target_stock_count"]),
        "duplicate_zero": probe["metrics"]["duplicate_key_groups"] == 0,
        "bj_zero": probe["metrics"]["bj_row_count"] == 0 and probe["metrics"]["target_bj_row_count"] == 0,
        "columns_expected": probe["column_count"] == int(feature_contract["column_count"]),
        "schema_matches_feature_contract": probe["schema_hash"] == feature_contract["schema_hash"],
        "historical_keys_unchanged": probe["historical_key_metrics"] == before["historical_key_metrics"],
        "qfq_schema": all(value for key, value in schema_gate.items() if key != "details"),
        "qfq_source_values_preserved": qfq_audit["mismatch_total"] == 0,
        "source_limited_null_preserved": qfq_audit["null_pattern_preserved"],
    }
    return {
        **probe,
        "schema_gate": schema_gate,
        "qfq_source_audit": qfq_audit,
        "feature_contract": feature_contract,
        "gates": gates,
    }


def _pre_active_switch_gate(
    args: argparse.Namespace,
    precheck: dict[str, Any],
    candidate: Path,
    candidate_validation: dict[str, Any],
    lease: dict[str, Any],
    code_provenance: dict[str, Any],
    evidence_path: Path,
) -> dict[str, Any]:
    """Re-run writer, WAL, lease and immutable asset gates immediately before apply."""
    feature_path = Path(precheck["active_feature_before"]["path"])
    label_path = Path(precheck["active_label_before"]["path"])
    l2_path = Path(precheck["l2"]["path"])
    errors: list[str] = []
    process_scan: dict[str, Any]
    lease_validation: dict[str, Any] | None = None
    states: dict[str, Any] = {}
    wal_paths: dict[str, list[str]] = {}
    code_state_checks: list[dict[str, Any]] = []
    try:
        process_scan = _scan_l3_processes(args)
    except BaseException as error:
        process_scan = {
            "passed": False,
            "blocking_writers": [],
            "unknown_relevant_processes": [{
                "pid": os.getpid(),
                "decision": "block_unknown",
                "reason": f"process scan failed: {type(error).__name__}: {error}",
            }],
        }
        errors.append(f"process_scan_failed:{type(error).__name__}:{error}")
    if not process_scan.get("passed"):
        errors.append("process_or_unknown_writer_gate_failed")
    try:
        lease_validation = _validate_active_feature_writer_lease(lease, args, feature_path)
    except BaseException as error:
        errors.append(f"writer_lease_invalid:{type(error).__name__}:{error}")

    for role, path in (
        ("active_feature", feature_path),
        ("active_label", label_path),
        ("candidate_feature", candidate),
    ):
        wal_paths[role] = [str(item) for item in _active_wal_paths(path) if item.exists()]
        if wal_paths[role]:
            errors.append(f"{role}_wal_present:{wal_paths[role]}")

    expected_states = {
        "active_feature": precheck["active_feature_before"]["file_state"],
        "active_label": precheck["active_label_before"]["file_state"],
        "active_l2": precheck["l2"]["file_state"],
        "production_registry": precheck["registry_before"],
        "candidate_feature": candidate_validation["file_state"],
    }
    paths = {
        "active_feature": feature_path,
        "active_label": label_path,
        "active_l2": l2_path,
        "production_registry": REGISTRY_PATH,
        "candidate_feature": candidate,
    }
    for role, path in paths.items():
        try:
            actual = _file_state(path)
            expected = expected_states[role]
            states[role] = {"expected": expected, "actual": actual, "matches": _same_file_state(actual, expected)}
            if not states[role]["matches"]:
                errors.append(f"{role}_hash_size_mtime_drift")
        except BaseException as error:
            states[role] = {"error_type": type(error).__name__, "error": str(error), "matches": False}
            errors.append(f"{role}_state_probe_failed:{type(error).__name__}:{error}")

    for item in code_provenance.get("files") or []:
        source_expected = item["source_after"]
        source_path = Path(source_expected["path"])
        try:
            source_actual = _file_state(source_path)
            matches = _same_file_state(source_actual, source_expected)
            code_state_checks.append({
                "path": str(source_path),
                "expected": source_expected,
                "actual": source_actual,
                "matches": matches,
            })
            if not matches:
                errors.append(f"code_source_drift:{source_path}")
        except BaseException as error:
            code_state_checks.append({
                "path": str(source_path),
                "error_type": type(error).__name__,
                "error": str(error),
                "matches": False,
            })
            errors.append(f"code_source_probe_failed:{source_path}:{type(error).__name__}:{error}")

    evidence = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed_closed",
        "generated_at": _now(),
        "workflow_run_id": args.workflow_run_id,
        "target_trade_date": args.target_trade_date,
        "process_scan": process_scan,
        "lease_validation": lease_validation,
        "wal_paths": wal_paths,
        "asset_state_checks": states,
        "code_state_checks": code_state_checks,
        "errors": errors,
        "passed": not errors,
        "active_switch_called": False,
    }
    _write_json(evidence_path, evidence)
    if errors:
        raise RuntimeError(f"pre-active-switch gate failed: {errors}; evidence={evidence_path}")
    return evidence


def _atomic_feature_switch(
    candidate: Path,
    snapshot: Path,
    active_path: Path,
    expected_before_hash: str,
    *,
    pre_switch_gate: dict[str, Any],
    expected_lease_nonce: str,
    final_gate_recheck: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if not pre_switch_gate.get("passed") or pre_switch_gate.get("status") != "passed":
        raise RuntimeError("atomic feature switch requires a passed pre-switch gate")
    gate_nonce = ((pre_switch_gate.get("lease_validation") or {}).get("lease_nonce"))
    if not gate_nonce or gate_nonce != expected_lease_nonce:
        raise RuntimeError("atomic feature switch lease nonce does not match the validated writer lease")
    active_gate = ((pre_switch_gate.get("asset_state_checks") or {}).get("active_feature") or {})
    if not active_gate.get("matches") or (active_gate.get("actual") or {}).get("sha256") != expected_before_hash:
        raise RuntimeError("atomic feature switch active state was not locked by the pre-switch gate")
    final_gate = final_gate_recheck()
    if not final_gate.get("passed") or final_gate.get("status") != "passed":
        raise RuntimeError("atomic feature switch final interlock did not pass")
    final_nonce = ((final_gate.get("lease_validation") or {}).get("lease_nonce"))
    if final_nonce != expected_lease_nonce:
        raise RuntimeError("atomic feature switch final interlock lease nonce mismatch")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        raise RuntimeError(f"rollback snapshot already exists: {snapshot}")
    _copy_file_verified(active_path, snapshot, expected_before_hash)
    snapshot_state = _file_state(snapshot)
    if snapshot_state["sha256"] != expected_before_hash:
        raise RuntimeError("rollback snapshot hash mismatch")
    os.replace(candidate, active_path)
    return {
        "snapshot": snapshot_state,
        "active_switch_called": True,
        "switched_at": _now(),
        "final_atomic_switch_gate": final_gate,
    }


def _rollback(snapshot: Path, active_path: Path, report_dir: Path, error: BaseException) -> dict[str, Any]:
    rollback_copy = report_dir / f"rollback_restore_{time.time_ns()}.duckdb"
    snapshot_hash = _file_sha256(snapshot)
    _copy_file_verified(snapshot, rollback_copy, snapshot_hash)
    os.replace(rollback_copy, active_path)
    return {
        "status": "rolled_back",
        "error_type": type(error).__name__,
        "error": str(error),
        "restored_state": _file_state(active_path),
        "restored_at": _now(),
    }


def _build_contract(report: dict[str, Any], args: argparse.Namespace, path: Path) -> dict[str, Any]:
    active_switch_called = bool(report.get("active_switch_called"))
    feature_output = report["active_feature_after"] if active_switch_called else report["candidate_feature"]
    gate_checks = []
    for section in (report["precheck"]["gates"], report["target_frame_validation"]["gates"], report["candidate_validation"]["gates"], report["post_switch_gates"]):
        gate_checks.extend({"name": name, "passed": bool(value), "severity": "blocker"} for name, value in section.items())
    evidence_paths = [str(path), report["progress_path"], feature_output["path"]]
    if report["rollback_snapshot"].get("path"):
        evidence_paths.append(report["rollback_snapshot"]["path"])
    contract = build_layer_handoff_contract(
        workflow_run_id=args.workflow_run_id,
        layer="L3",
        target_trade_date=args.target_trade_date,
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=[f"{report['precheck']['l2']['path']}::{report['precheck']['l2']['table']}"],
        active_output_assets=[
            f"{feature_output['path']}::{feature_output['table']}",
            f"{report['active_label_after']['path']}::{report['active_label_after']['table']}",
        ],
        gate_checks=gate_checks,
        handoff_constraints=[
            (
                "target-date feature delivery completed the controlled atomic active switch; "
                "no full-history rebuild claim"
                if active_switch_called
                else "candidate-only target-date feature delivery; no full-history rebuild claim and no active switch"
            ),
            "active label remains read-only and is not advanced to the target date",
            "L4 remains blocked until independent audit approval",
        ],
        evidence_paths=evidence_paths,
        residual_risk=[{
            "severity": "P2",
            "item": "target_date_only_delivery",
            "detail": (
                f"history was not recomputed; only {args.target_trade_date} was inserted atomically into active feature"
                if active_switch_called
                else f"history was not recomputed; only {args.target_trade_date} was written to the candidate"
            ),
        }],
        boundaries={
            "processing_mode": PROCESSING_MODE,
            "full_history_rebuild": False,
            "target_date_only": True,
            "candidate_only": not active_switch_called,
            "active_switch_called": active_switch_called,
            "label_write_called": False,
            "no_training": True,
            "no_prediction": True,
            "no_signal": True,
            "no_backtest": True,
        },
        layer_payload=report,
        hard_rules=["no-BJ", "DuckDB-only", "one-table-one-file", "explicit-qfq", "source-limited/null", "label-not-forwarded", "fail-closed"],
    )
    errors = validate_layer_handoff_contract(contract, expected_layer="L3", expected_owner_agent="factor-agent")
    if errors:
        raise RuntimeError(f"workflow contract validation failed: {errors}")
    _write_json(path, contract)
    return contract


def _write_markdown_legacy(path: Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    feature = report["candidate_feature"]
    label = report["active_label_after"]
    qfq = report["candidate_validation"]["qfq_source_audit"]
    active_switch_called = bool(report.get("active_switch_called"))
    delivery_state = "已完成受控 active 原子切换" if active_switch_called else "仅生成候选，尚未切换 active"
    snapshot_path = report.get("rollback_snapshot", {}).get("path") or "candidate 阶段未创建"
    text = f"""# {args.target_trade_date} L3 目标日增量交付报告

## 结论

- 本轮仅替换 `{args.target_trade_date}` feature 截面，不是全历史重建；{delivery_state}。
- feature 目标日为 `{feature['metrics']['target_row_count']}` 行 / `{feature['metrics']['target_stock_count']}` 只股票，重复键 `{feature['metrics']['duplicate_key_groups']}`，`.BJ` 行数 `{feature['metrics']['target_bj_row_count']}`。
- production feature 共 `{feature['column_count']}` 列，5 个 qfq 价格、74 个 L2 qfq 技术字段和 191 个 GTJA qfq 因子门禁通过。
- L2 qfq 字段逐单元格差异 `{qfq['mismatch_total']}`，74 个技术字段空值总数 L2/feature 为 `{qfq['l2_null_total']}/{qfq['feature_null_total']}`。
- active label 保持到 `{label['metrics']['max_trade_date']}`，目标日行数 `{label['metrics']['target_row_count']}`，没有前推或伪造标签。
- raw-derived 兼容桥只允许 `cci`，来源为裸 `high/low/close` 原始市场价公式；裸价格、裸 qfq 技术别名、裸 GTJA、future/label 和未知缺列均不允许补入。

## 资产与恢复协议

- feature 交付对象：`{report['active_feature_after']['path']}::{report['active_feature_after']['table']}`。
- active label：`{label['path']}::{label['table']}`。
- 回滚快照：`{snapshot_path}`。
- registry 未修改；审计通过前 `allow_next_layer_continue=false`。

## 边界

本轮未读取 quarantine、probe、legacy parquet、SQLite、`odb.db` 或 shared DuckDB；未启动 full-history attempt-6；未训练、预测、生成信号、回测或交易。`ready_for_audit_review=true`，`allow_next_layer_continue=false`。
"""
    path.write_text(text, encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    feature = report["candidate_feature"]
    label = report["active_label_after"]
    qfq = report["candidate_validation"]["qfq_source_audit"]
    switched = bool(report.get("active_switch_called"))
    state = (
        "\u5df2\u5b8c\u6210\u53d7\u63a7 active \u539f\u5b50\u5207\u6362"
        if switched
        else "\u4ec5\u751f\u6210\u5019\u9009\uff0c\u5c1a\u672a\u5207\u6362 active"
    )
    snapshot = report.get("rollback_snapshot", {}).get("path") or "candidate \u9636\u6bb5\u672a\u521b\u5efa"
    lines = [
        f"# {args.target_trade_date} L3 \u76ee\u6807\u65e5\u589e\u91cf\u4ea4\u4ed8\u62a5\u544a",
        "",
        "## \u7ed3\u8bba",
        "",
        f"- \u672c\u8f6e\u4ec5\u66ff\u6362 `{args.target_trade_date}` feature \u622a\u9762\uff0c\u4e0d\u662f\u5168\u5386\u53f2\u91cd\u5efa\uff1b{state}\u3002",
        f"- feature \u76ee\u6807\u65e5\u4e3a `{feature['metrics']['target_row_count']}` \u884c / `{feature['metrics']['target_stock_count']}` \u53ea\u80a1\u7968\uff0c\u91cd\u590d\u952e `{feature['metrics']['duplicate_key_groups']}`\uff0c`.BJ` \u884c\u6570 `{feature['metrics']['target_bj_row_count']}`\u3002",
        f"- production feature \u5171 `{feature['column_count']}` \u5217\uff0c5 \u4e2a qfq \u4ef7\u683c\u300174 \u4e2a L2 qfq \u6280\u672f\u5b57\u6bb5\u548c 191 \u4e2a GTJA qfq \u56e0\u5b50\u95e8\u7981\u901a\u8fc7\u3002",
        f"- L2 qfq \u5b57\u6bb5\u9010\u5355\u5143\u683c\u5dee\u5f02 `{qfq['mismatch_total']}`\uff0c74 \u4e2a\u6280\u672f\u5b57\u6bb5\u7a7a\u503c\u603b\u6570 L2/feature \u4e3a `{qfq['l2_null_total']}/{qfq['feature_null_total']}`\u3002",
        f"- active label \u4fdd\u6301\u5230 `{label['metrics']['max_trade_date']}`\uff0c\u76ee\u6807\u65e5\u884c\u6570 `{label['metrics']['target_row_count']}`\uff0c\u6ca1\u6709\u524d\u63a8\u6216\u4f2a\u9020\u6807\u7b7e\u3002",
        "",
        "## \u8d44\u4ea7\u4e0e\u6062\u590d\u534f\u8bae",
        "",
        f"- feature \u4ea4\u4ed8\u5bf9\u8c61\uff1a`{report['active_feature_after']['path']}::{report['active_feature_after']['table']}`\u3002",
        f"- active label\uff1a`{label['path']}::{label['table']}`\u3002",
        f"- \u56de\u6eda\u5feb\u7167\uff1a`{snapshot}`\u3002",
        "- registry \u672a\u4fee\u6539\uff1b\u5ba1\u8ba1\u901a\u8fc7\u524d `allow_next_layer_continue=false`\u3002",
        "",
        "## \u8fb9\u754c",
        "",
        "\u672c\u8f6e\u672a\u8bfb\u53d6 quarantine\u3001probe\u3001legacy parquet\u3001SQLite\u3001`odb.db` \u6216 shared DuckDB\uff1b\u672a\u542f\u52a8 full-history attempt-6\uff1b\u672a\u8bad\u7ec3\u3001\u9884\u6d4b\u3001\u751f\u6210\u4fe1\u53f7\u3001\u56de\u6d4b\u6216\u4ea4\u6613\u3002`ready_for_audit_review=true`\uff0c`allow_next_layer_continue=false`\u3002",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace, precheck: dict[str, Any], *, apply_active: bool) -> dict[str, Any]:
    workspace = Path(args.workspace_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    _assert_authoritative_execute_workspace(workspace)
    candidate = workspace / f"l3_feature_candidate_{args.target_trade_date}.duckdb"
    snapshot = report_dir / "rollback" / f"l3_feature_before_{args.target_trade_date}_{precheck['active_feature_before']['file_state']['sha256'][:12]}.duckdb"
    feature_path = Path(precheck["active_feature_before"]["path"])
    feature_table = precheck["active_feature_before"]["table"]
    active_columns = [item["name"] for item in precheck["active_feature_before"]["schema"]]
    feature_contract = _expected_feature_contract(precheck, args.feature_contract_version)
    switched = False
    completed = False
    lease: dict[str, Any] | None = None
    lease_release: dict[str, Any] | None = None
    candidate_base_snapshot: dict[str, Any] | None = None
    code_provenance: dict[str, Any] | None = None
    pre_switch_gate: dict[str, Any] | None = None
    final_switch_gate: dict[str, Any] | None = None
    contract_apply: dict[str, Any] | None = None
    lease_release_path = report_dir / f"l3_target_date_{args.target_trade_date}_writer_lease_release.json"
    try:
        lease = _acquire_active_feature_writer_lease(args, feature_path)
        progress = report_dir / f"l3_target_date_{args.target_trade_date}_progress.jsonl"
        _append_progress(progress, "execute_start", processing_mode=PROCESSING_MODE)
        execution_process_scan = _scan_l3_processes(args)
        execution_process_gate_path = report_dir / f"l3_target_date_{args.target_trade_date}_process_gate_snapshot.json"
        _write_json(execution_process_gate_path, execution_process_scan)
        if not execution_process_scan.get("passed"):
            raise RuntimeError(
                "active feature writer/WAL/unknown-process gate failed before candidate snapshot: "
                f"blocking_writers={execution_process_scan.get('blocking_writers', [])} "
                f"unknown={execution_process_scan.get('unknown_relevant_processes', [])}"
            )
        candidate_base_snapshot = _snapshot_active_feature_candidate_base(
            feature_path,
            candidate,
            precheck["active_feature_before"]["file_state"],
        )
        _append_progress(
            progress,
            "candidate_base_snapshot_verified_before_long_compute",
            candidate_sha256=candidate_base_snapshot["candidate"]["sha256"],
            copy_duration_seconds=candidate_base_snapshot["copy_duration_seconds"],
            lease_path=lease["path"],
        )
        code_provenance = _capture_code_provenance(report_dir, args.workflow_run_id)
        _append_progress(
            progress,
            "code_provenance_snapshot_written",
            manifest_path=code_provenance["manifest_path"],
            manifest_sha256=code_provenance["manifest_sha256"],
            file_count=code_provenance["file_count"],
        )
        raw_target, gtja_input, bucket_metrics = _compute_l2_only_frames(args, precheck, progress)
        gtja_target = _compute_gtja_target(gtja_input, args.target_trade_date, progress)
        feature_target, build_meta = _build_feature_target(
            raw_target,
            gtja_target,
            active_columns,
            feature_contract_version=args.feature_contract_version,
        )
        target_validation = _target_frame_gates(
            feature_target,
            precheck,
            feature_contract_version=args.feature_contract_version,
        )
        if not all(target_validation["gates"].values()):
            raise RuntimeError(f"target feature gates failed: {target_validation['gates']}")
        _append_progress(progress, "target_feature_validated", rows=int(feature_target.shape[0]), columns=len(feature_target.columns))

        with closing(duckdb.connect(str(candidate))) as conn:
            contract_apply = _apply_feature_contract_schema(conn, feature_table, feature_contract)
            conn.register("target_feature_frame", feature_target)
            try:
                conn.execute("BEGIN TRANSACTION")
                conn.execute(f"DELETE FROM {_quote_ident(feature_table)} WHERE CAST(trade_date AS VARCHAR)=?", [args.target_trade_date])
                conn.execute(f"INSERT INTO {_quote_ident(feature_table)} BY NAME SELECT * FROM target_feature_frame")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.unregister("target_feature_frame")
        candidate_validation = _validate_candidate(
            candidate,
            feature_table,
            args.target_trade_date,
            precheck,
            feature_contract_version=args.feature_contract_version,
        )
        if not all(candidate_validation["gates"].values()):
            raise RuntimeError(f"candidate gates failed: {candidate_validation['gates']}")
        _append_progress(progress, "candidate_validated", sha256=candidate_validation["file_state"]["sha256"])

        if apply_active:
            apply_guard = report_dir / "rollback" / f"apply_guard_{time.time_ns()}.duckdb"
            pre_switch_gate_path = report_dir / f"l3_target_date_{args.target_trade_date}_pre_active_switch_gate.json"
            pre_switch_gate = _pre_active_switch_gate(
                args,
                precheck,
                candidate,
                candidate_validation,
                lease,
                code_provenance,
                pre_switch_gate_path,
            )
            _append_progress(
                progress,
                "pre_active_switch_gate_passed",
                evidence_path=str(pre_switch_gate_path),
                lease_nonce=pre_switch_gate["lease_validation"]["lease_nonce"],
            )
            final_switch_gate_path = report_dir / f"l3_target_date_{args.target_trade_date}_final_atomic_switch_gate.json"

            def final_gate_recheck() -> dict[str, Any]:
                return _pre_active_switch_gate(
                    args,
                    precheck,
                    candidate,
                    candidate_validation,
                    lease,
                    code_provenance,
                    final_switch_gate_path,
                )

            switch = _atomic_feature_switch(
                candidate,
                apply_guard,
                feature_path,
                precheck["active_feature_before"]["file_state"]["sha256"],
                pre_switch_gate=pre_switch_gate,
                expected_lease_nonce=str(lease["lease_nonce"]),
                final_gate_recheck=final_gate_recheck,
            )
            final_switch_gate = switch["final_atomic_switch_gate"]
            switched = True
            snapshot_state = switch["snapshot"]
        else:
            snapshot_state = {
                "path": None,
                "created": False,
                "active_source_state": precheck["active_feature_before"]["file_state"],
                "protocol": "create verified snapshot immediately before separately authorized atomic active replacement",
            }
            switch = {"snapshot": snapshot_state, "active_switch_called": False, "switched_at": None}
            pre_switch_gate_path = None
            final_switch_gate_path = None
            final_switch_gate = None
        active_after = _asset_probe(feature_path, feature_table, args.target_trade_date)
        label_path = Path(precheck["active_label_before"]["path"])
        label_after = _asset_probe(label_path, precheck["active_label_before"]["table"], args.target_trade_date)
        registry_after = _file_state(REGISTRY_PATH)
        l2_after_hash = _file_sha256(Path(precheck["l2"]["path"]))
        post_gates = {
            "label_hash_size_mtime_unchanged": label_after["file_state"] == precheck["active_label_before"]["file_state"],
            "label_not_forwarded": label_after["metrics"]["max_trade_date"] == precheck["active_label_before"]["metrics"]["max_trade_date"] and label_after["metrics"]["target_row_count"] == 0,
            "registry_unchanged": registry_after == precheck["registry_before"],
            "l2_hash_unchanged": l2_after_hash == precheck["l2"]["file_state"]["sha256"],
        }
        if apply_active:
            post_gates.update({
                "active_matches_validated_candidate": active_after["file_state"]["sha256"] == candidate_validation["file_state"]["sha256"],
                "active_target_matches_l2_target": (
                    active_after["metrics"]["target_row_count"] == int(precheck["target_expected_metrics"]["target_row_count"])
                    and active_after["metrics"]["target_stock_count"] == int(precheck["target_expected_metrics"]["target_stock_count"])
                ),
                "active_duplicate_zero": active_after["metrics"]["duplicate_key_groups"] == 0,
                "active_bj_zero": active_after["metrics"]["bj_row_count"] == 0,
                "active_historical_keys_unchanged": active_after["historical_key_metrics"] == precheck["active_feature_before"]["historical_key_metrics"],
            })
        else:
            post_gates.update({
                "active_feature_unchanged": active_after["file_state"] == precheck["active_feature_before"]["file_state"],
                "active_feature_metrics_unchanged": active_after["metrics"] == precheck["active_feature_before"]["metrics"],
            })
        if not all(post_gates.values()):
            raise RuntimeError(f"post-switch gates failed: {post_gates}")

        report = {
            "schema_version": 1,
            "status": "ready_for_audit_review" if not apply_active else "active_delivery_completed_waiting_for_audit",
            "generated_at": _now(),
            "workflow_run_id": args.workflow_run_id,
            "target_trade_date": args.target_trade_date,
            "processing_mode": PROCESSING_MODE,
            "precheck": precheck,
            "raw_bucket_metrics": bucket_metrics,
            "raw_target_rows": int(raw_target.shape[0]),
            "gtja_target_rows": int(gtja_target.shape[0]),
            "feature_contract": feature_contract,
            "feature_contract_apply": contract_apply,
            "target_feature_build": build_meta,
            "target_frame_validation": target_validation,
            "candidate_base_snapshot": candidate_base_snapshot,
            "feature_contract": feature_contract,
            "feature_contract_apply": contract_apply,
            "code_provenance": code_provenance,
            "execution_process_gate": execution_process_scan,
            "execution_process_gate_path": str(execution_process_gate_path),
            "pre_active_switch_gate": pre_switch_gate,
            "pre_active_switch_gate_path": str(pre_switch_gate_path) if pre_switch_gate_path else None,
            "final_atomic_switch_gate": final_switch_gate,
            "final_atomic_switch_gate_path": str(final_switch_gate_path) if final_switch_gate_path else None,
            "writer_lease": {
                **lease,
                "held_during_delivery": True,
                "release_in_finally": True,
                "release_evidence_path": str(lease_release_path),
            },
            "candidate_validation": candidate_validation,
            "rollback_snapshot": snapshot_state,
            "candidate_feature": {**candidate_validation, "approved_for_active": False, "approved_for_downstream": False},
            "active_feature_after": active_after,
            "active_label_after": label_after,
            "registry_after": registry_after,
            "post_switch_gates": post_gates,
            "progress_path": str(progress),
            "active_switch_called": switched,
            "label_write_called": False,
            "pair_change_called": False,
            "ready_for_audit_review": True,
            "allow_next_layer_continue": False,
        }
        report_path = report_dir / f"l3_target_date_candidate_{args.target_trade_date}.json"
        _write_json(report_path, report)
        contract_path = report_dir / f"l3_to_l4_handoff_contract_{args.target_trade_date}.json"
        _build_contract(report, args, contract_path)
        report["contract_path"] = str(contract_path)
        report["report_path"] = str(report_path)
        _write_json(report_path, report)
        _write_markdown(report_dir / f"l3_target_date_candidate_{args.target_trade_date}.md", report, args)
        _append_progress(progress, "report_written", report_path=str(report_path), contract_path=str(contract_path))
        completed = True
        return report
    except BaseException as error:
        rollback = None
        if switched and snapshot.exists():
            rollback = _rollback(snapshot, feature_path, report_dir, error)
        incident = {
            "schema_version": 1,
            "status": "failed_closed",
            "generated_at": _now(),
            "target_trade_date": args.target_trade_date,
            "processing_mode": PROCESSING_MODE,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "rollback": rollback,
            "writer_lease": lease,
            "candidate_base_snapshot": candidate_base_snapshot,
            "code_provenance": code_provenance,
            "pre_active_switch_gate": pre_switch_gate,
            "final_atomic_switch_gate": final_switch_gate,
            "active_switch_called": switched,
            "label_write_called": False,
            "ready_for_audit_review": False,
            "allow_next_layer_continue": False,
        }
        _write_json(report_dir / f"l3_target_date_delivery_{args.target_trade_date}_incident.json", incident)
        raise
    finally:
        if lease is not None:
            try:
                lease_release = _release_active_feature_writer_lease(lease)
                _write_json(lease_release_path, lease_release)
            except BaseException as release_error:
                _write_json(
                    lease_release_path,
                    {
                        "schema_version": 1,
                        "status": "release_failed",
                        "generated_at": _now(),
                        "error_type": type(release_error).__name__,
                        "error": str(release_error),
                        "lease": lease,
                    },
                )
                if completed:
                    raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DuckDB-only L3 target-date feature delivery.")
    parser.add_argument("--mode", choices=["precheck", "candidate", "execute"], required=True)
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--read-start", default="20250101")
    parser.add_argument("--lookback-trading-days", type=int, default=300)
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--expected-l2-sha256", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--raw-buckets", type=int, default=16)
    parser.add_argument(
        "--feature-contract-version",
        choices=FEATURE_CONTRACT_CHOICES,
        default=FEATURE_CONTRACT_ACTIVE,
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 2:
        parser.error("--workers must be 1 or 2")
    return args


def main(argv: list[str] | None = None) -> None:
    _configure_multiprocessing_executable()
    args = parse_args(argv)
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "precheck":
        precheck = _precheck(args)
        path = report_dir / f"l3_target_date_{args.target_trade_date}_precheck.json"
        _write_json(path, precheck)
        print(json.dumps({"status": precheck["status"], "report": str(path)}, ensure_ascii=False))
        return

    precheck_path = report_dir / f"l3_target_date_{args.target_trade_date}_precheck.json"
    if not precheck_path.is_file():
        raise RuntimeError(f"non-destructive precheck report missing: {precheck_path}")
    baseline = json.loads(precheck_path.read_text(encoding="utf-8"))
    workspace = Path(args.workspace_dir).resolve()
    startup_gate_path = report_dir / f"l3_target_date_{args.target_trade_date}_execute_startup_root.json"
    startup_state, startup_gates = _workspace_gate(workspace, "absent")
    startup_payload = {
        "schema_version": 1,
        "generated_at": _now(),
        "workflow_run_id": args.workflow_run_id,
        "target_trade_date": args.target_trade_date,
        "mode": args.mode,
        "workspace_gate": startup_state,
        "gates": startup_gates,
    }
    if not all(startup_gates.values()):
        startup_payload["status"] = "failed_closed_startup_root"
        _write_json(startup_gate_path, startup_payload)
        raise RuntimeError(
            "workspace startup root must be absent before authoritative execute create: "
            f"path={workspace} entries={startup_state['entries']}"
        )
    workspace.mkdir(parents=True)
    startup_state_after, startup_gates_after = _workspace_gate(workspace, "authoritative_empty")
    startup_payload.update(
        {
            "status": "authoritative_workspace_created_for_execute",
            "workspace_after_create": startup_state_after,
            "workspace_after_create_gates": startup_gates_after,
        }
    )
    _write_json(startup_gate_path, startup_payload)
    live = _precheck(args, suffix="_execute", workspace_policy="authoritative_empty")
    locked = {
        "l2": live["l2"]["file_state"] == baseline["l2"]["file_state"],
        "feature": live["active_feature_before"]["file_state"] == baseline["active_feature_before"]["file_state"],
        "label": live["active_label_before"]["file_state"] == baseline["active_label_before"]["file_state"],
        "registry": live["registry_before"] == baseline["registry_before"],
    }
    if not all(locked.values()):
        raise RuntimeError(f"precheck-to-execute asset drift: {locked}")
    report = execute(args, live, apply_active=args.mode == "execute")
    print(json.dumps({
        "status": report["status"],
        "target_rows": report["candidate_feature"]["metrics"]["target_row_count"],
        "target_stocks": report["candidate_feature"]["metrics"]["target_stock_count"],
        "label_max_trade_date": report["active_label_after"]["metrics"]["max_trade_date"],
        "report": report["report_path"],
        "contract": report["contract_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
