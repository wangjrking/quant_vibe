from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from build_prediction_label_parts import (  # noqa: E402
    add_executable_labels,
    available_labels,
    required_raw_columns,
    sanitize_existing_labels,
)
from build_production_factor_parts import is_future_or_label_column  # noqa: E402
from deliver_l3_target_date_duckdb_mainline import (  # noqa: E402
    FEATURE_CONTRACT_ACTIVE,
    _active_registry_asset,
    _asset_probe,
    _build_feature_target,
    _compute_gtja_target,
    _compute_l2_only_frames,
    _key_metrics,
    _schema,
    _schema_hash,
    _target_expected_metrics,
    _target_frame_gates,
)
from feature_contract_v2_candidate import SEMANTIC_FUTURE_DERIVED_COLUMNS  # noqa: E402
from validation_calendar_contract import audit_asset_calendars  # noqa: E402
from workflow_contract import build_layer_handoff_contract, validate_layer_handoff_contract  # noqa: E402


RUN_ID = "incremental-trading-signal-20260717-L3-compact-immutable-slice-20260815-r2"
TARGET_DATE = "20260717"
WORKERS = 1
RAW_BUCKETS = 16
LOOKBACK_TRADING_DAYS = 300

ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / RUN_ID
SLICE_DUCKDB_PATH = REPORT_DIR / "l3_compact_slice_20260717.duckdb"
FEATURE_PARQUET_PATH = REPORT_DIR / "l3_feature_slice_20260717.parquet"
LABEL_PARQUET_PATH = REPORT_DIR / "l3_label_slice_20260717.parquet"
FEATURE_TABLE = "l3_feature_slice_20260717"
LABEL_TABLE = "l3_label_slice_20260717"
PROGRESS_PATH = REPORT_DIR / "l3_compact_slice_20260717_progress.jsonl"

L1_HANDOFF_PATH = DATA_DIR / "reports" / "l1_handoff_contract_20260717.json"
L1_VALIDATION_PATH = DATA_DIR / "reports" / "l1_handoff_contract_20260717_validation.json"
L1_COMPLETE_PATH = DATA_DIR / "reports" / "l1_incremental_raw_ingest_20260717_complete.json"
L2_HANDOFF_PATH = DATA_DIR / "reports" / "l2_handoff_contract_20260717.json"
L2_VALIDATION_PATH = DATA_DIR / "reports" / "l2_handoff_contract_20260717_validation.json"
L3_HISTORICAL_INCIDENT_PATH = (
    DATA_DIR
    / "reports"
    / "l3_target_date_delivery_20260717_candidate_20260719"
    / "l3_target_date_delivery_20260717_incident.json"
)
L3_HISTORICAL_PROGRESS_PATH = (
    DATA_DIR
    / "reports"
    / "l3_target_date_delivery_20260717_candidate_20260719"
    / "l3_target_date_20260717_progress.jsonl"
)

REGISTRY_PATH = DATA_DIR / "asset_registry" / "production_assets.json"
CURRENT_L2_PATH = DATA_DIR / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
CURRENT_LABEL_PATH = DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb"

EXPECTED_HISTORICAL_L2_SHA = "7836534c8339128e51c3080daf4ec511857802ce7f5ba2250da822047e616aa6"
EXPECTED_TARGET_ROWS = 5195


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path: Path, include_hash: bool = True) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(path) if include_hash else None,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_value(value):
    if value is None:
        return "__NULL__"
    try:
        if pd.isna(value):
            return "__NULL__"
    except Exception:
        pass
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return "__NaN__"
        if math.isinf(float(value)):
            return "__INF__" if float(value) > 0 else "__-INF__"
        return format(float(value), ".17g")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def digest_records(frame: pd.DataFrame, columns: list[str]) -> dict:
    ordered = frame.sort_values(["stock_code", "trade_date"], kind="mergesort").reset_index(
        drop=True
    )
    key_columns = ["stock_code", "trade_date"]
    value_columns = [column for column in columns if column not in key_columns]
    row_hasher = hashlib.sha256()
    key_hasher = hashlib.sha256()
    value_hasher = hashlib.sha256()
    for row in ordered[columns].itertuples(index=False, name=None):
        payload = [normalize_value(value) for value in row]
        mapping = dict(zip(columns, payload))
        key_payload = [mapping[column] for column in key_columns]
        value_payload = [mapping[column] for column in value_columns]
        row_hasher.update(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        row_hasher.update(b"\n")
        key_hasher.update(
            json.dumps(key_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        key_hasher.update(b"\n")
        value_hasher.update(
            json.dumps(value_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        value_hasher.update(b"\n")
    return {
        "row_digest_sha256": row_hasher.hexdigest(),
        "key_digest_sha256": key_hasher.hexdigest(),
        "value_digest_sha256": value_hasher.hexdigest(),
        "key_columns": key_columns,
        "sort_order": key_columns,
        "serialization": "json-array-per-row utf8 newline-delimited with __NULL__/__NaN__/__INF__ sentinels",
    }


def null_nonfinite_summary(frame: pd.DataFrame) -> dict:
    null_counts = {
        column: int(value)
        for column, value in frame.isna().sum().items()
        if int(value) > 0
    }
    nonfinite_counts = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            array = series.to_numpy(dtype="float64", na_value=np.nan)
            count = int((~np.isfinite(array) & ~np.isnan(array)).sum())
            if count > 0:
                nonfinite_counts[column] = count
    return {
        "null_summary": {
            "columns_with_nulls": len(null_counts),
            "total_null_cells": int(sum(null_counts.values())),
            "top10_columns": dict(
                sorted(null_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ),
        },
        "nonfinite_columns": nonfinite_counts,
        "nonfinite_zero": len(nonfinite_counts) == 0,
    }


def build_label_slice(raw_target: pd.DataFrame, active_label_columns: list[str]) -> pd.DataFrame:
    label_columns = [column for column in active_label_columns if column not in ("trade_date", "stock_code")]
    available = available_labels(list(raw_target.columns), label_columns)
    missing = [column for column in label_columns if column not in available]
    if missing:
        raise RuntimeError(f"label rebuild missing required columns: {missing}")
    frame = raw_target[required_raw_columns(list(raw_target.columns), label_columns)].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = add_executable_labels(frame, label_columns)
    frame = sanitize_existing_labels(frame, label_columns)
    keep = [column for column in active_label_columns if column in frame.columns]
    frame = frame[keep].copy()
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.sort_values(["trade_date", "stock_code"], inplace=True)
    return frame


def assert_inputs_exist() -> None:
    required = [
        L1_HANDOFF_PATH,
        L1_VALIDATION_PATH,
        L1_COMPLETE_PATH,
        L2_HANDOFF_PATH,
        L2_VALIDATION_PATH,
        L3_HISTORICAL_INCIDENT_PATH,
        L3_HISTORICAL_PROGRESS_PATH,
        CURRENT_L2_PATH,
        CURRENT_LABEL_PATH,
        REGISTRY_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required evidence missing: {missing}")


def main() -> int:
    assert_inputs_exist()
    if REPORT_DIR.exists():
        raise SystemExit(f"report_dir already exists: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=False)

    l1_handoff = json.loads(L1_HANDOFF_PATH.read_text(encoding="utf-8"))
    l2_handoff = json.loads(L2_HANDOFF_PATH.read_text(encoding="utf-8"))
    l3_historical_incident = json.loads(L3_HISTORICAL_INCIDENT_PATH.read_text(encoding="utf-8"))

    if (l2_handoff.get("status") or "").lower() != "ready_for_audit_review":
        raise SystemExit("l2 handoff is not ready_for_audit_review")
    historical_l2_sha = (
        (((l2_handoff.get("layer_payload") or {}).get("active_asset") or {}).get("file_state") or {}).get("sha256")
    )
    if str(historical_l2_sha).lower() != EXPECTED_HISTORICAL_L2_SHA:
        raise SystemExit(
            f"historical l2 sha mismatch: expected={EXPECTED_HISTORICAL_L2_SHA} actual={historical_l2_sha}"
        )
    layer_payload = l2_handoff.get("layer_payload") or {}
    target_metrics = (
        layer_payload.get("target_day_metrics")
        or ((layer_payload.get("active_asset") or {}).get("metrics"))
        or {}
    )
    if int(target_metrics.get("target_row_count") or 0) != EXPECTED_TARGET_ROWS:
        raise SystemExit(f"unexpected L2 target rows: {target_metrics}")
    if int(target_metrics.get("target_stock_count") or 0) != EXPECTED_TARGET_ROWS:
        raise SystemExit(f"unexpected L2 target stocks: {target_metrics}")
    if int(target_metrics.get("target_bj_row_count") or 0) != 0:
        raise SystemExit("historical L2 target BJ rows not zero")
    if int(target_metrics.get("duplicate_key_groups") or 0) != 0:
        raise SystemExit("historical L2 duplicate groups not zero")

    active_feature_asset, active_feature_path, active_feature_table = _active_registry_asset("L3_features")
    active_label_asset, active_label_path, active_label_table = _active_registry_asset("L3_labels")

    current_l2_before = file_state(CURRENT_L2_PATH, include_hash=True)
    current_feature_before = file_state(active_feature_path, include_hash=True)
    current_label_before = file_state(active_label_path, include_hash=True)
    registry_before = file_state(REGISTRY_PATH, include_hash=True)

    active_feature_probe = _asset_probe(active_feature_path, active_feature_table, TARGET_DATE)
    active_label_probe = _asset_probe(active_label_path, active_label_table, TARGET_DATE)
    current_l2_probe = _asset_probe(CURRENT_L2_PATH, "STOCK_DAILY_DATA", TARGET_DATE)

    l2_qfq_technical = (
        (l2_handoff.get("layer_payload") or {}).get("qfq_contract") or {}
    ).get("technical_columns") or current_l2_probe.get("schema", [])
    if l2_qfq_technical and isinstance(l2_qfq_technical[0], dict):
        l2_qfq_technical = [
            item["name"]
            for item in current_l2_probe["schema"]
            if "_qfq" in item["name"] and item["name"] not in {"open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"}
        ]

    precheck = {
        "l2": {
            "path": str(CURRENT_L2_PATH),
            "table": "STOCK_DAILY_DATA",
            "qfq_technical_columns": l2_qfq_technical,
            "metrics": current_l2_probe["metrics"],
        },
        "target_expected_metrics": _target_expected_metrics(current_l2_probe),
        "active_feature_before": active_feature_probe,
        "active_label_before": active_label_probe,
        "registry_before": registry_before,
    }

    args = type(
        "Args",
        (),
        {
            "target_trade_date": TARGET_DATE,
            "raw_buckets": RAW_BUCKETS,
            "workers": WORKERS,
            "lookback_trading_days": LOOKBACK_TRADING_DAYS,
        },
    )()

    raw_target, gtja_input, raw_bucket_metrics = _compute_l2_only_frames(args, precheck, PROGRESS_PATH)
    gtja_target = _compute_gtja_target(gtja_input, TARGET_DATE, PROGRESS_PATH)
    active_feature_columns = [item["name"] for item in active_feature_probe["schema"]]
    feature_frame, feature_lineage = _build_feature_target(
        raw_target,
        gtja_target,
        active_feature_columns,
        feature_contract_version=FEATURE_CONTRACT_ACTIVE,
    )
    feature_gate = _target_frame_gates(
        feature_frame,
        precheck,
        feature_contract_version=FEATURE_CONTRACT_ACTIVE,
    )
    if not all(feature_gate["gates"].values()):
        raise SystemExit(f"feature gate failed: {feature_gate['gates']}")

    active_label_columns = [item["name"] for item in active_label_probe["schema"]]
    label_frame = build_label_slice(raw_target, active_label_columns)
    if int(label_frame.shape[0]) != EXPECTED_TARGET_ROWS:
        raise SystemExit(f"unexpected label rows: {label_frame.shape[0]}")
    if int(label_frame["stock_code"].nunique()) != EXPECTED_TARGET_ROWS:
        raise SystemExit("unexpected label key domain")
    if int(label_frame.duplicated(["trade_date", "stock_code"]).sum()) != 0:
        raise SystemExit("label duplicate key groups not zero")
    if int(label_frame["stock_code"].astype(str).str.endswith(".BJ").sum()) != 0:
        raise SystemExit("label BJ rows not zero")

    feature_frame.to_parquet(FEATURE_PARQUET_PATH, index=False)
    label_frame.to_parquet(LABEL_PARQUET_PATH, index=False)
    with closing(duckdb.connect(str(SLICE_DUCKDB_PATH))) as conn:
        conn.register("feature_frame", feature_frame)
        conn.register("label_frame", label_frame)
        try:
            conn.execute(f"CREATE TABLE {FEATURE_TABLE} AS SELECT * FROM feature_frame")
            conn.execute(f"CREATE TABLE {LABEL_TABLE} AS SELECT * FROM label_frame")
        finally:
            conn.unregister("feature_frame")
            conn.unregister("label_frame")
        feature_slice_schema = _schema(conn, FEATURE_TABLE)
        label_slice_schema = _schema(conn, LABEL_TABLE)
        feature_metrics = _key_metrics(conn, FEATURE_TABLE, TARGET_DATE)
        label_metrics = _key_metrics(conn, LABEL_TABLE, TARGET_DATE)

    current_l2_after = file_state(CURRENT_L2_PATH, include_hash=False)
    current_feature_after = file_state(active_feature_path, include_hash=False)
    current_label_after = file_state(active_label_path, include_hash=False)
    registry_after = file_state(REGISTRY_PATH, include_hash=False)

    for before, after, label in [
        (current_l2_before, current_l2_after, "current_l2"),
        (current_feature_before, current_feature_after, "active_feature"),
        (current_label_before, current_label_after, "active_label"),
        (registry_before, registry_after, "registry"),
    ]:
        if before["size_bytes"] != after["size_bytes"] or before["mtime_ns"] != after["mtime_ns"]:
            raise SystemExit(f"{label} changed during compact slice build")

    feature_qfq_mismatch_total = 0
    with closing(duckdb.connect(str(SLICE_DUCKDB_PATH), read_only=True)) as conn:
        conn.execute(f"ATTACH '{CURRENT_L2_PATH.as_posix()}' AS l2 (READ_ONLY)")
        try:
            qfq_columns = [
                "open_qfq",
                "high_qfq",
                "low_qfq",
                "close_qfq",
                "pre_close_qfq",
                *l2_qfq_technical,
            ]
            mismatch_expr = ",".join(
                f"SUM(CASE WHEN f.\"{column}\" IS DISTINCT FROM l.\"{column}\" THEN 1 ELSE 0 END) AS \"{column}\""
                for column in qfq_columns
            )
            mismatch_row = conn.execute(
                f"""
                SELECT {mismatch_expr}
                FROM {FEATURE_TABLE} f
                JOIN l2.STOCK_DAILY_DATA l
                  ON CAST(f.trade_date AS VARCHAR)=CAST(l.trade_date AS VARCHAR)
                 AND f.stock_code=l.stock_code
                WHERE CAST(f.trade_date AS VARCHAR)=?
                """,
                [TARGET_DATE],
            ).fetchone()
            feature_qfq_mismatch_total = int(sum(int(value or 0) for value in mismatch_row))
        finally:
            conn.execute("DETACH l2")

    feature_future_columns = sorted(
        {
            column
            for column in feature_frame.columns
            if is_future_or_label_column(column) or column in SEMANTIC_FUTURE_DERIVED_COLUMNS
        }
    )
    feature_digests = digest_records(feature_frame, list(feature_frame.columns))
    label_digests = digest_records(label_frame, list(label_frame.columns))
    feature_quality = null_nonfinite_summary(feature_frame)
    label_quality = null_nonfinite_summary(label_frame)
    calendar_validation = audit_asset_calendars(
        [TARGET_DATE],
        {"compact_slice_20260717": [TARGET_DATE]},
        start=TARGET_DATE,
        end=TARGET_DATE,
    )

    feature_state = file_state(FEATURE_PARQUET_PATH, include_hash=True)
    label_state = file_state(LABEL_PARQUET_PATH, include_hash=True)
    slice_duckdb_state = file_state(SLICE_DUCKDB_PATH, include_hash=True)

    report = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "passed",
        "generated_at": now_iso(),
        "target_trade_date": TARGET_DATE,
        "operation": "compact_immutable_slice_from_audited_l1_l2_and_frozen_l3_rules",
        "explicit_non_claims": {
            "reused_historical_candidate_or_quarantine": False,
            "modified_active_feature": False,
            "modified_active_label": False,
            "modified_registry": False,
            "training_started": False,
            "prediction_started": False,
            "strategy_started": False,
            "read_strategy_returns": False,
        },
        "audit_bindings": {
            "l1_handoff": str(L1_HANDOFF_PATH),
            "l1_validation": str(L1_VALIDATION_PATH),
            "l1_complete": str(L1_COMPLETE_PATH),
            "l2_handoff": str(L2_HANDOFF_PATH),
            "l2_validation": str(L2_VALIDATION_PATH),
            "historical_l3_failed_incident": str(L3_HISTORICAL_INCIDENT_PATH),
            "historical_l3_progress": str(L3_HISTORICAL_PROGRESS_PATH),
            "historical_l2_sha256": EXPECTED_HISTORICAL_L2_SHA,
            "current_l2_sha256": current_l2_before["sha256"],
            "current_active_feature_asset_id": active_feature_asset.get("asset_id"),
            "current_active_label_asset_id": active_label_asset.get("asset_id"),
        },
        "inputs": {
            "current_l2_before": current_l2_before,
            "current_l2_after": current_l2_after,
            "active_feature_before": current_feature_before,
            "active_feature_after": current_feature_after,
            "active_label_before": current_label_before,
            "active_label_after": current_label_after,
            "registry_before": registry_before,
            "registry_after": registry_after,
            "l2_probe": current_l2_probe,
            "active_feature_probe": active_feature_probe,
            "active_label_probe": active_label_probe,
            "raw_bucket_metrics": raw_bucket_metrics,
            "gtja_input_rows": int(gtja_input.shape[0]),
        },
        "feature_slice": {
            "duckdb_path": str(SLICE_DUCKDB_PATH),
            "duckdb_table": FEATURE_TABLE,
            "parquet_path": str(FEATURE_PARQUET_PATH),
            "parquet_sha256": feature_state["sha256"],
            "schema_hash": _schema_hash(feature_slice_schema),
            "schema": feature_slice_schema,
            "column_count": len(feature_slice_schema),
            "metrics": feature_metrics,
            "digests": feature_digests,
            "future_use_columns": feature_future_columns,
            "future_use_zero": len(feature_future_columns) == 0,
            "feature_gate": feature_gate,
            "qfq_source_mismatch_total": feature_qfq_mismatch_total,
            "lineage": feature_lineage,
            **feature_quality,
        },
        "label_slice": {
            "duckdb_path": str(SLICE_DUCKDB_PATH),
            "duckdb_table": LABEL_TABLE,
            "parquet_path": str(LABEL_PARQUET_PATH),
            "parquet_sha256": label_state["sha256"],
            "schema_hash": _schema_hash(label_slice_schema),
            "schema": label_slice_schema,
            "column_count": len(label_slice_schema),
            "metrics": label_metrics,
            "digests": label_digests,
            "key_domain_matches_feature": feature_digests["key_digest_sha256"] == label_digests["key_digest_sha256"],
            "maturity_rule": "historical target date is prior to current label maturity frontier; labels are rebuilt under frozen production label rules only",
            **label_quality,
        },
        "slice_asset": {
            "duckdb_sha256": slice_duckdb_state["sha256"],
            "duckdb_size_bytes": slice_duckdb_state["size_bytes"],
        },
        "calendar_validation": calendar_validation,
        "handoff_scope": {
            "allow_model_agent_l4_backfill_request": True,
            "allow_next_layer_continue": False,
            "target_date_only": TARGET_DATE,
            "consumer_thread_id": "019ed0ad-8d91-74d0-8022-3dbe2006dc6a",
        },
    }

    contract = build_layer_handoff_contract(
        workflow_run_id=RUN_ID,
        layer="L3",
        target_trade_date=TARGET_DATE,
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=[
            str(L1_HANDOFF_PATH),
            str(L2_HANDOFF_PATH),
            f"{CURRENT_L2_PATH}::STOCK_DAILY_DATA",
            f"{active_feature_path}::{active_feature_table}",
            f"{active_label_path}::{active_label_table}",
        ],
        active_output_assets=[
            f"{SLICE_DUCKDB_PATH}::{FEATURE_TABLE}",
            f"{SLICE_DUCKDB_PATH}::{LABEL_TABLE}",
            str(FEATURE_PARQUET_PATH),
            str(LABEL_PARQUET_PATH),
        ],
        gate_checks=[
            {"name": "l2_target_rows_5195", "passed": int(current_l2_probe["metrics"]["target_row_count"]) == EXPECTED_TARGET_ROWS},
            {"name": "feature_target_rows_5195", "passed": int(feature_metrics["target_row_count"]) == EXPECTED_TARGET_ROWS},
            {"name": "label_target_rows_5195", "passed": int(label_metrics["target_row_count"]) == EXPECTED_TARGET_ROWS},
            {"name": "feature_duplicate_zero", "passed": int(feature_metrics["duplicate_key_groups"]) == 0},
            {"name": "label_duplicate_zero", "passed": int(label_metrics["duplicate_key_groups"]) == 0},
            {"name": "feature_bj_zero", "passed": int(feature_metrics["target_bj_row_count"]) == 0},
            {"name": "label_bj_zero", "passed": int(label_metrics["target_bj_row_count"]) == 0},
            {"name": "feature_future_use_zero", "passed": len(feature_future_columns) == 0},
            {"name": "feature_qfq_mismatch_zero", "passed": feature_qfq_mismatch_total == 0},
            {"name": "feature_label_key_domain_equal", "passed": feature_digests["key_digest_sha256"] == label_digests["key_digest_sha256"]},
            {"name": "active_assets_unchanged", "passed": True},
        ],
        handoff_constraints=[
            "compact immutable slice is rebuilt from audited historical L1/L2 inputs plus frozen production L3 factor and label rules",
            "it does not claim byte-equivalence to any missing historical candidate or quarantine artifact",
            "active feature, active label, registry and production assets were not modified in this run",
            "consumer may use only for historical 20260717 L4 backfill after fixed readonly audit approval",
        ],
        evidence_paths=[
            str(L1_HANDOFF_PATH),
            str(L2_HANDOFF_PATH),
            str(L3_HISTORICAL_INCIDENT_PATH),
            str(REPORT_DIR / "l3_compact_immutable_slice_20260717.json"),
            str(REPORT_DIR / "audit_handoff.json"),
        ],
        boundaries={
            "read_only_inputs": True,
            "no_active_write": True,
            "no_label_write": True,
            "no_registry_change": True,
            "no_strategy": True,
            "no_training": True,
            "no_prediction": True,
        },
        layer_payload=report,
    )
    errors = validate_layer_handoff_contract(
        contract,
        expected_layer="L3",
        expected_owner_agent="factor-agent",
    )
    if errors:
        raise SystemExit("contract validation failed: " + "; ".join(errors))

    handoff = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "ready_for_audit_review",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "compact_slice_report": str(REPORT_DIR / "l3_compact_immutable_slice_20260717.json"),
        "workflow_contract": str(REPORT_DIR / "l3_to_l4_handoff_contract_20260717_compact_slice.json"),
        "asset_digest": {
            "duckdb_sha256": slice_duckdb_state["sha256"],
            "feature_parquet_sha256": feature_state["sha256"],
            "label_parquet_sha256": label_state["sha256"],
            "feature_row_digest_sha256": feature_digests["row_digest_sha256"],
            "feature_key_digest_sha256": feature_digests["key_digest_sha256"],
            "feature_value_digest_sha256": feature_digests["value_digest_sha256"],
            "label_row_digest_sha256": label_digests["row_digest_sha256"],
            "label_key_digest_sha256": label_digests["key_digest_sha256"],
            "label_value_digest_sha256": label_digests["value_digest_sha256"],
        },
        "message_for_downstream": (
            "Compact immutable 20260717 L3 slice rebuilt from audited L1/L2 plus frozen L3 rules; "
            "use only for historical gap closure after fixed readonly audit."
        ),
        "data_integration_thread_id": "019ed0ad-8d91-74d0-8022-3dbe2006dc6a",
    }

    audit = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "passed",
        "generated_at": now_iso(),
        "checks": {
            "feature_rows_expected": int(feature_metrics["target_row_count"]) == EXPECTED_TARGET_ROWS,
            "label_rows_expected": int(label_metrics["target_row_count"]) == EXPECTED_TARGET_ROWS,
            "feature_schema_matches_active": _schema_hash(feature_slice_schema) == active_feature_probe["schema_hash"],
            "label_schema_matches_active": _schema_hash(label_slice_schema) == active_label_probe["schema_hash"],
            "feature_qfq_mismatch_zero": feature_qfq_mismatch_total == 0,
            "feature_future_use_zero": len(feature_future_columns) == 0,
            "feature_duplicate_zero": int(feature_metrics["duplicate_key_groups"]) == 0,
            "label_duplicate_zero": int(label_metrics["duplicate_key_groups"]) == 0,
            "feature_bj_zero": int(feature_metrics["target_bj_row_count"]) == 0,
            "label_bj_zero": int(label_metrics["target_bj_row_count"]) == 0,
            "feature_nonfinite_zero": feature_quality["nonfinite_zero"],
            "label_nonfinite_zero": label_quality["nonfinite_zero"],
            "active_feature_unchanged": current_feature_before["sha256"] == current_feature_after["sha256"],
            "active_label_unchanged": current_label_before["sha256"] == current_label_after["sha256"],
            "registry_unchanged": registry_before["sha256"] == registry_after["sha256"],
        },
    }
    if not all(audit["checks"].values()):
        raise SystemExit(f"audit checks failed: {audit['checks']}")

    write_json(REPORT_DIR / "l3_compact_immutable_slice_20260717.json", report)
    write_json(REPORT_DIR / "l3_to_l4_handoff_contract_20260717_compact_slice.json", contract)
    write_json(REPORT_DIR / "audit_handoff.json", handoff)
    write_json(REPORT_DIR / "l3_compact_immutable_slice_20260717_audit.json", audit)

    summary = {
        "status": "passed",
        "run_id": RUN_ID,
        "report_dir": str(REPORT_DIR),
        "duckdb_path": str(SLICE_DUCKDB_PATH),
        "duckdb_sha256": slice_duckdb_state["sha256"],
        "feature_parquet_sha256": feature_state["sha256"],
        "label_parquet_sha256": label_state["sha256"],
        "feature_row_digest_sha256": feature_digests["row_digest_sha256"],
        "label_row_digest_sha256": label_digests["row_digest_sha256"],
        "feature_rows": int(feature_metrics["target_row_count"]),
        "label_rows": int(label_metrics["target_row_count"]),
        "feature_columns": len(feature_slice_schema),
        "label_columns": len(label_slice_schema),
        "feature_future_use_zero": len(feature_future_columns) == 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
