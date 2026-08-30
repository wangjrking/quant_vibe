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

from build_production_factor_parts import is_future_or_label_column
from feature_contract_v2_candidate import SEMANTIC_FUTURE_DERIVED_COLUMNS
from rebuild_l3_full_duckdb_mainline import _key_metrics, _schema, _schema_hash
from validation_calendar_contract import audit_asset_calendars
from workflow_contract import build_layer_handoff_contract, validate_layer_handoff_contract


RUN_ID = "incremental-trading-signal-20260806-L3-compact-immutable-slice-20260815-r1"
TARGET_DATE = "20260806"
SOURCE_ACTIVE_PATH = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\l3_feature_current.duckdb"
)
SOURCE_ACTIVE_TABLE = "prod_l3_production_factor_parts_20260625"
CURRENT_L2_PATH = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\l2_stock_daily_data.duckdb"
)
CURRENT_LABEL_PATH = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\l3_label_current.duckdb"
)
REGISTRY_PATH = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\asset_registry\production_assets.json"
)
REPORT_DIR = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\reports\incremental-trading-signal-20260806-L3-compact-immutable-slice-20260815-r1"
)
PARQUET_PATH = REPORT_DIR / "l3_feature_slice_20260806_compact.parquet"
DUCKDB_PATH = REPORT_DIR / "l3_feature_slice_20260806_compact.duckdb"
TABLE_NAME = "l3_feature_slice_20260806_compact"
CONTINUITY_PATH = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\reports\incremental-trading-signal-20260807-L3-continuity-auto-advance-0806-r1\l3_continuity_auto_advance_0806.json"
)
CONTINUITY_GATE_PATH = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\reports\incremental-trading-signal-20260807-L3-continuity-auto-advance-0806-r1\l3_target_date_20260806_final_atomic_switch_gate.json"
)
ORIGINAL_HANDOFF_PATH = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\reports\incremental-trading-signal-20260806-L3-target-date-candidate-only-compliant-runtime-20260807-r3\l3_to_l4_handoff_contract_20260806.json"
)
ORIGINAL_CANDIDATE_SHA = "8e4b5e431726841225e3388ba4b0b40750edbd443b8b3e50a61ebf1f7c0f2d51"
EXPECTED_HISTORICAL_L2_SHA = "a4d8df3510da2bd3ed618d56a9ad4204cd209e0984c505236b32f01cd517be95"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


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
            json.dumps(key_payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        key_hasher.update(b"\n")
        value_hasher.update(
            json.dumps(value_payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
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


def main() -> int:
    if REPORT_DIR.exists():
        raise SystemExit(f"report_dir already exists: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=False)

    source_active_before = file_state(SOURCE_ACTIVE_PATH, include_hash=True)
    current_l2_before = file_state(CURRENT_L2_PATH, include_hash=True)
    current_label_before = file_state(CURRENT_LABEL_PATH, include_hash=True)
    registry_before = file_state(REGISTRY_PATH, include_hash=True)

    continuity = json.loads(CONTINUITY_PATH.read_text(encoding="utf-8"))
    original_handoff = json.loads(ORIGINAL_HANDOFF_PATH.read_text(encoding="utf-8"))

    if continuity.get("candidate_sha") != ORIGINAL_CANDIDATE_SHA:
        raise SystemExit("continuity candidate sha mismatch")
    if not bool(continuity.get("active_switch_called")):
        raise SystemExit("continuity active switch not recorded as true")
    post_switch = continuity.get("post_switch") or {}
    if post_switch.get("active_sha") != ORIGINAL_CANDIDATE_SHA:
        raise SystemExit("continuity active sha does not bind original candidate sha")
    if int(post_switch.get("active_target_rows") or 0) != 5200:
        raise SystemExit("continuity target rows mismatch")
    if int(post_switch.get("active_columns") or 0) != 838:
        raise SystemExit("continuity active column count mismatch")
    if not bool(post_switch.get("active_matches_candidate")):
        raise SystemExit("continuity does not prove active matches candidate")
    if bool(original_handoff.get("allow_next_layer_continue")):
        raise SystemExit("original handoff unexpectedly allowed next layer")
    original_payload = original_handoff.get("layer_payload") or {}
    if bool(original_payload.get("active_switch_called")):
        raise SystemExit("original handoff unexpectedly recorded active_switch_called=true")
    if bool(original_payload.get("label_write_called")):
        raise SystemExit("original handoff unexpectedly recorded label_write_called=true")

    with closing(duckdb.connect(str(SOURCE_ACTIVE_PATH), read_only=True)) as conn:
        source_schema = _schema(conn, SOURCE_ACTIVE_TABLE)
        source_schema_hash = _schema_hash(source_schema)
        conn.execute(
            f"""
            COPY (
              SELECT * FROM {SOURCE_ACTIVE_TABLE}
              WHERE CAST(trade_date AS VARCHAR) = ?
              ORDER BY stock_code, trade_date
            ) TO '{PARQUET_PATH.as_posix()}' (FORMAT PARQUET)
            """,
            [TARGET_DATE],
        )

    with closing(duckdb.connect(str(DUCKDB_PATH))) as conn:
        conn.execute(
            f"""
            CREATE TABLE {TABLE_NAME} AS
            SELECT * FROM read_parquet('{PARQUET_PATH.as_posix()}')
            """
        )
        slice_schema = _schema(conn, TABLE_NAME)
        slice_metrics = _key_metrics(conn, TABLE_NAME, TARGET_DATE)
        frame = conn.execute(
            f"SELECT * FROM {TABLE_NAME} ORDER BY stock_code, trade_date"
        ).fetchdf()

    slice_columns = list(frame.columns)
    future_columns = sorted(
        {
            column
            for column in slice_columns
            if is_future_or_label_column(column)
            or column in SEMANTIC_FUTURE_DERIVED_COLUMNS
        }
    )
    null_counts = {
        column: int(value)
        for column, value in frame.isna().sum().items()
        if int(value) > 0
    }
    null_summary = {
        "columns_with_nulls": len(null_counts),
        "total_null_cells": int(sum(null_counts.values())),
        "top10_columns": dict(
            sorted(null_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ),
    }
    nonfinite_counts = {}
    for column in slice_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            array = series.to_numpy(dtype="float64", na_value=np.nan)
            count = int((~np.isfinite(array) & ~np.isnan(array)).sum())
            if count > 0:
                nonfinite_counts[column] = count

    industry_empty_string_rows = []
    if "industry" in frame.columns:
        subset = frame.loc[
            frame["industry"].fillna("").astype(str) == "",
            ["stock_code", "trade_date"],
        ]
        industry_empty_string_rows = subset.to_dict(orient="records")

    digests = digest_records(frame, slice_columns)
    calendar_validation = audit_asset_calendars(
        [TARGET_DATE],
        {"compact_slice": [TARGET_DATE]},
        start=TARGET_DATE,
        end=TARGET_DATE,
    )

    source_active_after = file_state(SOURCE_ACTIVE_PATH, include_hash=False)
    current_l2_after = file_state(CURRENT_L2_PATH, include_hash=False)
    current_label_after = file_state(CURRENT_LABEL_PATH, include_hash=False)
    registry_after = file_state(REGISTRY_PATH, include_hash=False)

    for before, after, label in [
        (source_active_before, source_active_after, "source_active"),
        (current_l2_before, current_l2_after, "current_l2"),
        (current_label_before, current_label_after, "current_label"),
        (registry_before, registry_after, "registry"),
    ]:
        if before["size_bytes"] != after["size_bytes"] or before["mtime_ns"] != after["mtime_ns"]:
            raise SystemExit(f"{label} changed during compact extraction")

    parquet_state = file_state(PARQUET_PATH, include_hash=True)
    duckdb_state = file_state(DUCKDB_PATH, include_hash=True)

    report = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "passed",
        "generated_at": now_iso(),
        "target_trade_date": TARGET_DATE,
        "operation": "compact_immutable_slice_from_current_active_l3",
        "authorized_route": "architect_passed_p2_residual_compact_extraction",
        "source_mode": "current_active_l3_readonly_extraction",
        "explicit_non_claims": {
            "recomputed_original_candidate": False,
            "byte_equivalent_to_missing_candidate": False,
            "active_switch_called": False,
            "label_write_called": False,
            "registry_change": False,
            "training_started": False,
            "prediction_started": False,
            "strategy_started": False,
        },
        "source_bindings": {
            "source_active_asset": f"{SOURCE_ACTIVE_PATH}::{SOURCE_ACTIVE_TABLE}",
            "source_active_before": source_active_before,
            "source_active_after": source_active_after,
            "current_l2_before": current_l2_before,
            "current_l2_after": current_l2_after,
            "current_label_before": current_label_before,
            "current_label_after": current_label_after,
            "registry_before": registry_before,
            "registry_after": registry_after,
            "historical_expected_l2_sha_20260806": EXPECTED_HISTORICAL_L2_SHA,
            "continuity_evidence": str(CONTINUITY_PATH),
            "continuity_final_gate": str(CONTINUITY_GATE_PATH),
            "original_candidate_only_handoff": str(ORIGINAL_HANDOFF_PATH),
            "original_candidate_sha": ORIGINAL_CANDIDATE_SHA,
            "original_handoff_active_switch_called": bool(
                original_payload.get("active_switch_called")
            ),
            "original_handoff_label_write_called": bool(
                original_payload.get("label_write_called")
            ),
            "continuity_active_sha_after_switch": str(post_switch.get("active_sha") or ""),
            "continuity_active_matches_candidate": bool(
                post_switch.get("active_matches_candidate")
            ),
        },
        "slice_asset": {
            "duckdb_path": str(DUCKDB_PATH),
            "duckdb_table": TABLE_NAME,
            "duckdb_sha256": duckdb_state["sha256"],
            "parquet_path": str(PARQUET_PATH),
            "parquet_sha256": parquet_state["sha256"],
        },
        "schema_contract": {
            "source_schema_hash": source_schema_hash,
            "slice_schema_hash": _schema_hash(slice_schema),
            "column_count": len(slice_schema),
            "columns_match_source_order": [item["name"] for item in source_schema]
            == [item["name"] for item in slice_schema],
            "schema": slice_schema,
            "feature_contract_version": "v2-continuity-compact-extraction",
        },
        "quality": {
            "row_count": int(slice_metrics["row_count"]),
            "stock_count": int(slice_metrics["stock_count"]),
            "target_row_count": int(slice_metrics["target_row_count"]),
            "target_stock_count": int(slice_metrics["target_stock_count"]),
            "duplicate_key_groups": int(slice_metrics["duplicate_key_groups"]),
            "bj_row_count": int(slice_metrics["bj_row_count"]),
            "target_bj_row_count": int(slice_metrics["target_bj_row_count"]),
            "logical_key_hash": slice_metrics["logical_key_hash"],
            "logical_key_hash_components": slice_metrics["logical_key_hash_components"],
            "future_use_columns": future_columns,
            "future_use_zero": len(future_columns) == 0,
            "null_summary": null_summary,
            "nonfinite_columns": nonfinite_counts,
            "nonfinite_zero": len(nonfinite_counts) == 0,
            "calendar_validation": calendar_validation,
        },
        "approved_normalization": {
            "industry_null_to_empty_string": continuity.get(
                "approved_industry_normalization"
            ),
            "current_empty_string_rows": industry_empty_string_rows,
        },
        "digests": digests,
        "model_agent_scope": {
            "historical_target_date_only": TARGET_DATE,
            "allow_l4_backfill_request": True,
            "requires_fixed_readonly_audit": True,
            "allow_l4_without_audit": False,
        },
        "evidence_paths": [
            str(CONTINUITY_PATH),
            str(CONTINUITY_GATE_PATH),
            str(ORIGINAL_HANDOFF_PATH),
            str(DUCKDB_PATH),
            str(PARQUET_PATH),
        ],
    }

    contract = build_layer_handoff_contract(
        workflow_run_id=RUN_ID,
        layer="L3",
        target_trade_date=TARGET_DATE,
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=[
            f"{SOURCE_ACTIVE_PATH}::{SOURCE_ACTIVE_TABLE}",
            str(CONTINUITY_PATH),
            str(CONTINUITY_GATE_PATH),
            str(ORIGINAL_HANDOFF_PATH),
        ],
        active_output_assets=[
            f"{DUCKDB_PATH}::{TABLE_NAME}",
            str(PARQUET_PATH),
        ],
        gate_checks=[
            {"name": "continuity_active_switch_bound_to_original_candidate", "passed": True},
            {"name": "target_row_count_5200", "passed": int(slice_metrics["target_row_count"]) == 5200},
            {"name": "column_count_838", "passed": len(slice_schema) == 838},
            {
                "name": "duplicate_key_groups_zero",
                "passed": int(slice_metrics["duplicate_key_groups"]) == 0,
            },
            {
                "name": "bj_rows_zero",
                "passed": int(slice_metrics["bj_row_count"]) == 0
                and int(slice_metrics["target_bj_row_count"]) == 0,
            },
            {"name": "future_use_zero", "passed": len(future_columns) == 0},
            {"name": "nonfinite_zero", "passed": len(nonfinite_counts) == 0},
            {"name": "calendar_exact_match", "passed": bool(calendar_validation.get("ready"))},
            {"name": "active_label_registry_unchanged", "passed": True},
        ],
        handoff_constraints=[
            "compact extraction is derived from current active L3 after audited continuity active-switch",
            "this asset is not a recomputed byte-equivalent replacement for the missing original candidate file",
            "model-agent may only use it for a historical 20260806 L4 backfill after fixed readonly audit approval",
            "no active/label/registry modification occurred in this run",
        ],
        evidence_paths=report["evidence_paths"]
        + [str(REPORT_DIR / "l3_compact_immutable_slice_20260806.json")],
        residual_risk=[
            {
                "severity": "P2",
                "item": "compact_extraction_not_original_candidate_bytes",
                "detail": "The original candidate file is gone. This package binds a current active 20260806 slice to the audited continuity active-switch, not to direct candidate byte identity.",
            }
        ],
        boundaries={
            "read_only_source_active": True,
            "no_active_write": True,
            "no_label_write": True,
            "no_registry_change": True,
            "no_manifest_write": True,
            "no_training": True,
            "no_prediction": True,
            "no_strategy": True,
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
        "compact_slice_report": str(REPORT_DIR / "l3_compact_immutable_slice_20260806.json"),
        "workflow_contract": str(
            REPORT_DIR / "l3_to_l4_handoff_contract_20260806_compact_slice.json"
        ),
        "asset_digest": {
            "duckdb_sha256": duckdb_state["sha256"],
            "parquet_sha256": parquet_state["sha256"],
            "row_digest_sha256": digests["row_digest_sha256"],
            "key_digest_sha256": digests["key_digest_sha256"],
            "value_digest_sha256": digests["value_digest_sha256"],
        },
        "message_for_model_agent": (
            "Use only for historical 20260806 L4 backfill after fixed readonly audit approval; "
            "do not treat as original candidate byte identity."
        ),
    }

    write_json(REPORT_DIR / "l3_compact_immutable_slice_20260806.json", report)
    write_json(
        REPORT_DIR / "l3_to_l4_handoff_contract_20260806_compact_slice.json",
        contract,
    )
    write_json(REPORT_DIR / "audit_handoff.json", handoff)

    summary = {
        "status": "passed",
        "run_id": RUN_ID,
        "report_dir": str(REPORT_DIR),
        "duckdb_path": str(DUCKDB_PATH),
        "duckdb_sha256": duckdb_state["sha256"],
        "parquet_path": str(PARQUET_PATH),
        "parquet_sha256": parquet_state["sha256"],
        "row_digest_sha256": digests["row_digest_sha256"],
        "key_digest_sha256": digests["key_digest_sha256"],
        "value_digest_sha256": digests["value_digest_sha256"],
        "row_count": int(slice_metrics["target_row_count"]),
        "column_count": len(slice_schema),
        "future_use_zero": len(future_columns) == 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
