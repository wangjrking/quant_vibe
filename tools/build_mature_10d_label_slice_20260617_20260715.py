from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from build_prediction_label_parts import add_executable_labels, available_labels, required_raw_columns  # noqa: E402
from workflow_contract import build_layer_handoff_contract, validate_layer_handoff_contract  # noqa: E402


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
EVIDENCE_ROOT = DATA_DIR / "runtime" / "agent_workspaces" / "data-integration-agent" / "work" / "v260_rolling252_top10_ltr_10d_mature_label_binding_20260817_r1"
RUN_ID = "v260_rolling252_top10_ltr_10d_mature_label_slice_20260817_r2"

ACTIVE_FEATURE_PATH = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
ACTIVE_FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
ACTIVE_LABEL_PATH = DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb"
ACTIVE_LABEL_TABLE = "prod_l3_prediction_label_parts_current"
ACTIVE_L2_PATH = DATA_DIR / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
ACTIVE_L2_TABLE = "STOCK_DAILY_DATA"
REGISTRY_PATH = DATA_DIR / "asset_registry" / "production_assets.json"

BINDING_PATH = EVIDENCE_ROOT / "mature_10d_training_label_binding_v1.json"
CALENDAR_DESCRIPTOR_PATH = (
    DATA_DIR
    / "runtime"
    / "agent_workspaces"
    / "data-ingestion-agent"
    / "work"
    / "v7_blind_validation_official_trade_cal_20260105_20260813_r1"
    / "official_trade_cal_descriptor_v1.json"
)
CALENDAR_DUCKDB_PATH = (
    DATA_DIR
    / "runtime"
    / "agent_workspaces"
    / "data-ingestion-agent"
    / "work"
    / "v7_blind_validation_official_trade_cal_20260105_20260813_r1"
    / "official_trade_cal.duckdb"
)
CALENDAR_TABLE = "official_trade_cal"

LABEL_NAME = "executable_10d_open_return"
PREDICTION_START = "20260803"

SLICE_DUCKDB_PATH = EVIDENCE_ROOT / "mature_10d_label_slice_20260617_20260715.duckdb"
SLICE_TABLE = "mature_10d_label_slice_20260617_20260715"
SLICE_PARQUET_PATH = EVIDENCE_ROOT / "mature_10d_label_slice_20260617_20260715.parquet"
REPORT_PATH = EVIDENCE_ROOT / "mature_10d_label_slice_20260617_20260715_report.json"
HANDOFF_PATH = EVIDENCE_ROOT / "mature_10d_label_slice_20260617_20260715_handoff.json"
AUDIT_PATH = EVIDENCE_ROOT / "mature_10d_label_slice_20260617_20260715_audit.json"
WORKFLOW_CONTRACT_PATH = EVIDENCE_ROOT / "mature_10d_label_slice_20260617_20260715_workflow_contract.json"


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    return str(value)


def digest_records(frame: pd.DataFrame, columns: list[str]) -> dict:
    ordered = frame.sort_values(["trade_date", "stock_code"], kind="mergesort").reset_index(drop=True)
    key_columns = ["trade_date", "stock_code"]
    value_columns = [column for column in columns if column not in key_columns]
    row_hasher = hashlib.sha256()
    key_hasher = hashlib.sha256()
    value_hasher = hashlib.sha256()
    for row in ordered[columns].itertuples(index=False, name=None):
        payload = [normalize_value(value) for value in row]
        mapping = dict(zip(columns, payload))
        key_payload = [mapping[column] for column in key_columns]
        value_payload = [mapping[column] for column in value_columns]
        row_hasher.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        row_hasher.update(b"\n")
        key_hasher.update(json.dumps(key_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        key_hasher.update(b"\n")
        value_hasher.update(json.dumps(value_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        value_hasher.update(b"\n")
    return {
        "row_digest_sha256": row_hasher.hexdigest(),
        "key_digest_sha256": key_hasher.hexdigest(),
        "value_digest_sha256": value_hasher.hexdigest(),
        "key_columns": key_columns,
        "sort_order": key_columns,
    }


def null_summary(frame: pd.DataFrame) -> dict:
    null_counts = {column: int(value) for column, value in frame.isna().sum().items() if int(value) > 0}
    return {
        "columns_with_nulls": len(null_counts),
        "total_null_cells": int(sum(null_counts.values())),
        "top10_columns": dict(sorted(null_counts.items(), key=lambda item: (-item[1], item[0]))[:10]),
    }


def schema_snapshot(con: duckdb.DuckDBPyConnection, table: str) -> list[dict]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()
    schema = []
    for index, row in enumerate(rows):
        schema.append(
            {
                "index": index,
                "name": row[0],
                "type": row[1],
                "not_null": False if row[2] is None else str(row[2]).upper() == "NO",
            }
        )
    return schema


def schema_hash(schema: list[dict]) -> str:
    payload = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_paths() -> None:
    required = [
        ACTIVE_FEATURE_PATH,
        ACTIVE_LABEL_PATH,
        ACTIVE_L2_PATH,
        REGISTRY_PATH,
        BINDING_PATH,
        CALENDAR_DESCRIPTOR_PATH,
        CALENDAR_DUCKDB_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required inputs: {missing}")


def load_missing_dates() -> list[str]:
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    dates = (((binding.get("collective_gap") or {}).get("label_date_gap")) or {}).get("missing_official_open_dates") or []
    if len(dates) != 20:
        raise SystemExit(f"expected 20 missing mature dates, got {len(dates)}")
    return [str(item) for item in dates]


def load_open_dates() -> list[str]:
    with closing(duckdb.connect(str(CALENDAR_DUCKDB_PATH), read_only=True)) as con:
        rows = con.execute(
            f"""
            SELECT cal_date
            FROM {CALENDAR_TABLE}
            WHERE is_open = 1
            ORDER BY cal_date
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def maturity_proof(target_dates: list[str], open_dates: list[str]) -> tuple[list[dict], str]:
    index_map = {value: idx for idx, value in enumerate(open_dates)}
    checks = []
    max_end = None
    for signal_date in target_dates:
        if signal_date not in index_map:
            raise SystemExit(f"signal date missing from official calendar: {signal_date}")
        end_idx = index_map[signal_date] + 12
        if end_idx >= len(open_dates):
            raise SystemExit(f"official calendar cannot prove +12 open-session maturity for {signal_date}")
        label_end = open_dates[end_idx]
        matured = label_end < PREDICTION_START
        if not matured:
            raise SystemExit(f"unmatured signal date in requested range: {signal_date} -> {label_end}")
        checks.append(
            {
                "signal_date": signal_date,
                "label_end_date": label_end,
                "mature_before_prediction_start": matured,
            }
        )
        if max_end is None or label_end > max_end:
            max_end = label_end
    if max_end is None:
        raise SystemExit("no maturity proof generated")
    return checks, max_end


def build_label_slice(target_dates: list[str], max_required_date: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    target_date_values = ",".join(f"'{item}'" for item in target_dates)
    with closing(duckdb.connect()) as con:
        con.execute(f"ATTACH '{ACTIVE_FEATURE_PATH.as_posix()}' AS feat (READ_ONLY)")
        con.execute(f"ATTACH '{ACTIVE_L2_PATH.as_posix()}' AS l2 (READ_ONLY)")
        feature_keys = con.execute(
            f"""
            SELECT trade_date, stock_code
            FROM feat.{ACTIVE_FEATURE_TABLE}
            WHERE trade_date IN ({target_date_values})
            ORDER BY trade_date, stock_code
            """
        ).fetchdf()
        raw_window = con.execute(
            f"""
            SELECT stock_code, trade_date, open
            FROM l2.{ACTIVE_L2_TABLE}
            WHERE trade_date BETWEEN '{target_dates[0]}' AND '{max_required_date}'
            ORDER BY stock_code, trade_date
            """
        ).fetchdf()

    if feature_keys.empty:
        raise SystemExit("feature key-domain empty for requested mature dates")
    if raw_window.empty:
        raise SystemExit("l2 raw window empty for requested mature dates")

    raw_window["trade_date"] = raw_window["trade_date"].astype(str)
    raw_window["post_open"] = raw_window.groupby("stock_code", sort=False)["open"].shift(-1)
    raw_window["post12_open"] = raw_window.groupby("stock_code", sort=False)["open"].shift(-12)
    raw_target = raw_window[raw_window["trade_date"].isin(target_dates)].copy()

    labels = [LABEL_NAME]
    available = available_labels(list(raw_target.columns), labels)
    if LABEL_NAME not in available:
        raise SystemExit(f"{LABEL_NAME} is not available from raw target columns")
    label_frame = raw_target[required_raw_columns(list(raw_target.columns), labels)].copy()
    label_frame = add_executable_labels(label_frame, labels)
    label_frame = label_frame[["trade_date", "stock_code", LABEL_NAME]].copy()
    label_frame["trade_date"] = label_frame["trade_date"].astype(str)
    label_frame.sort_values(["trade_date", "stock_code"], inplace=True)

    merged = feature_keys.merge(label_frame, on=["trade_date", "stock_code"], how="left", sort=True)
    missing_label_rows = merged[merged[LABEL_NAME].isna()][["trade_date", "stock_code"]]
    if not missing_label_rows.empty:
        sample = missing_label_rows.head(10).to_dict(orient="records")
        raise SystemExit(
            f"label generation produced missing values for feature keys: count={len(missing_label_rows)} sample={sample}"
        )
    extras = label_frame.merge(feature_keys, on=["trade_date", "stock_code"], how="outer", indicator=True)
    extra_rows = extras[extras["_merge"] != "both"]
    if not extra_rows.empty:
        sample = extra_rows.head(10).to_dict(orient="records")
        raise SystemExit(f"feature/label key-domain mismatch: count={len(extra_rows)} sample={sample}")

    daily_rows = (
        merged.groupby("trade_date", sort=True)
        .size()
        .reset_index(name="row_count")
        .to_dict(orient="records")
    )
    return merged, feature_keys, daily_rows


def main() -> int:
    assert_paths()

    feature_before = file_state(ACTIVE_FEATURE_PATH, include_hash=True)
    label_before = file_state(ACTIVE_LABEL_PATH, include_hash=True)
    l2_before = file_state(ACTIVE_L2_PATH, include_hash=True)
    registry_before = file_state(REGISTRY_PATH, include_hash=True)

    target_dates = load_missing_dates()
    open_dates = load_open_dates()
    maturity_checks, max_required_date = maturity_proof(target_dates, open_dates)
    label_slice, feature_keys, daily_rows = build_label_slice(target_dates, max_required_date)

    if int(label_slice.duplicated(["trade_date", "stock_code"]).sum()) != 0:
        raise SystemExit("duplicate trade_date/stock_code keys detected in label slice")
    if int(label_slice["stock_code"].astype(str).str.endswith(".BJ").sum()) != 0:
        raise SystemExit("BJ rows detected in label slice")
    if not np.isfinite(label_slice[LABEL_NAME].to_numpy(dtype="float64")).all():
        raise SystemExit("non-finite label values detected")

    label_slice.to_parquet(SLICE_PARQUET_PATH, index=False)
    with closing(duckdb.connect(str(SLICE_DUCKDB_PATH))) as con:
        con.register("label_slice_frame", label_slice)
        try:
            con.execute(f"CREATE TABLE {SLICE_TABLE} AS SELECT * FROM label_slice_frame")
            slice_schema = schema_snapshot(con, SLICE_TABLE)
            slice_metrics = con.execute(
                f"""
                SELECT
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT stock_code) AS stock_count,
                    COUNT(*) - COUNT(DISTINCT (CAST(trade_date AS VARCHAR) || '|' || stock_code)) AS duplicate_key_rows,
                    SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) AS bj_rows
                FROM {SLICE_TABLE}
                """
            ).fetchone()
        finally:
            con.unregister("label_slice_frame")

    feature_after = file_state(ACTIVE_FEATURE_PATH, include_hash=False)
    label_after = file_state(ACTIVE_LABEL_PATH, include_hash=False)
    l2_after = file_state(ACTIVE_L2_PATH, include_hash=False)
    registry_after = file_state(REGISTRY_PATH, include_hash=False)
    for before, after, name in [
        (feature_before, feature_after, "active_feature"),
        (label_before, label_after, "active_label"),
        (l2_before, l2_after, "active_l2"),
        (registry_before, registry_after, "registry"),
    ]:
        if before["size_bytes"] != after["size_bytes"] or before["mtime_ns"] != after["mtime_ns"]:
            raise SystemExit(f"{name} changed during immutable label slice build")

    digests = digest_records(label_slice, ["trade_date", "stock_code", LABEL_NAME])
    feature_key_digests = digest_records(feature_keys, ["trade_date", "stock_code"])
    parquet_state = file_state(SLICE_PARQUET_PATH, include_hash=True)
    duckdb_state = file_state(SLICE_DUCKDB_PATH, include_hash=True)

    report = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "passed",
        "generated_at": now_iso(),
        "operation": "immutable_mature_10d_label_slice_from_audited_pit_rule",
        "target_label": LABEL_NAME,
        "target_dates": target_dates,
        "target_date_count": len(target_dates),
        "prediction_month_start": PREDICTION_START,
        "maturity_rule": "T+1 open entry and +12 official open-session label_end, strictly before prediction month start",
        "explicit_non_claims": {
            "modified_active_feature": False,
            "modified_active_label": False,
            "modified_registry": False,
            "training_started": False,
            "prediction_started": False,
            "strategy_started": False,
            "returns_summary_started": False,
            "candidate_performance_read": False,
        },
        "bindings": {
            "collective_gap_binding": str(BINDING_PATH),
            "calendar_descriptor": str(CALENDAR_DESCRIPTOR_PATH),
            "calendar_duckdb": f"{CALENDAR_DUCKDB_PATH}::{CALENDAR_TABLE}",
            "active_feature": f"{ACTIVE_FEATURE_PATH}::{ACTIVE_FEATURE_TABLE}",
            "active_label": f"{ACTIVE_LABEL_PATH}::{ACTIVE_LABEL_TABLE}",
            "active_l2": f"{ACTIVE_L2_PATH}::{ACTIVE_L2_TABLE}",
        },
        "inputs": {
            "active_feature_before": feature_before,
            "active_feature_after": feature_after,
            "active_label_before": label_before,
            "active_label_after": label_after,
            "active_l2_before": l2_before,
            "active_l2_after": l2_after,
            "registry_before": registry_before,
            "registry_after": registry_after,
            "max_required_future_trade_date": max_required_date,
        },
        "maturity_checks": maturity_checks,
        "slice_asset": {
            "duckdb_path": str(SLICE_DUCKDB_PATH),
            "duckdb_table": SLICE_TABLE,
            "duckdb_sha256": duckdb_state["sha256"],
            "parquet_path": str(SLICE_PARQUET_PATH),
            "parquet_sha256": parquet_state["sha256"],
            "schema": slice_schema,
            "schema_hash": schema_hash(slice_schema),
            "column_count": len(slice_schema),
            "row_count": int(slice_metrics[0]),
            "stock_count": int(slice_metrics[1]),
            "duplicate_key_rows": int(slice_metrics[2]),
            "bj_rows": int(slice_metrics[3]),
            "daily_rows": daily_rows,
            "null_summary": null_summary(label_slice),
            "finite_values": True,
            "digests": digests,
            "feature_key_digest_sha256": feature_key_digests["key_digest_sha256"],
            "key_domain_matches_feature": digests["key_digest_sha256"] == feature_key_digests["key_digest_sha256"],
        },
        "handoff_scope": {
            "allow_data_integration_binding": True,
            "allow_next_layer_continue": False,
            "production_write_allowed": False,
            "strategy_or_model_allowed": False,
        },
    }

    contract = build_layer_handoff_contract(
        workflow_run_id=RUN_ID,
        layer="L3",
        target_trade_date="20260715",
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=[
            f"{ACTIVE_FEATURE_PATH}::{ACTIVE_FEATURE_TABLE}",
            f"{ACTIVE_LABEL_PATH}::{ACTIVE_LABEL_TABLE}",
            f"{ACTIVE_L2_PATH}::{ACTIVE_L2_TABLE}",
            f"{CALENDAR_DUCKDB_PATH}::{CALENDAR_TABLE}",
            str(BINDING_PATH),
        ],
        active_output_assets=[
            f"{SLICE_DUCKDB_PATH}::{SLICE_TABLE}",
            str(SLICE_PARQUET_PATH),
        ],
        gate_checks=[
            {"name": "target_date_count_20", "passed": len(target_dates) == 20},
            {"name": "duplicate_zero", "passed": int(slice_metrics[2]) == 0},
            {"name": "bj_zero", "passed": int(slice_metrics[3]) == 0},
            {"name": "feature_key_domain_equal", "passed": digests["key_digest_sha256"] == feature_key_digests["key_digest_sha256"]},
            {"name": "finite_values", "passed": True},
            {"name": "active_feature_unchanged", "passed": feature_before["sha256"] == feature_after["sha256"]},
            {"name": "active_label_unchanged", "passed": label_before["sha256"] == label_after["sha256"]},
            {"name": "active_l2_unchanged", "passed": l2_before["sha256"] == l2_after["sha256"]},
            {"name": "registry_unchanged", "passed": registry_before["sha256"] == registry_after["sha256"]},
        ],
        handoff_constraints=[
            "this asset is a read-only immutable label slice for 20 mature signal dates only",
            "it is computed from audited PIT label formula and official open-session calendar evidence",
            "it does not claim byte-equivalence to any lost historical candidate",
            "active feature, active label, registry and production assets were not modified",
            "consumer may use only for blind binding gap closure and downstream readonly verification",
        ],
        evidence_paths=[
            str(BINDING_PATH),
            str(CALENDAR_DESCRIPTOR_PATH),
            str(REPORT_PATH),
            str(HANDOFF_PATH),
        ],
        boundaries={
            "read_only_inputs": True,
            "no_active_write": True,
            "no_label_write": True,
            "no_registry_change": True,
            "no_training": True,
            "no_prediction": True,
            "no_strategy": True,
        },
        layer_payload=report,
    )
    errors = validate_layer_handoff_contract(contract, expected_layer="L3", expected_owner_agent="factor-agent")
    if errors:
        raise SystemExit("contract validation failed: " + "; ".join(errors))

    handoff = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "ready_for_audit_review",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "label_slice_asset": f"{SLICE_DUCKDB_PATH}::{SLICE_TABLE}",
        "label_slice_parquet": str(SLICE_PARQUET_PATH),
        "asset_digest": {
            "duckdb_sha256": duckdb_state["sha256"],
            "parquet_sha256": parquet_state["sha256"],
            "row_digest_sha256": digests["row_digest_sha256"],
            "key_digest_sha256": digests["key_digest_sha256"],
            "value_digest_sha256": digests["value_digest_sha256"],
        },
        "message_for_data_integration": (
            "Immutable mature 10D label slice for 20260617..20260715 built from current authoritative L2 plus frozen PIT label formula; "
            "use only for blind-binding gap closure after readonly audit."
        ),
    }

    audit = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "passed",
        "generated_at": now_iso(),
        "checks": {
            "target_dates_20": len(target_dates) == 20,
            "row_count_matches_feature_keys": int(slice_metrics[0]) == int(feature_keys.shape[0]),
            "feature_key_domain_equal": digests["key_digest_sha256"] == feature_key_digests["key_digest_sha256"],
            "duplicate_zero": int(slice_metrics[2]) == 0,
            "bj_zero": int(slice_metrics[3]) == 0,
            "finite_values": True,
            "active_feature_unchanged": feature_before["sha256"] == feature_after["sha256"],
            "active_label_unchanged": label_before["sha256"] == label_after["sha256"],
            "active_l2_unchanged": l2_before["sha256"] == l2_after["sha256"],
            "registry_unchanged": registry_before["sha256"] == registry_after["sha256"],
        },
    }
    if not all(audit["checks"].values()):
        raise SystemExit(f"audit checks failed: {audit['checks']}")

    write_json(REPORT_PATH, report)
    write_json(WORKFLOW_CONTRACT_PATH, contract)
    write_json(HANDOFF_PATH, handoff)
    write_json(AUDIT_PATH, audit)

    print(
        json.dumps(
            {
                "status": "passed",
                "run_id": RUN_ID,
                "duckdb_path": str(SLICE_DUCKDB_PATH),
                "parquet_path": str(SLICE_PARQUET_PATH),
                "row_count": int(slice_metrics[0]),
                "stock_count": int(slice_metrics[1]),
                "row_digest_sha256": digests["row_digest_sha256"],
                "value_digest_sha256": digests["value_digest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
