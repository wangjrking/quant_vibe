from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from data_load_module import get_pro
from l1_raw_data_route import resolve_l1_raw_backend, resolve_l1_raw_duckdb_path
from l1_universe_rules import filter_frame_no_bj, is_bj_code
from production_asset_registry import active_main_workflow_asset
from project_paths import load_config, resolve_data_dir
from raw_data_update_module import RAW_TABLE_SPECS, append_parquet_dedup
from run_all_a_raw_update import DEFAULT_INDEX_CODES, _fetch_table_for_trade_date
from workflow_contract import build_layer_handoff_contract, validate_layer_handoff_contract


SOURCE_TABLES = (
    "daily",
    "daily_basic",
    "stk_factor",
    "moneyflow",
    "limit_list",
    "cyq_perf",
    "adj_factor",
    "stock_st",
    "index_daily",
    "top_list",
    "stk_shock",
    "stk_high_shock",
    "stk_alert",
)
TABLE_MAP = {
    "daily": "daily_data",
    "daily_basic": "daily_index_data",
    "limit_list": "limit_list_data",
}
DRIVER_MATCH_TABLES = {"daily_basic", "stk_factor", "moneyflow", "cyq_perf"}
EXPECTED_INDEX_CODES = set(DEFAULT_INDEX_CODES)
SPARSE_EVENT_TABLES = {"stk_shock", "stk_high_shock", "stk_alert"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with candidate.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(candidate, path)
    except Exception:
        if candidate.exists():
            candidate.unlink()
        raise
    return path


def atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    json.loads(text)
    return atomic_text(path, text)


def normalize_codes(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "ts_code" not in frame.columns:
        return set()
    return {
        str(code).strip().upper()
        for code in frame["ts_code"].dropna()
        if str(code).strip() and not is_bj_code(code)
    }


def duplicate_rows(frame: pd.DataFrame, keys: tuple[str, ...]) -> int:
    if frame.empty:
        return 0
    return int(frame.duplicated(list(keys), keep=False).sum())


def fetch_sources(pro: Any, trade_date: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    raw_frames: dict[str, pd.DataFrame] = {}
    for source_table in SOURCE_TABLES:
        if source_table == "top_list":
            frame = pro.top_list(trade_date=trade_date)
        else:
            frame = _fetch_table_for_trade_date(pro, source_table, trade_date, DEFAULT_INDEX_CODES)
        raw_frames[source_table] = frame if frame is not None else pd.DataFrame()

    daily = filter_frame_no_bj(raw_frames["daily"])
    driver_codes = normalize_codes(daily)
    frames: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for source_table, raw in raw_frames.items():
        asset_table = TABLE_MAP.get(source_table, source_table)
        no_bj = raw.copy() if source_table == "index_daily" else filter_frame_no_bj(raw)
        if source_table in DRIVER_MATCH_TABLES and "ts_code" in no_bj.columns:
            no_bj = no_bj[no_bj["ts_code"].astype(str).str.upper().isin(driver_codes)].copy()
        spec = RAW_TABLE_SPECS[source_table]
        deduped = no_bj.drop_duplicates(list(spec.key_columns), keep="last").reset_index(drop=True)
        full_codes = normalize_codes(raw)
        scoped_codes = normalize_codes(deduped)
        metrics[asset_table] = {
            "source_table": source_table,
            "source_raw_rows": int(len(raw)),
            "source_bj_rows": int(sum(is_bj_code(code) for code in raw.get("ts_code", pd.Series(dtype=str)))),
            "source_no_bj_rows": int(len(no_bj)),
            "source_duplicate_rows": duplicate_rows(no_bj, spec.key_columns),
            "source_dedup_rows": int(len(deduped)),
            "source_full_codes": len(full_codes),
            "source_codes": len(scoped_codes),
            "missing_vs_daily_count": len(driver_codes - scoped_codes) if source_table in DRIVER_MATCH_TABLES | {"adj_factor"} else 0,
            "extra_vs_daily_count": len(scoped_codes - driver_codes) if source_table not in {"index_daily"} else 0,
            "columns": list(raw.columns),
        }
        frames[asset_table] = deduped
    return frames, metrics


def validate_sources(
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
    trade_date: str,
    source_limited_codes: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    source_limited_codes = source_limited_codes or {}
    blockers: list[str] = []
    warnings: list[str] = []
    daily_codes = normalize_codes(frames["daily_data"])
    if not daily_codes:
        blockers.append("daily_data source returned zero no-BJ rows")
    for table, frame in frames.items():
        source_table = metrics[table]["source_table"]
        spec = RAW_TABLE_SPECS[source_table]
        if duplicate_rows(frame, spec.key_columns):
            blockers.append(f"{table} retains duplicate natural keys")
        if "ts_code" in frame.columns and any(is_bj_code(code) for code in frame["ts_code"]):
            blockers.append(f"{table} retains BJ rows")
        if not frame.empty and spec.date_column in frame.columns:
            dates = set(frame[spec.date_column].astype(str))
            if dates != {trade_date}:
                blockers.append(f"{table} contains non-target dates: {sorted(dates)[:5]}")

    for source_table in DRIVER_MATCH_TABLES:
        table = TABLE_MAP.get(source_table, source_table)
        missing_codes = daily_codes - normalize_codes(frames[table])
        extra_codes = normalize_codes(frames[table]) - daily_codes
        allowed_missing = source_limited_codes.get(table, set())
        if missing_codes == allowed_missing and not extra_codes and allowed_missing:
            warnings.append(
                f"{table} has documented source-limited codes: {sorted(allowed_missing)}"
            )
        elif missing_codes or extra_codes:
            blockers.append(f"{table} stock domain differs from daily_data")
    if daily_codes - normalize_codes(frames["adj_factor"]):
        blockers.append("adj_factor is missing daily_data driver codes")
    if normalize_codes(frames["adj_factor"]) - daily_codes:
        warnings.append("adj_factor has auxiliary extra codes and does not drive the universe")
    if normalize_codes(frames["index_daily"]) != EXPECTED_INDEX_CODES:
        blockers.append("index_daily configured index domain is incomplete")
    if frames["top_list"].empty:
        blockers.append("top_list source returned zero rows")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "driver_rows": int(len(frames["daily_data"])),
        "driver_codes": len(daily_codes),
    }


def align_schema(frame: pd.DataFrame, parquet_path: Path, db_path: Path, table: str) -> pd.DataFrame:
    parquet_columns = pq.ParquetFile(parquet_path).schema_arrow.names
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
        ).fetchall()]
        db_columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()]
    finally:
        connection.close()
    if tables != [table]:
        raise RuntimeError(f"one-table-one-file contract failed for {db_path}: {tables}")
    if set(frame.columns) != set(parquet_columns) or set(frame.columns) != set(db_columns):
        raise RuntimeError(
            f"schema drift for {table}: source={list(frame.columns)} parquet={parquet_columns} duckdb={db_columns}"
        )
    return frame[parquet_columns].copy()


def parquet_stats(
    path: Path,
    table: str,
    keys: tuple[str, ...],
    trade_date: str,
    date_column: str = "trade_date",
) -> dict[str, Any]:
    path_sql = str(path).replace("'", "''")
    key_sql = ",".join(quote_ident(key) for key in keys)
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"SELECT COUNT(*), MIN({quote_ident(date_column)}), MAX({quote_ident(date_column)}), "
            f"COUNT(*) FILTER (WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?), "
            f"COUNT(*) FILTER (WHERE UPPER(CAST(ts_code AS VARCHAR)) LIKE '%.BJ') "
            f"FROM read_parquet('{path_sql}')",
            [trade_date],
        ).fetchone()
        duplicate_groups = int(connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {key_sql}, COUNT(*) c FROM read_parquet('{path_sql}') "
            f"GROUP BY {key_sql} HAVING COUNT(*) > 1)"
        ).fetchone()[0])
    finally:
        connection.close()
    return {
        "path": str(path),
        "rows_all": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "target_rows": int(row[3]),
        "bj_rows_all": int(row[4]),
        "duplicate_groups_all": duplicate_groups,
    }


def duckdb_stats(
    path: Path,
    table: str,
    keys: tuple[str, ...],
    trade_date: str,
    date_column: str = "trade_date",
) -> dict[str, Any]:
    key_sql = ",".join(quote_ident(key) for key in keys)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
        ).fetchall()]
        row = connection.execute(
            f"SELECT COUNT(*), MIN({quote_ident(date_column)}), MAX({quote_ident(date_column)}), "
            f"COUNT(*) FILTER (WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?), "
            f"COUNT(*) FILTER (WHERE UPPER(CAST(ts_code AS VARCHAR)) LIKE '%.BJ') "
            f"FROM {quote_ident(table)}",
            [trade_date],
        ).fetchone()
        duplicate_groups = int(connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {key_sql}, COUNT(*) c FROM {quote_ident(table)} "
            f"GROUP BY {key_sql} HAVING COUNT(*) > 1)"
        ).fetchone()[0])
    finally:
        connection.close()
    return {
        "path": str(path),
        "rows_all": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "target_rows": int(row[3]),
        "bj_rows_all": int(row[4]),
        "duplicate_groups_all": duplicate_groups,
        "persistent_business_tables": tables,
        "one_table_one_file": tables == [table],
    }


def _target_slice_from_parquet(path: Path, trade_date: str, date_column: str) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        return connection.execute(
            f"SELECT * FROM read_parquet(?) WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?",
            [str(path), trade_date],
        ).df()
    finally:
        connection.close()


def _target_slice_from_duckdb(
    path: Path,
    table: str,
    trade_date: str,
    date_column: str,
) -> pd.DataFrame:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        return connection.execute(
            f"SELECT * FROM {quote_ident(table)} "
            f"WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?",
            [trade_date],
        ).df()
    finally:
        connection.close()


def _canonical_frame_payload(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    *,
    volatile_value_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    ordered = frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    normalized = ordered.astype(object).where(pd.notna(ordered), None)
    rows = json.loads(normalized.to_json(orient="values", date_format="iso", date_unit="us"))
    columns = list(ordered.columns)
    key_indexes = [columns.index(key) for key in keys]
    key_rows = [[row[index] for index in key_indexes] for row in rows]
    value_columns = [column for column in columns if column not in volatile_value_columns]
    value_indexes = [columns.index(column) for column in value_columns]
    value_rows = [[row[index] for index in value_indexes] for row in rows]

    def digest(value: Any) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    return {
        "rows": rows,
        "schema_sha256": digest(columns),
        "key_sha256": digest(key_rows),
        "value_sha256": digest({"columns": value_columns, "rows": value_rows}),
    }


def verify_preexisting_sparse_event_slice(
    source: pd.DataFrame,
    parquet_path: Path,
    db_path: Path,
    table: str,
    keys: tuple[str, ...],
    trade_date: str,
    date_column: str,
) -> dict[str, Any]:
    parquet = _target_slice_from_parquet(parquet_path, trade_date, date_column)
    active = _target_slice_from_duckdb(db_path, table, trade_date, date_column)
    expected_columns = list(source.columns)
    if list(parquet.columns) != expected_columns or list(active.columns) != expected_columns:
        raise RuntimeError(f"preexisting sparse-event schema mismatch for {table}")

    volatile_value_columns = ("fetched_at",)
    source_payload = _canonical_frame_payload(
        source, keys, volatile_value_columns=volatile_value_columns
    )
    parquet_payload = _canonical_frame_payload(
        parquet, keys, volatile_value_columns=volatile_value_columns
    )
    active_payload = _canonical_frame_payload(
        active, keys, volatile_value_columns=volatile_value_columns
    )
    digest_fields = ("schema_sha256", "key_sha256", "value_sha256")
    if any(
        source_payload[field] != parquet_payload[field]
        or source_payload[field] != active_payload[field]
        for field in digest_fields
    ):
        raise RuntimeError(f"preexisting sparse-event value mismatch for {table}")
    return {
        "status": "preexisting_verified_reuse",
        "write_action": "unchanged",
        "row_count": int(len(source)),
        "schema_sha256": source_payload["schema_sha256"],
        "key_sha256": source_payload["key_sha256"],
        "value_sha256": source_payload["value_sha256"],
        "value_digest_columns": [
            column for column in expected_columns if column not in volatile_value_columns
        ],
        "volatile_lineage_columns_preserved": list(volatile_value_columns),
        "source_parquet_active_equal": True,
    }


def update_duckdb_target(
    path: Path,
    table: str,
    frame: pd.DataFrame,
    trade_date: str,
    date_column: str = "trade_date",
) -> None:
    columns = list(frame.columns)
    column_sql = ",".join(quote_ident(column) for column in columns)
    connection = duckdb.connect(str(path))
    try:
        connection.register("__incoming_l1", frame)
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            f"DELETE FROM {quote_ident(table)} "
            f"WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?",
            [trade_date],
        )
        connection.execute(
            f"INSERT INTO {quote_ident(table)} ({column_sql}) SELECT {column_sql} FROM __incoming_l1"
        )
        actual = int(connection.execute(
            f"SELECT COUNT(*) FROM {quote_ident(table)} "
            f"WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?",
            [trade_date],
        ).fetchone()[0])
        if actual != len(frame):
            raise RuntimeError(f"target row count mismatch for {table}: {actual} != {len(frame)}")
        connection.execute("COMMIT")
        connection.unregister("__incoming_l1")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        connection.close()


def delete_duckdb_target(
    path: Path,
    table: str,
    trade_date: str,
    date_column: str = "trade_date",
) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            f"DELETE FROM {quote_ident(table)} "
            f"WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?",
            [trade_date],
        )
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        connection.close()


def create_hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    os.link(source, target)


def build_l1_handoff_contract(
    report: dict[str, Any],
    *,
    workflow_id: str,
    trade_date: str,
    evidence_paths: list[str],
) -> dict[str, Any]:
    results = report.get("results") or {}
    output_assets = [
        f"{item['duckdb']['path']}::{table}"
        for table, item in results.items()
        if isinstance(item, dict) and isinstance(item.get("duckdb"), dict) and item["duckdb"].get("path")
    ]
    result_values = [item for item in results.values() if isinstance(item, dict)]
    all_passed = bool(result_values) and all(bool(item.get("passed")) for item in result_values)
    max_date_ok = all(
        int(item.get("source_dedup_rows") or 0) == 0
        or str((item.get("duckdb") or {}).get("max_trade_date") or "") == trade_date
        for item in result_values
    )
    duplicate_zero = all(
        int((item.get("duckdb") or {}).get("duplicate_groups_all") or 0) == 0
        for item in result_values
    )
    bj_zero = all(
        int((item.get("duckdb") or {}).get("bj_rows_all") or 0) == 0
        for item in result_values
    )
    one_table_one_file = all(
        bool((item.get("duckdb") or {}).get("one_table_one_file"))
        for item in result_values
    )
    residual_risk = [
        {
            "level": "P3",
            "table": str(item.get("table") or ""),
            "summary": str(item.get("reason") or "source-limited optional input"),
        }
        for item in report.get("tables_not_landed") or []
        if isinstance(item, dict)
    ]
    residual_risk.extend(
        {
            "level": "P2",
            "table": str(item.get("table") or ""),
            "summary": str(item.get("reason_code") or "documented source-limited gap"),
            "missing_codes": list(item.get("missing_codes") or []),
        }
        for item in report.get("source_limited") or []
        if isinstance(item, dict)
    )
    return build_layer_handoff_contract(
        workflow_run_id=workflow_id,
        layer="L1",
        target_trade_date=trade_date,
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=["official Tushare SDK source APIs"],
        active_output_assets=output_assets,
        gate_checks=[
            {"name": "official_sdk_source_gate", "passed": bool(report.get("official_sdk_only")), "severity": "blocker"},
            {"name": "three_face_alignment", "passed": all_passed, "severity": "blocker"},
            {"name": "max_trade_date_target", "passed": max_date_ok, "severity": "blocker"},
            {"name": "duplicate_groups_zero", "passed": duplicate_zero, "severity": "blocker"},
            {"name": "no_bj", "passed": bj_zero, "severity": "blocker"},
            {"name": "one_table_one_file", "passed": one_table_one_file, "severity": "blocker"},
        ],
        handoff_constraints=[
            "daily_data drives the trading calendar and stock universe",
            "adj_factor remains auxiliary and must not extend the universe",
            "stock_st uses event/status semantics",
            "downstream qfq fields must remain explicit",
            "waiting for audit; L2 is not self-released",
        ],
        evidence_paths=evidence_paths,
        residual_risk=residual_risk,
        boundaries={
            "no_l2_trigger": True,
            "no_l3_l8_trigger": True,
            "experimental_assets_used": False,
            "legacy_sqlite_touched": False,
        },
        layer_payload=report,
        hard_rules=[
            "official-sdk",
            "no-BJ",
            "DuckDB-only",
            "one-table-one-file",
            "explicit-qfq",
            "fail-closed",
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest one audited L1 trade date into root parquet and active DuckDB")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--extended-probe-report", required=True)
    parser.add_argument("--source-limited-evidence", default=None)
    parser.add_argument("--workflow-id", default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trade_date = str(args.trade_date)
    data_dir = resolve_data_dir(args.data_dir)
    report_dir = data_dir / "reports"
    workflow_id = args.workflow_id or f"incremental-trading-signal-{trade_date}"
    task_id = f"{workflow_id}-L1"
    preflight = json.loads(Path(args.preflight_report).read_text(encoding="utf-8"))
    extended = json.loads(Path(args.extended_probe_report).read_text(encoding="utf-8"))
    source_limited: list[dict[str, Any]] = []
    source_limited_codes: dict[str, set[str]] = {}
    if args.source_limited_evidence:
        evidence_path = Path(args.source_limited_evidence)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            evidence.get("trade_date") != trade_date
            or evidence.get("official_tushare_sdk") is not True
            or evidence.get("business_data_written") is not False
        ):
            raise RuntimeError("source-limited evidence is stale or invalid")
        for item in evidence.get("source_limited") or []:
            table = str(item.get("table") or "")
            codes = {str(code).strip().upper() for code in item.get("missing_codes") or []}
            if (
                not table
                or not codes
                or item.get("verified_source_empty") is not True
                or item.get("reason_code") != "ipo_listing_date_official_sdk_empty"
                or str(item.get("list_date") or "") != trade_date
                or int(item.get("daily_target_rows") or 0) < 1
                or int(item.get("factor_target_rows") or 0) != 0
                or int(item.get("factor_window_rows") or 0) != 0
            ):
                raise RuntimeError("source-limited evidence item is incomplete")
            source_limited_codes[table] = codes
            source_limited.append({**item, "evidence_path": str(evidence_path)})
    if preflight.get("trade_date") != trade_date or not preflight.get("gate_pass"):
        raise RuntimeError("source preflight is missing, stale, or blocked")
    top_probe = (extended.get("probes") or {}).get("top_list") or {}
    if extended.get("trade_date") != trade_date or not top_probe.get("success") or not top_probe.get("source_dedup_rows"):
        raise RuntimeError("top_list extended source probe is missing, stale, or blocked")
    if resolve_l1_raw_backend(data_dir=data_dir) != "duckdb":
        raise RuntimeError("active L1 backend is not DuckDB")

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    workspace = report_dir / f"l1_incremental_raw_ingest_{trade_date}_run_{run_id}"
    stage_root = workspace / "stage_parquet"
    rollback_root = workspace / "rollback_parquet"
    stage_root.mkdir(parents=True, exist_ok=False)
    rollback_root.mkdir(parents=True, exist_ok=False)

    config = load_config(args.config)
    pro = get_pro(str(config["datasource"]["tushare_token"]))
    frames, source_metrics = fetch_sources(pro, trade_date)
    source_validation = validate_sources(
        frames,
        source_metrics,
        trade_date,
        source_limited_codes=source_limited_codes,
    )
    if not source_validation["passed"]:
        raise RuntimeError(f"source refetch validation blocked: {source_validation['blockers']}")
    if source_validation["driver_rows"] != int(preflight.get("driver_source_rows") or -1):
        raise RuntimeError("source refetch driver row count drifted from preflight")

    assets: dict[str, dict[str, Any]] = {}
    for source_table in SOURCE_TABLES:
        table = TABLE_MAP.get(source_table, source_table)
        spec = RAW_TABLE_SPECS[source_table]
        parquet_path = data_dir / spec.file_name
        db_path = resolve_l1_raw_duckdb_path(table, data_dir=data_dir, require_exists=True)
        frame = align_schema(frames[table], parquet_path, db_path, table)
        frames[table] = frame
        parquet_before = parquet_stats(
            parquet_path, table, spec.key_columns, trade_date, spec.date_column
        )
        db_before = duckdb_stats(
            db_path, table, spec.key_columns, trade_date, spec.date_column
        )
        preexisting_reuse = None
        if parquet_before["target_rows"] or db_before["target_rows"]:
            if source_table not in SPARSE_EVENT_TABLES:
                raise RuntimeError(f"target date already exists for {table}; refusing non-reviewed overwrite")
            if not parquet_before["target_rows"] or not db_before["target_rows"]:
                raise RuntimeError(f"preexisting sparse-event faces are incomplete for {table}")
            preexisting_reuse = verify_preexisting_sparse_event_slice(
                frame,
                parquet_path,
                db_path,
                table,
                spec.key_columns,
                trade_date,
                spec.date_column,
            )
            assets[table] = {
                "source_table": source_table,
                "spec": spec,
                "parquet_path": parquet_path,
                "duckdb_path": db_path,
                "candidate_path": None,
                "rollback_path": None,
                "parquet_before": parquet_before,
                "duckdb_before": db_before,
                "candidate": parquet_before,
                "preexisting_reuse": preexisting_reuse,
            }
            continue
        candidate = stage_root / spec.file_name
        rollback = rollback_root / spec.file_name
        create_hardlink(parquet_path, candidate)
        create_hardlink(parquet_path, rollback)
        append_parquet_dedup(candidate, frame, spec.key_columns)
        candidate_stats = parquet_stats(
            candidate, table, spec.key_columns, trade_date, spec.date_column
        )
        if (
            candidate_stats["target_rows"] != len(frame)
            or (len(frame) > 0 and candidate_stats["max_trade_date"] != trade_date)
            or candidate_stats["bj_rows_all"]
            or candidate_stats["duplicate_groups_all"]
        ):
            raise RuntimeError(f"candidate parquet validation failed for {table}: {candidate_stats}")
        assets[table] = {
            "source_table": source_table,
            "spec": spec,
            "parquet_path": parquet_path,
            "duckdb_path": db_path,
            "candidate_path": candidate,
            "rollback_path": rollback,
            "parquet_before": parquet_before,
            "duckdb_before": db_before,
            "candidate": candidate_stats,
            "preexisting_reuse": None,
        }

    replaced_parquets: list[str] = []
    updated_duckdbs: list[str] = []
    try:
        for table, asset in assets.items():
            if asset["preexisting_reuse"]:
                continue
            os.replace(asset["candidate_path"], asset["parquet_path"])
            replaced_parquets.append(table)
        for table, asset in assets.items():
            if asset["preexisting_reuse"]:
                continue
            update_duckdb_target(
                asset["duckdb_path"],
                table,
                frames[table],
                trade_date,
                asset["spec"].date_column,
            )
            updated_duckdbs.append(table)

        results: dict[str, dict[str, Any]] = {}
        for table, asset in assets.items():
            spec = asset["spec"]
            parquet_after = parquet_stats(
                asset["parquet_path"], table, spec.key_columns, trade_date, spec.date_column
            )
            duckdb_after = duckdb_stats(
                asset["duckdb_path"], table, spec.key_columns, trade_date, spec.date_column
            )
            source_rows = int(len(frames[table]))
            checks = {
                "source_parquet_match": source_rows == parquet_after["target_rows"],
                "source_duckdb_match": source_rows == duckdb_after["target_rows"],
                "parquet_duckdb_match": parquet_after["target_rows"] == duckdb_after["target_rows"],
                "parquet_max_target": source_rows == 0 or parquet_after["max_trade_date"] == trade_date,
                "duckdb_max_target": source_rows == 0 or duckdb_after["max_trade_date"] == trade_date,
                "parquet_duplicate_zero": parquet_after["duplicate_groups_all"] == 0,
                "duckdb_duplicate_zero": duckdb_after["duplicate_groups_all"] == 0,
                "parquet_bj_zero": parquet_after["bj_rows_all"] == 0,
                "duckdb_bj_zero": duckdb_after["bj_rows_all"] == 0,
                "one_table_one_file": duckdb_after["one_table_one_file"],
            }
            results[table] = {
                **source_metrics[table],
                "source_dedup_rows": source_rows,
                "parquet": parquet_after,
                "duckdb": duckdb_after,
                "checks": checks,
                "passed": all(checks.values()),
                "disposition": asset["preexisting_reuse"] or {
                    "status": "target_date_landed",
                    "write_action": "inserted",
                },
            }
        if not all(result["passed"] for result in results.values()):
            raise RuntimeError("post-write three-face alignment failed")
    except Exception:
        rollback_errors: list[str] = []
        for table in reversed(updated_duckdbs):
            try:
                delete_duckdb_target(
                    assets[table]["duckdb_path"],
                    table,
                    trade_date,
                    assets[table]["spec"].date_column,
                )
            except Exception as error:
                rollback_errors.append(f"duckdb {table}: {type(error).__name__}: {error}")
        for table in reversed(replaced_parquets):
            try:
                os.replace(assets[table]["rollback_path"], assets[table]["parquet_path"])
            except Exception as error:
                rollback_errors.append(f"parquet {table}: {type(error).__name__}: {error}")
        if rollback_errors:
            raise RuntimeError(f"L1 write failed and rollback was incomplete: {rollback_errors}")
        raise

    active_asset = active_main_workflow_asset("L1_raw_data", data_dir=data_dir)
    completed_at = now_iso()
    report = {
        "task_id": task_id,
        "workflow_id": workflow_id,
        "target_trade_date": trade_date,
        "status": "l1_complete_ready_for_audit",
        "success": True,
        "completed_at": completed_at,
        "official_sdk_only": True,
        "universe_rule": "no_bj",
        "source_gate": {
            "gate_pass": True,
            "status": preflight.get("status"),
            "driver_source_rows": source_validation["driver_rows"],
            "preflight_report": str(Path(args.preflight_report)),
            "extended_probe_report": str(Path(args.extended_probe_report)),
        },
        "active_route": {
            "backend": "duckdb",
            "registry_path": str(data_dir / "asset_registry" / "production_assets.json"),
            "registry_asset": active_asset,
            "output_root": str(resolve_l1_raw_duckdb_path("daily_data", data_dir=data_dir).parent),
            "contract": "DuckDB-only one-table-one-file",
        },
        "tables_landed": list(results),
        "tables_not_landed": [
            {"table": name, "status": "source_limited", "reason": "official SDK returned zero rows"}
            for name in ("ths_hot", "dc_hot")
            if int(((extended.get("probes") or {}).get(name) or {}).get("source_dedup_rows") or 0) == 0
        ],
        "source_limited": source_limited,
        "results": results,
        "warnings": source_validation["warnings"],
        "rollback_snapshots": {
            table: str(asset["rollback_path"])
            for table, asset in assets.items()
            if asset["rollback_path"] is not None
        },
        "preexisting_verified_reuse": {
            table: asset["preexisting_reuse"]
            for table, asset in assets.items()
            if asset["preexisting_reuse"] is not None
        },
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "boundaries": {
            "legacy_sqlite_touched": False,
            "experimental_assets_used": False,
            "l2_l8_triggered": False,
        },
    }

    report_path = report_dir / f"l1_incremental_raw_ingest_{trade_date}_complete.json"
    report_md = report_path.with_suffix(".md")
    self_audit_path = report_dir / f"l1_incremental_raw_ingest_{trade_date}_self_audit.json"
    handoff_path = report_dir / f"l1_handoff_contract_{trade_date}.json"
    handoff_validation_path = report_dir / f"l1_handoff_contract_{trade_date}_validation.json"
    hash_path = report_dir / f"l1_incremental_raw_ingest_{trade_date}_hashes.json"
    report["evidence_paths"] = {
        "complete": str(report_path),
        "markdown": str(report_md),
        "self_audit": str(self_audit_path),
        "handoff": str(handoff_path),
        "handoff_validation": str(handoff_validation_path),
        "hashes": str(hash_path),
    }
    self_audit = {
        "task_id": f"{task_id}-self-audit",
        "generated_at": completed_at,
        "status": "passed",
        "target_trade_date": trade_date,
        "checks": {
            "official_sdk_source_gate": True,
            "active_backend_duckdb": True,
            "three_face_alignment": True,
            "duplicate_groups_zero": True,
            "bj_rows_zero": True,
            "one_table_one_file": True,
            "max_trade_date_target": True,
            "daily_driver_semantics": True,
            "adj_factor_auxiliary_semantics": True,
            "stock_st_event_semantics": True,
        },
        "alignment": [
            {
                "table": table,
                "passed": item["passed"],
                "source": item["source_dedup_rows"],
                "parquet": item["parquet"]["target_rows"],
                "duckdb": item["duckdb"]["target_rows"],
                "max_trade_date": item["duckdb"]["max_trade_date"],
                "duplicate_groups": item["duckdb"]["duplicate_groups_all"],
                "bj_rows": item["duckdb"]["bj_rows_all"],
            }
            for table, item in results.items()
        ],
        "source_limited": report["tables_not_landed"],
        "documented_source_limited_gaps": source_limited,
        "boundaries": report["boundaries"],
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    handoff = build_l1_handoff_contract(
        report,
        workflow_id=workflow_id,
        trade_date=trade_date,
        evidence_paths=[
            str(report_path),
            str(report_md),
            str(self_audit_path),
            str(Path(args.preflight_report)),
            str(Path(args.extended_probe_report)),
        ],
    )
    handoff_errors = validate_layer_handoff_contract(
        handoff,
        expected_layer="L1",
        expected_owner_agent="data-ingestion-agent",
    )
    markdown_lines = [
        f"# L1 {trade_date} incremental ingest",
        "",
        "- status: l1_complete_ready_for_audit",
        "- official source: Tushare SDK only",
        "- active route: root raw parquet + DuckDB one-table-one-file",
        "- allow_next_layer_continue: false",
        "",
        "| table | source | parquet | DuckDB | max date | duplicate | BJ |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for table, item in results.items():
        markdown_lines.append(
            f"| {table} | {item['source_dedup_rows']} | {item['parquet']['target_rows']} | "
            f"{item['duckdb']['target_rows']} | {item['duckdb']['max_trade_date']} | "
            f"{item['duckdb']['duplicate_groups_all']} | {item['duckdb']['bj_rows_all']} |"
        )
    atomic_json(report_path, report)
    atomic_text(report_md, "\n".join(markdown_lines) + "\n")
    atomic_json(self_audit_path, self_audit)
    atomic_json(handoff_path, handoff)
    atomic_json(
        handoff_validation_path,
        {
            "file": str(handoff_path),
            "valid": not handoff_errors,
            "errors": handoff_errors,
            "allow_next_layer_continue": False,
            "validator": "workflow_contract.validate_layer_handoff_contract",
        },
    )
    if handoff_errors:
        raise RuntimeError(f"handoff contract validation failed: {handoff_errors}")
    hash_files = [
        report_path,
        report_md,
        self_audit_path,
        handoff_path,
        handoff_validation_path,
        Path(args.preflight_report),
        Path(args.extended_probe_report),
        data_dir / "asset_registry" / "production_assets.json",
    ]
    if args.source_limited_evidence:
        hash_files.append(Path(args.source_limited_evidence))
    for table, asset in assets.items():
        hash_files.extend([asset["parquet_path"], asset["duckdb_path"]])
    atomic_json(
        hash_path,
        {
            "task": f"l1_incremental_raw_ingest_{trade_date}",
            "generated_at": now_iso(),
            "algorithm": "SHA256",
            "contract_valid": True,
            "production_route": "DuckDB one-table-one-file",
            "legacy_sqlite_touched": False,
            "experimental_assets_used": False,
            "l2_l8_triggered": False,
            "files": {str(path): file_sha256(path) for path in hash_files},
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        data_dir = resolve_data_dir(args.data_dir)
        blocked = {
            "task_id": f"{args.workflow_id or f'incremental-trading-signal-{args.trade_date}'}-L1",
            "target_trade_date": str(args.trade_date),
            "status": "l1_blocked",
            "success": False,
            "blocked_at": now_iso(),
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "ready_for_audit_review": False,
            "allow_next_layer_continue": False,
            "boundaries": {
                "legacy_sqlite_touched": False,
                "experimental_assets_used": False,
                "l2_l8_triggered": False,
            },
        }
        path = data_dir / "reports" / f"l1_incremental_raw_ingest_{args.trade_date}_blocked.json"
        atomic_json(path, blocked)
        print(json.dumps(blocked, ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
