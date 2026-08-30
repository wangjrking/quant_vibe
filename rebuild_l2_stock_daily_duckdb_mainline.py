from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from database_module import sql as legacy_stock_daily_sql
from l1_raw_data_route import resolve_l1_raw_duckdb_path
from production_asset_registry import active_main_workflow_asset, split_asset_path
from project_paths import resolve_data_dir
from stock_daily_data_route import resolve_stock_daily_duckdb_path


RAW_TABLES = {
    "daily_data": "daily_data",
    "daily_index_data": "daily_index_data",
    "STOCK_BASIC_DATA": "stock_basic_data",
    "FINAN_DATA_SEASON": "finan_data_season",
    "FINAN_DATA_YEAR": "finan_data_year",
    "limit_list_data": "limit_list_data",
    "adj_factor": "adj_factor",
    "moneyflow": "moneyflow",
    "stk_factor": "stk_factor",
    "top_list": "top_list",
    "THS_HOT": "ths_hot",
    "CYQ_PERF": "cyq_perf",
    "stock_st": "stock_st",
    "index_daily": "index_daily",
}

DEFAULT_APPROVED_BASELINE_MIN_TRADE_DATE = "20100104"
DEFAULT_SHRINK_ROW_RATIO = 0.95
SPECIAL_TARGET_CODE = "301583.SZ"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _default_temp_target(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp_rebuild{path.suffix}")


def _default_spill_dir(data_dir: Path) -> Path:
    return data_dir / "runtime" / "duckdb_spill" / "l2_rebuild"


def _source_select_sql(table_name: str, alias: str, *, exclude_bj: bool) -> str:
    select_sql = f"SELECT * FROM {quote_ident(alias)}.{quote_ident(table_name)}"
    if exclude_bj and table_name in {"daily_data", "daily_index_data"}:
        select_sql += " WHERE ts_code NOT LIKE '%.BJ'"
    return select_sql


def _attach_l1_sources(
    conn: duckdb.DuckDBPyConnection,
    *,
    data_dir: str | Path,
    raw_tables: dict[str, str],
    exclude_bj: bool,
) -> dict[str, str]:
    attached: dict[str, str] = {}
    for view_name, table_name in raw_tables.items():
        duckdb_path = resolve_l1_raw_duckdb_path(table_name, data_dir=data_dir, require_exists=True)
        alias = f"src_{table_name}"
        conn.execute(f"ATTACH {quote_literal(str(duckdb_path))} AS {quote_ident(alias)} (READ_ONLY)")
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW {quote_ident(view_name)} AS "
            f"{_source_select_sql(table_name, alias, exclude_bj=exclude_bj)}"
        )
        attached[view_name] = str(duckdb_path)
    return attached


def _file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "mtime": None,
            "mtime_ns": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_snapshot_db_path(snapshot_manifest_path: Path) -> Path:
    if snapshot_manifest_path.suffix.lower() == ".json":
        return snapshot_manifest_path.with_name(snapshot_manifest_path.stem + ".duckdb")
    return snapshot_manifest_path.with_name(snapshot_manifest_path.name + ".duckdb")


def _gate(name: str, passed: bool, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"name": name, "passed": bool(passed), "reason": reason}
    payload.update(extra)
    return payload


def _safe_int(value: Any) -> int:
    return int(value or 0)


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _collect_table_metrics(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    *,
    target_trade_date: str | None = None,
    code_column: str = "stock_code",
    trade_column: str = "trade_date",
    special_code: str | None = None,
) -> dict[str, Any]:
    table_ident = quote_ident(table_name)
    row_count, stock_count, min_trade_date, max_trade_date, bj_row_count = conn.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT {quote_ident(code_column)}),
            MIN({quote_ident(trade_column)}),
            MAX({quote_ident(trade_column)}),
            SUM(CASE WHEN {quote_ident(code_column)} LIKE '%.BJ' THEN 1 ELSE 0 END)
        FROM {table_ident}
        """
    ).fetchone()
    duplicate_key_groups = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {quote_ident(code_column)}, {quote_ident(trade_column)}, COUNT(*) AS c
            FROM {table_ident}
            GROUP BY 1, 2
            HAVING c > 1
        )
        """
    ).fetchone()[0]
    metrics = {
        "row_count": _safe_int(row_count),
        "stock_count": _safe_int(stock_count),
        "min_trade_date": _safe_str(min_trade_date),
        "max_trade_date": _safe_str(max_trade_date),
        "bj_row_count": _safe_int(bj_row_count),
        "duplicate_key_groups": _safe_int(duplicate_key_groups),
    }
    if target_trade_date is not None:
        target_row = conn.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT {quote_ident(code_column)}),
                SUM(CASE WHEN {quote_ident(code_column)} LIKE '%.BJ' THEN 1 ELSE 0 END)
            FROM {table_ident}
            WHERE {quote_ident(trade_column)} = ?
            """,
            [target_trade_date],
        ).fetchone()
        target_dup = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT {quote_ident(code_column)}, {quote_ident(trade_column)}, COUNT(*) AS c
                FROM {table_ident}
                WHERE {quote_ident(trade_column)} = ?
                GROUP BY 1, 2
                HAVING c > 1
            )
            """,
            [target_trade_date],
        ).fetchone()[0]
        metrics.update(
            {
                "target_trade_date": str(target_trade_date),
                "target_row_count": _safe_int(target_row[0]),
                "target_stock_count": _safe_int(target_row[1]),
                "target_bj_row_count": _safe_int(target_row[2]),
                "target_duplicate_key_groups": _safe_int(target_dup),
            }
        )
        if special_code:
            metrics["target_has_special_code"] = bool(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table_ident}
                    WHERE {quote_ident(trade_column)} = ?
                      AND {quote_ident(code_column)} = ?
                    """,
                    [target_trade_date, special_code],
                ).fetchone()[0]
            )
    return metrics


def collect_output_metrics(
    target_path: Path,
    table_name: str = "STOCK_DAILY_DATA",
    *,
    target_trade_date: str | None = None,
    special_code: str | None = None,
) -> dict[str, Any]:
    with duckdb.connect(str(target_path), read_only=True) as conn:
        return _collect_table_metrics(
            conn,
            table_name,
            target_trade_date=target_trade_date,
            special_code=special_code,
        )


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _discover_latest_approved_baseline_report(
    reports_dir: Path,
    *,
    target_trade_date: str | None,
) -> Path | None:
    candidates = sorted(reports_dir.glob("l2_incremental_full_qfq_prices_*_validation.json"))
    latest_path: Path | None = None
    latest_date = ""
    for path in candidates:
        data = _load_json_file(path)
        if not data:
            continue
        report_target = str(data.get("target_trade_date") or "")
        if target_trade_date and report_target and report_target > str(target_trade_date):
            continue
        if report_target >= latest_date and data.get("overall"):
            latest_date = report_target
            latest_path = path
    return latest_path


def _load_approved_baseline(
    reports_dir: Path,
    *,
    target_trade_date: str | None,
    approved_baseline_report_json: str | Path | None = None,
    approved_baseline_min_trade_date: str | None = None,
) -> dict[str, Any]:
    baseline_path = Path(approved_baseline_report_json) if approved_baseline_report_json else _discover_latest_approved_baseline_report(
        reports_dir,
        target_trade_date=target_trade_date,
    )
    baseline_data = _load_json_file(baseline_path) if baseline_path else None
    baseline_metrics: dict[str, Any] = {
        "source": str(baseline_path) if baseline_path else "",
        "exists": bool(baseline_data),
        "target_trade_date": str(baseline_data.get("target_trade_date") or "") if baseline_data else "",
        "row_count": 0,
        "stock_count": 0,
        "min_trade_date": approved_baseline_min_trade_date or DEFAULT_APPROVED_BASELINE_MIN_TRADE_DATE,
        "max_trade_date": "",
        "bj_row_count": 0,
    }
    if baseline_data and baseline_data.get("overall"):
        overall = baseline_data["overall"]
        baseline_metrics.update(
            {
                "row_count": _safe_int(overall[0]),
                "stock_count": _safe_int(overall[1]),
                "min_trade_date": approved_baseline_min_trade_date or _safe_str(overall[2]),
                "max_trade_date": _safe_str(overall[3]),
                "bj_row_count": _safe_int(overall[4]),
            }
        )
    return baseline_metrics


def _collect_source_coverage_metrics(
    conn: duckdb.DuckDBPyConnection,
    *,
    target_trade_date: str,
    special_code: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for table_name in ("daily_data", "daily_index_data"):
        metrics[table_name] = _collect_table_metrics(
            conn,
            table_name,
            target_trade_date=target_trade_date,
            code_column="ts_code",
            trade_column="trade_date",
            special_code=special_code,
        )
    return metrics


def _evaluate_route_semantic_gate(*, data_dir: Path, target_path: Path) -> dict[str, Any]:
    asset = active_main_workflow_asset("L2_stock_daily_base", data_dir=data_dir)
    if not asset:
        return _gate(
            "route_semantic_gate",
            target_path.suffix.lower() == ".duckdb",
            "production registry unavailable; explicit DuckDB target path accepted for non-production precheck"
            if target_path.suffix.lower() == ".duckdb"
            else "production registry unavailable and target path is not a DuckDB asset",
            asset_path="",
        )
    asset_path, asset_table = split_asset_path(asset.get("asset_path"))
    passed = (
        asset_path is not None
        and asset_path.resolve() == target_path.resolve()
        and asset_path.suffix.lower() == ".duckdb"
        and str(asset_table or "") == "STOCK_DAILY_DATA"
    )
    return _gate(
        "route_semantic_gate",
        passed,
        "active route is DuckDB-only and points to STOCK_DAILY_DATA" if passed else "active route does not match expected DuckDB L2 mainline asset",
        asset_id=str(asset.get("asset_id") or ""),
        asset_path=str(asset.get("asset_path") or ""),
        status=str(asset.get("status") or ""),
    )


def _evaluate_approved_baseline_gate(
    *,
    approved_baseline: dict[str, Any],
) -> dict[str, Any]:
    passed = bool(approved_baseline.get("exists")) and bool(approved_baseline.get("min_trade_date"))
    return _gate(
        "approved_baseline_gate",
        passed,
        "approved baseline report discovered" if passed else "approved baseline report not available; precheck cannot trust current active L2 as a valid baseline",
        baseline=approved_baseline,
    )


def _evaluate_source_coverage_gate(
    *,
    source_metrics: dict[str, Any],
    approved_baseline: dict[str, Any],
    target_trade_date: str,
    special_code: str,
) -> dict[str, Any]:
    baseline_min = str(approved_baseline.get("min_trade_date") or DEFAULT_APPROVED_BASELINE_MIN_TRADE_DATE)
    daily = source_metrics["daily_data"]
    index_daily = source_metrics["daily_index_data"]
    failures: list[str] = []
    if str(daily["min_trade_date"]) > baseline_min:
        failures.append(f"daily_data.min_trade_date={daily['min_trade_date']} > approved_baseline.min_trade_date={baseline_min}")
    if str(index_daily["min_trade_date"]) > baseline_min:
        failures.append(
            f"daily_index_data.min_trade_date={index_daily['min_trade_date']} > approved_baseline.min_trade_date={baseline_min}"
        )
    if str(daily["max_trade_date"]) < target_trade_date:
        failures.append(f"daily_data.max_trade_date={daily['max_trade_date']} < target_trade_date={target_trade_date}")
    if str(index_daily["max_trade_date"]) < target_trade_date:
        failures.append(
            f"daily_index_data.max_trade_date={index_daily['max_trade_date']} < target_trade_date={target_trade_date}"
        )
    if not daily.get("target_has_special_code"):
        failures.append(f"{special_code} missing from daily_data target day")
    passed = not failures
    return _gate(
        "source_coverage_gate",
        passed,
        "source coverage meets approved baseline" if passed else "; ".join(failures),
        approved_baseline_min_trade_date=baseline_min,
        source_metrics=source_metrics,
    )


def _evaluate_driver_consistency_gate(
    *,
    source_metrics: dict[str, Any],
    target_trade_date: str,
) -> dict[str, Any]:
    daily = source_metrics["daily_data"]
    index_daily = source_metrics["daily_index_data"]
    failures: list[str] = []
    if _safe_int(daily.get("target_row_count")) <= 0:
        failures.append("daily_data target row count is 0")
    if _safe_int(index_daily.get("target_row_count")) <= 0:
        failures.append("daily_index_data target row count is 0")
    if _safe_int(daily.get("target_row_count")) != _safe_int(index_daily.get("target_row_count")):
        failures.append(
            "daily_data and daily_index_data target row counts differ "
            f"({daily.get('target_row_count')} vs {index_daily.get('target_row_count')})"
        )
    if str(daily.get("max_trade_date") or "") != str(index_daily.get("max_trade_date") or ""):
        failures.append(
            "daily_data and daily_index_data max_trade_date differ "
            f"({daily.get('max_trade_date')} vs {index_daily.get('max_trade_date')})"
        )
    passed = not failures
    return _gate(
        "driver_consistency_gate",
        passed,
        "daily drivers are consistent for target day" if passed else "; ".join(failures),
        target_trade_date=target_trade_date,
        daily_data=source_metrics["daily_data"],
        daily_index_data=source_metrics["daily_index_data"],
    )


def _evaluate_output_integrity_gate(
    *,
    output_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    special_code: str,
    exclude_bj: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    if exclude_bj and _safe_int(output_metrics.get("bj_row_count")) != 0:
        failures.append(f"output bj_row_count={output_metrics.get('bj_row_count')} != 0")
    if _safe_int(output_metrics.get("duplicate_key_groups")) != 0:
        failures.append(
            f"output duplicate_key_groups={output_metrics.get('duplicate_key_groups')} != 0"
        )
    if _safe_int(output_metrics.get("target_row_count")) != _safe_int(source_metrics["daily_data"].get("target_row_count")):
        failures.append(
            "output target_row_count does not match daily_data target_row_count "
            f"({output_metrics.get('target_row_count')} vs {source_metrics['daily_data'].get('target_row_count')})"
        )
    if not output_metrics.get("target_has_special_code"):
        failures.append(f"{special_code} missing from output target day")
    passed = not failures
    return _gate(
        "output_integrity_gate",
        passed,
        "temp rebuild output passes integrity checks" if passed else "; ".join(failures),
        output_metrics=output_metrics,
    )


def _evaluate_shrink_gate(
    *,
    output_metrics: dict[str, Any],
    approved_baseline: dict[str, Any],
    current_active_metrics: dict[str, Any] | None,
    min_row_ratio: float,
) -> dict[str, Any]:
    baseline = approved_baseline if approved_baseline.get("exists") else (current_active_metrics or {})
    baseline_min = str(baseline.get("min_trade_date") or "")
    baseline_rows = _safe_int(baseline.get("row_count"))
    baseline_stocks = _safe_int(baseline.get("stock_count"))
    failures: list[str] = []
    if baseline_min and str(output_metrics.get("min_trade_date") or "") > baseline_min:
        failures.append(
            f"output.min_trade_date={output_metrics.get('min_trade_date')} > baseline.min_trade_date={baseline_min}"
        )
    if baseline_rows and _safe_int(output_metrics.get("row_count")) < int(baseline_rows * min_row_ratio):
        failures.append(
            f"output.row_count={output_metrics.get('row_count')} < baseline.row_count*{min_row_ratio:.2f}={int(baseline_rows * min_row_ratio)}"
        )
    if baseline_stocks and _safe_int(output_metrics.get("stock_count")) < int(baseline_stocks * min_row_ratio):
        failures.append(
            f"output.stock_count={output_metrics.get('stock_count')} < baseline.stock_count*{min_row_ratio:.2f}={int(baseline_stocks * min_row_ratio)}"
        )
    passed = not failures
    return _gate(
        "shrink_gate",
        passed,
        "temp rebuild output does not shrink below approved baseline" if passed else "; ".join(failures),
        min_row_ratio=min_row_ratio,
        baseline=baseline,
        output_metrics=output_metrics,
    )


def _evaluate_snapshot_before_replace_gate(
    *,
    target_path: Path,
    snapshot_manifest_path: str | Path | None,
    precheck_only: bool,
) -> dict[str, Any]:
    configured = bool(snapshot_manifest_path)
    snapshot_manifest = Path(snapshot_manifest_path) if snapshot_manifest_path else None
    snapshot_db_path = _derive_snapshot_db_path(snapshot_manifest) if snapshot_manifest else None
    manifest_is_distinct = bool(snapshot_manifest) and snapshot_manifest.resolve() != target_path.resolve()
    snapshot_is_distinct = bool(snapshot_db_path) and snapshot_db_path.resolve() != target_path.resolve()
    path_is_distinct = manifest_is_distinct and snapshot_is_distinct
    passed = configured and path_is_distinct and not precheck_only
    if not configured:
        reason = "snapshot manifest path is not configured"
    elif not path_is_distinct:
        reason = "snapshot path must be different from active target path"
    else:
        reason = "snapshot path configured and distinct from active target"
    return _gate(
        "snapshot_before_replace_gate",
        passed,
        reason,
        execute_only_requirement=True,
        would_block_execute=not (configured and path_is_distinct),
        snapshot_manifest_path=str(snapshot_manifest_path or ""),
        snapshot_db_path=str(snapshot_db_path or ""),
    )


def _write_snapshot_before_replace(
    *,
    target_path: Path,
    snapshot_manifest_path: Path,
    table_name: str,
    target_trade_date: str,
    special_code: str,
) -> dict[str, Any]:
    snapshot_db_path = _derive_snapshot_db_path(snapshot_manifest_path)
    if snapshot_db_path.resolve() == target_path.resolve():
        raise ValueError("snapshot path must be different from active target path")
    if not target_path.exists():
        raise FileNotFoundError(f"active target does not exist: {target_path}")
    snapshot_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_db_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_db_path.exists():
        snapshot_db_path.unlink()
    shutil.copy2(target_path, snapshot_db_path)
    file_state = _file_state(snapshot_db_path)
    metrics = collect_output_metrics(
        snapshot_db_path,
        table_name=table_name,
        target_trade_date=target_trade_date,
        special_code=special_code,
    )
    sha256 = _sha256_file(snapshot_db_path)
    manifest = {
        "manifest_type": "l2_snapshot_before_replace",
        "captured_at": datetime.now().astimezone().isoformat(),
        "asset_path": str(snapshot_db_path),
        "active_target_path": str(target_path),
        "table_name": table_name,
        "target_trade_date": str(target_trade_date),
        "file_state": file_state,
        "sha256": sha256,
        "table_metrics": metrics,
    }
    snapshot_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _evaluate_quarantine_on_fail_gate(
    *,
    incident_output_json: str | Path | None,
    precheck_only: bool,
) -> dict[str, Any]:
    configured = bool(incident_output_json)
    return _gate(
        "quarantine_on_fail_gate",
        configured,
        "incident output path configured for failure reporting" if configured else "incident output path is not configured",
        execute_only_requirement=not precheck_only,
        would_block_execute=not configured and not precheck_only,
        incident_output_json=str(incident_output_json or ""),
    )


def _render_gate_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# L2 重建硬门禁预检报告",
        "",
        f"- approval_status: {report.get('approval_status')}",
        f"- status: {report.get('status')}",
        f"- active_written: {str(report.get('active_written')).lower()}",
        f"- target_trade_date: {report.get('target_trade_date')}",
        f"- target_path: {report.get('target_path')}",
        "",
        "## 门禁结果",
        "",
    ]
    for gate in report.get("gates", []):
        lines.append(
            f"- {gate.get('name')}: {'通过' if gate.get('passed') else '失败'}；{gate.get('reason')}"
        )
    lines.extend(
        [
            "",
            "## 当前 Active 目标资产",
            "",
            f"- before_file_state: {json.dumps(report.get('active_target_before_file_state', {}), ensure_ascii=False)}",
            f"- after_file_state: {json.dumps(report.get('active_target_after_file_state', {}), ensure_ascii=False)}",
            f"- before_metrics: {json.dumps(report.get('current_active_target_metrics_before', {}), ensure_ascii=False)}",
            f"- after_metrics: {json.dumps(report.get('current_active_target_metrics_after', {}), ensure_ascii=False)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_report_artifacts(
    report: dict[str, Any],
    *,
    gate_report_json: str | Path | None = None,
    gate_report_md: str | Path | None = None,
    incident_output_json: str | Path | None = None,
) -> None:
    if gate_report_json:
        gate_report_json_path = Path(gate_report_json)
        gate_report_json_path.parent.mkdir(parents=True, exist_ok=True)
        gate_report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if gate_report_md:
        gate_report_md_path = Path(gate_report_md)
        gate_report_md_path.parent.mkdir(parents=True, exist_ok=True)
        gate_report_md_path.write_text(_render_gate_report_markdown(report), encoding="utf-8")
    if incident_output_json and (not gate_report_json or Path(incident_output_json) != Path(gate_report_json)):
        incident_output_path = Path(incident_output_json)
        incident_output_path.parent.mkdir(parents=True, exist_ok=True)
        incident_output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_stock_daily_duckdb_mainline(
    *,
    data_dir: str | Path | None = None,
    target_path: str | Path | None = None,
    temp_target_path: str | Path | None = None,
    table_name: str = "STOCK_DAILY_DATA",
    raw_tables: dict[str, str] | None = None,
    build_sql: str | None = None,
    exclude_bj: bool = True,
    target_trade_date: str | None = None,
    precheck_only: bool = False,
    gate_report_json: str | Path | None = None,
    gate_report_md: str | Path | None = None,
    incident_output_json: str | Path | None = None,
    approved_baseline_report_json: str | Path | None = None,
    approved_baseline_min_trade_date: str | None = None,
    require_approved_baseline_gate: bool = True,
    require_full_history_gate: bool = True,
    require_snapshot_before_replace: bool = True,
    snapshot_manifest_path: str | Path | None = None,
    min_row_ratio: float = DEFAULT_SHRINK_ROW_RATIO,
    special_code: str = SPECIAL_TARGET_CODE,
    keep_temp_target: bool = False,
    duckdb_memory_limit: str = "8GB",
    duckdb_threads: int = 4,
    duckdb_temp_directory: str | Path | None = None,
) -> dict[str, Any]:
    if not target_trade_date:
        raise ValueError("target_trade_date is required for gated L2 rebuild")

    resolved_data_dir = resolve_data_dir(data_dir)
    reports_dir = resolved_data_dir / "reports"
    target = Path(target_path) if target_path else resolve_stock_daily_duckdb_path(data_dir=resolved_data_dir)
    temp_target = Path(temp_target_path) if temp_target_path else _default_temp_target(target)
    spill_dir = Path(duckdb_temp_directory) if duckdb_temp_directory else _default_spill_dir(resolved_data_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    build_statement = build_sql or legacy_stock_daily_sql
    selected_raw_tables = raw_tables or RAW_TABLES

    target.parent.mkdir(parents=True, exist_ok=True)
    active_target_before_file_state = _file_state(target)
    current_active_target_metrics_before = (
        collect_output_metrics(target, table_name=table_name, target_trade_date=target_trade_date, special_code=special_code)
        if target.exists()
        else {}
    )
    approved_baseline = _load_approved_baseline(
        reports_dir,
        target_trade_date=target_trade_date,
        approved_baseline_report_json=approved_baseline_report_json,
        approved_baseline_min_trade_date=approved_baseline_min_trade_date,
    )
    if temp_target.exists():
        temp_target.unlink()

    route_gate = _evaluate_route_semantic_gate(data_dir=resolved_data_dir, target_path=target)
    approved_baseline_gate = _evaluate_approved_baseline_gate(approved_baseline=approved_baseline)
    source_coverage_gate: dict[str, Any] | None = None
    driver_consistency_gate: dict[str, Any] | None = None
    output_integrity_gate: dict[str, Any] | None = None
    shrink_gate: dict[str, Any] | None = None
    build_error = ""
    attached: dict[str, str] = {}
    source_metrics: dict[str, Any] = {}
    temp_output_metrics: dict[str, Any] = {}
    snapshot_manifest: dict[str, Any] = {}

    try:
        with duckdb.connect(str(temp_target)) as conn:
            conn.execute("PRAGMA preserve_insertion_order=false")
            conn.execute(f"PRAGMA threads={int(duckdb_threads)}")
            conn.execute(f"PRAGMA memory_limit='{duckdb_memory_limit}'")
            conn.execute(f"PRAGMA temp_directory={quote_literal(str(spill_dir))}")
            attached = _attach_l1_sources(
                conn,
                data_dir=resolved_data_dir,
                raw_tables=selected_raw_tables,
                exclude_bj=exclude_bj,
            )
            source_metrics = _collect_source_coverage_metrics(
                conn,
                target_trade_date=target_trade_date,
                special_code=special_code,
            )
            source_coverage_gate = _evaluate_source_coverage_gate(
                source_metrics=source_metrics,
                approved_baseline=approved_baseline,
                target_trade_date=target_trade_date,
                special_code=special_code,
            )
            driver_consistency_gate = _evaluate_driver_consistency_gate(
                source_metrics=source_metrics,
                target_trade_date=target_trade_date,
            )
            conn.execute(build_statement)
            if table_name != "STOCK_DAILY_DATA":
                conn.execute(f"ALTER TABLE STOCK_DAILY_DATA RENAME TO {quote_ident(table_name)}")
            conn.execute("CHECKPOINT")
        temp_output_metrics = collect_output_metrics(
            temp_target,
            table_name=table_name,
            target_trade_date=target_trade_date,
            special_code=special_code,
        )
        output_integrity_gate = _evaluate_output_integrity_gate(
            output_metrics=temp_output_metrics,
            source_metrics=source_metrics,
            special_code=special_code,
            exclude_bj=exclude_bj,
        )
        shrink_gate = _evaluate_shrink_gate(
            output_metrics=temp_output_metrics,
            approved_baseline=approved_baseline,
            current_active_metrics=current_active_target_metrics_before or None,
            min_row_ratio=min_row_ratio,
        )
    except Exception as exc:
        build_error = f"{type(exc).__name__}: {exc}"

    snapshot_gate = _evaluate_snapshot_before_replace_gate(
        target_path=target,
        snapshot_manifest_path=snapshot_manifest_path,
        precheck_only=precheck_only,
    )
    quarantine_gate = _evaluate_quarantine_on_fail_gate(
        incident_output_json=incident_output_json,
        precheck_only=precheck_only,
    )
    gates = [route_gate]
    if require_approved_baseline_gate:
        gates.append(approved_baseline_gate)
    if source_coverage_gate is not None:
        gates.append(source_coverage_gate)
    if driver_consistency_gate is not None:
        gates.append(driver_consistency_gate)
    if output_integrity_gate is not None:
        gates.append(output_integrity_gate)
    if shrink_gate is not None:
        gates.append(shrink_gate)
    gates.extend([snapshot_gate, quarantine_gate])

    precheck_required_failures = [
        gate
        for gate in gates
        if gate["name"] in {
            "route_semantic_gate",
            "approved_baseline_gate",
            "source_coverage_gate",
            "driver_consistency_gate",
            "output_integrity_gate",
            "shrink_gate",
        }
        and not gate["passed"]
        and (gate["name"] != "approved_baseline_gate" or require_approved_baseline_gate)
    ]
    if not require_full_history_gate:
        precheck_required_failures = [
            gate for gate in precheck_required_failures if gate["name"] != "source_coverage_gate"
        ]

    status = "precheck_failed"
    active_written = False
    execute_blockers: list[dict[str, Any]] = []
    if require_snapshot_before_replace and not snapshot_gate["passed"]:
        execute_blockers.append(snapshot_gate)
    if not build_error and not precheck_required_failures:
        status = "precheck_passed" if precheck_only else "ready_to_replace"
    if build_error:
        status = "precheck_failed"
    if not precheck_only and status == "ready_to_replace" and not execute_blockers:
        try:
            if require_snapshot_before_replace:
                snapshot_manifest = _write_snapshot_before_replace(
                    target_path=target,
                    snapshot_manifest_path=Path(str(snapshot_manifest_path)),
                    table_name=table_name,
                    target_trade_date=target_trade_date,
                    special_code=special_code,
                )
            os.replace(temp_target, target)
            active_written = True
            status = "completed"
        except Exception as exc:
            build_error = f"{type(exc).__name__}: {exc}"
            status = "precheck_failed"
    elif temp_target.exists() and not keep_temp_target:
        temp_target.unlink()

    active_target_after_file_state = _file_state(target)
    current_active_target_metrics_after = (
        collect_output_metrics(target, table_name=table_name, target_trade_date=target_trade_date, special_code=special_code)
        if target.exists()
        else {}
    )
    report = {
        "task_id": "incremental-trading-signal-20260710-L2-hard-gate-precheck",
        "status": status,
        "approval_status": "waiting_for_user_approval" if precheck_only else "execution_authorized_scope_limited",
        "precheck_only": bool(precheck_only),
        "active_written": bool(active_written),
        "target_trade_date": str(target_trade_date),
        "target_path": str(target),
        "temp_target_path": str(temp_target),
        "table_name": table_name,
        "build_error": build_error,
        "attached_l1_sources": attached,
        "approved_baseline": approved_baseline,
        "current_active_target_metrics_before": current_active_target_metrics_before,
        "current_active_target_metrics_after": current_active_target_metrics_after,
        "active_target_before_file_state": active_target_before_file_state,
        "active_target_after_file_state": active_target_after_file_state,
        "source_metrics": source_metrics,
        "temp_output_metrics": temp_output_metrics,
        "snapshot_manifest": snapshot_manifest,
        "gates": gates,
        "summary": {
            "required_precheck_gate_failures": [gate["name"] for gate in precheck_required_failures],
            "execute_blockers": [gate["name"] for gate in execute_blockers],
            "active_unchanged": active_target_before_file_state == active_target_after_file_state
            and current_active_target_metrics_before == current_active_target_metrics_after,
        },
        "generated_at": datetime.now().astimezone().isoformat(),
        "duckdb_runtime": {
            "memory_limit": duckdb_memory_limit,
            "threads": int(duckdb_threads),
            "temp_directory": str(spill_dir),
        },
    }
    _write_report_artifacts(
        report,
        gate_report_json=gate_report_json,
        gate_report_md=gate_report_md,
        incident_output_json=incident_output_json,
    )
    if not precheck_only and status != "completed":
        raise RuntimeError(
            "L2 rebuild blocked before active replace: "
            + (build_error or ", ".join(report["summary"]["required_precheck_gate_failures"] + report["summary"]["execute_blockers"]))
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DuckDB-only rebuild for L2 STOCK_DAILY_DATA from L1 DuckDB raw table files with hard gates."
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--target-path")
    parser.add_argument("--temp-target-path")
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--gate-report-json")
    parser.add_argument("--gate-report-md")
    parser.add_argument("--incident-output-json")
    parser.add_argument("--approved-baseline-report-json")
    parser.add_argument("--approved-baseline-min-trade-date", default=DEFAULT_APPROVED_BASELINE_MIN_TRADE_DATE)
    parser.add_argument("--snapshot-manifest-path")
    parser.add_argument("--include-bj", action="store_true", help="Allow .BJ rows into the rebuilt L2 table.")
    parser.add_argument("--dry-run", action="store_true", help="Build a temp target and run hard gates without replacing active L2.")
    parser.add_argument("--precheck-only", action="store_true", help="Alias of --dry-run.")
    parser.add_argument("--keep-temp-target", action="store_true")
    parser.add_argument("--duckdb-memory-limit", default="8GB")
    parser.add_argument("--duckdb-threads", type=int, default=4)
    parser.add_argument("--duckdb-temp-directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = rebuild_stock_daily_duckdb_mainline(
        data_dir=args.data_dir,
        target_path=args.target_path,
        temp_target_path=args.temp_target_path,
        target_trade_date=args.target_trade_date,
        gate_report_json=args.gate_report_json,
        gate_report_md=args.gate_report_md,
        incident_output_json=args.incident_output_json,
        approved_baseline_report_json=args.approved_baseline_report_json,
        approved_baseline_min_trade_date=args.approved_baseline_min_trade_date,
        snapshot_manifest_path=args.snapshot_manifest_path,
        exclude_bj=not args.include_bj,
        precheck_only=bool(args.dry_run or args.precheck_only),
        keep_temp_target=args.keep_temp_target,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.duckdb_threads,
        duckdb_temp_directory=args.duckdb_temp_directory,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
