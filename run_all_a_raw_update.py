from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from l1_universe_rules import filter_frame_no_bj, filter_stock_codes_no_bj, is_bj_code
from data_load_module import date_range_pandas, get_pro
from project_paths import load_config, resolve_data_dir, resolve_data_path
from raw_table_db_module import replace_raw_table_full
from raw_data_update_module import (
    RAW_TABLE_SPECS,
    append_parquet_dedup,
    load_existing_stock_codes,
    load_stock_codes,
    next_start_date,
)


DEFAULT_TABLES = (
    "daily,daily_basic,stk_factor,moneyflow,limit_list,cyq_perf,adj_factor,"
    "stock_st,index_daily,stk_shock,stk_high_shock,stk_alert"
)
from tushare_stock_risk_event_contract import (
    API_SPEC_BY_NAME as RISK_EVENT_API_SPECS,
    fetch_complete_range as fetch_complete_risk_event_range,
    normalize_source_frame as normalize_risk_event_source,
)
from l1_raw_data_route import resolve_l1_raw_duckdb_path
DEFAULT_INDEX_CODES = ("000300.SH", "000905.SH", "000852.SH", "932000.CSI")
SQLITE_TABLE_NAMES = {
    "daily": "daily_data",
    "daily_basic": "daily_index_data",
    "limit_list": "limit_list_data",
}
DRIVER_TABLE = "daily"
DRIVER_COVERAGE_TABLES = {"daily_basic", "stk_factor", "cyq_perf"}
AUXILIARY_TABLES = {
    "moneyflow",
    "limit_list",
    "stock_st",
    "stk_shock",
    "stk_high_shock",
    "stk_alert",
}
MAX_MISSING_SOURCE_PROBES = 50


def _query_retry(call):
    import time

    while True:
        try:
            return call()
        except Exception:
            time.sleep(61)


def _fetch_stock_table(ts_pro, table: str, code: str, start: str, end: str) -> pd.DataFrame:
    if table == "daily":
        return _query_retry(lambda: ts_pro.query("daily", ts_code=code, start_date=start, end_date=end))
    if table == "daily_basic":
        return _query_retry(lambda: ts_pro.query("daily_basic", ts_code=code, start_date=start, end_date=end))
    if table == "moneyflow":
        return _query_retry(lambda: ts_pro.moneyflow(ts_code=code, start_date=start, end_date=end))
    if table == "stk_factor":
        return _query_retry(lambda: ts_pro.stk_factor_pro(ts_code=code, start_date=start, end_date=end))
    if table == "limit_list":
        return _query_retry(lambda: ts_pro.limit_list_d(ts_code=code, limit_type="U", start_date=start, end_date=end))
    if table == "cyq_perf":
        return _query_retry(lambda: ts_pro.cyq_perf(ts_code=code, start_date=start, end_date=end))
    raise ValueError(f"unsupported per-stock table: {table}")


def _fetch_date_table(ts_pro, table: str, start: str, end: str, index_codes: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    if table == "adj_factor":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.adj_factor(trade_date=trade_date)))
    elif table == "top_list":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.top_list(trade_date=trade_date)))
    elif table == "ths_hot":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.ths_hot(trade_date=trade_date, market="热股")))
    elif table == "dc_hot":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.dc_hot(trade_date=trade_date, market="A股市场", hot_type="人气榜")))
    elif table == "stock_st":
        for trade_date in date_range_pandas(start, end):
            frames.append(_query_retry(lambda trade_date=trade_date: ts_pro.stock_st(trade_date=trade_date)))
    elif table == "index_daily":
        for index_code in index_codes:
            frames.append(_query_retry(lambda index_code=index_code: ts_pro.query("index_daily", ts_code=index_code, start_date=start, end_date=end)))
    elif table in {"stk_shock", "stk_high_shock", "stk_alert"}:
        frames.append(fetch_complete_risk_event_range(ts_pro, table, start, end))
    else:
        raise ValueError(f"unsupported date table: {table}")
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if table in RISK_EVENT_API_SPECS:
        source = pd.concat(frames, ignore_index=True) if frames else None
        return filter_frame_no_bj(normalize_risk_event_source(table, source))
    if not frames:
        return pd.DataFrame()
    return filter_frame_no_bj(pd.concat(frames, ignore_index=True))


def _fetch_table_for_trade_date(ts_pro, table: str, trade_date: str, index_codes: tuple[str, ...]) -> pd.DataFrame:
    if table == "daily":
        return _query_retry(lambda: ts_pro.query("daily", trade_date=trade_date))
    if table == "daily_basic":
        return _query_retry(lambda: ts_pro.query("daily_basic", trade_date=trade_date))
    if table == "moneyflow":
        return _query_retry(lambda: ts_pro.moneyflow(trade_date=trade_date))
    if table == "stk_factor":
        return _query_retry(lambda: ts_pro.stk_factor_pro(trade_date=trade_date))
    if table == "limit_list":
        return _query_retry(lambda: ts_pro.limit_list_d(trade_date=trade_date, limit_type="U"))
    if table == "cyq_perf":
        return _query_retry(lambda: ts_pro.cyq_perf(trade_date=trade_date))
    if table == "adj_factor":
        return _query_retry(lambda: ts_pro.adj_factor(trade_date=trade_date))
    if table == "stock_st":
        return _query_retry(lambda: ts_pro.stock_st(trade_date=trade_date))
    if table == "index_daily":
        frames = [
            _query_retry(lambda index_code=index_code: ts_pro.query("index_daily", ts_code=index_code, start_date=trade_date, end_date=trade_date))
            for index_code in index_codes
        ]
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if table in {"stk_shock", "stk_high_shock", "stk_alert"}:
        return normalize_risk_event_source(
            table,
            _query_retry(
                lambda: ts_pro.query(table, start_date=trade_date, end_date=trade_date)
            ),
        )
    raise ValueError(f"unsupported preflight table: {table}")


def _code_set(frame: pd.DataFrame) -> set[str]:
    if frame is None or frame.empty or "ts_code" not in frame.columns:
        return set()
    return {
        str(value).strip().upper()
        for value in frame["ts_code"].dropna()
        if str(value).strip() and not is_bj_code(value)
    }


def _duplicate_key_groups(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> int:
    if frame is None or frame.empty:
        return 0
    if not set(key_columns).issubset(frame.columns):
        return 0
    return int(frame.duplicated(subset=list(key_columns), keep=False).sum())


def _filter_to_driver_codes(frame: pd.DataFrame, driver_codes: set[str]) -> pd.DataFrame:
    if frame is None or frame.empty or "ts_code" not in frame.columns:
        return pd.DataFrame()
    normalized = filter_frame_no_bj(frame)
    return normalized[normalized["ts_code"].isin(driver_codes)].copy()


def _probe_missing_source_codes(ts_pro, table: str, missing_codes: list[str], trade_date: str) -> dict:
    if not missing_codes or table not in DRIVER_COVERAGE_TABLES:
        return {"probed": 0, "returned_codes": [], "skipped": len(missing_codes)}
    if len(missing_codes) > MAX_MISSING_SOURCE_PROBES:
        return {"probed": 0, "returned_codes": [], "skipped": len(missing_codes)}
    returned = []
    for code in missing_codes:
        frame = _fetch_stock_table(ts_pro, table, code, trade_date, trade_date)
        if frame is not None and not frame.empty:
            returned.append(code)
    return {"probed": len(missing_codes), "returned_codes": returned, "skipped": 0}


def build_source_preflight_report(
    ts_pro,
    tables: list[str],
    trade_date: str,
    stock_codes: list[str],
    index_codes: tuple[str, ...] = DEFAULT_INDEX_CODES,
) -> dict:
    source_frames = {table: _fetch_table_for_trade_date(ts_pro, table, trade_date, index_codes) for table in tables}
    source_frames = {
        table: filter_frame_no_bj(frame) if table != "index_daily" else frame
        for table, frame in source_frames.items()
    }
    daily_frame = source_frames.get(DRIVER_TABLE, pd.DataFrame())
    daily_codes = _code_set(daily_frame)
    stock_pool_codes = set(filter_stock_codes_no_bj(stock_codes))
    report = {
        "trade_date": trade_date,
        "status": "source_ready",
        "gate_pass": True,
        "blockers": [],
        "warnings": [],
        "universe_rule": "no_bj",
        "driver_table": DRIVER_TABLE,
        "driver_source_rows": int(daily_frame.shape[0]),
        "driver_source_codes": len(daily_codes),
        "stock_pool_missing_from_driver_count": len(stock_pool_codes - daily_codes),
        "stock_pool_missing_from_driver_sample": sorted(stock_pool_codes - daily_codes)[:20],
        "tables": {},
    }
    if not daily_codes:
        report["blockers"].append("daily source returned zero rows")

    for table in tables:
        frame = source_frames.get(table, pd.DataFrame())
        spec = RAW_TABLE_SPECS[table]
        source_codes = _code_set(frame)
        scoped_frame = _filter_to_driver_codes(frame, daily_codes) if table != DRIVER_TABLE else frame
        scoped_codes = _code_set(scoped_frame)
        missing_vs_daily = (
            sorted(daily_codes - scoped_codes)
            if table not in AUXILIARY_TABLES | {"index_daily"}
            else []
        )
        extra_vs_daily = sorted(source_codes - daily_codes) if table not in {"index_daily"} else []
        table_report = {
            "source_rows": int(scoped_frame.shape[0] if table != DRIVER_TABLE else frame.shape[0]),
            "source_codes": len(scoped_codes if table != DRIVER_TABLE else source_codes),
            "source_full_rows": int(frame.shape[0]),
            "source_full_codes": len(source_codes),
            "duplicate_key_groups": _duplicate_key_groups(frame, spec.key_columns),
            "missing_vs_daily_count": len(missing_vs_daily),
            "missing_vs_daily_sample": missing_vs_daily[:20],
            "extra_vs_daily_count": len(extra_vs_daily),
            "extra_vs_daily_sample": extra_vs_daily[:20],
        }
        if table in DRIVER_COVERAGE_TABLES and missing_vs_daily:
            probe = _probe_missing_source_codes(ts_pro, table, missing_vs_daily, trade_date)
            table_report["missing_source_probe"] = probe
            if probe["returned_codes"]:
                report["blockers"].append(f"{table} direct trade_date source missed codes that per-code source returned")
            elif len(missing_vs_daily) > 1:
                report["blockers"].append(
                    f"{table} source coverage gap vs daily_data exceeds single-code source-limited review scope: "
                    f"{len(missing_vs_daily)} codes"
                )
                table_report["gap_nature"] = "source_not_ready_multi_code_coverage_gap"
            else:
                report["warnings"].append(
                    f"{table} has one-code coverage gap requiring authoritative source-limited review"
                )
                table_report["gap_nature"] = "single_code_source_limited_review_required"
        elif table == "adj_factor":
            if missing_vs_daily:
                report["blockers"].append("adj_factor is missing daily_data driver codes")
            if extra_vs_daily:
                report["warnings"].append("adj_factor has extra codes and must not drive trading calendar")
                table_report["calendar_governance_note"] = "daily_data remains the driver"
        elif table == "moneyflow" and missing_vs_daily:
            table_report["gap_nature"] = "source_coverage_gap"
            report["warnings"].append("moneyflow has auxiliary source coverage gap vs daily_data")
        elif table == "index_daily":
            index_source_codes = _code_set(frame)
            missing_indices = sorted(set(index_codes) - index_source_codes)
            table_report["missing_index_codes"] = missing_indices
            if missing_indices:
                report["blockers"].append("index_daily is missing configured index codes")

        if table in {DRIVER_TABLE} | DRIVER_COVERAGE_TABLES | {"adj_factor"} and table_report["source_rows"] == 0:
            report["blockers"].append(f"{table} source returned zero driver-scope rows")
        if table_report["duplicate_key_groups"]:
            report["blockers"].append(f"{table} source has duplicate key groups")
        report["tables"][table] = table_report

    if report["blockers"]:
        report["status"] = "source_not_ready"
        report["gate_pass"] = False
    return report


def write_source_preflight_report(data_dir: Path, report: dict, report_path: str | None = None) -> Path:
    if report_path:
        path = Path(report_path)
        if not path.is_absolute():
            path = data_dir / report_path
    else:
        path = data_dir / "reports" / f"l1_source_preflight_{report['trade_date']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _sync_sqlite_table(data_dir: Path, file_name: str, table_name: str) -> None:
    frame = pd.read_parquet(data_dir / file_name)
    replace_raw_table_full(data_dir, table_name, frame)


def _sync_risk_event_duckdb(data_dir: Path, table: str, parquet_path: Path) -> Path:
    target = resolve_l1_raw_duckdb_path(table, data_dir=data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    frame = filter_frame_no_bj(pd.read_parquet(parquet_path))
    spec = RAW_TABLE_SPECS[table]
    if frame.duplicated(list(spec.key_columns)).any():
        raise RuntimeError(f"{table} parquet contains duplicate natural keys")
    connection = duckdb.connect(str(candidate))
    try:
        connection.register("payload", frame)
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM payload')
    finally:
        connection.close()
    os.replace(candidate, target)
    return target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Incrementally update local raw data for the all-A-share universe.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--stock-pool", default="stock_pool_all_a.csv")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", required=True)
    parser.add_argument("--tables", default=DEFAULT_TABLES)
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--limit-stocks", type=int)
    parser.add_argument("--sync-sqlite", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip source completeness preflight. Use only for audited recovery runs.")
    parser.add_argument("--preflight-only", action="store_true", help="Run source completeness preflight and exit before any local writes.")
    parser.add_argument("--preflight-report", default=None, help="Optional preflight report path. Relative paths are under data-dir.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    config = load_config(args.config)
    ts_pro = get_pro(config["datasource"]["tushare_token"])
    stock_pool_path = resolve_data_path(args.stock_pool, data_dir=data_dir)
    stock_codes = load_stock_codes(stock_pool_path)
    stock_codes = filter_stock_codes_no_bj(stock_codes)
    if args.limit_stocks:
        stock_codes = stock_codes[: args.limit_stocks]
    tables = [item.strip() for item in args.tables.split(",") if item.strip()]
    print(
        f"raw_update_start stocks={len(stock_codes)} tables={','.join(tables)} end={args.end} "
        f"stock_pool={stock_pool_path}",
        flush=True,
    )
    if not args.skip_preflight:
        preflight = build_source_preflight_report(ts_pro, tables, args.end, stock_codes, DEFAULT_INDEX_CODES)
        preflight_path = write_source_preflight_report(data_dir, preflight, args.preflight_report)
        print(f"source_preflight_report path={preflight_path} gate_pass={preflight['gate_pass']}", flush=True)
        if not preflight["gate_pass"]:
            print(f"source_preflight_blocked blockers={json.dumps(preflight['blockers'], ensure_ascii=False)}", flush=True)
            raise SystemExit(3)
        if args.preflight_only:
            print("source_preflight_done", flush=True)
            return

    for table in tables:
        spec = RAW_TABLE_SPECS[table]
        path = data_dir / spec.file_name
        start = args.start if args.full_refresh else next_start_date(path, args.start, spec.date_column)
        if spec.per_stock:
            frames = []
            existing_codes = set() if args.full_refresh else load_existing_stock_codes(path, spec.key_columns[0])
            incremental_codes = stock_codes if args.full_refresh else [code for code in stock_codes if code in existing_codes]
            missing_codes = [] if args.full_refresh else [code for code in stock_codes if code not in existing_codes]

            if start <= args.end and incremental_codes:
                for idx, code in enumerate(incremental_codes, start=1):
                    frame = _fetch_stock_table(ts_pro, table, code, start, args.end)
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                    if idx % 100 == 0:
                        print(
                            f"table_progress table={table} phase=incremental stocks={idx}/{len(incremental_codes)}",
                            flush=True,
                        )
            elif start > args.end and not missing_codes:
                print(f"table_skip table={table} start={start} end={args.end}", flush=True)
                continue

            if missing_codes:
                for idx, code in enumerate(missing_codes, start=1):
                    frame = _fetch_stock_table(ts_pro, table, code, args.start, args.end)
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                    if idx % 100 == 0:
                        print(
                            f"table_progress table={table} phase=missing stocks={idx}/{len(missing_codes)}",
                            flush=True,
                        )
            rows = filter_frame_no_bj(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
        else:
            if start > args.end:
                print(f"table_skip table={table} start={start} end={args.end}", flush=True)
                continue
            rows = _fetch_date_table(ts_pro, table, start, args.end, DEFAULT_INDEX_CODES)
        if table != "index_daily":
            rows = filter_frame_no_bj(rows)
        added = append_parquet_dedup(path, rows, spec.key_columns)
        if table in RISK_EVENT_API_SPECS and not path.exists():
            rows.to_parquet(path, index=False)
        print(f"table_done table={table} downloaded={added} file={path}", flush=True)
        if table in RISK_EVENT_API_SPECS:
            duckdb_path = _sync_risk_event_duckdb(data_dir, table, path)
            print(f"duckdb_synced table={table} file={duckdb_path}", flush=True)
        if args.sync_sqlite:
            sqlite_name = SQLITE_TABLE_NAMES.get(table, table)
            _sync_sqlite_table(data_dir, spec.file_name, sqlite_name)
            print(f"sqlite_synced table={sqlite_name}", flush=True)
    print("raw_update_done", flush=True)


if __name__ == "__main__":
    main()
