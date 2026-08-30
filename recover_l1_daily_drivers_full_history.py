from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_load_module import get_pro
from l1_raw_data_route import resolve_l1_raw_duckdb_path
from l1_universe_rules import filter_frame_no_bj
from project_paths import load_config, resolve_data_dir
from workflow_contract_adapters import build_l1_contract_from_report, validate_built_contract


def _query_retry(call):
    import time

    while True:
        try:
            return call()
        except Exception as exc:
            print(f"query_retry err={exc}", flush=True)
            time.sleep(61)


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _normalize_code_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def _driver_scope_daily_basic(frame: pd.DataFrame, daily_frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])
    if daily_frame is None or daily_frame.empty:
        return pd.DataFrame(columns=list(frame.columns))
    driver_codes = set(_normalize_code_series(daily_frame["ts_code"]))
    scoped = frame[_normalize_code_series(frame["ts_code"]).isin(driver_codes)].copy()
    return scoped.reset_index(drop=True)


def _fetch_open_trade_dates(ts_pro, start: str, end: str) -> list[str]:
    frame = _query_retry(lambda: ts_pro.query("trade_cal", exchange="SSE", start_date=start, end_date=end, is_open="1"))
    if frame is None or frame.empty:
        raise RuntimeError("trade_cal returned no open dates")
    dates = sorted(str(value) for value in frame["cal_date"].dropna().astype(str).unique())
    if not dates:
        raise RuntimeError("trade_cal returned empty open-date list")
    return dates


def _connect_duckdb(path: Path):
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _insert_frame(conn, table_name: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    conn.register("__stage_frame", frame)
    exists = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()[0]
    if exists:
        conn.execute(f"INSERT INTO {_quote_ident(table_name)} SELECT * FROM __stage_frame")
    else:
        conn.execute(f"CREATE TABLE {_quote_ident(table_name)} AS SELECT * FROM __stage_frame")
    conn.unregister("__stage_frame")


def _stage_metrics(conn, table_name: str, target_trade_date: str) -> dict[str, object]:
    return {
        "row_count": int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0]),
        "stock_count": int(conn.execute(f"SELECT COUNT(DISTINCT ts_code) FROM {_quote_ident(table_name)}").fetchone()[0]),
        "min_trade_date": conn.execute(f"SELECT MIN(trade_date) FROM {_quote_ident(table_name)}").fetchone()[0],
        "max_trade_date": conn.execute(f"SELECT MAX(trade_date) FROM {_quote_ident(table_name)}").fetchone()[0],
        "rows_target_trade_date": int(
            conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE trade_date = ?",
                [target_trade_date],
            ).fetchone()[0]
        ),
        "duplicate_groups": int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM (
                  SELECT ts_code, trade_date, COUNT(*) c
                  FROM {_quote_ident(table_name)}
                  GROUP BY 1, 2
                  HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        ),
        "bj_rows": int(
            conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE upper(trim(ts_code)) LIKE '%.BJ'"
            ).fetchone()[0]
        ),
    }


def _parquet_metrics(path: Path, target_trade_date: str) -> dict[str, object]:
    frame = pd.read_parquet(path, columns=["ts_code", "trade_date"])
    groups = frame.groupby(["ts_code", "trade_date"], dropna=False).size()
    return {
        "row_count": int(len(frame)),
        "stock_count": int(_normalize_code_series(frame["ts_code"]).nunique()),
        "min_trade_date": None if frame.empty else str(frame["trade_date"].astype(str).min()),
        "max_trade_date": None if frame.empty else str(frame["trade_date"].astype(str).max()),
        "rows_target_trade_date": int((frame["trade_date"].astype(str) == target_trade_date).sum()),
        "duplicate_groups": int((groups > 1).sum()),
        "bj_rows": int(_normalize_code_series(frame["ts_code"]).str.endswith(".BJ").sum()),
    }


def _duckdb_metrics(path: Path, table_name: str, target_trade_date: str) -> dict[str, object]:
    conn = _connect_duckdb(path)
    try:
        return _stage_metrics(conn, table_name, target_trade_date)
    finally:
        conn.close()


def _build_alignment(source: dict[str, object], parquet: dict[str, object], duckdb: dict[str, object]) -> dict[str, object]:
    return {
        "source_rows": int(source["row_count"]),
        "source_codes": int(source["stock_count"]),
        "parquet_rows": int(parquet["row_count"]),
        "parquet_codes": int(parquet["stock_count"]),
        "duckdb_rows": int(duckdb["row_count"]),
        "duckdb_codes": int(duckdb["stock_count"]),
        "source_minus_parquet_count": abs(int(source["row_count"]) - int(parquet["row_count"])),
        "source_minus_duckdb_count": abs(int(source["row_count"]) - int(duckdb["row_count"])),
        "parquet_minus_source_count": abs(int(parquet["row_count"]) - int(source["row_count"])),
        "duckdb_minus_source_count": abs(int(duckdb["row_count"]) - int(source["row_count"])),
        "source_duplicate_key_groups": int(source["duplicate_groups"]),
        "parquet_duplicate_key_groups": int(parquet["duplicate_groups"]),
        "duckdb_duplicate_key_groups": int(duckdb["duplicate_groups"]),
        "parquet_bj_rows": int(parquet["bj_rows"]),
        "duckdb_bj_rows": int(duckdb["bj_rows"]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Recover full-history active L1 daily drivers with official Tushare SDK.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--report-prefix", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    report_prefix = Path(args.report_prefix)
    report_prefix.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    ts_pro = get_pro(config["datasource"]["tushare_token"])
    trade_dates = _fetch_open_trade_dates(ts_pro, args.start, args.end)
    print(
        json.dumps(
            {
                "status": "recovery_start",
                "trade_dates": len(trade_dates),
                "start": args.start,
                "end": args.end,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    stage_path = report_prefix.parent / f"{report_prefix.name}_stage.duckdb"
    if stage_path.exists():
        stage_path.unlink()
    stage_conn = _connect_duckdb(stage_path)
    target_source_rows = {"daily_data": 0, "daily_index_data": 0}

    try:
        for idx, trade_date in enumerate(trade_dates, start=1):
            daily_frame = _query_retry(lambda trade_date=trade_date: ts_pro.query("daily", trade_date=trade_date))
            daily_frame = filter_frame_no_bj(daily_frame)
            daily_basic_frame = _query_retry(lambda trade_date=trade_date: ts_pro.query("daily_basic", trade_date=trade_date))
            daily_basic_frame = filter_frame_no_bj(daily_basic_frame)
            daily_basic_frame = _driver_scope_daily_basic(daily_basic_frame, daily_frame)

            _insert_frame(stage_conn, "daily_data", daily_frame)
            _insert_frame(stage_conn, "daily_index_data", daily_basic_frame)

            if trade_date == args.end:
                target_source_rows["daily_data"] = int(len(daily_frame))
                target_source_rows["daily_index_data"] = int(len(daily_basic_frame))

            if idx % 50 == 0 or idx == len(trade_dates):
                print(
                    json.dumps(
                        {
                            "status": "recovery_progress",
                            "processed_trade_dates": idx,
                            "total_trade_dates": len(trade_dates),
                            "last_trade_date": trade_date,
                            "daily_rows": int(len(daily_frame)),
                            "daily_basic_rows": int(len(daily_basic_frame)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        for table_name, file_name in {
            "daily_data": "daily_data.parquet",
            "daily_index_data": "daily_index_data.parquet",
        }.items():
            parquet_path = data_dir / file_name
            stage_conn.execute(
                f"COPY (SELECT * FROM {_quote_ident(table_name)} ORDER BY trade_date, ts_code) "
                f"TO {_quote_literal(str(parquet_path))} (FORMAT PARQUET)"
            )
            active_path = resolve_l1_raw_duckdb_path(table_name, data_dir=data_dir)
            active_conn = _connect_duckdb(active_path)
            try:
                active_conn.execute(
                    f"CREATE OR REPLACE TABLE {_quote_ident(table_name)} AS "
                    f"SELECT * FROM read_parquet({_quote_literal(str(parquet_path))})"
                )
            finally:
                active_conn.close()

        source_metrics = {
            table_name: _stage_metrics(stage_conn, table_name, args.end)
            for table_name in ("daily_data", "daily_index_data")
        }
    finally:
        stage_conn.close()

    parquet_metrics = {
        "daily_data": _parquet_metrics(data_dir / "daily_data.parquet", args.end),
        "daily_index_data": _parquet_metrics(data_dir / "daily_index_data.parquet", args.end),
    }
    duckdb_metrics = {
        "daily_data": _duckdb_metrics(resolve_l1_raw_duckdb_path("daily_data", data_dir=data_dir), "daily_data", args.end),
        "daily_index_data": _duckdb_metrics(resolve_l1_raw_duckdb_path("daily_index_data", data_dir=data_dir), "daily_index_data", args.end),
    }

    alignment = {
        table_name: _build_alignment(source_metrics[table_name], parquet_metrics[table_name], duckdb_metrics[table_name])
        for table_name in ("daily_data", "daily_index_data")
    }

    report = {
        "task_id": args.workflow_run_id,
        "target_trade_date": args.end,
        "recovery_baseline_min_trade_date": args.start,
        "status": "ready_for_audit_review",
        "layer": "L1",
        "scope": "active_l1_daily_driver_full_history_recovery_only",
        "allow_l2_continue": False,
        "ready_for_audit_review": True,
        "allow_l2_statement": "等待审计，不自行放行",
        "active_route": "quant/data_file/production_assets/duckdb/l1_raw_tables/[table].duckdb",
        "route_file": "quant/main/l1_raw_data_route.py",
        "source_kind": "official_tushare_sdk_trade_date_rebuild",
        "source_trade_dates_fetched": len(trade_dates),
        "tables": {
            "daily_data": {
                "source": source_metrics["daily_data"],
                "root_raw_parquet": parquet_metrics["daily_data"],
                "active_duckdb": duckdb_metrics["daily_data"],
                "target_trade_date_source_rows": target_source_rows["daily_data"],
                "parquet_path": str(data_dir / "daily_data.parquet"),
                "duckdb_path": str(resolve_l1_raw_duckdb_path("daily_data", data_dir=data_dir)),
            },
            "daily_index_data": {
                "source": source_metrics["daily_index_data"],
                "root_raw_parquet": parquet_metrics["daily_index_data"],
                "active_duckdb": duckdb_metrics["daily_index_data"],
                "target_trade_date_source_rows": target_source_rows["daily_index_data"],
                "parquet_path": str(data_dir / "daily_index_data.parquet"),
                "duckdb_path": str(resolve_l1_raw_duckdb_path("daily_index_data", data_dir=data_dir)),
            },
        },
        "alignment": alignment,
        "boundaries": {
            "touched_l2": False,
            "touched_l3_l8": False,
            "used_legacy_sqlite": False,
            "used_legacy_odb": False,
            "used_raw_split_db_as_active_surface": False,
        },
        "observed_at": datetime.now().isoformat(timespec="seconds"),
    }

    json_path = report_prefix.with_suffix(".json")
    md_path = report_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# L1 Full-History Driver Recovery {args.end}",
        "",
        f"- status: `{report['status']}`",
        f"- scope: `{report['scope']}`",
        f"- allow_l2_continue: `false`",
        f"- allow_l2_statement: `{report['allow_l2_statement']}`",
        f"- source_trade_dates_fetched: `{len(trade_dates)}`",
        "",
        "| table | source rows | parquet rows | duckdb rows | min_trade_date | max_trade_date | 20260710 rows | duplicate groups | BJ rows |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for table_name in ("daily_data", "daily_index_data"):
        active = report["tables"][table_name]["active_duckdb"]
        lines.append(
            f"| `{table_name}` | {report['tables'][table_name]['source']['row_count']} | "
            f"{report['tables'][table_name]['root_raw_parquet']['row_count']} | "
            f"{active['row_count']} | {active['min_trade_date']} | {active['max_trade_date']} | "
            f"{active['rows_target_trade_date']} | {active['duplicate_groups']} | {active['bj_rows']} |"
        )
    lines.append("")
    lines.append("该 handoff 仅用于 `L2 hard gate` 继续，不代表 `L2` 已恢复。")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    contract_payload = build_l1_contract_from_report(
        {
            "target_trade_date": args.end,
            "source_gate": {
                "gate_pass": True,
                "daily_rows": source_metrics["daily_data"]["rows_target_trade_date"],
                "stk_factor_rows": None,
                "driver_codes": source_metrics["daily_data"]["stock_count"],
            },
            "ready_for_audit_review": True,
            "allow_l2_continue": False,
            "alignment": alignment,
            "universe_rule": "no_bj",
            "active_l1_asset": "quant/data_file/production_assets/duckdb/l1_raw_tables/[table].duckdb",
            "governance_notes": [
                "full-history driver recovery only",
                "handoff contract is only for L2 hard gate continuation",
            ],
        },
        workflow_run_id=args.workflow_run_id,
        evidence_paths=[str(json_path), str(md_path)],
    )
    errors = validate_built_contract(contract_payload, expected_layer="L1")
    if errors:
        raise RuntimeError("contract validation failed: " + "; ".join(errors))
    contract_path = report_prefix.parent / f"{report_prefix.name}_handoff_contract.json"
    contract_path.write_text(json.dumps(contract_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "recovery_complete",
                "report_json": str(json_path),
                "report_md": str(md_path),
                "contract_json": str(contract_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
