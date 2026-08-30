from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from data_load_module import get_pro
from l1_raw_data_route import resolve_l1_raw_duckdb_path
from project_paths import load_config, resolve_data_dir
from tushare_stock_risk_event_contract import (
    API_SPECS,
    build_l2_events,
    build_signal_slice,
    normalize_yyyymmdd,
)


EVENT_TABLE = "stock_risk_events"
SIGNAL_TABLE = "stock_risk_signal"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path, table: str, where: str | None = None, params=None) -> pd.DataFrame:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        sql = f"SELECT * FROM {quote_ident(table)}"
        if where:
            sql += f" WHERE {where}"
        return connection.execute(sql, params or []).df()
    finally:
        connection.close()


def load_target_l1(data_dir: Path, target_trade_date: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for spec in API_SPECS:
        path = resolve_l1_raw_duckdb_path(
            spec.api,
            data_dir=data_dir,
            require_exists=True,
        )
        frames[spec.api] = _read_table(
            path,
            spec.api,
            f"CAST({quote_ident(spec.date_column)} AS VARCHAR)=?",
            [target_trade_date],
        )
    return frames


def load_all_l1(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        spec.api: _read_table(
            resolve_l1_raw_duckdb_path(
                spec.api,
                data_dir=data_dir,
                require_exists=True,
            ),
            spec.api,
        )
        for spec in API_SPECS
    }


def load_official_calendar(pro: Any, start_date: str, end_date: str) -> list[str]:
    frame = pro.query(
        "trade_cal",
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
        fields="cal_date,is_open",
    )
    dates = sorted(frame["cal_date"].astype(str).unique().tolist())
    if not dates:
        raise RuntimeError("official SSE trading calendar returned zero open sessions")
    return dates


def _copy_or_bootstrap(
    target: Path,
    *,
    bootstrap: Path | None,
    candidate: Path,
) -> None:
    candidate.parent.mkdir(parents=True, exist_ok=True)
    source = target if target.exists() else bootstrap
    if source is None or not source.is_file():
        raise FileNotFoundError(
            f"missing L2 asset {target}; first run requires an audited --bootstrap-root"
        )
    shutil.copy2(source, candidate)


def _write_one_table(path: Path, table: str, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        connection.register("payload", frame)
        connection.execute(f"CREATE TABLE {quote_ident(table)} AS SELECT * FROM payload")
    finally:
        connection.close()


def _commit_candidates(pairs: list[tuple[Path, Path]]) -> None:
    token = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    existed = {target: target.exists() for _, target in pairs}
    try:
        for _, target in pairs:
            if target.exists():
                backup = target.with_name(f".{target.name}.{token}.rollback")
                shutil.copy2(target, backup)
                backups[target] = backup
        for candidate, target in pairs:
            os.replace(candidate, target)
    except Exception:
        for _, target in reversed(pairs):
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
            elif not existed[target] and target.exists():
                target.unlink()
        raise
    finally:
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def _build_signal_history(
    events: pd.DataFrame,
    open_dates: list[str],
    end_date: str,
) -> pd.DataFrame:
    slices = [
        build_signal_slice(events, open_dates, signal_date)
        for signal_date in open_dates
        if signal_date <= end_date
    ]
    slices = [frame for frame in slices if not frame.empty]
    if slices:
        return pd.concat(slices, ignore_index=True)
    return build_signal_slice(events, open_dates, end_date)


def _replace_target_slice(
    path: Path,
    table: str,
    date_column: str,
    target_date: str,
    frame: pd.DataFrame,
) -> None:
    connection = duckdb.connect(str(path))
    try:
        columns = [row[1] for row in connection.execute(
            f"PRAGMA table_info({quote_ident(table)})"
        ).fetchall()]
        if set(columns) != set(frame.columns):
            raise RuntimeError(
                f"schema drift for {table}: asset={columns}, incoming={list(frame.columns)}"
            )
        payload = frame.loc[:, columns].copy()
        connection.register("__incoming", payload)
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            f"DELETE FROM {quote_ident(table)} "
            f"WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?",
            [target_date],
        )
        selected = ",".join(quote_ident(column) for column in columns)
        connection.execute(
            f"INSERT INTO {quote_ident(table)} ({selected}) "
            f"SELECT {selected} FROM __incoming"
        )
        actual = int(connection.execute(
            f"SELECT COUNT(*) FROM {quote_ident(table)} "
            f"WHERE CAST({quote_ident(date_column)} AS VARCHAR)=?",
            [target_date],
        ).fetchone()[0])
        if actual != len(payload):
            raise RuntimeError(f"target row mismatch for {table}: {actual} != {len(payload)}")
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        connection.close()


def _quality(path: Path, table: str, keys: list[str], target_column: str, target: str) -> dict[str, Any]:
    key_sql = ",".join(quote_ident(key) for key in keys)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
        ).fetchall()]
        rows, target_rows, future_rows, duplicates, bj_rows = connection.execute(
            f"SELECT COUNT(*), "
            f"COUNT(*) FILTER (WHERE CAST({quote_ident(target_column)} AS VARCHAR)=?), "
            f"COUNT(*) FILTER (WHERE CAST({quote_ident(target_column)} AS VARCHAR)>?), "
            f"(SELECT COUNT(*) FROM (SELECT {key_sql}, COUNT(*) c "
            f"FROM {quote_ident(table)} GROUP BY {key_sql} HAVING COUNT(*)>1)), "
            f"COUNT(*) FILTER (WHERE UPPER(CAST(stock_code AS VARCHAR)) LIKE '%.BJ') "
            f"FROM {quote_ident(table)}",
            [target, target],
        ).fetchone()
    finally:
        connection.close()
    return {
        "path": str(path),
        "table": table,
        "rows": int(rows),
        "target_rows": int(target_rows),
        "future_rows": int(future_rows),
        "duplicate_groups": int(duplicates),
        "bj_rows": int(bj_rows),
        "one_table_one_file": tables == [table],
        "sha256": sha256_file(path),
    }


def integrate(
    *,
    target_trade_date: str,
    data_dir: Path,
    output_root: Path,
    pro: Any,
    bootstrap_root: Path | None = None,
) -> dict[str, Any]:
    target_trade_date = normalize_yyyymmdd(target_trade_date)
    raw_by_api = load_target_l1(data_dir, target_trade_date)
    output_root.mkdir(parents=True, exist_ok=True)
    event_target = output_root / "l2_stock_risk_events.duckdb"
    signal_target = output_root / "l2_stock_risk_signal.duckdb"
    token = uuid.uuid4().hex
    event_candidate = output_root / f".{event_target.name}.{token}.tmp"
    signal_candidate = output_root / f".{signal_target.name}.{token}.tmp"
    bootstrap_event = (
        bootstrap_root / "l2_stock_risk_events.duckdb" if bootstrap_root else None
    )
    try:
        event_source = event_target if event_target.exists() else bootstrap_event
        if event_source is not None and event_source.is_file():
            shutil.copy2(event_source, event_candidate)
            existing_events = _read_table(event_candidate, EVENT_TABLE)
            dates = existing_events["event_date"].dropna().astype(str)
            calendar_start = min(
                dates.min() if not dates.empty else target_trade_date,
                target_trade_date,
            )
            all_raw = None
        else:
            all_raw = load_all_l1(data_dir)
            source_dates = [
                str(frame[spec.date_column].min())
                for spec in API_SPECS
                for frame in [all_raw[spec.api]]
                if not frame.empty
            ]
            calendar_start = min(source_dates + [target_trade_date])
        calendar_end = (
            pd.to_datetime(target_trade_date, format="%Y%m%d") + pd.Timedelta(days=45)
        ).strftime("%Y%m%d")
        open_dates = load_official_calendar(pro, calendar_start, calendar_end)
        if target_trade_date not in open_dates:
            raise RuntimeError(f"target date is not an official open session: {target_trade_date}")
        if all_raw is not None:
            existing_events = build_l2_events(all_raw, open_dates)
            _write_one_table(event_candidate, EVENT_TABLE, existing_events)

        incoming_events = build_l2_events(raw_by_api, open_dates)
        _replace_target_slice(
            event_candidate,
            EVENT_TABLE,
            "event_date",
            target_trade_date,
            incoming_events,
        )
        all_events = _read_table(event_candidate, EVENT_TABLE)
        events_as_of_target = all_events[
            all_events["event_date"].astype(str).le(target_trade_date)
        ].copy()
        signal_history = _build_signal_history(
            events_as_of_target,
            open_dates,
            target_trade_date,
        )
        _write_one_table(signal_candidate, SIGNAL_TABLE, signal_history)
        event_quality = _quality(
            event_candidate,
            EVENT_TABLE,
            ["event_id"],
            "event_date",
            target_trade_date,
        )
        signal_quality = _quality(
            signal_candidate,
            SIGNAL_TABLE,
            ["signal_date", "stock_code"],
            "signal_date",
            target_trade_date,
        )
        if any(
            (
                event_quality["duplicate_groups"],
                event_quality["bj_rows"],
                signal_quality["future_rows"],
                signal_quality["duplicate_groups"],
                signal_quality["bj_rows"],
            )
        ) or not event_quality["one_table_one_file"] or not signal_quality["one_table_one_file"]:
            raise RuntimeError("L2 stock risk event quality gate failed")
        _commit_candidates(
            [(event_candidate, event_target), (signal_candidate, signal_target)]
        )
        event_quality = _quality(
            event_target,
            EVENT_TABLE,
            ["event_id"],
            "event_date",
            target_trade_date,
        )
        signal_quality = _quality(
            signal_target,
            SIGNAL_TABLE,
            ["signal_date", "stock_code"],
            "signal_date",
            target_trade_date,
        )
    finally:
        for path in (event_candidate, signal_candidate):
            if path.exists():
                path.unlink()
    return {
        "status": "completed",
        "target_trade_date": target_trade_date,
        "l1_target_rows": {api: int(len(frame)) for api, frame in raw_by_api.items()},
        "events": event_quality,
        "signal": signal_quality,
        "pit": {
            "same_day_source_event_visible": False,
            "availability": "first official open session strictly after event date",
            "alert_end_date_inclusive": True,
        },
        "boundary": {
            "stock_daily_data_modified": False,
            "strategy_or_trading_triggered": False,
            "network_used_only_for_tushare_and_official_trade_calendar": True,
        },
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update L2 Tushare stock-risk event and signal tables for one trade date."
    )
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--bootstrap-root")
    parser.add_argument("--summary-json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else data_dir / "production_assets" / "duckdb"
    )
    bootstrap_root = Path(args.bootstrap_root) if args.bootstrap_root else None
    pro = get_pro(str(load_config(args.config)["datasource"]["tushare_token"]))
    try:
        result = integrate(
            target_trade_date=args.target_trade_date,
            data_dir=data_dir,
            output_root=output_root,
            pro=pro,
            bootstrap_root=bootstrap_root,
        )
    except Exception as exc:
        result = {
            "status": "blocked",
            "target_trade_date": str(args.target_trade_date),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "active_written": False,
        }
        if args.summary_json:
            Path(args.summary_json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(result, ensure_ascii=False))
        return 3
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
