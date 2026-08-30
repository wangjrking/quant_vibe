from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import duckdb
import pandas as pd

import akshare_l1_experimental as base


TASK_ID = "investment-platform-akshare-test-l1-scaleout-20260715"
ROOT = base.DEFAULT_OUTPUT_ROOT
CHECKPOINT_ROOT = ROOT / "checkpoints"
REPORT_ROOT = base.DEFAULT_REPORT_ROOT
TRADE_GATE = REPORT_ROOT / "investment_platform_akshare_l1_scaleout_trade_gate_20260715.json"
REPORT_JSON = REPORT_ROOT / "investment_platform_akshare_l1_scaleout_20260715.json"
REPORT_MD = REPORT_JSON.with_suffix(".md")
HASH_JSON = REPORT_ROOT / "investment_platform_akshare_l1_scaleout_20260715_hashes.json"
FINANCIAL_SCALEOUT_REPORT = REPORT_ROOT / "investment_platform_akshare_financial_scaleout_latest.json"
MANIFEST = ROOT / "manifest.json"
USER_DB = base.PROJECT_ROOT / "site" / "backend" / "platform.sqlite"
ANNOUNCEMENT_CHECKPOINT = CHECKPOINT_ROOT / "company_announcements_30d.json"
FINANCIAL_CHECKPOINT = CHECKPOINT_ROOT / "financial_indicator_batch.json"
NEWS_CHECKPOINT = CHECKPOINT_ROOT / "stock_news_on_demand_cache.json"
MINUTE_CHECKPOINT = CHECKPOINT_ROOT / "stock_minutes_1m_on_demand_cache.json"
NEWS_TTL_SECONDS = 6 * 60 * 60
MINUTE_TTL_SECONDS = 7 * 24 * 60 * 60
NEWS_MAX_STOCKS = 10
NEWS_MAX_CALLS = 10
MINUTE_MAX_STOCKS = 5
MINUTE_MAX_CALLS = 5
FINANCIAL_BATCH_SIZE = 200


class SourceCallFailure(RuntimeError):
    def __init__(self, error: Exception, attempts: int) -> None:
        super().__init__(str(error))
        self.error_type = type(error).__name__
        self.attempts = attempts


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(
    path: Path,
    payload: dict[str, Any],
    max_replace_attempts: int = 4,
    base_delay_seconds: float = 0.05,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Publish JSON without reusing the shared ``.tmp`` path.

    A unique candidate avoids consuming a stale file left by an interrupted
    run.  Windows can briefly deny ``os.replace`` while a reader releases its
    handle, so only that permission failure is retried with bounded backoff.
    """
    if max_replace_attempts < 1:
        raise ValueError("max_replace_attempts must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid.uuid4().hex}.candidate")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    with candidate.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    json.loads(candidate.read_text(encoding="utf-8"))
    for attempt in range(1, max_replace_attempts + 1):
        try:
            os.replace(candidate, path)
            return path
        except PermissionError:
            if attempt == max_replace_attempts:
                raise
            sleep(base_delay_seconds * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def retry_call(
    call: Callable[[], pd.DataFrame],
    max_retries: int = 2,
    base_delay_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, int]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            return call(), attempt
        except Exception as exc:  # source boundary
            last_error = exc
            if attempt <= max_retries:
                sleep(base_delay_seconds * (2 ** (attempt - 1)))
    assert last_error is not None
    raise SourceCallFailure(last_error, max_retries + 1)


def cache_fresh(cached_at: str | None, ttl_seconds: int, current: datetime | None = None) -> bool:
    if not cached_at:
        return False
    current = current or datetime.now().astimezone()
    cached = datetime.fromisoformat(cached_at)
    if cached.tzinfo is None:
        cached = cached.replace(tzinfo=current.tzinfo)
    return 0 <= (current - cached).total_seconds() < ttl_seconds


def bounded_codes(codes: list[str], max_stocks: int, max_calls: int) -> list[str]:
    normalized = list(dict.fromkeys(base._normalize_code(code) for code in codes))
    if any(code.upper().endswith(".BJ") for code in normalized):
        raise ValueError("no-BJ cache contract rejects Beijing Stock Exchange codes")
    if len(normalized) > max_stocks or len(normalized) > max_calls:
        raise ValueError(f"request exceeds cache limits: stocks={len(normalized)}, max_stocks={max_stocks}, max_calls={max_calls}")
    return normalized


def table_path(table: str) -> Path:
    return ROOT / table / f"{table}.duckdb"


def read_table(table: str) -> pd.DataFrame:
    path = table_path(table)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        return connection.execute(f'SELECT * FROM "{table}"').fetchdf()
    finally:
        connection.close()


def enforce_no_bj(frame: pd.DataFrame) -> pd.DataFrame:
    if "ts_code" not in frame.columns:
        return frame.copy()
    result = frame.loc[~frame["ts_code"].astype(str).str.upper().str.endswith(".BJ")].copy()
    if result["ts_code"].astype(str).str.upper().str.endswith(".BJ").any():
        raise RuntimeError("no-BJ write gate failed")
    return result.reset_index(drop=True)


def atomic_write_table(table: str, frame: pd.DataFrame) -> Path:
    root = base.validate_output_root(ROOT)
    frame = enforce_no_bj(frame)
    target = table_path(table)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.scaleout.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = duckdb.connect(str(temporary))
    try:
        connection.register("incoming", frame)
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM incoming')
        connection.unregister("incoming")
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE'"
        ).fetchall()
        if tables != [(table,)]:
            raise RuntimeError(f"one-table-one-file failed: {tables}")
    finally:
        connection.close()
    os.replace(temporary, target)
    return target


def merge_deduplicated(current: pd.DataFrame, incoming: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if incoming.empty:
        return current
    combined = pd.concat([current, incoming], ignore_index=True, sort=False)
    return combined.drop_duplicates(keys, keep="last").reset_index(drop=True)


def create_trade_gate() -> dict[str, Any]:
    checked = datetime.now().astimezone()
    sys.path.insert(0, str(base.PROJECT_ROOT / "quant" / "main"))
    from data_load_module import get_pro
    from project_paths import load_config

    token = str(load_config(None)["datasource"]["tushare_token"])
    pro = get_pro(token)
    natural_date = checked.strftime("%Y%m%d")
    previous_date = (checked - timedelta(days=1)).strftime("%Y%m%d")
    calendar = pro.query("trade_cal", exchange="SSE", start_date=previous_date, end_date=natural_date)
    source_daily_rows = {
        previous_date: int(len(pro.query("daily", trade_date=previous_date))),
        natural_date: int(len(pro.query("daily", trade_date=natural_date))),
    }
    connection = duckdb.connect(str(base.PRODUCTION_ROOT / "duckdb" / "l1_raw_tables" / "daily_data.duckdb"), read_only=True)
    try:
        active_max, active_rows = connection.execute(
            "SELECT MAX(trade_date), COUNT(*) FILTER (WHERE trade_date=(SELECT MAX(trade_date) FROM daily_data)) FROM daily_data"
        ).fetchone()
    finally:
        connection.close()
    completed = str(active_max)
    if source_daily_rows.get(natural_date, 0) > 0 and checked.hour >= 16:
        completed = natural_date
    payload = {
        "task_id": TASK_ID,
        "checked_at": checked.isoformat(timespec="seconds"),
        "natural_date": natural_date,
        "official_tushare_sdk": True,
        "trade_calendar": calendar.to_dict("records"),
        "source_daily_rows": source_daily_rows,
        "active_daily_data_max_trade_date": str(active_max),
        "active_daily_data_max_date_rows_no_bj": int(active_rows),
        "latest_completed_trade_date": completed,
        "minute_cache_target_allowed": True,
        "reason": "natural date daily source is empty or session is not completed; use latest completed active/source-aligned date",
        "business_data_written": False,
    }
    atomic_json(TRADE_GATE, payload)
    return payload


def formal_no_bj_pool() -> list[str]:
    path = base.PRODUCTION_ROOT / "duckdb" / "l1_raw_tables" / "stock_basic_data.duckdb"
    connection = duckdb.connect(str(path), read_only=True)
    try:
        rows = connection.execute(
            "SELECT DISTINCT ts_code FROM stock_basic_data WHERE ts_code NOT LIKE '%.BJ' ORDER BY ts_code"
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def platform_watch_pool() -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{USER_DB.as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT user_id, stock_code, created_at FROM watchlist "
            "WHERE user_id NOT LIKE 'watch_%' ORDER BY user_id, id"
        ).fetchall()
    finally:
        connection.close()
    codes = list(dict.fromkeys(base._normalize_code(row[1]) for row in rows))
    return {
        "trigger_source": "test_platform_current_watchlist",
        "read_only_user_state_source": str(USER_DB),
        "codes": codes,
        "records": [{"stock_code": base._normalize_code(row[1]), "created_at": row[2]} for row in rows],
        "legacy_odb_used": False,
    }


def announcement_checkpoint(start: str, end: str) -> dict[str, Any]:
    dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D").strftime("%Y%m%d").tolist()
    default = {
        "task_id": TASK_ID,
        "table": "company_announcements",
        "window": {"start": start, "end": end, "calendar_days": len(dates)},
        "resume_policy": "skip completed/empty dates; retry pending/running/failed dates; merge by announcement_id",
        "dates": {date: {"status": "pending", "attempts": 0} for date in dates},
        "updated_at": now_iso(),
    }
    checkpoint = load_json(ANNOUNCEMENT_CHECKPOINT, default)
    if checkpoint["window"] != default["window"]:
        raise RuntimeError("announcement checkpoint window mismatch")
    return checkpoint


def run_announcements(ak: Any, start: str, end: str) -> dict[str, Any]:
    checkpoint = announcement_checkpoint(start, end)
    current = read_table("company_announcements")
    current_time = datetime.now().astimezone()
    for date, item in checkpoint["dates"].items():
        if item["status"] in {"completed", "empty"}:
            refresh_after = item.get("refresh_after")
            should_refresh = bool(
                item.get("provisional_current_natural_day")
                and refresh_after
                and current_time >= datetime.fromisoformat(refresh_after)
            )
            if not should_refresh:
                continue
        item.update({"status": "running", "started_at": now_iso()})
        atomic_json(ANNOUNCEMENT_CHECKPOINT, checkpoint)
        try:
            source, attempts = retry_call(lambda date=date: ak.stock_notice_report(symbol="全部", date=date))
            item["attempts"] = int(item.get("attempts", 0)) + attempts
            item["source_rows"] = int(len(source))
            provisional = date == current_time.strftime("%Y%m%d")
            refresh_after = (current_time + timedelta(days=1)).replace(hour=0, minute=15, second=0, microsecond=0)
            if source.empty:
                item.update({"status": "empty", "source_empty": True, "completed_at": now_iso(), "landed_rows": 0})
            else:
                transformed = base.transform_announcements(source, now_iso())
                transformed = transformed[~transformed["ts_code"].astype(str).str.endswith(".BJ")]
                before_ids = set(current["announcement_id"].astype(str))
                current = merge_deduplicated(current, transformed, ["announcement_id"])
                atomic_write_table("company_announcements", current)
                landed = int((~transformed["announcement_id"].astype(str).isin(before_ids)).sum())
                item.update(
                    {
                        "status": "completed",
                        "source_empty": False,
                        "completed_at": now_iso(),
                        "source_bj_rows": int(source["代码"].astype(str).map(base._normalize_code).str.endswith(".BJ").sum()),
                        "transformed_no_bj_rows": int(len(transformed)),
                        "new_unique_rows": landed,
                    }
                )
            item["provisional_current_natural_day"] = provisional
            item["refresh_after"] = refresh_after.isoformat() if provisional else None
        except SourceCallFailure as exc:
            item.update(
                {
                    "status": "failed",
                    "attempts": int(item.get("attempts", 0)) + exc.attempts,
                    "error_type": exc.error_type,
                    "last_error": str(exc)[:500],
                    "failed_at": now_iso(),
                    "next_resume_point": date,
                }
            )
        checkpoint["updated_at"] = now_iso()
        atomic_json(ANNOUNCEMENT_CHECKPOINT, checkpoint)
    statuses = [item["status"] for item in checkpoint["dates"].values()]
    provisional_dates = [
        date for date, item in checkpoint["dates"].items()
        if item.get("provisional_current_natural_day")
    ]
    return {
        "checkpoint_path": str(ANNOUNCEMENT_CHECKPOINT),
        "completed_days": statuses.count("completed"),
        "empty_days": statuses.count("empty"),
        "failed_days": statuses.count("failed"),
        "pending_days": statuses.count("pending") + statuses.count("running"),
        "provisional_days": len(provisional_dates),
        "provisional_dates": provisional_dates,
        "next_resume_point": next((date for date, item in checkpoint["dates"].items() if item["status"] in {"pending", "running", "failed"}), None),
    }


def financial_checkpoint(pool: list[str], current: pd.DataFrame) -> dict[str, Any]:
    pool_hash = hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()
    existing_rows = current.groupby("ts_code").size().to_dict()
    default = {
        "task_id": TASK_ID,
        "table": "financial_indicator",
        "pool": {"source": "formal stock_basic_data read-only no-BJ", "total": len(pool), "sha256": pool_hash},
        "batch_size": FINANCIAL_BATCH_SIZE,
        "resume_policy": "skip completed/source_empty stocks; retry failed/pending with bounded exponential backoff",
        "stocks": {
            code: {
                "status": "completed" if code in existing_rows else "pending",
                "rows": int(existing_rows.get(code, 0)),
                "origin": "preexisting_experimental_asset" if code in existing_rows else None,
                "attempts": 0,
            }
            for code in pool
        },
        "updated_at": now_iso(),
    }
    checkpoint = load_json(FINANCIAL_CHECKPOINT, default)
    if checkpoint["pool"]["sha256"] != pool_hash:
        raise RuntimeError("formal no-BJ stock pool changed; start a reviewed checkpoint generation")
    return checkpoint


def run_financial_batch(
    ak: Any,
    pool: list[str],
    batch_size: int = FINANCIAL_BATCH_SIZE,
    flush_size: int = 25,
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 500:
        raise ValueError("financial batch size must be between 1 and 500")
    if flush_size < 1 or flush_size > 100:
        raise ValueError("financial flush size must be between 1 and 100")
    started = time.monotonic()
    current = read_table("financial_indicator")
    columns = list(current.columns)
    checkpoint = financial_checkpoint(pool, current)
    candidates = [
        code
        for code in pool
        if checkpoint["stocks"][code]["status"] in {"pending", "failed", "running", "staged"}
    ][:batch_size]
    processed = 0
    consecutive_failures = 0
    staged: list[tuple[str, pd.DataFrame]] = []

    def flush_staged() -> None:
        nonlocal current
        if not staged:
            return
        codes = [code for code, _ in staged]
        try:
            incoming = pd.concat([frame for _, frame in staged], ignore_index=True, sort=False)
            current = merge_deduplicated(current, incoming, ["ts_code", "report_date"])
            atomic_write_table("financial_indicator", current)
            completed_at = now_iso()
            for staged_code, frame in staged:
                checkpoint["stocks"][staged_code].update(
                    {
                        "status": "completed",
                        "rows": int(len(frame)),
                        "source_empty": False,
                        "completed_at": completed_at,
                    }
                )
            checkpoint["updated_at"] = completed_at
            atomic_json(FINANCIAL_CHECKPOINT, checkpoint)
            staged.clear()
        except Exception as exc:
            failed_at = now_iso()
            for staged_code in codes:
                checkpoint["stocks"][staged_code].update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "last_error": str(exc)[:500],
                        "failed_at": failed_at,
                        "next_resume_point": staged_code,
                    }
                )
            checkpoint["updated_at"] = failed_at
            atomic_json(FINANCIAL_CHECKPOINT, checkpoint)
            raise

    for code in candidates:
        item = checkpoint["stocks"][code]
        item.update({"status": "running", "started_at": now_iso()})
        atomic_json(FINANCIAL_CHECKPOINT, checkpoint)
        call_started = time.monotonic()
        try:
            source, attempts = retry_call(lambda code=code: ak.stock_financial_analysis_indicator_em(symbol=code, indicator="按报告期"))
            item["attempts"] = int(item.get("attempts", 0)) + attempts
            item["source_rows"] = int(len(source))
            if source.empty:
                item.update({"status": "source_empty", "rows": 0, "source_empty": True, "completed_at": now_iso()})
            else:
                transformed = base.transform_financial_indicator(source)
                if set(transformed.columns) != set(columns):
                    raise RuntimeError(f"SchemaDrift: expected={len(columns)} actual={len(transformed.columns)}")
                transformed = transformed[columns]
                staged.append((code, transformed))
                item.update({"status": "staged", "rows": int(len(transformed)), "source_empty": False})
            consecutive_failures = 0
        except (SourceCallFailure, Exception) as exc:
            failure = exc if isinstance(exc, SourceCallFailure) else SourceCallFailure(exc, 1)
            item.update(
                {
                    "status": "failed",
                    "attempts": int(item.get("attempts", 0)) + failure.attempts,
                    "error_type": failure.error_type,
                    "last_error": str(failure)[:500],
                    "failed_at": now_iso(),
                    "next_resume_point": code,
                }
            )
            consecutive_failures += 1
        item["elapsed_seconds"] = round(time.monotonic() - call_started, 3)
        checkpoint["updated_at"] = now_iso()
        atomic_json(FINANCIAL_CHECKPOINT, checkpoint)
        if len(staged) >= flush_size:
            flush_staged()
        processed += 1
        if consecutive_failures >= 5:
            break
    flush_staged()
    statuses = [checkpoint["stocks"][code]["status"] for code in pool]
    completed = statuses.count("completed")
    empty = statuses.count("source_empty")
    failed = statuses.count("failed")
    remaining = statuses.count("pending") + statuses.count("running") + statuses.count("staged") + failed
    elapsed = time.monotonic() - started
    terminal_in_run = max(processed, 1)
    avg = elapsed / terminal_in_run
    return {
        "checkpoint_path": str(FINANCIAL_CHECKPOINT),
        "pool_total": len(pool),
        "batch_size": batch_size,
        "flush_size": flush_size,
        "processed_this_run": processed,
        "completed_with_data": completed,
        "source_empty": empty,
        "failed": failed,
        "remaining": remaining,
        "elapsed_seconds": round(elapsed, 3),
        "average_seconds_per_stock_this_run": round(avg, 3),
        "estimated_total_batches": math.ceil(len(pool) / batch_size),
        "estimated_remaining_seconds_at_current_rate": round(remaining * avg, 1),
        "next_resume_point": next(
            (
                code
                for code in pool
                if checkpoint["stocks"][code]["status"] in {"pending", "failed", "running", "staged"}
            ),
            None,
        ),
    }


def update_financial_manifest_only(financial: dict[str, Any], pool_total: int) -> dict[str, Any]:
    """Refresh only the test financial asset contract after a resumable batch."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    old_asset = manifest["assets"]["financial_indicator"]
    asset = inspect_asset(
        "financial_indicator",
        ["ts_code", "report_date"],
        "report_date",
        old_asset["mapping_status"],
    )
    asset.update(
        {
            "status": "experimental/test_partial_resumable_batch_not_approved_for_production",
            "source": base.ASSET_CONTRACTS["financial_indicator"]["source"],
            "source_params": {"symbol": "one formal no-BJ ts_code per call", "indicator": "按报告期"},
            "date_semantics": "financial report period, not market trade date",
            "checkpoint_path": str(FINANCIAL_CHECKPOINT),
            "resume_policy": "per-stock skip completed/source_empty; bounded retry for failed/pending",
            "sample_scope": {
                "formal_no_bj_pool_total": pool_total,
                "latest_batch_size": financial["batch_size"],
            },
            "coverage": financial,
            "ttl_seconds": None,
            "adjustment": "not_applicable",
        }
    )
    manifest["assets"]["financial_indicator"] = asset
    manifest["financial_first_batch"] = financial
    manifest["coverage_summary"]["financial_terminal_stock_coverage"] = {
        "covered_stocks": financial["completed_with_data"] + financial["source_empty"],
        "pool_total": pool_total,
        "ratio": round((financial["completed_with_data"] + financial["source_empty"]) / pool_total, 6),
    }
    manifest["generated_at"] = now_iso()
    manifest["production_assets_touched"] = False
    manifest["route_registry_touched"] = False
    manifest["l2_l8_triggered"] = False
    manifest["production_approved"] = False
    atomic_json(MANIFEST, manifest)
    report = {
        "task_id": TASK_ID,
        "status": "experimental_financial_scaleout_completed_for_test_only",
        "generated_at": now_iso(),
        "financial": financial,
        "asset": asset,
        "production_assets_touched": False,
        "route_registry_touched": False,
        "l2_l8_triggered": False,
        "production_approved": False,
        "requires_read_only_reaudit": True,
    }
    atomic_json(FINANCIAL_SCALEOUT_REPORT, report)
    return report


def run_news_cache(ak: Any, trigger: dict[str, Any]) -> dict[str, Any]:
    codes = bounded_codes(trigger["codes"], NEWS_MAX_STOCKS, NEWS_MAX_CALLS)
    current = read_table("stock_news")
    checkpoint = load_json(NEWS_CHECKPOINT, {"task_id": TASK_ID, "table": "stock_news", "entries": {}, "requests": []})
    request = {"requested_at": now_iso(), "trigger_source": trigger["trigger_source"], "codes": codes, "hits": [], "misses": [], "calls": 0}
    for code in codes:
        latest = current.loc[current["ts_code"] == code, "snapshot_at"]
        cached_at = str(latest.max()) if not latest.empty else None
        if cache_fresh(cached_at, NEWS_TTL_SECONDS):
            request["hits"].append(code)
            continue
        request["misses"].append(code)
        if request["calls"] >= NEWS_MAX_CALLS:
            break
        try:
            remaining_budget = NEWS_MAX_CALLS - request["calls"]
            source, attempts = retry_call(
                lambda code=code: ak.stock_news_em(symbol=code[:6]),
                max_retries=min(2, remaining_budget - 1),
            )
            request["calls"] += attempts
            if source.empty:
                checkpoint["entries"][code] = {"status": "source_empty", "cached_at": now_iso(), "attempts": attempts}
            else:
                transformed = base.transform_stock_news(source, code, now_iso())
                transformed = transformed[~transformed["ts_code"].astype(str).str.endswith(".BJ")]
                current = merge_deduplicated(current, transformed, ["news_id"])
                atomic_write_table("stock_news", current)
                checkpoint["entries"][code] = {"status": "refreshed", "cached_at": now_iso(), "attempts": attempts, "rows": int(len(transformed))}
        except SourceCallFailure as exc:
            request["calls"] += exc.attempts
            checkpoint["entries"][code] = {"status": "failed", "attempts": exc.attempts, "error_type": exc.error_type, "last_error": str(exc)[:500]}
    request["completed_at"] = now_iso()
    checkpoint["contract"] = {
        "ttl_seconds": NEWS_TTL_SECONDS,
        "max_stocks_per_request": NEWS_MAX_STOCKS,
        "max_calls_per_run": NEWS_MAX_CALLS,
        "retry": {"max_retries": 2, "backoff_seconds": [1, 2]},
        "request_deduplication": "normalized unique ts_code preserving first occurrence",
        "expiration": "refresh only on trigger after TTL expiry",
        "natural_key": ["news_id"],
    }
    checkpoint["requests"].append(request)
    atomic_json(NEWS_CHECKPOINT, checkpoint)
    return {"checkpoint_path": str(NEWS_CHECKPOINT), **request, "ttl_seconds": NEWS_TTL_SECONDS}


def run_minute_cache(ak: Any, trigger: dict[str, Any], trade_date: str) -> dict[str, Any]:
    codes = bounded_codes(trigger["codes"], MINUTE_MAX_STOCKS, MINUTE_MAX_CALLS)
    current = read_table("stock_minutes_1m")
    checkpoint = load_json(MINUTE_CHECKPOINT, {"task_id": TASK_ID, "table": "stock_minutes_1m", "entries": {}, "requests": []})
    request = {"requested_at": now_iso(), "trigger_source": trigger["trigger_source"], "trade_date": trade_date, "codes": codes, "hits": [], "misses": [], "calls": 0}
    for code in codes:
        key = f"{code}|{trade_date}"
        entry = checkpoint["entries"].get(key, {})
        physical_rows = int(((current["ts_code"] == code) & (pd.to_datetime(current["trade_time"]).dt.strftime("%Y%m%d") == trade_date)).sum())
        if entry.get("status") == "completed" and physical_rows > 0 and cache_fresh(entry.get("cached_at"), MINUTE_TTL_SECONDS):
            request["hits"].append(key)
            continue
        request["misses"].append(key)
        if request["calls"] >= MINUTE_MAX_CALLS:
            break
        try:
            remaining_budget = MINUTE_MAX_CALLS - request["calls"]
            source, attempts = retry_call(
                lambda code=code: ak.stock_zh_a_minute(symbol=base._sina_code(code), period="1", adjust=""),
                max_retries=min(2, remaining_budget - 1),
            )
            request["calls"] += attempts
            transformed = base.transform_minutes(source, code)
            transformed = transformed[pd.to_datetime(transformed["trade_time"]).dt.strftime("%Y%m%d") == trade_date]
            if transformed.empty:
                checkpoint["entries"][key] = {"status": "source_empty", "cached_at": now_iso(), "attempts": attempts, "rows": 0}
            else:
                current = merge_deduplicated(current, transformed, ["ts_code", "trade_time"])
                atomic_write_table("stock_minutes_1m", current)
                checkpoint["entries"][key] = {"status": "completed", "cached_at": now_iso(), "attempts": attempts, "rows": int(len(transformed))}
        except SourceCallFailure as exc:
            request["calls"] += exc.attempts
            checkpoint["entries"][key] = {"status": "failed", "attempts": exc.attempts, "error_type": exc.error_type, "last_error": str(exc)[:500]}
    request["completed_at"] = now_iso()
    checkpoint["contract"] = {
        "cache_key": ["stock_code", "latest_completed_trade_date"],
        "ttl_seconds": MINUTE_TTL_SECONDS,
        "max_stocks_per_request": MINUTE_MAX_STOCKS,
        "max_calls_per_run": MINUTE_MAX_CALLS,
        "retry": {"max_retries": 2, "backoff_seconds": [1, 2]},
        "source": "AKShare.stock_zh_a_minute",
        "fallback_from": "AKShare.stock_zh_a_hist_min_em",
        "source_parameters": {"period": "1", "adjust": ""},
        "adjustment": "unadjusted/raw_price",
        "date_semantics": "historical bars for latest completed trade date; not real-time",
        "natural_key": ["ts_code", "trade_time"],
    }
    checkpoint["requests"].append(request)
    atomic_json(MINUTE_CHECKPOINT, checkpoint)
    return {"checkpoint_path": str(MINUTE_CHECKPOINT), **request, "ttl_seconds": MINUTE_TTL_SECONDS}


def inspect_asset(table: str, keys: list[str], date_field: str, mapping_status: str) -> dict[str, Any]:
    frame = read_table(table)
    entry = base.build_asset_entry(table, table_path(table), frame, keys, date_field, mapping_status)
    entry["bj_rows"] = int(frame["ts_code"].astype(str).str.upper().str.endswith(".BJ").sum()) if "ts_code" in frame else 0
    entry["natural_key_null_counts"] = {
        key: int(frame[key].isna().sum()) for key in keys
    }
    return entry


def update_manifest_and_report(
    gate: dict[str, Any],
    trigger: dict[str, Any],
    announcements: dict[str, Any],
    financial: dict[str, Any],
    news: dict[str, Any],
    minutes: dict[str, Any],
    pool_total: int,
) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    old_assets = manifest["assets"]
    assets = {
        "company_announcements": inspect_asset("company_announcements", ["announcement_id"], "announcement_date", old_assets["company_announcements"]["mapping_status"]),
        "stock_news": inspect_asset("stock_news", ["news_id"], "published_at", old_assets["stock_news"]["mapping_status"]),
        "social_sentiment_snapshot": inspect_asset("social_sentiment_snapshot", ["snapshot_at", "rank"], "snapshot_at", old_assets["social_sentiment_snapshot"]["mapping_status"]),
        "stock_minutes_1m": inspect_asset("stock_minutes_1m", ["ts_code", "trade_time"], "trade_time", old_assets["stock_minutes_1m"]["mapping_status"]),
        "financial_indicator": inspect_asset("financial_indicator", ["ts_code", "report_date"], "report_date", old_assets["financial_indicator"]["mapping_status"]),
    }
    contracts = {
        "company_announcements": {
            "status": "experimental/test_30d_backfill_not_approved_for_production",
            "source_params": {"symbol": "全部", "date": "one YYYYMMDD per call"},
            "date_semantics": "source disclosure event date; natural-day window",
            "checkpoint_path": str(ANNOUNCEMENT_CHECKPOINT),
            "resume_policy": "skip completed/empty dates; retry pending/running/failed; idempotent announcement_id merge",
            "sample_scope": {"window": "20260616-20260715", "market_scope": "all source rows filtered no-BJ"},
            "coverage": announcements,
            "ttl_seconds": None,
            "adjustment": "not_applicable",
        },
        "financial_indicator": {
            "status": "experimental/test_partial_resumable_batch_not_approved_for_production",
            "source_params": {"symbol": "one formal no-BJ ts_code per call", "indicator": "按报告期"},
            "date_semantics": "financial report period, not market trade date",
            "checkpoint_path": str(FINANCIAL_CHECKPOINT),
            "resume_policy": "per-stock skip completed/source_empty; bounded retry for failed/pending",
            "sample_scope": {"formal_no_bj_pool_total": pool_total, "first_batch_size": FINANCIAL_BATCH_SIZE},
            "coverage": financial,
            "ttl_seconds": None,
            "adjustment": "not_applicable",
        },
        "stock_news": {
            "status": "experimental/test_on_demand_cache_not_approved_for_production",
            "source_params": {"symbol": "one 6-digit stock code per call"},
            "date_semantics": "source news event time; natural-day event, never a formal trade-date quote",
            "checkpoint_path": str(NEWS_CHECKPOINT),
            "resume_policy": "trigger-only; hit before TTL, refresh expired misses, dedup news_id",
            "sample_scope": {"trigger": trigger["trigger_source"], "codes": trigger["codes"]},
            "coverage": news,
            "ttl_seconds": NEWS_TTL_SECONDS,
            "adjustment": "not_applicable",
        },
        "stock_minutes_1m": {
            "status": "experimental/test_completed_session_on_demand_cache_not_approved_for_production",
            "source_params": {"symbol": "Sina-prefixed code", "period": "1", "adjust": ""},
            "date_semantics": "historical completed-session bars; not real-time行情 or current price",
            "checkpoint_path": str(MINUTE_CHECKPOINT),
            "resume_policy": "cache key stock_code+latest_completed_trade_date; refresh only expired/missing trigger",
            "sample_scope": {"trigger": trigger["trigger_source"], "codes": trigger["codes"], "trade_date": gate["latest_completed_trade_date"]},
            "coverage": minutes,
            "ttl_seconds": MINUTE_TTL_SECONDS,
            "adjustment": base.ASSET_CONTRACTS["stock_minutes_1m"]["adjustment"],
            "fallback_from": "AKShare.stock_zh_a_hist_min_em",
        },
        "social_sentiment_snapshot": {
            "status": "blocked_no_security_code_or_source_event_time",
            "source_params": {"time_period": "CNHOUR24"},
            "date_semantics": "ingestion snapshot only; source event time unavailable",
            "checkpoint_path": None,
            "resume_policy": "blocked; do not expand or map by display name",
            "sample_scope": {"existing_rows_only": True, "expanded": False},
            "coverage": {"stock_level_coverage": 0, "blocked": True},
            "ttl_seconds": None,
            "adjustment": "not_applicable",
            "blocked_reason": "source has no security code and no source event timestamp",
            "unblock_conditions": ["stable source security code", "source event timestamp", "audited mapping contract"],
        },
    }
    for table, contract in contracts.items():
        assets[table].update(contract)
        assets[table]["source"] = base.ASSET_CONTRACTS[table]["source"]
        assets[table]["source_params"] = contract["source_params"]
    announcement_total = sum(
        announcements.get(key, 0) for key in ("completed_days", "empty_days", "failed_days", "pending_days")
    )
    news_frame = read_table("stock_news")
    minute_frame = read_table("stock_minutes_1m")
    requested_codes = set(trigger["codes"])
    news_covered = requested_codes.intersection(set(news_frame.get("ts_code", pd.Series(dtype=str)).astype(str)))
    minute_dates = pd.to_datetime(minute_frame["trade_time"]).dt.strftime("%Y%m%d")
    minute_covered = requested_codes.intersection(
        set(minute_frame.loc[minute_dates == gate["latest_completed_trade_date"], "ts_code"].astype(str))
    )
    news_checkpoint = load_json(NEWS_CHECKPOINT, {"entries": {}})
    minute_checkpoint = load_json(MINUTE_CHECKPOINT, {"entries": {}})
    error_summary = {
        "announcement_failed_days": announcements.get("failed_days", 0),
        "financial_failed_stocks": financial.get("failed", 0),
        "news_failed_codes": sorted(
            code for code in requested_codes
            if news_checkpoint.get("entries", {}).get(code, {}).get("status") == "failed"
        ),
        "minute_failed_keys": sorted(
            key for key, item in minute_checkpoint.get("entries", {}).items()
            if key.split("|", 1)[0] in requested_codes and item.get("status") == "failed"
        ),
    }
    status = "partial_ready_for_read_only_reaudit"
    report = {
        "task_id": TASK_ID,
        "status": status,
        "generated_at": now_iso(),
        "trade_date_gate": gate,
        "watch_pool_trigger": trigger,
        "announcements_30d": announcements,
        "financial_first_batch": financial,
        "news_on_demand_cache": news,
        "minutes_on_demand_cache": minutes,
        "social_sentiment": contracts["social_sentiment_snapshot"],
        "assets": assets,
        "production_assets_touched": False,
        "route_registry_touched": False,
        "l2_l8_triggered": False,
        "legacy_odb_used": False,
        "website_display_expanded": False,
        "production_approved": False,
        "requires_read_only_reaudit": True,
        "validation": manifest.get("validation", {"status": "pending"}),
        "source_runtime": {
            "akshare_version": manifest.get("akshare_version", "unknown"),
            "scaleout_source_calls_completed_at": max(news.get("completed_at", ""), minutes.get("completed_at", "")),
        },
        "coverage_summary": {
            "announcement_call_completion": {
                "covered_days": announcements.get("completed_days", 0) + announcements.get("empty_days", 0),
                "total_days": announcement_total,
                "provisional_days": announcements.get("provisional_days", 0),
            },
            "financial_terminal_stock_coverage": {
                "covered_stocks": financial.get("completed_with_data", 0) + financial.get("source_empty", 0),
                "pool_total": pool_total,
                "ratio": round((financial.get("completed_with_data", 0) + financial.get("source_empty", 0)) / pool_total, 6),
            },
            "news_trigger_data_coverage": {"covered": len(news_covered), "requested": len(requested_codes)},
            "minute_trigger_trade_date_coverage": {"covered": len(minute_covered), "requested": len(requested_codes)},
        },
        "error_summary": error_summary,
    }
    manifest.update(report)
    atomic_json(MANIFEST, manifest)
    atomic_json(REPORT_JSON, report)
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AKShare experimental/test L1 scaleout",
        "",
        f"- Status: `{report['status']}`",
        f"- Latest completed trade date: `{report['trade_date_gate']['latest_completed_trade_date']}`",
        "- Scope: resumable experimental/test only; not approved for production or expanded website display",
        "",
        "## Progress",
        "",
        f"- Announcements: completed {report['announcements_30d']['completed_days']} days, empty {report['announcements_30d']['empty_days']}, failed {report['announcements_30d']['failed_days']}.",
        f"- Announcement provisional natural days: {report['announcements_30d'].get('provisional_dates', [])}.",
        f"- Financial: processed {report['financial_first_batch']['processed_this_run']}; completed with data {report['financial_first_batch']['completed_with_data']}; remaining {report['financial_first_batch']['remaining']}.",
        f"- News cache: hits {len(report['news_on_demand_cache']['hits'])}, misses {len(report['news_on_demand_cache']['misses'])}, calls {report['news_on_demand_cache']['calls']}.",
        f"- Minute cache: trade date {report['minutes_on_demand_cache']['trade_date']}; hits {len(report['minutes_on_demand_cache']['hits'])}, misses {len(report['minutes_on_demand_cache']['misses'])}, calls {report['minutes_on_demand_cache']['calls']}.",
        "- Sentiment: blocked; no security code or source event timestamp.",
        f"- Tests: {report.get('validation', {}).get('tests', 'pending')} ({report.get('validation', {}).get('status', 'pending')}).",
        "",
        "## Assets",
        "",
        "| Table | Rows | Stocks | Date range | Duplicate groups | Status |",
        "|---|---:|---:|---|---:|---|",
    ]
    for table, asset in report["assets"].items():
        lines.append(f"| `{table}` | {asset['rows']} | {asset.get('stock_coverage')} | {asset.get('min_date')} to {asset.get('max_date')} | {asset['duplicate_key_groups']} | `{asset['status']}` |")
    lines.extend(["", "Production assets/route/registry touched: `false`; L2-L8 triggered: `false`; legacy odb used: `false`."])
    return "\n".join(lines) + "\n"


def write_hashes(report: dict[str, Any]) -> dict[str, str]:
    files = [
        Path(__file__),
        Path(__file__).parent / "akshare_l1_experimental.py",
        Path(__file__).parent / "test_akshare_l1_experimental.py",
        Path(__file__).parent / "test_akshare_l1_scaleout.py",
        MANIFEST,
        TRADE_GATE,
        REPORT_JSON,
        REPORT_MD,
        ANNOUNCEMENT_CHECKPOINT,
        FINANCIAL_CHECKPOINT,
        NEWS_CHECKPOINT,
        MINUTE_CHECKPOINT,
        *(table_path(table) for table in report["assets"]),
    ]
    hashes = {str(path): sha256(path) for path in files if path.exists()}
    atomic_json(HASH_JSON, {"task_id": TASK_ID, "generated_at": now_iso(), "algorithm": "SHA256", "files": hashes, "note": "evidence file intentionally does not self-hash"})
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--financial-only", action="store_true")
    parser.add_argument("--financial-batch-size", type=int, default=FINANCIAL_BATCH_SIZE)
    parser.add_argument("--financial-flush-size", type=int, default=25)
    parser.add_argument("--gate-report", type=Path, default=TRADE_GATE)
    parser.add_argument("--announcement-start", default="20260616")
    parser.add_argument("--announcement-end", default="20260715")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gate_only:
        print(json.dumps(create_trade_gate(), ensure_ascii=False, default=str))
        return 0
    if args.financial_only:
        import akshare as ak

        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        pool = formal_no_bj_pool()
        financial = run_financial_batch(ak, pool, args.financial_batch_size, args.financial_flush_size)
        report = update_financial_manifest_only(financial, len(pool))
        print(json.dumps({"status": report["status"], "report": str(FINANCIAL_SCALEOUT_REPORT), "financial": financial}, ensure_ascii=False))
        return 0
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    if not gate.get("latest_completed_trade_date"):
        raise RuntimeError("valid trade-date gate is required")
    import akshare as ak

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    trigger = platform_watch_pool()
    announcements = run_announcements(ak, args.announcement_start, args.announcement_end)
    pool = formal_no_bj_pool()
    financial = run_financial_batch(ak, pool, args.financial_batch_size, args.financial_flush_size)
    news = run_news_cache(ak, trigger)
    minutes = run_minute_cache(ak, trigger, gate["latest_completed_trade_date"])
    report = update_manifest_and_report(gate, trigger, announcements, financial, news, minutes, len(pool))
    hashes = write_hashes(report)
    print(json.dumps({"status": report["status"], "report": str(REPORT_JSON), "hash_files": len(hashes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
