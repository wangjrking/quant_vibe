from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import duckdb
import pandas as pd

from data_load_module import get_pro


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING_ROOT = PROJECT_ROOT / "quant/data_file/experimental_assets/l1_web_gap_staging_20260712"
PRODUCTION_L1_ROOT = PROJECT_ROOT / "quant/data_file/production_assets/duckdb/l1_raw_tables"
TRACKING_QUERY_KEYS = {"spm", "from", "source", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}


@dataclass(frozen=True)
class TableContract:
    source_api: str
    key: tuple[str, ...]
    date_columns: tuple[str, ...]


CONTRACTS = {
    "trade_calendar": TableContract("trade_cal", ("exchange", "cal_date"), ("cal_date", "pretrade_date")),
    "company_disclosure_calendar": TableContract(
        "disclosure_date", ("ts_code", "end_date", "revision_hash"),
        ("ann_date", "end_date", "pre_date", "actual_date", "modify_date", "source_snapshot_date"),
    ),
    "major_news": TableContract("major_news", ("news_id",), ("pub_date",)),
    "stock_mins_1m": TableContract("stk_mins", ("ts_code", "trade_time", "freq", "adj"), ("trade_date",)),
}


def utc8_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Shanghai")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    normalized = normalized.fillna("<NULL>").astype(str)
    normalized = normalized.sort_values(list(normalized.columns), kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def normalize_url(value: object) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in TRACKING_QUERY_KEYS]
    query.sort()
    return urlunsplit((scheme, host, parts.path or "/", urlencode(query), ""))


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_yyyymmdd(series: pd.Series, *, nullable: bool = True) -> pd.Series:
    values = series.astype("string")
    invalid = values.notna() & ~values.str.fullmatch(r"\d{8}")
    if invalid.any():
        raise ValueError(f"invalid YYYYMMDD values: {values[invalid].head(3).tolist()}")
    if not nullable and values.isna().any():
        raise ValueError("required date contains null")
    return values


def transform_trade_calendar(frame: pd.DataFrame, ingested_at: pd.Timestamp | None = None) -> pd.DataFrame:
    required = ["exchange", "cal_date", "is_open", "pretrade_date"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"trade_calendar missing fields: {sorted(missing)}")
    out = frame[required].copy()
    out["exchange"] = out["exchange"].astype("string").str.strip()
    out["cal_date"] = validate_yyyymmdd(out["cal_date"], nullable=False)
    out["pretrade_date"] = validate_yyyymmdd(out["pretrade_date"])
    out["is_open"] = pd.to_numeric(out["is_open"], errors="raise").astype("int8")
    if not out["is_open"].isin([0, 1]).all():
        raise ValueError("is_open must be 0/1")
    stamp = ingested_at or utc8_now()
    out["source_updated_at"] = stamp
    out["ingested_at"] = stamp
    return out.sort_values(["exchange", "cal_date"]).reset_index(drop=True)


def transform_disclosure(frame: pd.DataFrame, snapshot_date: str, ingested_at: pd.Timestamp | None = None) -> pd.DataFrame:
    required = ["ts_code", "ann_date", "end_date", "pre_date", "actual_date"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"disclosure_date missing fields: {sorted(missing)}")
    out = frame.copy()
    if "modify_date" not in out:
        out["modify_date"] = pd.NA
    out = out[[*required, "modify_date"]]
    out["ts_code"] = out["ts_code"].astype("string").str.upper().str.strip()
    # 本项目 production scope 明确排除北交所；保留源端计数到 source summary，落盘不得混入 .BJ。
    out = out.loc[~out["ts_code"].str.endswith(".BJ", na=False)].copy()
    for column in ["ann_date", "end_date", "pre_date", "actual_date", "modify_date"]:
        out[column] = validate_yyyymmdd(out[column], nullable=column != "end_date")
    out["source_snapshot_date"] = validate_yyyymmdd(pd.Series([snapshot_date] * len(out)), nullable=False)
    revision_fields = ["ts_code", "ann_date", "end_date", "pre_date", "actual_date", "modify_date"]
    out["revision_hash"] = out[revision_fields].fillna("").astype(str).agg("|".join, axis=1).map(digest_text)
    out["ingested_at"] = ingested_at or utc8_now()
    return out.sort_values(["ts_code", "end_date", "source_snapshot_date", "revision_hash"]).reset_index(drop=True)


def transform_major_news(frame: pd.DataFrame, ingested_at: pd.Timestamp | None = None) -> pd.DataFrame:
    required = ["title", "pub_time", "src", "url"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"major_news missing fields: {sorted(missing)}")
    out = frame[required].copy()
    out["title"] = out["title"].astype("string")
    out["src"] = out["src"].astype("string")
    out["url"] = out["url"].astype("string")
    out["normalized_url"] = out["url"].map(normalize_url)
    parsed = pd.to_datetime(out["pub_time"], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"unparseable pub_time rows: {int(parsed.isna().sum())}")
    out["pub_time_shanghai"] = parsed.dt.tz_localize("Asia/Shanghai", ambiguous="raise", nonexistent="raise")
    out["pub_date"] = parsed.dt.strftime("%Y%m%d")
    fallback = out[["src", "pub_time", "title"]].fillna("").astype(str).agg("|".join, axis=1)
    out["url_sha256"] = out["normalized_url"].map(lambda x: digest_text(x) if x else "")
    out["fallback_sha256"] = fallback.map(digest_text)
    out["news_id"] = out.apply(lambda row: row["url_sha256"] or row["fallback_sha256"], axis=1)
    out["ingested_at"] = ingested_at or utc8_now()
    return out.sort_values(["pub_time_shanghai", "news_id"]).reset_index(drop=True)


def transform_stock_mins(frame: pd.DataFrame, ingested_at: pd.Timestamp | None = None) -> pd.DataFrame:
    required = ["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"stk_mins missing fields: {sorted(missing)}")
    out = frame[required].copy()
    out["ts_code"] = out["ts_code"].astype("string").str.upper().str.strip()
    if out["ts_code"].str.endswith(".BJ").any():
        raise ValueError("stock_mins_1m staging forbids .BJ")
    parsed = pd.to_datetime(out["trade_time"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("unparseable trade_time")
    out["trade_time"] = parsed.dt.floor("min").dt.tz_localize("Asia/Shanghai")
    out["trade_date"] = parsed.dt.strftime("%Y%m%d")
    for column in ["close", "open", "high", "low", "vol", "amount"]:
        out[column] = pd.to_numeric(out[column], errors="raise")
    invalid_ohlc = (out["high"] < out[["open", "close"]].max(axis=1)) | (out["low"] > out[["open", "close"]].min(axis=1))
    if invalid_ohlc.any():
        raise ValueError(f"invalid OHLC rows: {int(invalid_ohlc.sum())}")
    invalid_nonnegative = (out[["open", "high", "low", "close", "vol", "amount"]] < 0).any(axis=1)
    if invalid_nonnegative.any():
        raise ValueError(f"negative price/volume/amount rows: {int(invalid_nonnegative.sum())}")
    out["freq"] = "1min"
    out["adj"] = "none"
    out["ingested_at"] = ingested_at or utc8_now()
    return out.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def merge_versions(existing: pd.DataFrame, incoming: pd.DataFrame, key: tuple[str, ...]) -> pd.DataFrame:
    result = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming.copy()
    duplicate = result.duplicated(list(key), keep=False)
    if duplicate.any():
        grouped = result.loc[duplicate].groupby(list(key), dropna=False).size()
        if (grouped > 1).any():
            result = result.drop_duplicates(list(key), keep="last")
    return result.sort_values(list(key)).reset_index(drop=True)


def content_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"ingested_at", "source_updated_at"}]


def write_staging_table(root: Path, table: str, incoming: pd.DataFrame) -> dict:
    if root.resolve() == PRODUCTION_L1_ROOT.resolve() or PRODUCTION_L1_ROOT.resolve() in root.resolve().parents:
        raise ValueError("production L1 output is forbidden")
    table_root = root / table
    table_root.mkdir(parents=True, exist_ok=True)
    parquet_path = table_root / f"{table}.parquet"
    duckdb_path = table_root / f"{table}.duckdb"
    contract = CONTRACTS[table]
    existing = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()
    merged = merge_versions(existing, incoming, contract.key)
    merged.to_parquet(parquet_path, index=False)
    con = duckdb.connect(str(duckdb_path))
    try:
        con.register("staging_frame", merged)
        con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM staging_frame')
        duck_frame = con.execute(f'SELECT * FROM "{table}"').df()
    finally:
        con.close()
    columns = content_columns(merged)
    left = merged[columns].fillna("<NULL>").astype(str).merge(
        duck_frame[columns].fillna("<NULL>").astype(str), how="left", indicator=True
    )
    right = duck_frame[columns].fillna("<NULL>").astype(str).merge(
        merged[columns].fillna("<NULL>").astype(str), how="left", indicator=True
    )
    duplicate_keys = int(merged.duplicated(list(contract.key)).sum())
    return {
        "table": table,
        "rows": len(merged),
        "columns": list(merged.columns),
        "key": list(contract.key),
        "duplicate_keys": duplicate_keys,
        "parquet_path": str(parquet_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "parquet_sha256": sha256_file(parquet_path),
        "duckdb_path": str(duckdb_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "duckdb_sha256": sha256_file(duckdb_path),
        "content_sha256": stable_frame_hash(merged[columns]),
        "parquet_minus_duckdb": int((left["_merge"] == "left_only").sum()),
        "duckdb_minus_parquet": int((right["_merge"] == "left_only").sum()),
    }


def load_pro():
    config = json.loads((PROJECT_ROOT / "quant/main/config.json").read_text(encoding="utf-8"))
    token = os.environ.get("TUSHARE_TOKEN") or config.get("datasource", {}).get("tushare_token")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    return get_pro(token)


def query_source(pro, api: str, params: dict) -> tuple[pd.DataFrame, dict]:
    started = utc8_now()
    try:
        frame = pro.query(api, **params)
        status, error = "ok", None
    except Exception as exc:
        frame, status, error = pd.DataFrame(), "error", f"{type(exc).__name__}: {exc}"
    ended = utc8_now()
    summary = {
        "api": api, "params": params, "status": status, "error": error,
        "started_at": started.isoformat(), "ended_at": ended.isoformat(),
        "rows": len(frame), "columns": list(frame.columns),
        "source_bj_rows": int(frame["ts_code"].astype("string").str.endswith(".BJ", na=False).sum()) if "ts_code" in frame else 0,
    }
    return frame, summary


def select_minute_pool(data_root: Path, as_of: str, max_symbols: int) -> pd.DataFrame:
    basic_path = data_root / "stock_basic_data.duckdb"
    st_path = data_root / "stock_st.duckdb"
    daily_path = data_root / "daily_data.duckdb"
    if not (basic_path.exists() and st_path.exists() and daily_path.exists()):
        raise FileNotFoundError("active stock_basic_data/stock_st/daily_data evidence is required")
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{basic_path.as_posix()}' AS basic (READ_ONLY)")
        con.execute(f"ATTACH '{st_path.as_posix()}' AS st (READ_ONLY)")
        con.execute(f"ATTACH '{daily_path.as_posix()}' AS daily (READ_ONLY)")
        candidates = con.execute(
            """
            WITH covered AS (
              SELECT ts_code, count(*) AS st_source_rows,
                     max(CASE WHEN type IS NOT NULL THEN 1 ELSE 0 END) AS is_st
              FROM st.stock_st WHERE trade_date=? GROUP BY ts_code
            ), traded AS (
              SELECT DISTINCT ts_code FROM daily.daily_data WHERE trade_date=?
            )
            SELECT b.ts_code, b.name, b.market, ? AS formed_as_of,
                   c.st_source_rows, c.is_st,
                   CASE WHEN c.is_st=1 THEN 'included_st_sample' ELSE 'included_non_st' END AS selection_reason
            FROM basic.stock_basic_data b
            JOIN covered c USING(ts_code)
            JOIN traded t USING(ts_code)
            WHERE NOT ends_with(b.ts_code, '.BJ')
            ORDER BY c.is_st DESC, b.market, b.ts_code
            LIMIT ?
            """, [as_of, as_of, as_of, max_symbols]
        ).df()
    finally:
        con.close()
    if candidates.empty:
        raise ValueError("minute pool is empty")
    return candidates


def date_range_summary(frame: pd.DataFrame) -> dict:
    result = {}
    for column in frame.columns:
        if "date" in column.lower() or "time" in column.lower():
            non_null = frame[column].dropna().astype(str)
            result[column] = {"min": non_null.min() if len(non_null) else None, "max": non_null.max() if len(non_null) else None}
    return result


def hourly_windows(start: str, end: str) -> list[tuple[str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("news start must not exceed news end")
    windows = []
    cursor = start_ts
    while cursor <= end_ts:
        window_end = min(cursor + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1), end_ts)
        windows.append((cursor.strftime("%Y-%m-%d %H:%M:%S"), window_end.strftime("%Y-%m-%d %H:%M:%S")))
        cursor = window_end + pd.Timedelta(seconds=1)
    return windows


def run(args: argparse.Namespace) -> dict:
    root = Path(args.staging_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pro = load_pro()
    source, tables = [], []
    stamp = utc8_now()

    frame, meta = query_source(pro, "trade_cal", {"exchange": args.exchange, "start_date": args.calendar_start, "end_date": args.calendar_end})
    source.append(meta)
    if meta["status"] == "ok":
        tables.append(write_staging_table(root, "trade_calendar", transform_trade_calendar(frame, stamp)))

    for end_date in args.disclosure_end_date:
        frame, meta = query_source(pro, "disclosure_date", {"end_date": end_date})
        source.append(meta)
        if meta["status"] == "ok":
            tables.append(write_staging_table(root, "company_disclosure_calendar", transform_disclosure(frame, args.snapshot_date, stamp)))

    news_frames = []
    for window_start, window_end in hourly_windows(args.news_start, args.news_end):
        frame, meta = query_source(pro, "major_news", {"src": args.news_src, "start_date": window_start, "end_date": window_end})
        source.append(meta)
        if meta["status"] == "ok" and not frame.empty:
            news_frames.append(frame)
    if news_frames:
        tables.append(write_staging_table(root, "major_news", transform_major_news(pd.concat(news_frames, ignore_index=True), stamp)))

    pool = select_minute_pool(PRODUCTION_L1_ROOT, args.minute_date, args.minute_symbols)
    pool_path = root / "stock_mins_1m/minute_pool_manifest.parquet"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(pool_path, index=False)
    minute_frames = []
    for idx, row in pool.iterrows():
        if idx and args.minute_throttle_seconds:
            time.sleep(args.minute_throttle_seconds)
        params = {
            "ts_code": row.ts_code, "freq": "1min",
            "start_date": f"{args.minute_date[:4]}-{args.minute_date[4:6]}-{args.minute_date[6:]} 09:00:00",
            "end_date": f"{args.minute_date[:4]}-{args.minute_date[4:6]}-{args.minute_date[6:]} 15:30:00",
        }
        frame, meta = query_source(pro, "stk_mins", params)
        source.append(meta)
        if meta["status"] == "ok" and not frame.empty:
            minute_frames.append(transform_stock_mins(frame, stamp))
    if minute_frames:
        minute = pd.concat(minute_frames, ignore_index=True)
        tables.append(write_staging_table(root, "stock_mins_1m", minute))

    # Collapse repeated write summaries to the final state per table.
    tables = list({item["table"]: item for item in tables}.values())
    for item in tables:
        frame = pd.read_parquet(PROJECT_ROOT / item["parquet_path"])
        item["date_ranges"] = date_range_summary(frame)
        item["bj_rows"] = int(frame.get("ts_code", pd.Series(dtype="string")).astype("string").str.endswith(".BJ").sum()) if "ts_code" in frame else 0
        if item["table"] == "stock_mins_1m":
            item["ohlc_invalid"] = int(((frame.high < frame[["open", "close"]].max(axis=1)) | (frame.low > frame[["open", "close"]].min(axis=1))).sum())
            item["negative_vol_amount"] = int(((frame.vol < 0) | (frame.amount < 0)).sum())
    manifest = {
        "status": "staging_only_not_production", "generated_at": utc8_now().isoformat(),
        "python": sys.executable, "staging_root": str(root.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "audit_record": "quant/data_file/reports/audit_agent_web_missing_l1_data_plan_20260712.md",
        "source_calls": source, "tables": tables,
        "minute_pool_manifest": str(pool_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "minute_pool_rows": len(pool), "minute_pool_st_rows": int(pool.is_st.sum()),
        "production_write_allowed": False, "downstream_allowed": False,
    }
    manifest_path = root / "staging_manifest.json"
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_path.write_text(payload, encoding="utf-8")
    run_dir = root / "run_manifests"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = utc8_now().strftime("run_%Y%m%dT%H%M%S_%f+0800.json")
    (run_dir / run_name).write_text(payload, encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="网站缺失 L1 数据隔离 staging 接入，不得指向 production L1。")
    p.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--calendar-start", default="20260701")
    p.add_argument("--calendar-end", default="20261231")
    p.add_argument("--disclosure-end-date", action="append", default=[])
    p.add_argument("--snapshot-date", default="20260712")
    p.add_argument("--news-src", default="")
    p.add_argument("--news-start", default="2026-07-10 00:00:00")
    p.add_argument("--news-end", default="2026-07-10 23:59:59")
    p.add_argument("--minute-date", default="20260710")
    p.add_argument("--minute-symbols", type=int, default=1, choices=range(1, 11))
    p.add_argument("--minute-throttle-seconds", type=int, default=60)
    return p


def main() -> int:
    args = parser().parse_args()
    if not args.disclosure_end_date:
        args.disclosure_end_date = ["20260630"]
    manifest = run(args)
    print(json.dumps({"status": manifest["status"], "tables": [{"table": x["table"], "rows": x["rows"]} for x in manifest["tables"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
