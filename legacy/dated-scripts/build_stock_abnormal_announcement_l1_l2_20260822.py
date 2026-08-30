from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import duckdb
import pandas as pd
import requests


REPO = Path(r"D:/work/quant/quant_mcp")
DEFAULT_ROOT = REPO / "quant/data_file/experimental_assets/stock_abnormal_announcements_v1"
EASTMONEY_ENDPOINT = "https://np-anotice-stock.eastmoney.com/api/security/ann"
ABNORMAL_COLUMN_CODE = "001002004007"
ABNORMAL_COLUMN_NAME = "\u80a1\u7968\u4ea4\u6613\u5f02\u5e38\u6ce2\u52a8"
SOURCE_NAME = "Eastmoney.security.ann"
PAGE_SIZE = 100


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_stock_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 6:
        raise ValueError(f"unsupported stock code: {value!r}")
    if digits.startswith(("4", "8", "9")):
        return f"{digits}.BJ"
    if digits.startswith(("5", "6", "7")):
        return f"{digits}.SH"
    if digits.startswith(("0", "1", "2", "3")):
        return f"{digits}.SZ"
    raise ValueError(f"unsupported stock code: {value!r}")


def normalize_published_at(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for pattern in ("%Y-%m-%d %H:%M:%S:%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, pattern).strftime("%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            continue
    raise ValueError(f"unsupported announcement publication timestamp: {value!r}")


def parse_source_items(items: Iterable[dict[str, Any]], fetched_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        columns = item.get("columns") or []
        abnormal_columns = [
            column
            for column in columns
            if str(column.get("column_code")) == ABNORMAL_COLUMN_CODE
            or str(column.get("column_name")) == ABNORMAL_COLUMN_NAME
        ]
        if not abnormal_columns:
            continue
        art_code = str(item.get("art_code") or "").strip()
        if not art_code:
            raise ValueError("announcement without art_code")
        notice_date = pd.to_datetime(item.get("notice_date"), errors="raise").strftime("%Y%m%d")
        published_at = normalize_published_at(item.get("display_time"))
        codes = [code for code in item.get("codes") or [] if str(code.get("ann_type", "")).startswith("A")]
        if not codes:
            raise ValueError(f"A-share announcement without code: {art_code}")
        for code in codes:
            stock_code = normalize_stock_code(code.get("stock_code"))
            raw_digest = hashlib.sha256(
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "announcement_id": art_code,
                    "stock_code": stock_code,
                    "stock_name": str(code.get("short_name") or "").strip(),
                    "title": str(item.get("title") or "").strip(),
                    "category_code": ABNORMAL_COLUMN_CODE,
                    "category": ABNORMAL_COLUMN_NAME,
                    "announcement_date": notice_date,
                    "published_at": published_at,
                    "publication_time_available": published_at is not None,
                    "source_url": f"https://data.eastmoney.com/notices/detail/{code.get('stock_code')}/{art_code}.html",
                    "source": SOURCE_NAME,
                    "source_item_sha256": raw_digest,
                    "fetched_at": fetched_at,
                }
            )
    columns = [
        "announcement_id",
        "stock_code",
        "stock_name",
        "title",
        "category_code",
        "category",
        "announcement_date",
        "published_at",
        "publication_time_available",
        "source_url",
        "source",
        "source_item_sha256",
        "fetched_at",
    ]
    return pd.DataFrame(rows, columns=columns)


def request_json(
    params: dict[str, str],
    request_get: Callable[..., Any] = requests.get,
    attempts: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = request_get(EASTMONEY_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success") or not isinstance(payload.get("data"), dict):
                raise RuntimeError("Eastmoney announcement response is not successful")
            return payload
        except Exception as exc:  # bounded source retry
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"announcement source failed after {attempts} attempts") from last_error


def source_params(start_date: str, end_date: str, page_index: int) -> dict[str, str]:
    return {
        "sr": "-1",
        "page_size": str(PAGE_SIZE),
        "page_index": str(page_index),
        "ann_type": "A",
        "client_source": "web",
        "f_node": "3",
        "s_node": "0",
        "begin_time": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "end_time": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
    }


def fetch_period(start_date: str, end_date: str, workers: int = 8) -> tuple[pd.DataFrame, dict[str, Any]]:
    first = request_json(source_params(start_date, end_date, 1))
    total_hits = int(first["data"]["total_hits"])
    if total_hits >= 50000:
        raise RuntimeError(f"source result cap reached for {start_date}..{end_date}; split the period")
    total_pages = int(math.ceil(total_hits / PAGE_SIZE))
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    pages: dict[int, list[dict[str, Any]]] = {1: list(first["data"]["list"])}

    def fetch_page(page: int) -> tuple[int, list[dict[str, Any]]]:
        payload = request_json(source_params(start_date, end_date, page))
        return page, list(payload["data"]["list"])

    if total_pages > 1:
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 12))) as executor:
            futures = [executor.submit(fetch_page, page) for page in range(2, total_pages + 1)]
            for future in as_completed(futures):
                page, items = future.result()
                pages[page] = items

    items = [item for page in sorted(pages) for item in pages[page]]
    if len(items) != total_hits:
        raise RuntimeError(f"source pagination mismatch: expected={total_hits}, actual={len(items)}")
    frame = parse_source_items(items, fetched_at)
    return frame, {
        "start": start_date,
        "end": end_date,
        "source_rows": total_hits,
        "pages": total_pages,
        "abnormal_stock_rows": int(len(frame)),
    }


def year_periods(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    periods = []
    for year in range(start.year, end.year + 1):
        left = max(start, pd.Timestamp(year=year, month=1, day=1))
        right = min(end, pd.Timestamp(year=year, month=12, day=31))
        periods.append((left.strftime("%Y%m%d"), right.strftime("%Y%m%d")))
    return periods


def merge_l1(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return parse_source_items([], datetime.now().astimezone().isoformat())
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["announcement_date", "published_at", "announcement_id", "stock_code"])
    duplicate_conflicts = (
        frame.groupby(["announcement_id", "stock_code"])["source_item_sha256"].nunique().gt(1).sum()
    )
    if duplicate_conflicts:
        raise RuntimeError(f"conflicting duplicate announcement keys: {duplicate_conflicts}")
    return frame.drop_duplicates(["announcement_id", "stock_code"], keep="last").reset_index(drop=True)


def load_official_calendar(start_date: str, end_date: str) -> list[str]:
    main_root = REPO / "quant/main"
    sys.path.insert(0, str(main_root))
    try:
        from data_load_module import get_pro
        from project_paths import load_config

        token = str(load_config(None)["datasource"]["tushare_token"])
        pro = get_pro(token)
        frame = pro.query("trade_cal", exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
    finally:
        if sys.path and sys.path[0] == str(main_root):
            sys.path.pop(0)
    dates = sorted(frame["cal_date"].astype(str).unique().tolist())
    if not dates:
        raise RuntimeError("official trade calendar is empty")
    return dates


def next_open_date(announcement_date: str, open_dates: list[str]) -> str:
    index = pd.Index(open_dates).searchsorted(str(announcement_date), side="right")
    if int(index) >= len(open_dates):
        raise ValueError(f"no next open session after {announcement_date}")
    return str(open_dates[int(index)])


def classify_event(title: str) -> tuple[str, bool]:
    severe = "\u4e25\u91cd\u5f02\u5e38\u6ce2\u52a8" in title
    risk_warning = "\u98ce\u9669\u63d0\u793a" in title
    return ("severe_abnormal_volatility" if severe else "abnormal_volatility", risk_warning)


def build_l2_events(l1: pd.DataFrame, open_dates: list[str]) -> pd.DataFrame:
    rows = []
    eligible_l1 = l1[~l1["stock_code"].astype(str).str.endswith(".BJ")]
    for row in eligible_l1.itertuples(index=False):
        event_type, risk_warning = classify_event(str(row.title))
        rows.append(
            {
                "stock_code": str(row.stock_code),
                "announcement_id": str(row.announcement_id),
                "announcement_date": str(row.announcement_date),
                "published_at": None if pd.isna(row.published_at) else str(row.published_at),
                "publication_time_available": bool(row.publication_time_available),
                "available_signal_date": next_open_date(str(row.announcement_date), open_dates),
                "event_type": event_type,
                "risk_warning_in_title": bool(risk_warning),
                "title": str(row.title),
                "source_url": str(row.source_url),
                "availability_rule": "first_open_session_strictly_after_announcement_date",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["available_signal_date", "stock_code", "announcement_id"]
    ).reset_index(drop=True)


def build_signal_features(events: pd.DataFrame, open_dates: list[str], max_window: int = 10) -> pd.DataFrame:
    date_to_index = {date: index for index, date in enumerate(open_dates)}
    sparse: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        start = date_to_index[str(row.available_signal_date)]
        for age in range(max_window):
            if start + age >= len(open_dates):
                break
            sparse.append(
                {
                    "signal_date": open_dates[start + age],
                    "stock_code": str(row.stock_code),
                    "announcement_id": str(row.announcement_id),
                    "age_sessions": age,
                    "severe": row.event_type == "severe_abnormal_volatility",
                    "risk_warning": bool(row.risk_warning_in_title),
                }
            )
    expanded = pd.DataFrame(sparse)
    columns = ["signal_date", "stock_code"]
    if expanded.empty:
        return pd.DataFrame(columns=columns)
    grouped = expanded.groupby(columns, sort=True)
    result = grouped.agg(
        abnormal_count_10d=("announcement_id", "nunique"),
        severe_count_10d=("severe", "sum"),
        risk_warning_count_10d=("risk_warning", "sum"),
    ).reset_index()
    for window in (1, 3, 5):
        windowed = expanded[expanded["age_sessions"] < window].groupby(columns, sort=True).agg(
            **{
                f"abnormal_count_{window}d": ("announcement_id", "nunique"),
                f"severe_count_{window}d": ("severe", "sum"),
                f"risk_warning_count_{window}d": ("risk_warning", "sum"),
            }
        ).reset_index()
        result = result.merge(windowed, on=columns, how="left")
    count_columns = [column for column in result if column.endswith("d")]
    result[count_columns] = result[count_columns].fillna(0).astype("int32")
    ordered = columns + [
        f"{prefix}_{window}d"
        for window in (1, 3, 5, 10)
        for prefix in ("abnormal_count", "severe_count", "risk_warning_count")
    ]
    return result[ordered].sort_values(columns).reset_index(drop=True)


def write_table(path: Path, table: str, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = duckdb.connect(str(temporary))
    try:
        connection.register("payload", frame)
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM payload')
    finally:
        connection.close()
    temporary.replace(path)


def build(start_date: str, end_date: str, output_root: Path, workers: int) -> dict[str, Any]:
    if start_date > end_date:
        raise ValueError("start date must not exceed end date")
    period_frames = []
    fetch_audit = []
    for left, right in year_periods(start_date, end_date):
        frame, audit = fetch_period(left, right, workers)
        period_frames.append(frame)
        fetch_audit.append(audit)
    l1 = merge_l1(period_frames)
    calendar_end = (pd.Timestamp(end_date) + pd.Timedelta(days=45)).strftime("%Y%m%d")
    open_dates = load_official_calendar(start_date, calendar_end)
    l2 = build_l2_events(l1, open_dates)
    features = build_signal_features(l2, open_dates)

    l1_path = output_root / "l1_stock_abnormal_announcements.duckdb"
    l2_path = output_root / "l2_stock_abnormal_announcement_events.duckdb"
    feature_path = output_root / "l2_stock_abnormal_announcement_signal.duckdb"
    write_table(l1_path, "stock_abnormal_announcements", l1)
    write_table(l2_path, "stock_abnormal_announcement_events", l2)
    write_table(feature_path, "stock_abnormal_announcement_signal", features)

    duplicate_l1 = int(l1.duplicated(["announcement_id", "stock_code"]).sum())
    duplicate_l2 = int(l2.duplicated(["announcement_id", "stock_code"]).sum())
    same_day_visibility = int((l2["available_signal_date"] <= l2["announcement_date"]).sum())
    manifest = {
        "schema_version": 1,
        "status": "candidate_l1_l2_ready",
        "source": SOURCE_NAME,
        "source_filter": {"f_node": "3", "category_code": ABNORMAL_COLUMN_CODE},
        "coverage": {"start": start_date, "end": end_date},
        "fetch": fetch_audit,
        "pit_contract": {
            "publication_timestamp_retained": True,
            "strategy_availability": "first official open session strictly after announcement_date",
            "same_day_visibility_rows": same_day_visibility,
        },
        "quality": {
            "l1_rows": int(len(l1)),
            "l1_stocks": int(l1["stock_code"].nunique()),
            "l1_duplicate_keys": duplicate_l1,
            "l2_rows": int(len(l2)),
            "l2_duplicate_keys": duplicate_l2,
            "feature_rows": int(len(features)),
            "bj_rows": int(l1["stock_code"].str.endswith(".BJ").sum()),
            "l2_bj_rows": int(l2["stock_code"].str.endswith(".BJ").sum()),
            "publication_time_missing_rows": int((~l1["publication_time_available"]).sum()),
            "required_nulls": int(
                l1[["announcement_id", "stock_code", "title", "announcement_date"]]
                .isna()
                .sum()
                .sum()
            ),
        },
        "assets": {
            "l1": {"path": str(l1_path), "table": "stock_abnormal_announcements", "sha256": sha256_file(l1_path)},
            "l2_events": {"path": str(l2_path), "table": "stock_abnormal_announcement_events", "sha256": sha256_file(l2_path)},
            "l2_signal": {"path": str(feature_path), "table": "stock_abnormal_announcement_signal", "sha256": sha256_file(feature_path)},
        },
        "production_modified": False,
    }
    if (
        duplicate_l1
        or duplicate_l2
        or same_day_visibility
        or manifest["quality"]["required_nulls"]
        or manifest["quality"]["l2_bj_rows"]
    ):
        raise RuntimeError("L1/L2 quality gate failed")
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20220101")
    parser.add_argument("--end", default="20260821")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = build(args.start, args.end, args.output_root, args.workers)
    print(json.dumps(result["quality"], ensure_ascii=False))


if __name__ == "__main__":
    main()
