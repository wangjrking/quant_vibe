from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


WINDOWS = (1, 3, 5, 10)
SOURCE_LIMIT = 1000


@dataclass(frozen=True)
class RiskEventApiSpec:
    api: str
    event_type: str
    date_column: str
    required_columns: tuple[str, ...]


API_SPECS = (
    RiskEventApiSpec(
        "stk_shock",
        "abnormal_volatility",
        "trade_date",
        ("ts_code", "trade_date", "name", "trade_market", "reason", "period"),
    ),
    RiskEventApiSpec(
        "stk_high_shock",
        "severe_abnormal_volatility",
        "trade_date",
        ("ts_code", "trade_date", "name", "trade_market", "reason", "period"),
    ),
    RiskEventApiSpec(
        "stk_alert",
        "exchange_focus_security",
        "start_date",
        ("ts_code", "name", "start_date", "end_date", "type"),
    ),
)
API_SPEC_BY_NAME = {spec.api: spec for spec in API_SPECS}


def normalize_yyyymmdd(value: Any) -> str:
    raw = str(value or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"invalid YYYYMMDD value: {value!r}")
    return pd.to_datetime(raw, format="%Y%m%d", errors="raise").strftime("%Y%m%d")


def _row_digest(row: pd.Series, columns: tuple[str, ...]) -> str:
    payload = {
        column: None if pd.isna(row[column]) else str(row[column]).strip()
        for column in columns
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def normalize_source_frame(
    api: str,
    frame: pd.DataFrame | None,
    *,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    spec = API_SPEC_BY_NAME[api]
    value = pd.DataFrame() if frame is None else frame.copy()
    if value.empty and not set(spec.required_columns).issubset(value.columns):
        value = pd.DataFrame(columns=list(spec.required_columns))
    missing = sorted(set(spec.required_columns) - set(value.columns))
    if missing:
        raise RuntimeError(f"Tushare {api} schema missing columns: {missing}")
    value = value.loc[:, list(spec.required_columns)].copy()
    for column in (spec.date_column, "end_date"):
        if column in value.columns and not value.empty:
            value[column] = value[column].map(normalize_yyyymmdd)
    value["source_api"] = api
    if value.empty:
        value["source_row_sha256"] = pd.Series(dtype="object")
    else:
        value["source_row_sha256"] = value.apply(
            _row_digest,
            axis=1,
            columns=spec.required_columns,
        )
    value["fetched_at"] = fetched_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    return (
        value.sort_values([spec.date_column, "ts_code", "source_row_sha256"])
        .drop_duplicates("source_row_sha256", keep="last")
        .reset_index(drop=True)
    )


def _query_range_with_retry(
    pro: Any,
    api: str,
    start_date: str,
    end_date: str,
    attempts: int = 4,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return pro.query(api, start_date=start_date, end_date=end_date)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Tushare {api} failed for {start_date}..{end_date}") from last_error


def fetch_complete_range(
    pro: Any,
    api: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_date = normalize_yyyymmdd(start_date)
    end_date = normalize_yyyymmdd(end_date)
    frame = _query_range_with_retry(pro, api, start_date, end_date)
    if len(frame) < SOURCE_LIMIT:
        return frame
    if start_date == end_date:
        raise RuntimeError(
            f"Tushare {api} daily result reached the {SOURCE_LIMIT}-row cap on {start_date}"
        )
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    middle = start + (end - start) // 2
    left_end = middle.strftime("%Y%m%d")
    right_start = (middle + pd.Timedelta(days=1)).strftime("%Y%m%d")
    pieces = [
        fetch_complete_range(pro, api, start_date, left_end),
        fetch_complete_range(pro, api, right_start, end_date),
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def next_open_date(event_date: str, open_dates: list[str]) -> str:
    index = int(pd.Index(open_dates).searchsorted(str(event_date), side="right"))
    if index >= len(open_dates):
        raise ValueError(f"no official open session after {event_date}")
    return str(open_dates[index])


def build_l2_events(
    raw_by_api: dict[str, pd.DataFrame],
    open_dates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in API_SPECS:
        for row in raw_by_api[spec.api].itertuples(index=False):
            stock_code = str(row.ts_code).strip().upper()
            if stock_code.endswith(".BJ"):
                continue
            event_date = str(getattr(row, spec.date_column))
            source_end_date = (
                str(getattr(row, "end_date")) if spec.api == "stk_alert" else None
            )
            detail_value = (
                getattr(row, "type")
                if spec.api == "stk_alert"
                else getattr(row, "reason")
            )
            period_value = (
                None if spec.api == "stk_alert" else getattr(row, "period")
            )
            source_sha = str(row.source_row_sha256)
            rows.append(
                {
                    "event_id": f"{spec.api}:{source_sha}",
                    "stock_code": stock_code,
                    "stock_name": str(row.name),
                    "event_type": spec.event_type,
                    "source_api": spec.api,
                    "event_date": event_date,
                    "source_end_date": source_end_date,
                    "available_signal_date": next_open_date(event_date, open_dates),
                    "detail": None if pd.isna(detail_value) else str(detail_value),
                    "period": (
                        None
                        if period_value is None or pd.isna(period_value)
                        else str(period_value)
                    ),
                    "source_row_sha256": source_sha,
                    "availability_rule": (
                        "first_official_open_session_strictly_after_source_event_date"
                    ),
                }
            )
    columns = [
        "event_id",
        "stock_code",
        "stock_name",
        "event_type",
        "source_api",
        "event_date",
        "source_end_date",
        "available_signal_date",
        "detail",
        "period",
        "source_row_sha256",
        "availability_rule",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["available_signal_date", "stock_code", "source_api", "event_id"])
        .reset_index(drop=True)
    )


def build_signal_slice(
    events: pd.DataFrame,
    open_dates: list[str],
    signal_date: str,
) -> pd.DataFrame:
    signal_date = normalize_yyyymmdd(signal_date)
    if signal_date not in open_dates:
        raise ValueError(f"signal date is not an official open session: {signal_date}")
    index_by_date = {date: index for index, date in enumerate(open_dates)}
    signal_index = index_by_date[signal_date]
    visible = events[
        events["available_signal_date"].astype(str).le(signal_date)
    ].copy()
    if visible.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_code",
                *[
                    f"{prefix}_count_{window}d"
                    for window in WINDOWS
                    for prefix in ("shock", "high_shock", "alert_start")
                ],
                "alert_active_count",
            ]
        )
    visible["available_index"] = visible["available_signal_date"].map(index_by_date)
    if visible["available_index"].isna().any():
        raise RuntimeError("event availability date is outside official calendar")
    visible["age_sessions"] = signal_index - visible["available_index"].astype(int)
    visible = visible[visible["age_sessions"].ge(0)].copy()
    active_mask = (
        visible["source_api"].eq("stk_alert")
        & visible["source_end_date"].notna()
        & visible["source_end_date"].astype(str).ge(signal_date)
    )
    relevant = visible[visible["age_sessions"].lt(max(WINDOWS)) | active_mask]
    stocks = sorted(relevant["stock_code"].astype(str).unique())
    if not stocks:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_code",
                *[
                    f"{prefix}_count_{window}d"
                    for window in WINDOWS
                    for prefix in ("shock", "high_shock", "alert_start")
                ],
                "alert_active_count",
            ]
        )
    result = pd.DataFrame({"signal_date": signal_date, "stock_code": stocks})
    for window in WINDOWS:
        windowed = visible[visible["age_sessions"].lt(window)]
        for api, prefix in (
            ("stk_shock", "shock"),
            ("stk_high_shock", "high_shock"),
            ("stk_alert", "alert_start"),
        ):
            counts = (
                windowed[windowed["source_api"].eq(api)]
                .groupby("stock_code")["event_id"]
                .nunique()
                .rename(f"{prefix}_count_{window}d")
            )
            result = result.merge(counts, on="stock_code", how="left")
    active = visible[active_mask]
    active_counts = (
        active.groupby("stock_code")["event_id"]
        .nunique()
        .rename("alert_active_count")
    )
    result = result.merge(active_counts, on="stock_code", how="left")
    count_columns = [
        column for column in result.columns if column not in {"signal_date", "stock_code"}
    ]
    result[count_columns] = result[count_columns].fillna(0).astype("int32")
    return result.sort_values(["signal_date", "stock_code"]).reset_index(drop=True)
