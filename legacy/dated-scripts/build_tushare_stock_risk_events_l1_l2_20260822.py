from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
DEFAULT_ROOT = REPO / "quant/data_file/experimental_assets/tushare_stock_risk_events_v1"
SOURCE_START = "19900101"
SOURCE_LIMIT = 1000
WINDOWS = (1, 3, 5, 10)


@dataclass(frozen=True)
class ApiSpec:
    api: str
    event_type: str
    date_column: str
    required_columns: tuple[str, ...]


API_SPECS = (
    ApiSpec(
        api="stk_shock",
        event_type="abnormal_volatility",
        date_column="trade_date",
        required_columns=("ts_code", "trade_date", "name", "trade_market", "reason", "period"),
    ),
    ApiSpec(
        api="stk_high_shock",
        event_type="severe_abnormal_volatility",
        date_column="trade_date",
        required_columns=("ts_code", "trade_date", "name", "trade_market", "reason", "period"),
    ),
    ApiSpec(
        api="stk_alert",
        event_type="exchange_focus_security",
        date_column="start_date",
        required_columns=("ts_code", "name", "start_date", "end_date", "type"),
    ),
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_yyyymmdd(value: Any) -> str:
    raw = str(value or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"invalid YYYYMMDD value: {value!r}")
    return pd.to_datetime(raw, format="%Y%m%d", errors="raise").strftime("%Y%m%d")


def canonical_row_sha256(row: pd.Series, columns: tuple[str, ...]) -> str:
    payload = {column: None if pd.isna(row[column]) else str(row[column]).strip() for column in columns}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def query_with_retry(pro: Any, api: str, start_date: str, end_date: str, attempts: int = 4) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return pro.query(api, start_date=start_date, end_date=end_date)
        except Exception as exc:  # bounded source retry
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Tushare {api} failed for {start_date}..{end_date}") from last_error


def midpoint(left: str, right: str) -> tuple[str, str]:
    start = pd.to_datetime(left, format="%Y%m%d", errors="raise")
    end = pd.to_datetime(right, format="%Y%m%d", errors="raise")
    middle = start + (end - start) // 2
    return middle.strftime("%Y%m%d"), (middle + pd.Timedelta(days=1)).strftime("%Y%m%d")


def fetch_complete_range(
    pro: Any,
    spec: ApiSpec,
    start_date: str,
    end_date: str,
    audit: list[dict[str, Any]],
) -> list[pd.DataFrame]:
    frame = query_with_retry(pro, spec.api, start_date, end_date)
    audit.append({"start": start_date, "end": end_date, "rows": int(len(frame))})
    if len(frame) < SOURCE_LIMIT:
        return [frame]
    if start_date == end_date:
        raise RuntimeError(f"Tushare {spec.api} daily result reached the {SOURCE_LIMIT}-row cap on {start_date}")
    left_end, right_start = midpoint(start_date, end_date)
    return fetch_complete_range(pro, spec, start_date, left_end, audit) + fetch_complete_range(
        pro, spec, right_start, end_date, audit
    )


def normalize_raw(spec: ApiSpec, frames: list[pd.DataFrame], fetched_at: str) -> pd.DataFrame:
    if not frames:
        frame = pd.DataFrame(columns=spec.required_columns)
    else:
        frame = pd.concat(frames, ignore_index=True)
    missing = sorted(set(spec.required_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Tushare {spec.api} schema missing columns: {missing}")
    frame = frame.loc[:, list(spec.required_columns)].copy()
    for column in (spec.date_column, "end_date"):
        if column in frame:
            frame[column] = frame[column].map(normalize_yyyymmdd)
    frame["source_api"] = spec.api
    frame["source_row_sha256"] = frame.apply(
        canonical_row_sha256, axis=1, columns=spec.required_columns
    )
    frame["fetched_at"] = fetched_at
    frame = frame.sort_values([spec.date_column, "ts_code", "source_row_sha256"])
    frame = frame.drop_duplicates("source_row_sha256", keep="last").reset_index(drop=True)
    return frame


def get_tushare_pro() -> Any:
    sys.path.insert(0, str(MAIN_ROOT))
    try:
        from data_load_module import get_pro
        from project_paths import load_config

        return get_pro(str(load_config(None)["datasource"]["tushare_token"]))
    finally:
        if sys.path and sys.path[0] == str(MAIN_ROOT):
            sys.path.pop(0)


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
        raise RuntimeError("official SSE trade calendar is empty")
    return dates


def next_open_date(event_date: str, open_dates: list[str]) -> str:
    index = int(pd.Index(open_dates).searchsorted(str(event_date), side="right"))
    if index >= len(open_dates):
        raise ValueError(f"no official open session after {event_date}")
    return str(open_dates[index])


def build_l2_events(raw_by_api: dict[str, pd.DataFrame], open_dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in API_SPECS:
        frame = raw_by_api[spec.api]
        for row in frame.itertuples(index=False):
            stock_code = str(row.ts_code)
            if stock_code.endswith(".BJ"):
                continue
            event_date = str(getattr(row, spec.date_column))
            source_end_date = str(getattr(row, "end_date")) if spec.api == "stk_alert" else None
            detail_value = getattr(row, "type") if spec.api == "stk_alert" else getattr(row, "reason")
            period_value = None if spec.api == "stk_alert" else getattr(row, "period")
            detail = None if pd.isna(detail_value) else str(detail_value)
            period = None if period_value is None or pd.isna(period_value) else str(period_value)
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
                    "detail": detail,
                    "period": period,
                    "source_row_sha256": source_sha,
                    "availability_rule": "first_official_open_session_strictly_after_source_event_date",
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
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["available_signal_date", "stock_code", "source_api", "event_id"]
    ).reset_index(drop=True)


def build_signal_features(
    events: pd.DataFrame,
    open_dates: list[str],
    *,
    end_date: str | None = None,
) -> pd.DataFrame:
    if end_date is not None:
        end_date = normalize_yyyymmdd(end_date)
        open_dates = [date for date in open_dates if date <= end_date]
    date_to_index = {date: index for index, date in enumerate(open_dates)}
    expanded: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        available_signal_date = str(row.available_signal_date)
        if available_signal_date not in date_to_index:
            continue
        start = date_to_index[available_signal_date]
        for age in range(max(WINDOWS)):
            if start + age >= len(open_dates):
                break
            expanded.append(
                {
                    "signal_date": open_dates[start + age],
                    "stock_code": str(row.stock_code),
                    "event_id": str(row.event_id),
                    "source_api": str(row.source_api),
                    "age_sessions": age,
                    "alert_active": bool(
                        row.source_api == "stk_alert"
                        and row.source_end_date is not None
                        and open_dates[start + age] <= str(row.source_end_date)
                    ),
                }
            )
    if not expanded:
        return pd.DataFrame(columns=["signal_date", "stock_code"])
    sparse = pd.DataFrame(expanded)
    keys = ["signal_date", "stock_code"]
    result = sparse[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    for window in WINDOWS:
        windowed = sparse[sparse["age_sessions"] < window]
        for api, prefix in (
            ("stk_shock", "shock"),
            ("stk_high_shock", "high_shock"),
            ("stk_alert", "alert_start"),
        ):
            counts = (
                windowed[windowed["source_api"] == api]
                .groupby(keys, sort=True)["event_id"]
                .nunique()
                .rename(f"{prefix}_count_{window}d")
                .reset_index()
            )
            result = result.merge(counts, on=keys, how="left")
    active = (
        sparse[sparse["alert_active"]]
        .groupby(keys, sort=True)["event_id"]
        .nunique()
        .rename("alert_active_count")
        .reset_index()
    )
    result = result.merge(active, on=keys, how="left")
    count_columns = [column for column in result.columns if column not in keys]
    result[count_columns] = result[count_columns].fillna(0).astype("int32")
    return result.sort_values(keys).reset_index(drop=True)


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


def build(start_date: str, end_date: str, output_root: Path) -> dict[str, Any]:
    start_date = normalize_yyyymmdd(start_date)
    end_date = normalize_yyyymmdd(end_date)
    if start_date > end_date:
        raise ValueError("start date must not exceed end date")
    pro = get_tushare_pro()
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    raw_by_api: dict[str, pd.DataFrame] = {}
    request_audit: dict[str, list[dict[str, Any]]] = {}
    for spec in API_SPECS:
        request_audit[spec.api] = []
        pieces = fetch_complete_range(pro, spec, start_date, end_date, request_audit[spec.api])
        raw_by_api[spec.api] = normalize_raw(spec, pieces, fetched_at)

    calendar_end = (pd.to_datetime(end_date, format="%Y%m%d", errors="raise") + pd.Timedelta(days=45)).strftime("%Y%m%d")
    open_dates = load_official_calendar(pro, start_date, calendar_end)
    events = build_l2_events(raw_by_api, open_dates)
    features = build_signal_features(events, open_dates, end_date=end_date)

    assets: dict[str, dict[str, Any]] = {}
    for spec in API_SPECS:
        path = output_root / f"l1_{spec.api}.duckdb"
        write_table(path, spec.api, raw_by_api[spec.api])
        assets[f"l1_{spec.api}"] = {"path": str(path), "table": spec.api, "sha256": sha256_file(path)}
    event_path = output_root / "l2_stock_risk_events.duckdb"
    feature_path = output_root / "l2_stock_risk_signal.duckdb"
    write_table(event_path, "stock_risk_events", events)
    write_table(feature_path, "stock_risk_signal", features)
    assets["l2_events"] = {
        "path": str(event_path),
        "table": "stock_risk_events",
        "sha256": sha256_file(event_path),
    }
    assets["l2_signal"] = {
        "path": str(feature_path),
        "table": "stock_risk_signal",
        "sha256": sha256_file(feature_path),
    }

    api_quality: dict[str, Any] = {}
    for spec in API_SPECS:
        frame = raw_by_api[spec.api]
        key_columns = ["ts_code", spec.date_column]
        if spec.api == "stk_alert":
            key_columns.append("end_date")
        api_quality[spec.api] = {
            "rows": int(len(frame)),
            "stocks": int(frame["ts_code"].nunique()),
            "min_date": None if frame.empty else str(frame[spec.date_column].min()),
            "max_date": None if frame.empty else str(frame[spec.date_column].max()),
            "duplicate_source_rows": int(frame["source_row_sha256"].duplicated().sum()),
            "required_key_nulls": int(frame[key_columns].isna().sum().sum()),
            "descriptive_nulls": int(frame[list(spec.required_columns)].isna().sum().sum()),
            "request_count": int(len(request_audit[spec.api])),
        }
    same_day_visibility = int((events["available_signal_date"] <= events["event_date"]).sum())
    manifest = {
        "schema_version": 1,
        "status": "candidate_l1_l2_ready",
        "source": "Tushare Pro",
        "source_apis": [spec.api for spec in API_SPECS],
        "requested_full_range": {"start": start_date, "end": end_date},
        "source_limit_handling": "recursive_date_bisection_until_each_response_below_1000_rows",
        "requests": request_audit,
        "pit_contract": {
            "event_dates": {
                "stk_shock": "trade_date documented as announcement date",
                "stk_high_shock": "trade_date documented as announcement date",
                "stk_alert": "start_date documented as exchange focus start date",
            },
            "strategy_availability": "first official open session strictly after source event date",
            "alert_end_semantics": "official source end_date inclusive",
            "same_day_visibility_rows": same_day_visibility,
        },
        "quality": {
            "apis": api_quality,
            "l2_rows": int(len(events)),
            "l2_stocks": int(events["stock_code"].nunique()),
            "l2_duplicate_event_ids": int(events["event_id"].duplicated().sum()),
            "l2_bj_rows": int(events["stock_code"].str.endswith(".BJ").sum()),
            "feature_rows": int(len(features)),
            "feature_duplicate_keys": int(features.duplicated(["signal_date", "stock_code"]).sum()),
        },
        "assets": assets,
        "production_modified": False,
        "legacy_eastmoney_asset_consumed": False,
    }
    hard_failures = [
        same_day_visibility,
        manifest["quality"]["l2_duplicate_event_ids"],
        manifest["quality"]["l2_bj_rows"],
        manifest["quality"]["feature_duplicate_keys"],
        *[value["duplicate_source_rows"] for value in api_quality.values()],
        *[value["required_key_nulls"] for value in api_quality.values()],
    ]
    if any(hard_failures):
        raise RuntimeError(f"Tushare L1/L2 quality gate failed: {hard_failures}")
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=SOURCE_START)
    parser.add_argument("--end", default=pd.Timestamp.now().strftime("%Y%m%d"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = build(args.start, args.end, args.output_root)
    print(json.dumps(result["quality"], ensure_ascii=False))


if __name__ == "__main__":
    main()
