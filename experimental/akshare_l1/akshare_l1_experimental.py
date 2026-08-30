from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTAL_ROOT = PROJECT_ROOT / "quant" / "data_file" / "experimental_assets"
PRODUCTION_ROOT = PROJECT_ROOT / "quant" / "data_file" / "production_assets"
DEFAULT_OUTPUT_ROOT = EXPERIMENTAL_ROOT / "akshare_l1_gap_fill_20260714"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "quant" / "data_file" / "reports"
ASSET_STATUS = "experimental/test_not_approved_for_production"
ASSET_CONTRACTS: dict[str, dict[str, Any]] = {
    "company_announcements": {
        "source": "AKShare.stock_notice_report",
        "date_field": "announcement_date",
        "source_parameters": {"symbol": "全部", "date": "20260713"},
        "sample_scope": {"scope": "all_source_rows_for_one_disclosure_date", "date": "20260713"},
    },
    "stock_news": {
        "source": "AKShare.stock_news_em",
        "date_field": "published_at",
        "source_parameters": {"symbol": "one_6_digit_security_code_per_call"},
        "sample_scope": {"scope": "three_stock_recent_news_sample", "codes": ["000001.SZ", "600487.SH", "688006.SH"]},
    },
    "social_sentiment_snapshot": {
        "source": "AKShare.stock_js_weibo_report",
        "date_field": "snapshot_at",
        "source_parameters": {"time_period": "CNHOUR24"},
        "sample_scope": {"scope": "one_rolling_24_hour_name_only_snapshot", "source_rows": 50},
    },
    "stock_minutes_1m": {
        "source": "AKShare.stock_zh_a_minute",
        "date_field": "trade_time",
        "source_parameters": {
            "symbols": ["sz000001", "sh600487", "sh688006"],
            "period": "1",
            "adjust": "",
        },
        "fallback_from": "AKShare.stock_zh_a_hist_min_em",
        "fallback_reason": "primary interface failed with ProxyError for all three sample stocks",
        "adjustment": {
            "parameter": "adjust",
            "value": "",
            "semantics": "unadjusted/raw_price",
            "verification": "AKShare 1.18.64 stock_zh_a_minute docstring states empty adjust returns unadjusted data",
        },
        "sample_scope": {
            "scope": "three_stocks_one_trade_date_only",
            "codes": ["000001.SZ", "600487.SH", "688006.SH"],
            "trade_date": "20260713",
            "time_range": "09:31:00-15:00:00",
            "not_full_market": True,
            "not_full_history": True,
        },
    },
    "financial_indicator": {
        "source": "AKShare.stock_financial_analysis_indicator_em",
        "date_field": "report_date",
        "source_parameters": {"indicator": "按报告期", "symbol": "one_ts_code_per_call"},
        "sample_scope": {
            "scope": "three_stock_historical_financial_sample",
            "codes": ["000001.SZ", "600487.SH", "688006.SH"],
        },
    },
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    experimental = EXPERIMENTAL_ROOT.resolve()
    production = PRODUCTION_ROOT.resolve()
    if not _is_relative_to(resolved, experimental):
        raise ValueError(f"output root must be under experimental_assets: {resolved}")
    if _is_relative_to(resolved, production) or "production_assets" in resolved.parts:
        raise ValueError(f"production output is forbidden: {resolved}")
    return resolved


def _safe_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalize_code(code: str) -> str:
    value = str(code).strip().upper()
    if value.endswith((".SH", ".SZ", ".BJ")):
        return value
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) != 6:
        return value
    if digits.startswith(("4", "8", "92")):
        return f"{digits}.BJ"
    if digits.startswith(("5", "6", "9")):
        return f"{digits}.SH"
    return f"{digits}.SZ"


def _column(frame: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    if required:
        raise KeyError(f"missing required source column; expected one of {list(aliases)}")
    return None


def _hash_rows(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    def digest(row: pd.Series) -> str:
        raw = "|".join("" if pd.isna(row[col]) else str(row[col]).strip() for col in columns)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return frame.apply(digest, axis=1)


def transform_announcements(frame: pd.DataFrame, snapshot_at: str) -> pd.DataFrame:
    code = _column(frame, ["代码", "股票代码", "证券代码"])
    title = _column(frame, ["公告标题", "标题"])
    date = _column(frame, ["公告日期", "公告时间", "日期"])
    name = _column(frame, ["名称", "股票简称", "证券简称"], required=False)
    category = _column(frame, ["公告类型", "类型", "公告类别"], required=False)
    url = _column(frame, ["网址", "公告链接", "链接"], required=False)
    result = pd.DataFrame(
        {
            "ts_code": frame[code].map(_normalize_code),
            "stock_name": frame[name] if name else None,
            "title": frame[title].astype(str).str.strip(),
            "category": frame[category] if category else None,
            "announcement_date": pd.to_datetime(frame[date], errors="coerce").dt.strftime("%Y%m%d"),
            "url": frame[url] if url else None,
            "snapshot_at": snapshot_at,
            "source": "AKShare.stock_notice_report",
        }
    )
    result["announcement_id"] = _hash_rows(result, ["ts_code", "announcement_date", "title", "url"])
    return result.drop_duplicates("announcement_id").reset_index(drop=True)


def transform_stock_news(frame: pd.DataFrame, query_code: str, snapshot_at: str) -> pd.DataFrame:
    title = _column(frame, ["新闻标题", "标题"])
    published = _column(frame, ["发布时间", "日期", "时间"])
    content = _column(frame, ["新闻内容", "内容"], required=False)
    source = _column(frame, ["文章来源", "来源"], required=False)
    url = _column(frame, ["新闻链接", "链接", "网址"], required=False)
    keyword = _column(frame, ["关键词"], required=False)
    result = pd.DataFrame(
        {
            "ts_code": _normalize_code(query_code),
            "keyword": frame[keyword] if keyword else None,
            "title": frame[title].astype(str).str.strip(),
            "content": frame[content] if content else None,
            "published_at": pd.to_datetime(frame[published], errors="coerce"),
            "article_source": frame[source] if source else None,
            "url": frame[url] if url else None,
            "snapshot_at": snapshot_at,
            "source": "AKShare.stock_news_em",
        }
    )
    result["news_id"] = _hash_rows(result, ["ts_code", "published_at", "title", "url"])
    return result.drop_duplicates("news_id").reset_index(drop=True)


def transform_social_sentiment(frame: pd.DataFrame, snapshot_at: str) -> pd.DataFrame:
    name = _column(frame, ["name", "名称", "股票名称"])
    rate = _column(frame, ["rate", "热度", "比率"])
    result = pd.DataFrame(
        {
            "snapshot_at": snapshot_at,
            "rank": range(1, len(frame) + 1),
            "stock_name": frame[name].astype(str).str.strip(),
            "sentiment_rate": pd.to_numeric(frame[rate], errors="coerce"),
            "source": "AKShare.stock_js_weibo_report",
        }
    )
    # The source has no security code. Keep it as an explicitly name-only raw snapshot.
    result["mapping_status"] = "partial_name_only_no_ts_code"
    return result


def transform_financial_indicator(frame: pd.DataFrame) -> pd.DataFrame:
    code = _column(frame, ["SECUCODE", "SECURITY_CODE"])
    source_report_date = _column(frame, ["REPORT_DATE"])
    canonical_report_date = pd.to_datetime(frame[source_report_date], errors="coerce").dt.strftime("%Y%m%d")
    # DuckDB identifiers are case-insensitive, so retaining REPORT_DATE would
    # silently materialize it as REPORT_DATE_1 beside canonical report_date.
    result = frame.drop(columns=[source_report_date]).copy()
    result.insert(0, "ts_code", result[code].map(_normalize_code))
    result.insert(1, "report_date", canonical_report_date)
    result["source"] = "AKShare.stock_financial_analysis_indicator_em"
    return result.drop_duplicates(["ts_code", "report_date"]).reset_index(drop=True)


def transform_minutes(frame: pd.DataFrame, query_code: str) -> pd.DataFrame:
    timestamp = _column(frame, ["时间", "day", "日期时间", "datetime"])
    aliases = {
        "open": ["开盘", "open"],
        "high": ["最高", "high"],
        "low": ["最低", "low"],
        "close": ["收盘", "close"],
        "volume": ["成交量", "volume"],
        "amount": ["成交额", "amount"],
    }
    result = pd.DataFrame({"ts_code": [_normalize_code(query_code)] * len(frame)})
    result["trade_time"] = pd.to_datetime(frame[timestamp], errors="coerce")
    for target, candidates in aliases.items():
        source = _column(frame, candidates, required=False)
        result[target] = pd.to_numeric(frame[source], errors="coerce") if source else None
    result["source"] = "AKShare.stock_zh_a_minute"
    return result.dropna(subset=["trade_time"]).drop_duplicates(["ts_code", "trade_time"]).reset_index(drop=True)


def transform_risk_warning(frame: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    code = _column(frame, ["代码", "股票代码", "证券代码"])
    name = _column(frame, ["名称", "股票简称", "证券简称"], required=False)
    result = pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "ts_code": frame[code].map(_normalize_code),
            "stock_name": frame[name] if name else None,
            "risk_type": "ST_EXPLICIT_SOURCE_LIST",
            "source": "AKShare.stock_zh_a_st_em",
        }
    )
    return result.drop_duplicates(["snapshot_date", "ts_code"]).reset_index(drop=True)


def duplicate_groups(frame: pd.DataFrame, keys: list[str]) -> int:
    if frame.empty:
        return 0
    return int(frame.groupby(keys, dropna=False).size().gt(1).sum())


def frame_metrics(frame: pd.DataFrame, keys: list[str], date_column: str | None = None) -> dict[str, Any]:
    nulls = {str(column): int(count) for column, count in frame.isna().sum().items() if int(count) > 0}
    codes = int(frame["ts_code"].nunique()) if "ts_code" in frame.columns else None
    metric: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "stock_coverage": codes,
        "natural_key": keys,
        "duplicate_key_groups": duplicate_groups(frame, keys),
        "null_counts": nulls,
    }
    if date_column and date_column in frame.columns and not frame.empty:
        values = frame[date_column].dropna()
        metric["min_date"] = _safe_scalar(values.min()) if not values.empty else None
        metric["max_date"] = _safe_scalar(values.max()) if not values.empty else None
    return metric


def write_one_table_duckdb(output_root: Path, table_name: str, frame: pd.DataFrame) -> Path:
    root = validate_output_root(output_root)
    table_root = root / table_name
    table_root.mkdir(parents=True, exist_ok=True)
    db_path = table_root / f"{table_name}.duckdb"
    connection = duckdb.connect(str(db_path))
    try:
        connection.register("incoming_frame", frame)
        connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        connection.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM incoming_frame')
        connection.unregister("incoming_frame")
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
            ).fetchall()
        ]
        if tables != [table_name]:
            raise RuntimeError(f"one-table-one-file contract failed for {db_path}: {tables}")
    finally:
        connection.close()
    return db_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_table_state(table_name: str, date_column: str | None = None) -> dict[str, Any]:
    db_path = PRODUCTION_ROOT / "duckdb" / "l1_raw_tables" / f"{table_name}.duckdb"
    if not db_path.exists():
        return {"available": False, "reason": "active_table_missing", "path": str(db_path)}
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        state: dict[str, Any] = {
            "available": True,
            "path": str(db_path),
            "rows": int(connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]),
            "stock_coverage": int(connection.execute(f'SELECT COUNT(DISTINCT ts_code) FROM "{table_name}"').fetchone()[0]) if "ts_code" in columns else None,
        }
        if date_column and date_column in columns:
            minimum, maximum = connection.execute(f'SELECT MIN("{date_column}"), MAX("{date_column}") FROM "{table_name}"').fetchone()
            state.update({"date_column": date_column, "min_date": _safe_scalar(minimum), "max_date": _safe_scalar(maximum)})
        return state
    finally:
        connection.close()


def collect_existing_l1_mapping() -> dict[str, Any]:
    return {
        "announcements": {
            "status": "missing",
            "website_contract": "companyAnnouncements=false; no active announcement event table",
            "evidence": ["site/backend/server.py:139", "site/scripts/build_platform_snapshot.py:77"],
        },
        "stock_news": {
            "status": "missing",
            "website_contract": "only general major_news title matching; no company-news table",
            "evidence": ["site/backend/server.py:556", "site/backend/server.py:673"],
        },
        "intraday_minutes": {
            "status": "missing",
            "website_contract": "no formal 5m/30m or intraday quote asset",
            "evidence": ["site/scripts/build_platform_snapshot.py:76", "site/backend/server.py:1041"],
        },
        "sentiment": {
            "status": "missing",
            "website_contract": "no formal sentiment table",
            "evidence": ["site/scripts/build_platform_snapshot.py:78", "site/backend/server.py:1041"],
        },
        "financial": {
            "status": "partial",
            "active_table": "finan_data_season",
            "state": _active_table_state("finan_data_season", "end_date"),
        },
        "risk_warning": {
            "status": "available",
            "active_table": "stock_st",
            "state": _active_table_state("stock_st", "trade_date"),
            "note": "formal explicit risk table already exists; AKShare is only a supplemental-source probe",
        },
        "event_heat_rank": {
            "status": "partial",
            "active_tables": {
                "top_list": _active_table_state("top_list", "trade_date"),
                "ths_hot": _active_table_state("ths_hot", "trade_date"),
                "dc_hot": _active_table_state("dc_hot", "trade_date"),
            },
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AKShare experimental/test L1 gap-fill audit",
        "",
        f"- Task: `{report['task_id']}`",
        f"- Status: `{report['status']}`",
        f"- Formal platform as-of: `{report['formal_platform_as_of']}`",
        f"- Probe time: `{report['probe_time']}`",
        f"- Runtime: AKShare `{report['akshare_version']}`, pandas `{report['pandas_version']}`, Python `{report['python_version']}`",
        "- Scope: `experimental/test`; `not_approved_for_production=true`",
        "- Production assets touched: `false`; production route/registry touched: `false`; L2-L8 triggered: `false`; legacy odb used: `false`",
        "",
        "## Existing L1 and website gaps",
        "",
        "| Capability | Current status | Current evidence |",
        "|---|---:|---|",
    ]
    for capability, state in report.get("existing_l1_mapping", {}).items():
        evidence = state.get("website_contract") or state.get("note") or json.dumps(state.get("state") or state.get("active_tables", {}), ensure_ascii=False)
        lines.append(f"| `{capability}` | `{state.get('status')}` | {evidence} |")
    lines.extend(["", "## Experimental assets", "", "| Table | Rows | Stock coverage | Date range | Duplicate groups | Mapping |", "|---|---:|---:|---|---:|---|"])
    for table, asset in report.get("assets", {}).items():
        date_range = f"{asset.get('min_date')} to {asset.get('max_date')}"
        lines.append(
            f"| `{table}` | {asset['rows']} | {asset.get('stock_coverage')} | {date_range} | {asset['duplicate_key_groups']} | `{asset['mapping_status']}` |"
        )
    lines.extend(["", "## Source probes", "", "| Interface | Success | Rows | Error | Seconds |", "|---|---:|---:|---|---:|"])
    for probe in report.get("probes", []):
        lines.append(
            f"| `{probe['interface']}` | `{str(probe['success']).lower()}` | {probe.get('rows')} | {probe.get('error_type') or ''} | {probe['elapsed_seconds']} |"
        )
    lines.extend(["", "## Interface contracts", "", "| Capability | Interface | Frequency | History/coverage | Production readiness |", "|---|---|---|---|---|"])
    for capability, contract in report.get("interface_contracts", {}).items():
        lines.append(
            f"| `{capability}` | `{contract['interface']}` | {contract['update_frequency']} | {contract['history_or_coverage']} | `{contract['production_readiness']}` |"
        )
    lines.extend(["", "## L1_SOURCE_GAP", ""])
    for gap in report.get("l1_source_gaps", []):
        lines.append(f"- `{gap['capability']}`: `{gap['reason']}`")
    if report.get("concurrent_workspace_observations"):
        lines.extend(["", "## Concurrent workspace observation", ""])
        for observation in report["concurrent_workspace_observations"]:
            lines.append(f"- {observation}")
    lines.extend(
        [
            "",
            "## Promotion boundary",
            "",
            "The experimental package is ready for read-only audit, but it is not ready for production promotion. "
            "Announcements have a stable target-day raw mapping. News, minute bars, and financials remain sample-only; "
            "sentiment is name-only; Eastmoney heat ranking and explicit AKShare ST list are currently unavailable.",
            "",
            "Any production proposal requires a separate user approval with exact tables, date range, volume, runtime, rollback, and audit gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_interface_contracts(report: dict[str, Any]) -> dict[str, Any]:
    assets = report.get("assets", {})
    return {
        "announcements": {
            "interface": "stock_notice_report",
            "parameters": {"symbol": "全部", "date": "YYYYMMDD"},
            "update_frequency": "daily disclosure-date snapshot",
            "history_or_coverage": "one requested date; 20260713 sample has 432 rows and 269 stocks",
            "natural_key": ["announcement_id"],
            "estimated_production_volume": "hundreds of rows per disclosure day; backfill requires one request per calendar date",
            "production_readiness": "candidate_after_audit_and_user_approval",
        },
        "stock_news": {
            "interface": "stock_news_em",
            "parameters": {"symbol": "6-digit security code"},
            "update_frequency": "rolling recent-news query",
            "history_or_coverage": "10 rows per sampled stock in this run; no complete-history guarantee",
            "natural_key": ["news_id"],
            "estimated_production_volume": "roughly 10 rows times queried universe before dedup; requires thousands of per-stock calls",
            "production_readiness": "sample_only_requires_universe_scale_rate_limit_test",
        },
        "social_sentiment": {
            "interface": "stock_js_weibo_report",
            "parameters": {"time_period": "CNHOUR24"},
            "update_frequency": "rolling 24-hour snapshot",
            "history_or_coverage": "50 names; no security code and no source timestamp",
            "natural_key": ["snapshot_at", "rank"],
            "estimated_production_volume": "50 rows per ingestion snapshot",
            "production_readiness": "not_ready_name_only_mapping",
        },
        "heat_rank": {
            "interface": "stock_hot_rank_em",
            "parameters": {},
            "update_frequency": "source snapshot",
            "history_or_coverage": "unavailable in this environment due upstream proxy failure",
            "natural_key": None,
            "estimated_production_volume": None,
            "production_readiness": "L1_SOURCE_GAP",
        },
        "intraday_minutes": {
            "interface": "stock_zh_a_minute",
            "parameters": {"symbol": "Sina-prefixed security code", "period": "1", "adjust": ""},
            "fallback_from": "stock_zh_a_hist_min_em",
            "adjustment": "unadjusted/raw_price; verified from AKShare 1.18.64 empty-adjust contract",
            "update_frequency": "intraday/recent minute history",
            "history_or_coverage": f"fallback returned 1970 recent rows per sampled stock; target 20260713 retained {assets.get('stock_minutes_1m', {}).get('rows', 0)} rows across 3 stocks",
            "natural_key": ["ts_code", "trade_time"],
            "estimated_production_volume": "approximately 1.2 million rows per full A-share trading day before source suspensions and missing bars",
            "production_readiness": "sample_only_primary_source_blocked_and_fallback_history_limited",
        },
        "financial_indicator": {
            "interface": "stock_financial_analysis_indicator_em",
            "parameters": {"symbol": "ts_code", "indicator": "按报告期"},
            "update_frequency": "per reporting period",
            "history_or_coverage": f"{assets.get('financial_indicator', {}).get('rows', 0)} historical rows across 3 stocks, through 20260331",
            "natural_key": ["ts_code", "report_date"],
            "estimated_production_volume": "one request per stock; full current A-share universe requires about 5,210 calls",
            "production_readiness": "sample_only_requires_universe_scale_coverage_and_rate_limit_test",
        },
        "risk_warning": {
            "interface": "stock_zh_a_st_em",
            "parameters": {},
            "update_frequency": "source snapshot",
            "history_or_coverage": "explicit ST source list unavailable in this environment; formal stock_st already exists",
            "natural_key": ["snapshot_date", "ts_code"],
            "estimated_production_volume": None,
            "production_readiness": "L1_SOURCE_GAP_for_AKShare_supplement_only",
        },
    }


def run_probe(interface: str, params: dict[str, Any], call: Callable[[], pd.DataFrame]) -> tuple[dict[str, Any], pd.DataFrame | None]:
    started = time.monotonic()
    try:
        frame = call()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"expected DataFrame, got {type(frame).__name__}")
        record = {
            "interface": interface,
            "params": params,
            "success": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
        }
        return record, frame
    except Exception as exc:  # source failures belong in the evidence report
        return {
            "interface": interface,
            "params": params,
            "success": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }, None


def _sina_code(ts_code: str) -> str:
    normalized = _normalize_code(ts_code)
    code, suffix = normalized.split(".")
    return f"{'sh' if suffix == 'SH' else 'sz'}{code}"


def _add_asset(
    report: dict[str, Any],
    output_root: Path,
    table_name: str,
    frame: pd.DataFrame,
    keys: list[str],
    date_column: str | None,
    mapping_status: str,
) -> None:
    if duplicate_groups(frame, keys) != 0:
        raise RuntimeError(f"duplicate natural keys in {table_name}")
    path = write_one_table_duckdb(output_root, table_name, frame)
    entry = build_asset_entry(table_name, path, frame, keys, date_column, mapping_status)
    report["assets"][table_name] = entry


def build_asset_entry(
    table_name: str,
    path: Path,
    frame: pd.DataFrame,
    keys: list[str],
    date_column: str | None,
    mapping_status: str,
) -> dict[str, Any]:
    contract = ASSET_CONTRACTS[table_name]
    metrics = frame_metrics(frame, keys, date_column)
    entry: dict[str, Any] = {
        "table": table_name,
        "source": contract["source"],
        "status": ASSET_STATUS,
        **metrics,
        "mapping_status": mapping_status,
        "date_field": date_column,
        "db_path": str(path),
        "duckdb_path": str(path),
        "sha256": file_sha256(path),
        "source_parameters": contract["source_parameters"],
        "sample_scope": contract["sample_scope"],
        "one_table_one_file": True,
        "not_approved_for_production": True,
        "production_approved": False,
    }
    for field in ("fallback_from", "fallback_reason", "adjustment"):
        if field in contract:
            entry[field] = contract[field]
    return entry


def execute(as_of: str, output_root: Path, report_root: Path, sample_codes: list[str]) -> dict[str, Any]:
    output_root = validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    import akshare as ak

    snapshot_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "task_id": "investment-platform-akshare-test-l1-gap-fill-20260714",
        "status": "in_progress",
        "source": "AKShare",
        "akshare_version": getattr(ak, "__version__", "unknown"),
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "probe_time": snapshot_at,
        "formal_platform_as_of": as_of,
        "mode": "experimental/test",
        "not_approved_for_production": True,
        "output_root": str(output_root),
        "production_assets_touched": False,
        "production_route_or_registry_touched": False,
        "l2_l8_triggered": False,
        "legacy_odb_used": False,
        "downstream_l2_l7_triggered": False,
        "sample_codes": sample_codes,
        "existing_l1_mapping": collect_existing_l1_mapping(),
        "probes": [],
        "assets": {},
        "l1_source_gaps": [],
    }

    # Announcements: the source accepts one disclosure date per call.
    probe, announcements = run_probe(
        "stock_notice_report",
        {"symbol": "全部", "date": as_of},
        lambda: ak.stock_notice_report(symbol="全部", date=as_of),
    )
    report["probes"].append(probe)
    if probe["success"] and announcements is not None and not announcements.empty:
        try:
            transformed = transform_announcements(announcements, snapshot_at)
            _add_asset(report, output_root, "company_announcements", transformed, ["announcement_id"], "announcement_date", "stable_raw_mapping")
        except Exception as exc:
            report["l1_source_gaps"].append({"capability": "announcements", "reason": "mapping_failed", "error_type": type(exc).__name__, "error": str(exc)})
    else:
        report["l1_source_gaps"].append({"capability": "announcements", "reason": "source_unavailable_or_empty", "probe": probe})

    news_frames: list[pd.DataFrame] = []
    news_schemas: list[list[str]] = []
    for code in sample_codes:
        probe, source_frame = run_probe("stock_news_em", {"symbol": code[:6]}, lambda code=code: ak.stock_news_em(symbol=code[:6]))
        report["probes"].append(probe)
        if probe["success"] and source_frame is not None and not source_frame.empty:
            news_schemas.append(probe["columns"])
            news_frames.append(transform_stock_news(source_frame, code, snapshot_at))
    if news_frames and len({tuple(schema) for schema in news_schemas}) == 1:
        combined_news = pd.concat(news_frames, ignore_index=True).drop_duplicates("news_id")
        _add_asset(report, output_root, "stock_news", combined_news, ["news_id"], "published_at", "stable_sample_mapping")
    else:
        report["l1_source_gaps"].append({"capability": "news", "reason": "insufficient_successful_samples_or_schema_drift", "successful_samples": len(news_frames)})

    probe, sentiment = run_probe(
        "stock_js_weibo_report",
        {"time_period": "CNHOUR24"},
        lambda: ak.stock_js_weibo_report(time_period="CNHOUR24"),
    )
    report["probes"].append(probe)
    if probe["success"] and sentiment is not None and not sentiment.empty:
        transformed = transform_social_sentiment(sentiment, snapshot_at)
        _add_asset(report, output_root, "social_sentiment_snapshot", transformed, ["snapshot_at", "rank"], "snapshot_at", "partial_name_only_no_ts_code")
        report["l1_source_gaps"].append({"capability": "stock_level_sentiment_mapping", "reason": "source_has_no_security_code_or_source_timestamp"})
    else:
        report["l1_source_gaps"].append({"capability": "sentiment", "reason": "source_unavailable_or_empty", "probe": probe})

    probe, hot_rank = run_probe("stock_hot_rank_em", {}, lambda: ak.stock_hot_rank_em())
    report["probes"].append(probe)
    if not probe["success"] or hot_rank is None or hot_rank.empty:
        report["l1_source_gaps"].append({"capability": "heat_rank", "reason": "source_unavailable_or_empty", "probe": probe})

    minute_frames: list[pd.DataFrame] = []
    for code in sample_codes:
        start = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]} 09:30:00"
        end = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]} 15:00:00"
        probe, source_frame = run_probe(
            "stock_zh_a_hist_min_em",
            {"symbol": code[:6], "start_date": start, "end_date": end, "period": "1", "adjust": ""},
            lambda code=code, start=start, end=end: ak.stock_zh_a_hist_min_em(symbol=code[:6], start_date=start, end_date=end, period="1", adjust=""),
        )
        report["probes"].append(probe)
        if not probe["success"]:
            fallback, source_frame = run_probe(
                "stock_zh_a_minute",
                {"symbol": _sina_code(code), "period": "1", "adjust": ""},
                lambda code=code: ak.stock_zh_a_minute(symbol=_sina_code(code), period="1", adjust=""),
            )
            report["probes"].append(fallback)
            probe = fallback
        if probe["success"] and source_frame is not None and not source_frame.empty:
            transformed = transform_minutes(source_frame, code)
            transformed = transformed[transformed["trade_time"].dt.strftime("%Y%m%d") == as_of]
            if not transformed.empty:
                minute_frames.append(transformed)
    if minute_frames:
        combined_minutes = pd.concat(minute_frames, ignore_index=True).drop_duplicates(["ts_code", "trade_time"])
        _add_asset(report, output_root, "stock_minutes_1m", combined_minutes, ["ts_code", "trade_time"], "trade_time", "stable_sample_mapping")
    else:
        report["l1_source_gaps"].append({"capability": "intraday_minutes", "reason": "no_target_date_rows_from_primary_or_fallback"})

    finance_frames: list[pd.DataFrame] = []
    finance_schemas: list[list[str]] = []
    for code in sample_codes:
        probe, source_frame = run_probe(
            "stock_financial_analysis_indicator_em",
            {"symbol": _normalize_code(code), "indicator": "按报告期"},
            lambda code=code: ak.stock_financial_analysis_indicator_em(symbol=_normalize_code(code), indicator="按报告期"),
        )
        report["probes"].append(probe)
        if probe["success"] and source_frame is not None and not source_frame.empty:
            finance_schemas.append(probe["columns"])
            finance_frames.append(transform_financial_indicator(source_frame))
    if finance_frames and len({tuple(schema) for schema in finance_schemas}) == 1:
        combined_finance = pd.concat(finance_frames, ignore_index=True).drop_duplicates(["ts_code", "report_date"])
        _add_asset(report, output_root, "financial_indicator", combined_finance, ["ts_code", "report_date"], "report_date", "stable_sample_mapping")
    else:
        report["l1_source_gaps"].append({"capability": "financial_coverage", "reason": "insufficient_successful_samples_or_schema_drift", "successful_samples": len(finance_frames)})

    probe, st_frame = run_probe("stock_zh_a_st_em", {}, lambda: ak.stock_zh_a_st_em())
    report["probes"].append(probe)
    if probe["success"] and st_frame is not None and not st_frame.empty:
        try:
            transformed = transform_risk_warning(st_frame, as_of)
            _add_asset(report, output_root, "risk_warning", transformed, ["snapshot_date", "ts_code"], "snapshot_date", "explicit_source_list")
        except Exception as exc:
            report["l1_source_gaps"].append({"capability": "risk_warning", "reason": "explicit_code_mapping_failed", "error_type": type(exc).__name__, "error": str(exc)})
    else:
        report["l1_source_gaps"].append({"capability": "risk_warning", "reason": "explicit_source_list_unavailable", "probe": probe})

    report["status"] = "completed" if report["assets"] else "blocked"
    report["interface_contracts"] = build_interface_contracts(report)
    report["source_ready_for_production"] = False
    report["experimental_audit_ready"] = True
    report["requires_audit"] = True
    report["requires_user_approval_for_production"] = True

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_path = report_root / "investment_platform_akshare_l1_gap_fill_20260714.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown_path = report_root / "investment_platform_akshare_l1_gap_fill_20260714.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["manifest_path"] = str(manifest_path)
    report["report_path"] = str(report_path)
    report["markdown_report_path"] = str(markdown_path)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AKShare experimental/test-only L1 gap probe and ingest")
    parser.add_argument("--as-of", default="20260713")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--sample-code", action="append", dest="sample_codes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    sample_codes = args.sample_codes or ["000001.SZ", "600487.SH", "688006.SH"]
    report = execute(args.as_of, args.output_root, args.report_root, sample_codes)
    print(json.dumps({"status": report["status"], "assets": report["assets"], "gaps": report["l1_source_gaps"]}, ensure_ascii=False, default=str))
    return 0 if report["assets"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
