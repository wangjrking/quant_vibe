from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA = ROOT / "quant" / "data_file"
REPORT_ROOT = (
    DATA
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
    / "diversification_tune_20260624"
    / "strict_sync_liquidity_neighborhood_20260624"
    / "formal_horizon_entry_confirmation_20260624"
)
RULE_DIR = REPORT_ROOT / "dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624"
REPORT_DIR = RULE_DIR / "style_exposure_audit_20260624"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"

PRIMARY_CASE = "continue_c975"
BASE_CASE = "base_pos89_e970_c980_mh3_sl06_tp06_dd0712"
WINDOWS = {
    "full": None,
    "recent120": 120,
    "recent60": 60,
    "recent20": 20,
}


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _f(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _parse_date(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return datetime.strptime(value[:8], "%Y%m%d")
    except ValueError:
        return None


def _bucket_age(buy_date: str, list_date: str) -> str:
    buy = _parse_date(buy_date)
    listed = _parse_date(list_date)
    if buy is None or listed is None:
        return "missing"
    days = (buy - listed).days
    if days < 60:
        return "new_lt60d"
    if days < 250:
        return "new_60_250d"
    if days < 750:
        return "mid_250_750d"
    return "old_ge750d"


def _bucket_price(close: float | None) -> str:
    if close is None:
        return "missing"
    if close < 5:
        return "price_lt5"
    if close < 10:
        return "price_5_10"
    if close < 20:
        return "price_10_20"
    if close < 50:
        return "price_20_50"
    return "price_ge50"


def _quantile_bucket(rank_pct: float | None, prefix: str) -> str:
    if rank_pct is None:
        return f"{prefix}_missing"
    if rank_pct <= 0.2:
        return f"{prefix}_q1_low"
    if rank_pct <= 0.4:
        return f"{prefix}_q2"
    if rank_pct <= 0.6:
        return f"{prefix}_q3"
    if rank_pct <= 0.8:
        return f"{prefix}_q4"
    return f"{prefix}_q5_high"


def _load_market(dates: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in dates)
    query = f"""
        select trade_date, stock_code, name, industry, market, list_date,
               open, close, amount, turnover_rate, total_mv, circ_mv, atr_qfq
        from STOCK_DAILY_DATA
        where trade_date in ({placeholders})
    """
    with sqlite3.connect(MARKET_DB) as con:
        df = pd.read_sql_query(query, con, params=dates)
    for col in ["open", "close", "amount", "turnover_rate", "total_mv", "circ_mv", "atr_qfq"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["total_mv", "amount", "turnover_rate", "atr_qfq", "close"]:
        df[f"{col}_rank_pct"] = df.groupby("trade_date")[col].rank(pct=True, method="average")
    return df


def _enrich_signals(case_name: str, signal_path: Path, market: pd.DataFrame) -> list[dict]:
    rows = _read_csv(signal_path)
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    merged = df.merge(
        market,
        left_on=["buy_date", "stock_code"],
        right_on=["trade_date", "stock_code"],
        how="left",
        suffixes=("", "_buy"),
    )
    enriched: list[dict] = []
    for row in merged.to_dict("records"):
        buy_date = str(row.get("buy_date") or "")
        out = {
            "case_name": case_name,
            "signal_date": row.get("signal_date"),
            "buy_date": buy_date,
            "stock_code": row.get("stock_code"),
            "name": row.get("name") or row.get("name_buy"),
            "industry": row.get("industry") or "missing",
            "market": row.get("market") or "missing",
            "list_date": row.get("list_date") or "",
            "total_mv": row.get("total_mv"),
            "amount": row.get("amount"),
            "turnover_rate": row.get("turnover_rate"),
            "atr_qfq": row.get("atr_qfq"),
            "close": row.get("close"),
            "mv_bucket": _quantile_bucket(_f(row.get("total_mv_rank_pct")), "mv"),
            "amount_bucket": _quantile_bucket(_f(row.get("amount_rank_pct")), "amount"),
            "turnover_bucket": _quantile_bucket(_f(row.get("turnover_rate_rank_pct")), "turnover"),
            "atr_bucket": _quantile_bucket(_f(row.get("atr_qfq_rank_pct")), "atr"),
            "price_bucket": _bucket_price(_f(row.get("close"))),
            "age_bucket": _bucket_age(buy_date, str(row.get("list_date") or "")),
            "market_join_missing": 1 if pd.isna(row.get("trade_date")) else 0,
        }
        enriched.append(out)
    return enriched


def _distribution(rows: list[dict], window: str, dimension: str) -> list[dict]:
    counter = Counter(row.get(dimension) or "missing" for row in rows)
    total = sum(counter.values()) or 1
    out = []
    for bucket, count in counter.most_common():
        out.append(
            {
                "window": window,
                "dimension": dimension,
                "bucket": bucket,
                "count": count,
                "share": count / total,
            }
        )
    return out


def _window_rows(rows: list[dict], window_size: int | None) -> list[dict]:
    if window_size is None:
        return rows
    dates = sorted({str(row.get("buy_date")) for row in rows})
    keep = set(dates[-window_size:])
    return [row for row in rows if str(row.get("buy_date")) in keep]


def _case_summary(case_rows: dict[str, list[dict]]) -> list[dict]:
    out = []
    dims = ["mv_bucket", "amount_bucket", "turnover_bucket", "atr_bucket", "price_bucket", "age_bucket", "industry"]
    for case, rows in case_rows.items():
        row = {"case_name": case, "rows": len(rows), "market_join_missing": sum(int(r["market_join_missing"]) for r in rows)}
        for dim in dims:
            counter = Counter(r.get(dim) or "missing" for r in rows)
            top, count = counter.most_common(1)[0] if counter else ("missing", 0)
            row[f"{dim}_top"] = top
            row[f"{dim}_top_share"] = count / len(rows) if rows else None
        out.append(row)
    return out


def _drift_summary(primary_rows: list[dict]) -> list[dict]:
    dims = ["mv_bucket", "amount_bucket", "turnover_bucket", "atr_bucket", "price_bucket", "age_bucket", "industry"]
    out = []
    for window, size in WINDOWS.items():
        rows = _window_rows(primary_rows, size)
        for dim in dims:
            counter = Counter(row.get(dim) or "missing" for row in rows)
            top, count = counter.most_common(1)[0] if counter else ("missing", 0)
            out.append(
                {
                    "window": window,
                    "dimension": dim,
                    "rows": len(rows),
                    "top_bucket": top,
                    "top_share": count / len(rows) if rows else None,
                    "bucket_count": len(counter),
                }
            )
    return out


def _flag(summary_rows: list[dict], drift_rows: list[dict]) -> dict:
    primary = next(row for row in summary_rows if row["case_name"] == PRIMARY_CASE)
    max_style = max(
        float(primary.get(key) or 0)
        for key in primary
        if key.endswith("_top_share")
    )
    high_concentration = max_style >= 0.50
    drift_flags = []
    by_dim: dict[str, dict[str, str]] = defaultdict(dict)
    for row in drift_rows:
        by_dim[row["dimension"]][row["window"]] = row["top_bucket"]
    for dim, by_window in by_dim.items():
        if by_window.get("full") and len(set(by_window.values())) > 1:
            drift_flags.append(dim)
    return {
        "primary_case": PRIMARY_CASE,
        "max_top_share": max_style,
        "high_concentration_ge50pct": high_concentration,
        "drift_dimensions": drift_flags,
        "style_warning": high_concentration or bool(drift_flags),
        "style_warning_reasons": {
            "high_concentration_ge50pct": high_concentration,
            "drift_dimensions_nonempty": bool(drift_flags),
        },
        "note": "该判断是策略侧风格提示，不单独阻断生产准入；若同时伴随参数敏感、低路径依赖、时间切片失效、未来信息、硬过滤或样本过窄问题，再按对应强准入项处理。",
    }


def _write_report(flag: dict, primary_summary: dict, drift_rows: list[dict]) -> None:
    drift_dims = ", ".join(flag["drift_dimensions"]) if flag["drift_dimensions"] else "无"
    lines = [
        "# pos89 风格暴露和风格漂移审计",
        "",
        "## 当前结论",
        "",
        f"- 审计对象：`{PRIMARY_CASE}`。",
        f"- 是否触发风格风险提示：`{flag['style_warning']}`。",
        f"- 最大单一风格集中度：`{flag['max_top_share']:.2%}`。",
        f"- 出现主导风格漂移的维度：{drift_dims}。",
        "",
        "该审计只使用买入日当时的 `STOCK_DAILY_DATA` 字段，不使用最新状态回填历史样本。风格暴露和风格漂移仅作为风险提示，不单独阻断生产准入。",
        "",
        "## 全周期主导风格",
        "",
        "| 维度 | 主导分组 | 占比 |",
        "| --- | --- | ---: |",
    ]
    for dim in ["mv_bucket", "amount_bucket", "turnover_bucket", "atr_bucket", "price_bucket", "age_bucket", "industry"]:
        lines.append(
            f"| `{dim}` | `{primary_summary.get(dim + '_top')}` | {float(primary_summary.get(dim + '_top_share') or 0):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 窗口漂移",
            "",
            "| 窗口 | 维度 | 主导分组 | 占比 | 分组数量 |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in drift_rows:
        lines.append(
            f"| `{row['window']}` | `{row['dimension']}` | `{row['top_bucket']}` | "
            f"{float(row.get('top_share') or 0):.2%} | {row['bucket_count']} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 明细：`{REPORT_DIR / 'style_enriched_signals.csv'}`",
            f"- 全周期分布：`{REPORT_DIR / 'style_distribution.csv'}`",
            f"- 候选一致性：`{REPORT_DIR / 'style_case_summary.csv'}`",
            f"- 窗口漂移：`{REPORT_DIR / 'style_drift_summary.csv'}`",
            f"- 结论 JSON：`{REPORT_DIR / 'style_warning_summary.json'}`",
        ]
    )
    (REPORT_DIR / "style_exposure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    signal_paths = sorted((RULE_DIR / "signals").glob("*.csv"))
    if not signal_paths:
        raise FileNotFoundError(RULE_DIR / "signals")
    dates = sorted({row["buy_date"] for path in signal_paths for row in _read_csv(path)})
    market = _load_market(dates)

    all_rows: list[dict] = []
    case_rows: dict[str, list[dict]] = {}
    for path in signal_paths:
        case = path.stem
        rows = _enrich_signals(case, path, market)
        case_rows[case] = rows
        all_rows.extend(rows)

    primary_rows = case_rows[PRIMARY_CASE]
    distribution_rows: list[dict] = []
    for window, size in WINDOWS.items():
        rows = _window_rows(primary_rows, size)
        for dim in ["mv_bucket", "amount_bucket", "turnover_bucket", "atr_bucket", "price_bucket", "age_bucket", "industry"]:
            distribution_rows.extend(_distribution(rows, window, dim))

    summary_rows = _case_summary(case_rows)
    drift_rows = _drift_summary(primary_rows)
    flag = _flag(summary_rows, drift_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORT_DIR / "style_enriched_signals.csv", all_rows)
    _write_csv(REPORT_DIR / "style_distribution.csv", distribution_rows)
    _write_csv(REPORT_DIR / "style_case_summary.csv", summary_rows)
    _write_csv(REPORT_DIR / "style_drift_summary.csv", drift_rows)
    (REPORT_DIR / "style_warning_summary.json").write_text(json.dumps(flag, ensure_ascii=False, indent=2), encoding="utf-8")
    primary_summary = next(row for row in summary_rows if row["case_name"] == PRIMARY_CASE)
    _write_report(flag, primary_summary, drift_rows)
    print(json.dumps(flag, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
