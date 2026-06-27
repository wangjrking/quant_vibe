from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pandas as pd

import tune_strict_sync_liquidity_refill_20260624 as liq


SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "formal_horizon_entry_confirmation_20260624" / "dynamic_candidate_proxy_analysis_20260624"
SIGNALS = {
    "dynamic_h2_m3": SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_hold_20260624"
    / "signals"
    / "dynamic_h2_m3_c098_w78_5d12_3d10_pos56_e097_mh1.csv",
    "fixed_h3": SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "entry_confirm_exit_neighborhood_20260624"
    / "signals"
    / "entry_confirm_w78_5d12_3d10_pos56_h3_e097_mh1.csv",
}
HOLD_DAYS = {"dynamic_h2_m3": 3, "fixed_h3": 3}
TARGET_PCT = 0.56


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_rows(path: Path, rows: list[dict]) -> None:
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


def _market_dates() -> list[str]:
    conn = sqlite3.connect(liq.MARKET_DB)
    try:
        return [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")]
    finally:
        conn.close()


def _load_open_prices(codes: set[str], dates: set[str]) -> dict[tuple[str, str], float]:
    if not codes or not dates:
        return {}
    conn = sqlite3.connect(liq.MARKET_DB)
    try:
        code_q = ",".join("?" for _ in codes)
        date_q = ",".join("?" for _ in dates)
        rows = conn.execute(
            f"SELECT trade_date, stock_code, open FROM STOCK_DAILY_DATA WHERE stock_code IN ({code_q}) AND trade_date IN ({date_q})",
            [*codes, *dates],
        ).fetchall()
    finally:
        conn.close()
    return {(str(date), str(code)): float(open_price) for date, code, open_price in rows if open_price not in (None, "")}


def _enrich(name: str, rows: list[dict]) -> list[dict]:
    dates = _market_dates()
    pos = {date: idx for idx, date in enumerate(dates)}
    request_dates: set[str] = set()
    codes: set[str] = set()
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        buy_date = str(item.get("buy_date") or "")
        code = str(item.get("stock_code") or "")
        if buy_date not in pos:
            continue
        sell_idx = min(pos[buy_date] + HOLD_DAYS[name], len(dates) - 1)
        item["proxy_sell_date"] = dates[sell_idx]
        request_dates.add(buy_date)
        request_dates.add(dates[sell_idx])
        codes.add(code)
        out.append(item)
    prices = _load_open_prices(codes, request_dates)
    for item in out:
        buy_open = prices.get((str(item["buy_date"]), str(item["stock_code"])))
        sell_open = prices.get((str(item["proxy_sell_date"]), str(item["stock_code"])))
        item["proxy_open_return"] = ""
        item["proxy_weighted_return"] = ""
        if buy_open and sell_open and buy_open > 0:
            ret = sell_open / buy_open - 1.0
            item["proxy_open_return"] = ret
            item["proxy_weighted_return"] = TARGET_PCT * ret
        item["month"] = str(item.get("signal_date") or "")[:6]
        item["period"] = _period(str(item.get("buy_date") or ""))
    return out


def _period(buy_date: str) -> str:
    if buy_date >= "20260324":
        return "recent60"
    if buy_date >= "20251224":
        return "recent120_ex_recent60"
    if buy_date >= "20260105":
        return "2026ytd_ex_recent120"
    if buy_date >= "20250701":
        return "2025h2"
    if buy_date >= "20250102":
        return "2025h1"
    return "2024h2"


def _summaries(name: str, rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    frame = pd.DataFrame(rows)
    frame["proxy_open_return_num"] = pd.to_numeric(frame["proxy_open_return"], errors="coerce")
    frame["proxy_weighted_return_num"] = pd.to_numeric(frame["proxy_weighted_return"], errors="coerce")
    period_rows = []
    for period, part in frame.groupby("period", sort=False):
        rets = part["proxy_open_return_num"].dropna()
        period_rows.append(
            {
                "strategy": name,
                "period": period,
                "trades": int(len(part)),
                "mean_proxy_open_return": float(rets.mean()) if len(rets) else None,
                "median_proxy_open_return": float(rets.median()) if len(rets) else None,
                "win_rate": float((rets > 0).mean()) if len(rets) else None,
                "sum_weighted_return": float(part["proxy_weighted_return_num"].sum()),
                "top_positive_share": _top_share(part),
                "bottom_5_sum": float(part.nsmallest(max(1, int(len(part) * 0.05)), "proxy_weighted_return_num")["proxy_weighted_return_num"].sum()) if len(part) else None,
            }
        )
    month_rows = []
    for month, part in frame.groupby("month", sort=True):
        month_rows.append(
            {
                "strategy": name,
                "month": month,
                "trades": int(len(part)),
                "sum_weighted_return": float(part["proxy_weighted_return_num"].sum()),
                "mean_proxy_open_return": float(part["proxy_open_return_num"].mean()),
            }
        )
    top_rows = frame.sort_values("proxy_weighted_return_num", ascending=False).head(20).copy()
    top_rows.insert(0, "strategy", name)
    return period_rows, month_rows, top_rows.to_dict("records")


def _top_share(part: pd.DataFrame) -> float | None:
    pos = part["proxy_weighted_return_num"].clip(lower=0.0)
    total = float(pos.sum())
    if total <= 0:
        return None
    n = max(1, int(len(part) * 0.01))
    return float(pos.sort_values(ascending=False).head(n).sum()) / total


def _write_report(period_rows: list[dict], month_rows: list[dict], top_rows: list[dict]) -> None:
    lines = [
        "# 动态候选代理贡献分析",
        "",
        "## 说明",
        "",
        "本报告使用买入日开盘到持有期后开盘的代理收益，只用于定位 recent60 弱和最高 1% 贡献压力，不作为正式收益结论。正式回测结论仍以掘金为准。",
        "",
        "## 分段代理收益",
        "",
        "| 策略 | 分段 | 交易数 | 平均代理收益 | 中位代理收益 | 胜率 | 加权收益和 | 最高 1% 正贡献占比 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in period_rows:
        lines.append(
            "| {strategy} | {period} | {trades} | {mean_proxy_open_return:.2%} | {median_proxy_open_return:.2%} | {win_rate:.2%} | {sum_weighted_return:.2%} | {top_positive_share:.2%} |".format(
                **{key: (0 if value is None else value) for key, value in row.items()}
            )
        )
    weak_months = sorted(month_rows, key=lambda row: row["sum_weighted_return"])[:10]
    lines.extend([
        "",
        "## 最弱月份",
        "",
        "| 策略 | 月份 | 交易数 | 加权收益和 | 平均代理收益 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in weak_months:
        lines.append(f"| {row['strategy']} | {row['month']} | {row['trades']} | {row['sum_weighted_return']:.2%} | {row['mean_proxy_open_return']:.2%} |")
    lines.extend([
        "",
        "## 最高代理贡献交易",
        "",
        "| 策略 | signal_date | buy_date | stock_code | name | 代理收益 | 加权代理收益 | amount | total_mv |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in top_rows[:20]:
        lines.append(
            f"| {row.get('strategy')} | {row.get('signal_date')} | {row.get('buy_date')} | {row.get('stock_code')} | {row.get('name')} | "
            f"{float(row.get('proxy_open_return_num') or 0):.2%} | {float(row.get('proxy_weighted_return_num') or 0):.2%} | "
            f"{float(row.get('amount') or 0):.0f} | {float(row.get('total_mv') or 0):.0f} |"
        )
    lines.extend([
        "",
        "## 证据路径",
        "",
        f"- 分段汇总：`{REPORT_DIR / 'period_summary.csv'}`",
        f"- 月度汇总：`{REPORT_DIR / 'month_summary.csv'}`",
        f"- 明细：`{REPORT_DIR / 'proxy_detail.csv'}`",
    ])
    (REPORT_DIR / "proxy_contribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    all_detail = []
    all_period = []
    all_month = []
    all_top = []
    for name, path in SIGNALS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _enrich(name, _read_rows(path))
        all_detail.extend(dict(row, strategy=name) for row in rows)
        period_rows, month_rows, top_rows = _summaries(name, rows)
        all_period.extend(period_rows)
        all_month.extend(month_rows)
        all_top.extend(top_rows)
    _write_rows(REPORT_DIR / "proxy_detail.csv", all_detail)
    _write_rows(REPORT_DIR / "period_summary.csv", all_period)
    _write_rows(REPORT_DIR / "month_summary.csv", all_month)
    _write_rows(REPORT_DIR / "top_proxy_trades.csv", all_top)
    (REPORT_DIR / "summary.json").write_text(json.dumps({"periods": all_period}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(all_period, all_month, all_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
