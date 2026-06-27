from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
    / "diversification_tune_20260624"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"

STRATEGIES = [
    "div_top1_w90_5d10_h5_e098",
    "div_top1_w90_5d10_h5_e099",
    "div_top1_w90_5d10_h5_e099_pos93",
    "div_top1_w90_5d10_h5_e099_pos95",
    "div_top1_w90_5d10_h5_e099_pos98",
    "div_top1_w90_5d10_h5_e099_mh2",
    "div_top1_w89_5d11_h5_e099",
]

ANCHORS = ["20250701", "20251009", "20260105"]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


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


def _parse_nav(log_file: Path) -> list[dict]:
    pattern = re.compile(
        r"EXPOSURE\s+(?P<date>\d{8})\s+post_buy\s+invested_pct=(?P<invested>[0-9.]+)"
        r".*active_positions=(?P<active>\d+).*nav=(?P<nav>[0-9.]+)"
    )
    rows = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rows.append(
            {
                "date": match.group("date"),
                "nav": float(match.group("nav")),
                "invested_pct": float(match.group("invested")),
                "active_positions": int(match.group("active")),
            }
        )
    rows.sort(key=lambda row: row["date"])
    return rows


def _max_drawdown(values: list[float]) -> float:
    peak = None
    out = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak and peak > 0:
            out = max(out, 1.0 - value / peak)
    return out


def _nav_metrics(rows: list[dict]) -> dict:
    if len(rows) < 2:
        return {
            "period_days": len(rows),
            "annual": None,
            "pnl": None,
            "sharpe": None,
            "max_drawdown": None,
            "avg_invested_pct": None,
        }
    start = rows[0]["nav"]
    end = rows[-1]["nav"]
    returns = []
    for prev, curr in zip(rows, rows[1:]):
        if prev["nav"] > 0:
            returns.append(curr["nav"] / prev["nav"] - 1.0)
    mean = sum(returns) / len(returns) if returns else 0.0
    if len(returns) > 1:
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {
        "period_days": len(rows),
        "annual": (end / start) ** (252.0 / max(len(rows) - 1, 1)) - 1.0 if start > 0 else None,
        "pnl": end / start - 1.0 if start > 0 else None,
        "sharpe": mean / std * math.sqrt(252.0) if std > 0 else None,
        "max_drawdown": _max_drawdown([row["nav"] for row in rows]),
        "avg_invested_pct": sum(row["invested_pct"] for row in rows) / len(rows),
        "max_active_positions": max(row["active_positions"] for row in rows),
    }


def _nav_slice_rows(strategy: str) -> list[dict]:
    full_log = REPORT_DIR / "logs" / f"{strategy}_full_20240605.log"
    nav = _parse_nav(full_log)
    out = []
    for anchor in ANCHORS:
        sliced = [row for row in nav if row["date"] >= anchor]
        metrics = _nav_metrics(sliced)
        out.append({"strategy": strategy, "anchor": anchor, "source": "continuous_account_nav_slice", **metrics})
    return out


def _cold_rows(strategy: str) -> list[dict]:
    cases = _read_csv(REPORT_DIR / "cases.csv")
    out = []
    for row in cases:
        if row.get("name") != strategy or not str(row.get("start_name") or "").startswith("late_"):
            continue
        out.append(
            {
                "strategy": strategy,
                "anchor": str(row["start_name"]).replace("late_", ""),
                "source": "cold_start_juejin",
                "annual": _to_float(row.get("annual")),
                "pnl": _to_float(row.get("pnl_ratio")),
                "sharpe": _to_float(row.get("sharpe")),
                "max_drawdown": _to_float(row.get("max_drawdown")),
                "avg_invested_pct": _to_float(row.get("avg_invested_pct")),
                "max_active_positions": _to_float(row.get("max_active_positions")),
            }
        )
    return out


def _daily_and_month_concentration(strategy: str) -> dict:
    nav = _parse_nav(REPORT_DIR / "logs" / f"{strategy}_full_20240605.log")
    daily = []
    for prev, curr in zip(nav, nav[1:]):
        change = curr["nav"] - prev["nav"]
        daily.append({"date": curr["date"], "month": curr["date"][:6], "nav_change": change})
    total_gain = sum(row["nav_change"] for row in daily if row["nav_change"] > 0)
    top_day = max(daily, key=lambda row: row["nav_change"], default=None)
    month_gain = defaultdict(float)
    for row in daily:
        month_gain[row["month"]] += row["nav_change"]
    top_month_key = max(month_gain, key=lambda key: month_gain[key]) if month_gain else None
    return {
        "strategy": strategy,
        "total_positive_nav_change": total_gain,
        "top_day": top_day["date"] if top_day else "",
        "top_day_nav_change": top_day["nav_change"] if top_day else None,
        "top_day_positive_gain_share": top_day["nav_change"] / total_gain if top_day and total_gain > 0 else None,
        "top_month": top_month_key or "",
        "top_month_nav_change": month_gain[top_month_key] if top_month_key else None,
        "top_month_positive_gain_share": month_gain[top_month_key] / total_gain if top_month_key and total_gain > 0 else None,
    }


def _market_date_index() -> list[str]:
    conn = sqlite3.connect(MARKET_DB)
    try:
        dates = [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")]
    finally:
        conn.close()
    return dates


def _load_open_prices(stock_codes: set[str], dates: set[str]) -> dict[tuple[str, str], float]:
    if not stock_codes or not dates:
        return {}
    conn = sqlite3.connect(MARKET_DB)
    try:
        placeholders_code = ",".join("?" for _ in stock_codes)
        placeholders_date = ",".join("?" for _ in dates)
        sql = (
            f"SELECT trade_date, stock_code, open FROM STOCK_DAILY_DATA "
            f"WHERE stock_code IN ({placeholders_code}) AND trade_date IN ({placeholders_date})"
        )
        params = list(stock_codes) + list(dates)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return {(str(date), str(code)): float(open_price) for date, code, open_price in rows if open_price not in (None, "")}


def _signal_horizon_contribution(strategy: str) -> dict:
    signal_file = REPORT_DIR / "signals" / f"{strategy}.csv"
    signals = _read_csv(signal_file)
    market_dates = _market_date_index()
    date_pos = {date: index for index, date in enumerate(market_dates)}
    requested_dates = set()
    stock_codes = set()
    for row in signals:
        buy_date = str(row.get("buy_date") or "")
        stock_code = str(row.get("stock_code") or "")
        hold = int(float(row.get("holding_days") or 5))
        if buy_date not in date_pos:
            continue
        sell_index = min(date_pos[buy_date] + hold, len(market_dates) - 1)
        sell_date = market_dates[sell_index]
        requested_dates.add(buy_date)
        requested_dates.add(sell_date)
        stock_codes.add(stock_code)
        row["_sell_date_proxy"] = sell_date
    prices = _load_open_prices(stock_codes, requested_dates)
    stock_contrib = defaultdict(float)
    day_contrib = defaultdict(float)
    month_contrib = defaultdict(float)
    total_positive = 0.0
    usable = 0
    for row in signals:
        buy_date = str(row.get("buy_date") or "")
        sell_date = str(row.get("_sell_date_proxy") or "")
        stock_code = str(row.get("stock_code") or "")
        buy_open = prices.get((buy_date, stock_code))
        sell_open = prices.get((sell_date, stock_code))
        target = _to_float(row.get("target_pct"), 0.0) or 0.0
        if not buy_open or not sell_open or buy_open <= 0:
            continue
        contribution = target * (sell_open / buy_open - 1.0)
        usable += 1
        stock_contrib[stock_code] += contribution
        day_contrib[buy_date] += contribution
        month_contrib[buy_date[:6]] += contribution
        if contribution > 0:
            total_positive += contribution
    top_stock = max(stock_contrib, key=lambda key: stock_contrib[key]) if stock_contrib else None
    top_day = max(day_contrib, key=lambda key: day_contrib[key]) if day_contrib else None
    top_month = max(month_contrib, key=lambda key: month_contrib[key]) if month_contrib else None
    return {
        "strategy": strategy,
        "method": "proxy_open_to_holding_days_open_not_juejin_trade_pnl",
        "signals": len(signals),
        "usable_signals": usable,
        "total_positive_proxy_contribution": total_positive,
        "top_stock": top_stock or "",
        "top_stock_proxy_contribution": stock_contrib[top_stock] if top_stock else None,
        "top_stock_positive_share": stock_contrib[top_stock] / total_positive if top_stock and total_positive > 0 else None,
        "top_day": top_day or "",
        "top_day_proxy_contribution": day_contrib[top_day] if top_day else None,
        "top_day_positive_share": day_contrib[top_day] / total_positive if top_day and total_positive > 0 else None,
        "top_month": top_month or "",
        "top_month_proxy_contribution": month_contrib[top_month] if top_month else None,
        "top_month_positive_share": month_contrib[top_month] / total_positive if top_month and total_positive > 0 else None,
    }


def main() -> int:
    slice_rows = []
    cold_rows = []
    nav_concentration = []
    signal_concentration = []
    for strategy in STRATEGIES:
        slice_rows.extend(_nav_slice_rows(strategy))
        cold_rows.extend(_cold_rows(strategy))
        nav_concentration.append(_daily_and_month_concentration(strategy))
        signal_concentration.append(_signal_horizon_contribution(strategy))
    _write_csv(REPORT_DIR / "robustness_nav_slices.csv", slice_rows)
    _write_csv(REPORT_DIR / "robustness_cold_starts.csv", cold_rows)
    _write_csv(REPORT_DIR / "robustness_nav_concentration.csv", nav_concentration)
    _write_csv(REPORT_DIR / "robustness_signal_proxy_concentration.csv", signal_concentration)
    payload = {
        "nav_slices": slice_rows,
        "cold_starts": cold_rows,
        "nav_concentration": nav_concentration,
        "signal_proxy_concentration": signal_concentration,
    }
    (REPORT_DIR / "robustness_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
