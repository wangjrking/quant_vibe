from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
SIGNAL_FILE = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_trade_diagnostic_20260622"
)


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_signals() -> list[dict]:
    with SIGNAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_market(rows: list[dict]) -> tuple[dict[tuple[str, str], float], list[str]]:
    dates = sorted({str(row["buy_date"]) for row in rows if row.get("buy_date")})
    codes = sorted({str(row["stock_code"]) for row in rows if row.get("stock_code")})
    con = sqlite3.connect(MARKET_DB)
    trade_dates = [str(item[0]) for item in con.execute("select distinct trade_date from STOCK_DAILY_DATA order by trade_date")]
    start = min(dates)
    end = trade_dates[min(len(trade_dates) - 1, trade_dates.index(max(dates)) + 15)]
    market: dict[tuple[str, str], float] = {}
    placeholders = ",".join("?" for _ in codes)
    sql = f"""
        select trade_date, stock_code, open
        from STOCK_DAILY_DATA
        where trade_date >= ? and trade_date <= ? and stock_code in ({placeholders})
    """
    for trade_date, stock_code, open_price in con.execute(sql, [start, end, *codes]):
        if open_price is not None:
            market[(str(trade_date), str(stock_code))] = float(open_price)
    con.close()
    return market, trade_dates


def _exit_date(buy_date: str, trade_dates: list[str], holding_days: int) -> str | None:
    if buy_date not in trade_dates:
        return None
    idx = trade_dates.index(buy_date)
    exit_idx = idx + int(holding_days)
    if exit_idx >= len(trade_dates):
        return None
    return trade_dates[exit_idx]


def _bucket(row: dict, name: str) -> str:
    value = _to_float(row.get(name))
    if name == "rank":
        rank = int(value or 0)
        if rank <= 1:
            return "rank<=1"
        if rank <= 2:
            return "rank=2"
        if rank <= 3:
            return "rank=3"
        return "rank>=4"
    if name == "pred_gap":
        if value is None:
            return "missing"
        if value < 0.03:
            return "<0.03"
        if value < 0.05:
            return "0.03-0.05"
        if value < 0.07:
            return "0.05-0.07"
        return ">=0.07"
    if name == "pred_10d":
        if value is None:
            return "missing"
        if value < 0.08:
            return "<0.08"
        if value < 0.12:
            return "0.08-0.12"
        if value < 0.16:
            return "0.12-0.16"
        return ">=0.16"
    if name == "amount":
        if value is None:
            return "missing"
        if value < 10000:
            return "<1w"
        if value < 30000:
            return "1w-3w"
        if value < 60000:
            return "3w-6w"
        return ">=6w"
    if name == "turnover_rate":
        if value is None:
            return "missing"
        if value < 1:
            return "<1"
        if value < 3:
            return "1-3"
        if value < 6:
            return "3-6"
        return ">=6"
    if name == "total_mv":
        if value is None:
            return "missing"
        if value < 80000:
            return "<8w"
        if value < 120000:
            return "8w-12w"
        if value < 180000:
            return "12w-18w"
        return ">=18w"
    return str(row.get(name) or "")


def _summarize(rows: list[dict], group_field: str) -> list[dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    weights: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = _bucket(row, group_field)
        ret = _to_float(row.get("oo_return"))
        if ret is None:
            continue
        groups[key].append(ret)
        weights[key].append(_to_float(row.get("target_pct"), 0.0) or 0.0)
    out = []
    for key, values in groups.items():
        w = weights[key]
        weight_sum = sum(w)
        weighted_mean = sum(v * ww for v, ww in zip(values, w)) / weight_sum if weight_sum else None
        out.append(
            {
                "group_field": group_field,
                "bucket": key,
                "count": len(values),
                "mean_return": sum(values) / len(values),
                "weighted_mean_return": weighted_mean,
                "win_rate": sum(1 for value in values if value > 0) / len(values),
                "min_return": min(values),
                "max_return": max(values),
                "target_weight_sum": weight_sum,
            }
        )
    return sorted(out, key=lambda item: (item["group_field"], item["bucket"]))


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
        writer.writerows(rows)


def main() -> int:
    rows = _read_signals()
    market, trade_dates = _load_market(rows)
    enriched = []
    for row in rows:
        buy_date = str(row.get("buy_date") or "")
        stock_code = str(row.get("stock_code") or "")
        holding_days = int(_to_float(row.get("holding_days"), 6) or 6)
        exit_date = _exit_date(buy_date, trade_dates, holding_days)
        buy_open = market.get((buy_date, stock_code))
        exit_open = market.get((exit_date, stock_code)) if exit_date else None
        out = dict(row)
        out["diagnostic_exit_date"] = exit_date
        out["diagnostic_buy_open"] = buy_open
        out["diagnostic_exit_open"] = exit_open
        out["oo_return"] = (exit_open / buy_open - 1.0) if buy_open not in (None, 0) and exit_open is not None else None
        enriched.append(out)

    summary = []
    for field in ["pool_role", "rank", "pred_gap", "pred_10d", "amount", "turnover_rate", "total_mv"]:
        summary.extend(_summarize(enriched, field))
    _write_csv(REPORT_DIR / "trade_feature_bucket_summary.csv", summary)
    _write_csv(REPORT_DIR / "trade_feature_enriched_signals.csv", enriched)
    for row in sorted(summary, key=lambda item: item["weighted_mean_return"], reverse=True)[:20]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
