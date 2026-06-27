from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import research_10d_with_new5d_confirm_probe_20260621 as base
from research_10d_new5d_3d_confirm_grid_20260621 import TABLE_3D
from research_list_age_current_best_probe_20260620 import MARKET_DB, PRED_DB


ROOT = Path(__file__).resolve().parents[2]
SIGNAL_FILE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_dynamic_holding_fine_20260621"
    / "signals"
    / "h7_top_ge0p45_targetbase.csv"
)
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_signal_return_diagnostics_20260621"
)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


def _to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_trade_dates() -> list[str]:
    conn = _connect_readonly(MARKET_DB)
    try:
        dates = [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")]
    finally:
        conn.close()
    return dates


def _load_market_rows() -> dict[tuple[str, str], dict]:
    conn = _connect_readonly(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, open, close, amount, turnover_rate,
                   total_mv, index_2000_open, index_2000_close
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= ? AND trade_date <= ?
            """,
            (base.START_DATE, base.END_DATE),
        ).fetchall()
    finally:
        conn.close()
    return {(str(row["trade_date"]), str(row["stock_code"])): dict(row) for row in rows}


def _rank_by_day(table: str) -> dict[tuple[str, str], float]:
    conn = _connect_readonly(PRED_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(
            conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_prob
                FROM "{table}"
                WHERE trade_date >= ? AND trade_date <= ? AND pred_prob IS NOT NULL
                ORDER BY trade_date, pred_prob DESC, stock_code
                """,
                (base.START_DATE, base.END_DATE),
            )
        )
    finally:
        conn.close()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trade_date"])].append(row)
    ranks = {}
    for trade_date, day_rows in grouped.items():
        denom = max(len(day_rows) - 1, 1)
        for idx, row in enumerate(day_rows):
            ranks[(trade_date, str(row["stock_code"]))] = 1.0 - idx / denom
    return ranks


def _bucket(value: float | None, cuts: tuple[float, ...]) -> str:
    if value is None:
        return "missing"
    lower = "-inf"
    for cut in cuts:
        if value < cut:
            return f"{lower}_{cut}"
        lower = str(cut)
    return f"{lower}_inf"


def _summarize(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    out = []
    for value, items in grouped.items():
        returns = [float(item["open_return"]) for item in items if item.get("open_return") is not None]
        if not returns:
            continue
        out.append(
            {
                "feature": key,
                "bucket": value,
                "count": len(returns),
                "mean_return": sum(returns) / len(returns),
                "win_rate": sum(1 for ret in returns if ret > 0) / len(returns),
                "loss_tail_5pct": sorted(returns)[max(0, int(len(returns) * 0.05) - 1)],
            }
        )
    return sorted(out, key=lambda item: (item["feature"], item["bucket"]))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    trade_dates = _load_trade_dates()
    next_by_date = {date: trade_dates[idx + 1] for idx, date in enumerate(trade_dates[:-1])}
    idx_by_date = {date: idx for idx, date in enumerate(trade_dates)}
    market = _load_market_rows()
    rank3d = _rank_by_day(TABLE_3D)

    enriched = []
    with SIGNAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            buy_date = str(row["buy_date"])
            stock_code = str(row["stock_code"])
            holding_days = int(float(row.get("holding_days") or 5))
            buy_idx = idx_by_date.get(buy_date)
            if buy_idx is None or buy_idx + holding_days >= len(trade_dates):
                continue
            sell_date = trade_dates[buy_idx + holding_days]
            buy_row = market.get((buy_date, stock_code))
            sell_row = market.get((sell_date, stock_code))
            buy_open = _to_float((buy_row or {}).get("open"))
            sell_open = _to_float((sell_row or {}).get("open"))
            if buy_open is None or sell_open is None or buy_open <= 0:
                continue
            signal_date = str(row["signal_date"])
            signal_market = market.get((signal_date, stock_code), {})
            day_market = next((v for (day, code), v in market.items() if day == signal_date and code == stock_code), {})
            index_open = _to_float(signal_market.get("index_2000_open"))
            index_close = _to_float(signal_market.get("index_2000_close"))
            index_intraday = None
            if index_open and index_close:
                index_intraday = index_close / index_open - 1.0
            out = dict(row)
            out["open_return"] = sell_open / buy_open - 1.0
            out["sell_date"] = sell_date
            out["amount"] = signal_market.get("amount")
            out["turnover_rate"] = signal_market.get("turnover_rate")
            out["rank3d"] = rank3d.get((signal_date, stock_code))
            out["index_2000_intraday"] = index_intraday
            out["pred_bucket"] = _bucket(_to_float(row.get("pred_prob")), (0.25, 0.35, 0.45, 0.55, 0.65))
            out["close_rate_bucket"] = _bucket(_to_float(row.get("close_rate")), (0.96, 0.98, 1.0, 1.03, 1.06))
            out["mv_bucket"] = _bucket(_to_float(row.get("total_mv")), (50000, 80000, 120000, 200000))
            out["amount_bucket"] = _bucket(_to_float(out.get("amount")), (10000, 30000, 60000, 120000, 300000))
            out["turnover_bucket"] = _bucket(_to_float(out.get("turnover_rate")), (1.0, 2.0, 4.0, 8.0, 15.0))
            out["rank5d_bucket"] = _bucket(_to_float(row.get("rank5d_new")), (0.5, 0.7, 0.85, 0.95))
            out["rank3d_bucket"] = _bucket(_to_float(out.get("rank3d")), (0.3, 0.5, 0.7, 0.85))
            out["list_age_bucket"] = _bucket(_to_float(row.get("list_age_days")), (120, 250, 500, 1000, 2500))
            out["index_intraday_bucket"] = _bucket(index_intraday, (-0.03, -0.015, 0.0, 0.015, 0.03))
            enriched.append(out)

    detail_file = REPORT_DIR / "signal_forward_returns.csv"
    fieldnames = list(enriched[0].keys())
    with detail_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    summary = []
    for key in (
        "pred_bucket",
        "close_rate_bucket",
        "mv_bucket",
        "amount_bucket",
        "turnover_bucket",
        "rank5d_bucket",
        "rank3d_bucket",
        "list_age_bucket",
        "index_intraday_bucket",
    ):
        summary.extend(_summarize(enriched, key))
    with (REPORT_DIR / "bucket_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps({"rows": len(enriched), "detail_file": str(detail_file)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
