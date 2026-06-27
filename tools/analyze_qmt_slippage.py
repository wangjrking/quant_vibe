"""Analyze QMT trade slippage against daily open prices.

This tool parses local QMT `XtMiniQmt_YYYYMMDD.log` files, aggregates fills to
order level, joins them with `STOCK_DAILY_DATA.open`, and outputs both detail
and distribution summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable


DEFAULT_LOG_DIR = Path(r"D:\work\QMT\国金证券QMT交易端\userdata_mini\log")
DEFAULT_DB_PATH = Path(r"D:\work\quant\quant_mcp\quant\data_file\STOCK_DAILY_DATA.db")
DEFAULT_OUTPUT_DIR = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\runtime\trading_agent\slippage_reports"
)
DEFAULT_LEDGER_DIR = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\runtime\trading_agent\slippage_ledger"
)

LOG_NAME_RE = re.compile(r"XtMiniQmt_(\d{8})\.log$")
DEAL_RE = re.compile(
    r"push dealdetail, \[act: .*?market: (?P<market>\d+), stock: (?P<stock>\d+), "
    r"offset: (?P<offset>\d+), volume: (?P<volume>\d+), amount: (?P<amount>[0-9.]+), "
    r"sysid: (?P<sysid>\d+), bizno: (?P<bizno>[^,]+), .*?entype: (?P<entype>\d+), "
    r"prctype: (?P<prctype>\d+), remark:"
)


@dataclass
class Fill:
    trade_date: str
    timestamp: str
    stock_code: str
    side: str
    sysid: str
    bizno: str
    volume: int
    amount: float
    market: int


@dataclass
class OrderAgg:
    trade_date: str
    stock_code: str
    side: str
    sysid: str
    fill_count: int
    total_volume: int
    total_amount: float
    avg_fill_price: float
    open_price: float | None
    slippage_ratio: float | None
    slippage_amount: float | None
    first_fill_time: str
    last_fill_time: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze QMT slippage from local logs.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER_DIR)
    parser.add_argument("--start-date", type=str, default=None, help="YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYYMMDD")
    parser.add_argument(
        "--include-today-xtdata",
        action="store_true",
        help="Use local xtdata full tick open price when STOCK_DAILY_DATA lacks today's open.",
    )
    parser.add_argument(
        "--update-ledger",
        action="store_true",
        help="Append deduplicated order detail and daily summaries to the long-term ledger.",
    )
    return parser.parse_args()


def iter_log_files(log_dir: Path, start_date: str | None, end_date: str | None) -> Iterable[Path]:
    for path in sorted(log_dir.glob("XtMiniQmt_*.log")):
        match = LOG_NAME_RE.match(path.name)
        if not match:
            continue
        trade_date = match.group(1)
        if start_date and trade_date < start_date:
            continue
        if end_date and trade_date > end_date:
            continue
        yield path


def side_from_offset(offset: str) -> str | None:
    if offset == "48":
        return "BUY"
    if offset == "49":
        return "SELL"
    return None


def market_suffix(market: int) -> str:
    return ".SH" if market == 0 else ".SZ"


def parse_log_file(path: Path) -> list[Fill]:
    match = LOG_NAME_RE.match(path.name)
    if not match:
        return []
    trade_date = match.group(1)
    fills: list[Fill] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if "push dealdetail" not in raw_line:
                continue
            line = raw_line.strip()
            result = DEAL_RE.search(line)
            if not result:
                continue
            side = side_from_offset(result.group("offset"))
            if side is None:
                continue
            stock = result.group("stock")
            market = int(result.group("market"))
            stock_code = f"{stock}{market_suffix(market)}"
            fills.append(
                Fill(
                    trade_date=trade_date,
                    timestamp=line[:23],
                    stock_code=stock_code,
                    side=side,
                    sysid=result.group("sysid"),
                    bizno=result.group("bizno").strip(),
                    volume=int(result.group("volume")),
                    amount=float(result.group("amount")),
                    market=market,
                )
            )
    return fills


def load_open_prices(db_path: Path, stock_dates: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
    if not stock_dates:
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "select stock_code, trade_date, open from stock_daily_data "
            "where trade_date between ? and ?",
            (
                min(trade_date for _, trade_date in stock_dates),
                max(trade_date for _, trade_date in stock_dates),
            ),
        ).fetchall()
    finally:
        conn.close()
    prices: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["stock_code"], row["trade_date"])
        if key not in stock_dates:
            continue
        open_price = row["open"]
        if open_price is None:
            continue
        try:
            prices[key] = float(open_price)
        except (TypeError, ValueError):
            continue
    return prices


def load_today_open_from_xtdata(stock_codes: set[str]) -> dict[str, float]:
    try:
        from xtquant import xtdata
    except Exception:
        return {}

    if not stock_codes:
        return {}
    try:
        full_tick = xtdata.get_full_tick(sorted(stock_codes))
    except Exception:
        return {}

    result: dict[str, float] = {}
    for stock_code in stock_codes:
        tick = full_tick.get(stock_code) or {}
        open_price = tick.get("open")
        if open_price in (None, "", 0):
            continue
        result[stock_code] = float(open_price)
    return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def aggregate_orders(fills: list[Fill], open_prices: dict[tuple[str, str], float]) -> list[OrderAgg]:
    grouped: dict[tuple[str, str, str, str], list[Fill]] = defaultdict(list)
    for fill in fills:
        grouped[(fill.trade_date, fill.stock_code, fill.side, fill.sysid)].append(fill)

    orders: list[OrderAgg] = []
    for (trade_date, stock_code, side, sysid), group in sorted(grouped.items()):
        total_volume = sum(item.volume for item in group)
        total_amount = sum(item.amount for item in group)
        avg_fill_price = total_amount / total_volume if total_volume else 0.0
        open_price = open_prices.get((stock_code, trade_date))
        slippage_ratio = None
        slippage_amount = None
        if open_price and open_price > 0:
            if side == "BUY":
                slippage_ratio = avg_fill_price / open_price - 1.0
            else:
                slippage_ratio = open_price / avg_fill_price - 1.0
            slippage_amount = abs(avg_fill_price - open_price) * total_volume
        orders.append(
            OrderAgg(
                trade_date=trade_date,
                stock_code=stock_code,
                side=side,
                sysid=sysid,
                fill_count=len(group),
                total_volume=total_volume,
                total_amount=total_amount,
                avg_fill_price=avg_fill_price,
                open_price=open_price,
                slippage_ratio=slippage_ratio,
                slippage_amount=slippage_amount,
                first_fill_time=min(item.timestamp for item in group),
                last_fill_time=max(item.timestamp for item in group),
            )
        )
    return orders


def to_percent(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def write_detail_csv(path: Path, orders: list[OrderAgg]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trade_date",
                "stock_code",
                "side",
                "sysid",
                "fill_count",
                "total_volume",
                "total_amount",
                "avg_fill_price",
                "open_price",
                "slippage_ratio",
                "slippage_pct",
                "slippage_amount",
                "first_fill_time",
                "last_fill_time",
            ],
        )
        writer.writeheader()
        for item in orders:
            writer.writerow(
                {
                    "trade_date": item.trade_date,
                    "stock_code": item.stock_code,
                    "side": item.side,
                    "sysid": item.sysid,
                    "fill_count": item.fill_count,
                    "total_volume": item.total_volume,
                    "total_amount": round(item.total_amount, 6),
                    "avg_fill_price": round(item.avg_fill_price, 6),
                    "open_price": None if item.open_price is None else round(item.open_price, 6),
                    "slippage_ratio": item.slippage_ratio,
                    "slippage_pct": to_percent(item.slippage_ratio),
                    "slippage_amount": item.slippage_amount,
                    "first_fill_time": item.first_fill_time,
                    "last_fill_time": item.last_fill_time,
                }
            )


def write_summary_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def order_key(item: OrderAgg) -> tuple[str, str, str, str]:
    return (item.trade_date, item.stock_code, item.side, item.sysid)


def write_ledger(ledger_dir: Path, orders: list[OrderAgg]) -> dict[str, str]:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    detail_path = ledger_dir / "order_slippage_ledger.csv"
    summary_path = ledger_dir / "daily_slippage_summary.jsonl"

    existing: dict[tuple[str, str, str, str], dict[str, object]] = {}
    if detail_path.exists():
        with detail_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                existing[(row["trade_date"], row["stock_code"], row["side"], row["sysid"])] = row

    for item in orders:
        existing[order_key(item)] = {
            "trade_date": item.trade_date,
            "stock_code": item.stock_code,
            "side": item.side,
            "sysid": item.sysid,
            "fill_count": item.fill_count,
            "total_volume": item.total_volume,
            "total_amount": round(item.total_amount, 6),
            "avg_fill_price": round(item.avg_fill_price, 6),
            "open_price": None if item.open_price is None else round(item.open_price, 6),
            "slippage_ratio": item.slippage_ratio,
            "slippage_pct": to_percent(item.slippage_ratio),
            "slippage_amount": item.slippage_amount,
            "first_fill_time": item.first_fill_time,
            "last_fill_time": item.last_fill_time,
        }

    ordered_rows = [existing[key] for key in sorted(existing.keys())]
    with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trade_date",
                "stock_code",
                "side",
                "sysid",
                "fill_count",
                "total_volume",
                "total_amount",
                "avg_fill_price",
                "open_price",
                "slippage_ratio",
                "slippage_pct",
                "slippage_amount",
                "first_fill_time",
                "last_fill_time",
            ],
        )
        writer.writeheader()
        writer.writerows(ordered_rows)

    daily_existing: dict[str, dict[str, object]] = {}
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            daily_existing[str(payload["trade_date"])] = payload

    by_date: dict[str, list[OrderAgg]] = defaultdict(list)
    for item in orders:
        by_date[item.trade_date].append(item)

    for trade_date, items in by_date.items():
        buy_values = [x.slippage_ratio for x in items if x.side == "BUY" and x.slippage_ratio is not None]
        sell_values = [x.slippage_ratio for x in items if x.side == "SELL" and x.slippage_ratio is not None]
        all_values = [x.slippage_ratio for x in items if x.slippage_ratio is not None]
        daily_existing[trade_date] = {
            "trade_date": trade_date,
            "order_count": len(items),
            "buy_order_count": len([x for x in items if x.side == "BUY"]),
            "sell_order_count": len([x for x in items if x.side == "SELL"]),
            "buy_slippage": summarize(buy_values),
            "sell_slippage": summarize(sell_values),
            "all_slippage": summarize(all_values),
        }

    with summary_path.open("w", encoding="utf-8") as handle:
        for trade_date in sorted(daily_existing):
            handle.write(json.dumps(daily_existing[trade_date], ensure_ascii=False) + "\n")

    return {"detail_ledger": str(detail_path), "daily_summary_ledger": str(summary_path)}


def main() -> None:
    args = parse_args()
    files = list(iter_log_files(args.log_dir, args.start_date, args.end_date))
    fills: list[Fill] = []
    for path in files:
        fills.extend(parse_log_file(path))

    stock_dates = {(fill.stock_code, fill.trade_date) for fill in fills}
    open_prices = load_open_prices(args.db_path, stock_dates)

    if args.include_today_xtdata:
        missing_stock_dates = {
            (stock_code, trade_date)
            for stock_code, trade_date in stock_dates
            if (stock_code, trade_date) not in open_prices
        }
        by_date: dict[str, set[str]] = defaultdict(set)
        for stock_code, trade_date in missing_stock_dates:
            by_date[trade_date].add(stock_code)
        today = datetime.now().strftime("%Y%m%d")
        if today in by_date:
            today_prices = load_today_open_from_xtdata(by_date[today])
            for stock_code, open_price in today_prices.items():
                open_prices[(stock_code, today)] = open_price

    orders = aggregate_orders(fills, open_prices)
    covered_orders = [item for item in orders if item.slippage_ratio is not None]

    buy_values = [item.slippage_ratio for item in covered_orders if item.side == "BUY" and item.slippage_ratio is not None]
    sell_values = [item.slippage_ratio for item in covered_orders if item.side == "SELL" and item.slippage_ratio is not None]
    all_values = [item.slippage_ratio for item in covered_orders if item.slippage_ratio is not None]

    date_tag = (
        f"{args.start_date or (files[0].stem[-8:] if files else 'na')}"
        f"_{args.end_date or (files[-1].stem[-8:] if files else 'na')}"
    )
    output_dir = args.output_dir / f"qmt_slippage_{date_tag}"
    detail_path = output_dir / "order_slippage_detail.csv"
    summary_path = output_dir / "slippage_summary.json"

    write_detail_csv(detail_path, orders)
    summary_payload = {
        "log_dir": str(args.log_dir),
        "db_path": str(args.db_path),
        "files": [str(path) for path in files],
        "total_fill_count": len(fills),
        "total_order_count": len(orders),
        "covered_order_count": len(covered_orders),
        "missing_open_order_count": len(orders) - len(covered_orders),
        "buy_slippage": summarize(buy_values),
        "sell_slippage": summarize(sell_values),
        "all_slippage": summarize(all_values),
        "detail_csv": str(detail_path),
    }
    if args.update_ledger:
        summary_payload["ledger"] = write_ledger(args.ledger_dir, orders)
    write_summary_json(summary_path, summary_payload)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
