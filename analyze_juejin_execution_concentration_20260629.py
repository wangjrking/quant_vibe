from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
DEFAULT_ARTIFACT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_adaptive70w_fouryear_momentum_cooldown_20260629"
    / "trade_artifacts_pos50_cool2d20"
)

ARTIFACT_DIR = Path(os.environ.get("STRATEGY_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))
OUT_DIR = Path(os.environ.get("STRATEGY_EXEC_CONCENTRATION_OUT_DIR", str(ARTIFACT_DIR / "execution_concentration")))
INITIAL_CASH = float(os.environ.get("STRATEGY_INITIAL_CASH", "600000"))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_execution_reports(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            report = payload.get("report") or {}
            rows.append(report)
    rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("exec_id") or ""), str(row.get("order_id") or "")))
    return rows


def _to_float(value: Any) -> float:
    if value in (None, "", "None"):
        return 0.0
    return float(value)


def _to_int(value: Any) -> int:
    if value in (None, "", "None"):
        return 0
    return int(value)


def _dt_text(value: Any) -> str:
    return str(value or "")


def _month_of(dt_text: str) -> str:
    if not dt_text:
        return ""
    return dt_text[:7].replace("-", "")


def _round(value: float) -> float:
    return round(float(value), 10)


def _consume_sell(lots_by_symbol: dict[str, list[dict[str, Any]]], report: dict[str, Any], closed_rows: list[dict[str, Any]], anomalies: list[dict[str, Any]]) -> None:
    symbol = str(report.get("symbol") or "")
    remaining_volume = _to_int(report.get("volume"))
    amount = _to_float(report.get("amount"))
    commission = _to_float(report.get("commission"))
    created_at = _dt_text(report.get("created_at"))
    lots = lots_by_symbol.get(symbol, [])
    total_volume = sum(int(lot["remaining_volume"]) for lot in lots)
    if remaining_volume <= 0:
        return
    if total_volume < remaining_volume:
        anomalies.append(
            {
                "type": "sell_without_enough_inventory",
                "symbol": symbol,
                "sell_created_at": created_at,
                "sell_volume": remaining_volume,
                "available_volume": total_volume,
            }
        )
    sell_price = amount / remaining_volume if remaining_volume else 0.0
    sell_commission_per_share = commission / remaining_volume if remaining_volume else 0.0
    remaining = remaining_volume
    while remaining > 0 and lots:
        lot = lots[0]
        take = min(int(lot["remaining_volume"]), remaining)
        if take <= 0:
            lots.pop(0)
            continue
        buy_cost_per_share = float(lot["remaining_cost"]) / float(lot["remaining_volume"]) if int(lot["remaining_volume"]) else 0.0
        sell_net = take * sell_price - take * sell_commission_per_share
        buy_cost = take * buy_cost_per_share
        realized_pnl = sell_net - buy_cost
        lot["remaining_volume"] = int(lot["remaining_volume"]) - take
        lot["remaining_cost"] = float(lot["remaining_cost"]) - buy_cost
        lot["realized_pnl"] = float(lot["realized_pnl"]) + realized_pnl
        lot["sell_amount"] = float(lot["sell_amount"]) + take * sell_price
        lot["sell_commission"] = float(lot["sell_commission"]) + take * sell_commission_per_share
        lot["close_created_at"] = created_at
        lot["close_month"] = _month_of(created_at)
        closed_rows.append(
            {
                "lot_id": lot["lot_id"],
                "symbol": symbol,
                "buy_created_at": lot["buy_created_at"],
                "sell_created_at": created_at,
                "matched_volume": take,
                "buy_cost": _round(buy_cost),
                "sell_net": _round(sell_net),
                "realized_pnl": _round(realized_pnl),
            }
        )
        remaining -= take
        if int(lot["remaining_volume"]) <= 0:
            lots.pop(0)


def _build_lots(execution_reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lots_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_lots: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    lot_id = 0
    for report in execution_reports:
        side = _to_int(report.get("side"))
        symbol = str(report.get("symbol") or "")
        volume = _to_int(report.get("volume"))
        amount = _to_float(report.get("amount"))
        commission = _to_float(report.get("commission"))
        created_at = _dt_text(report.get("created_at"))
        if volume <= 0 or amount <= 0 or not symbol:
            continue
        if side == 1:
            lot_id += 1
            lot = {
                "lot_id": lot_id,
                "symbol": symbol,
                "buy_created_at": created_at,
                "buy_month": _month_of(created_at),
                "buy_volume": volume,
                "buy_amount": amount,
                "buy_commission": commission,
                "remaining_volume": volume,
                "remaining_cost": amount + commission,
                "realized_pnl": 0.0,
                "sell_amount": 0.0,
                "sell_commission": 0.0,
                "close_created_at": "",
                "close_month": "",
            }
            lots_by_symbol[symbol].append(lot)
            all_lots.append(lot)
        elif side == 2:
            _consume_sell(lots_by_symbol, report, closed_rows, anomalies)
    return all_lots, closed_rows, anomalies


def _mark_open_positions(lots: list[dict[str, Any]], positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions_by_symbol = {str(row.get("symbol") or ""): row for row in positions}
    open_rows: list[dict[str, Any]] = []
    for lot in lots:
        remaining_volume = int(lot["remaining_volume"])
        if remaining_volume <= 0:
            lot["floating_pnl"] = 0.0
            lot["market_value"] = 0.0
            lot["closed"] = True
            continue
        position = positions_by_symbol.get(str(lot["symbol"]))
        if not position:
            lot["floating_pnl"] = 0.0
            lot["market_value"] = 0.0
            lot["closed"] = False
            lot["mark_warning"] = "missing_final_position"
            continue
        total_position_volume = _to_int(position.get("volume"))
        total_position_market_value = _to_float(position.get("market_value"))
        if total_position_volume <= 0:
            lot["floating_pnl"] = 0.0
            lot["market_value"] = 0.0
            lot["closed"] = False
            lot["mark_warning"] = "invalid_final_position_volume"
            continue
        alloc_market_value = total_position_market_value * remaining_volume / total_position_volume
        floating_pnl = alloc_market_value - float(lot["remaining_cost"])
        lot["market_value"] = alloc_market_value
        lot["floating_pnl"] = floating_pnl
        lot["closed"] = False
        open_rows.append(
            {
                "lot_id": lot["lot_id"],
                "symbol": lot["symbol"],
                "buy_created_at": lot["buy_created_at"],
                "remaining_volume": remaining_volume,
                "remaining_cost": _round(lot["remaining_cost"]),
                "market_value": _round(alloc_market_value),
                "floating_pnl": _round(floating_pnl),
            }
        )
    return open_rows


def _trade_rows(lots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lot in lots:
        remaining_cost = float(lot["remaining_cost"])
        buy_cost_total = float(lot["buy_amount"]) + float(lot["buy_commission"])
        realized_pnl = float(lot.get("realized_pnl") or 0.0)
        floating_pnl = float(lot.get("floating_pnl") or 0.0)
        total_pnl = realized_pnl + floating_pnl
        rows.append(
            {
                "lot_id": lot["lot_id"],
                "symbol": lot["symbol"],
                "buy_created_at": lot["buy_created_at"],
                "buy_date": _dt_text(lot["buy_created_at"])[:10],
                "buy_month": lot["buy_month"],
                "close_created_at": lot.get("close_created_at") or "",
                "close_month": lot.get("close_month") or "",
                "buy_volume": int(lot["buy_volume"]),
                "remaining_volume": int(lot["remaining_volume"]),
                "buy_amount": _round(float(lot["buy_amount"])),
                "buy_commission": _round(float(lot["buy_commission"])),
                "buy_cost_total": _round(buy_cost_total),
                "sell_amount": _round(float(lot["sell_amount"])),
                "sell_commission": _round(float(lot["sell_commission"])),
                "remaining_cost": _round(remaining_cost),
                "market_value": _round(float(lot.get("market_value") or 0.0)),
                "realized_pnl": _round(realized_pnl),
                "floating_pnl": _round(floating_pnl),
                "total_pnl": _round(total_pnl),
                "closed": bool(lot.get("closed")),
                "mark_warning": lot.get("mark_warning") or "",
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        item = grouped.setdefault(value, {key: value, "count": 0, "total_pnl": 0.0})
        item["count"] += 1
        item["total_pnl"] += float(row["total_pnl"])
    for item in grouped.values():
        item["avg_pnl"] = item["total_pnl"] / item["count"] if item["count"] else None
    return sorted(grouped.values(), key=lambda item: item["total_pnl"], reverse=True)


def _summary(trade_rows: list[dict[str, Any]], artifacts: dict[str, Any], anomalies: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(row["total_pnl"]) for row in trade_rows]
    total_pnl = sum(pnls)
    positive = [value for value in pnls if value > 0]
    ranked = sorted(trade_rows, key=lambda row: float(row["total_pnl"]), reverse=True)
    top_1pct_n = max(1, int(len(ranked) * 0.01)) if ranked else 0
    top_5pct_n = max(1, int(len(ranked) * 0.05)) if ranked else 0
    top_1pct = sum(float(row["total_pnl"]) for row in ranked[:top_1pct_n])
    top_5pct = sum(float(row["total_pnl"]) for row in ranked[:top_5pct_n])
    by_stock = _aggregate(trade_rows, "symbol")
    by_month = _aggregate(trade_rows, "buy_month")
    by_day = _aggregate(trade_rows, "buy_date")
    indicator = artifacts.get("indicator") or {}
    official_total_pnl = _to_float(indicator.get("pnl_ratio")) * INITIAL_CASH
    closed_count = sum(1 for row in trade_rows if bool(row["closed"]))
    open_count = sum(1 for row in trade_rows if not bool(row["closed"]))
    top_stock_total = float(by_stock[0]["total_pnl"]) if by_stock else 0.0
    top_month_total = float(by_month[0]["total_pnl"]) if by_month else 0.0
    top_day_total = float(by_day[0]["total_pnl"]) if by_day else 0.0
    return {
        "artifact_dir": str(ARTIFACT_DIR),
        "trade_rows": len(trade_rows),
        "closed_lots": closed_count,
        "open_lots": open_count,
        "execution_total_pnl": _round(total_pnl),
        "official_total_pnl_from_indicator": _round(official_total_pnl),
        "indicator_pnl_gap": _round(official_total_pnl - total_pnl),
        "top_1pct_n": top_1pct_n,
        "top_1pct_total_pnl": _round(top_1pct),
        "top_1pct_share_of_positive_total": _round(top_1pct / sum(positive)) if positive else None,
        "top_5pct_n": top_5pct_n,
        "top_5pct_total_pnl": _round(top_5pct),
        "top_5pct_share_of_positive_total": _round(top_5pct / sum(positive)) if positive else None,
        "drop_top_1pct_total_pnl": _round(sum(float(row["total_pnl"]) for row in ranked[top_1pct_n:])),
        "drop_top_5pct_total_pnl": _round(sum(float(row["total_pnl"]) for row in ranked[top_5pct_n:])),
        "bottom_1pct_total_pnl": _round(sum(float(row["total_pnl"]) for row in sorted(trade_rows, key=lambda row: float(row["total_pnl"]))[:top_1pct_n])) if ranked else 0.0,
        "top_stock": by_stock[0] if by_stock else None,
        "top_month": by_month[0] if by_month else None,
        "top_day": by_day[0] if by_day else None,
        "drop_top_stock_total_pnl": _round(total_pnl - top_stock_total),
        "drop_top_month_total_pnl": _round(total_pnl - top_month_total),
        "drop_top_day_total_pnl": _round(total_pnl - top_day_total),
        "anomaly_count": len(anomalies),
        "anomaly_examples": anomalies[:20],
    }


def main() -> None:
    artifacts = _load_json(ARTIFACT_DIR / "backtest_trade_artifacts.json")
    execution_reports = _load_execution_reports(ARTIFACT_DIR / "execution_reports.jsonl")
    lots, closed_rows, anomalies = _build_lots(execution_reports)
    open_rows = _mark_open_positions(lots, artifacts.get("positions") or [])
    trades = _trade_rows(lots)
    by_stock = _aggregate(trades, "symbol")
    by_month = _aggregate(trades, "buy_month")
    by_day = _aggregate(trades, "buy_date")
    summary = _summary(trades, artifacts, anomalies)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_rows(OUT_DIR / "execution_trade_rows.csv", trades)
    _write_rows(OUT_DIR / "execution_closed_matches.csv", closed_rows)
    _write_rows(OUT_DIR / "execution_open_positions.csv", open_rows)
    _write_rows(OUT_DIR / "execution_by_stock.csv", by_stock)
    _write_rows(OUT_DIR / "execution_by_buy_month.csv", by_month)
    _write_rows(OUT_DIR / "execution_by_buy_date.csv", by_day)
    _write_rows(OUT_DIR / "execution_anomalies.csv", anomalies)
    (OUT_DIR / "execution_concentration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
