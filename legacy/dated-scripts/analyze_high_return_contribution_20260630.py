from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"

CASES = {
    "hef_top3_deep_skip_drop_scale": {
        "log_file": REPORT_DIR / "high_exposure_dayrisk_filter_logs" / "hef_top3_deep_skip_drop_scale.log",
        "signal_file": REPORT_DIR / "high_exposure_dayrisk_filter_signals" / "hef_top3_deep_skip_drop_scale.csv",
    },
    "thex_t55_top3_s1000_cap99_h2m3_ddoff": {
        "log_file": REPORT_DIR / "timing_high_exposure_extension_logs" / "thex_t55_top3_s1000_cap99_h2m3_ddoff.log",
        "signal_file": REPORT_DIR / "timing_high_exposure_extension_signals" / "thex_t55_top3_s1000_cap99_h2m3_ddoff.csv",
    },
}

OUT_TRADE_CSV = REPORT_DIR / "high_return_contribution_trades_20260630.csv"
OUT_STOCK_CSV = REPORT_DIR / "high_return_contribution_by_stock_20260630.csv"
OUT_MONTH_CSV = REPORT_DIR / "high_return_contribution_by_month_20260630.csv"
OUT_DAY_CSV = REPORT_DIR / "high_return_contribution_by_day_20260630.csv"
OUT_JSON = REPORT_DIR / "high_return_contribution_summary_20260630.json"
OUT_MD = REPORT_DIR / "high_return_contribution_summary_20260630.md"

BUY_RE = re.compile(
    r"BUY_ATTEMPT\s+(?P<date>\d{8})\s+(?P<symbol>\S+)\s+volume=(?P<volume>[-0-9.]+)\s+price=(?P<price>[-0-9.eE]+)"
)
SELL_RE = re.compile(
    r"SELL_ATTEMPT\s+(?P<date>\d{8})\s+(?P<symbol>\S+)\s+volume=(?P<volume>[-0-9.]+)\s+price=(?P<price>[-0-9.eE]+)"
)
EXPOSURE_RE = re.compile(
    r"EXPOSURE\s+(?P<date>\d{8})\s+post_buy\s+invested_pct=(?P<invested>[-0-9.]+)\s+"
    r"active_positions=(?P<positions>\d+)\s+market_value=(?P<mv>[-0-9.]+)\s+nav=(?P<nav>[-0-9.]+)\s+cash=(?P<cash>[-0-9.]+)"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _name_map(signal_file: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            symbol = str(row.get("symbol") or "")
            name = str(row.get("name") or "")
            if symbol and name and symbol not in out:
                out[symbol] = name
    return out


def _parse_log(case_name: str, log_file: Path, names: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trades: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        buy = BUY_RE.search(line)
        if buy:
            volume = float(buy.group("volume"))
            price = float(buy.group("price"))
            if volume > 0 and price > 0:
                positions[buy.group("symbol")].append(
                    {
                        "date": buy.group("date"),
                        "volume": volume,
                        "price": price,
                    }
                )
            continue
        sell = SELL_RE.search(line)
        if sell:
            symbol = sell.group("symbol")
            sell_date = sell.group("date")
            sell_volume = float(sell.group("volume"))
            sell_price = float(sell.group("price"))
            remaining = sell_volume
            lots = positions.get(symbol, [])
            while remaining > 1e-9 and lots:
                lot = lots[0]
                matched = min(remaining, float(lot["volume"]))
                buy_price = float(lot["price"])
                pnl = (sell_price - buy_price) * matched
                cost = buy_price * matched
                trades.append(
                    {
                        "case": case_name,
                        "symbol": symbol,
                        "name": names.get(symbol, ""),
                        "buy_date": lot["date"],
                        "sell_date": sell_date,
                        "month": sell_date[:6],
                        "volume": matched,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "cost": cost,
                        "pnl": pnl,
                        "return_pct": pnl / cost if cost else None,
                    }
                )
                lot["volume"] = float(lot["volume"]) - matched
                remaining -= matched
                if lot["volume"] <= 1e-9:
                    lots.pop(0)
            continue
        exposure = EXPOSURE_RE.search(line)
        if exposure:
            nav_rows.append(
                {
                    "case": case_name,
                    "date": exposure.group("date"),
                    "invested_pct": float(exposure.group("invested")),
                    "positions": int(exposure.group("positions")),
                    "market_value": float(exposure.group("mv")),
                    "nav": float(exposure.group("nav")),
                    "cash": float(exposure.group("cash")),
                }
            )
    previous_nav = None
    peak_nav = None
    peak_date = None
    for row in nav_rows:
        nav = float(row["nav"])
        if previous_nav is None:
            row["daily_return"] = None
        else:
            row["daily_return"] = nav / previous_nav - 1.0
        if peak_nav is None or nav > peak_nav:
            peak_nav = nav
            peak_date = str(row["date"])
        row["drawdown"] = nav / peak_nav - 1.0 if peak_nav else 0.0
        row["drawdown_peak_date"] = peak_date
        previous_nav = nav
    return trades, nav_rows


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(item, "") for item in keys)
        item = out.setdefault(key, {field: row.get(field, "") for field in keys})
        item["trade_count"] = int(item.get("trade_count", 0)) + 1
        item["gross_cost"] = float(item.get("gross_cost", 0.0)) + float(row.get("cost") or 0.0)
        item["pnl"] = float(item.get("pnl", 0.0)) + float(row.get("pnl") or 0.0)
        if row.get("pnl") is not None and float(row["pnl"]) > 0:
            item["positive_pnl"] = float(item.get("positive_pnl", 0.0)) + float(row["pnl"])
        if row.get("pnl") is not None and float(row["pnl"]) < 0:
            item["negative_pnl"] = float(item.get("negative_pnl", 0.0)) + float(row["pnl"])
    for item in out.values():
        cost = float(item.get("gross_cost") or 0.0)
        item["return_on_matched_cost"] = float(item.get("pnl") or 0.0) / cost if cost else None
    return list(out.values())


def _month_nav(case: str, nav_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nav_rows:
        grouped[str(row["date"])[:6]].append(row)
    rows: list[dict[str, Any]] = []
    for month, items in sorted(grouped.items()):
        first = items[0]
        last = items[-1]
        rows.append(
            {
                "case": case,
                "month": month,
                "start_nav": first["nav"],
                "end_nav": last["nav"],
                "month_return": float(last["nav"]) / float(first["nav"]) - 1.0 if first["nav"] else None,
                "min_drawdown_in_month": min(float(item["drawdown"]) for item in items),
                "days": len(items),
            }
        )
    return rows


def _summarize(case: str, trades: list[dict[str, Any]], nav_rows: list[dict[str, Any]], stock_rows: list[dict[str, Any]], month_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_realized = sum(float(row.get("pnl") or 0.0) for row in trades)
    total_positive = sum(float(row.get("pnl") or 0.0) for row in trades if float(row.get("pnl") or 0.0) > 0)
    sorted_stocks = sorted(stock_rows, key=lambda row: float(row.get("pnl") or 0.0), reverse=True)
    sorted_months = sorted(month_rows, key=lambda row: float(row.get("month_return") or -999.0), reverse=True)
    sorted_days = sorted(
        [row for row in nav_rows if row.get("daily_return") is not None],
        key=lambda row: float(row.get("daily_return") or 0.0),
        reverse=True,
    )
    worst_days = sorted(
        [row for row in nav_rows if row.get("daily_return") is not None],
        key=lambda row: float(row.get("daily_return") or 0.0),
    )
    max_dd_row = min(nav_rows, key=lambda row: float(row.get("drawdown") or 0.0)) if nav_rows else None
    top_stock = sorted_stocks[0] if sorted_stocks else {}
    top3_stock_pnl = sum(float(row.get("pnl") or 0.0) for row in sorted_stocks[:3])
    top_month = sorted_months[0] if sorted_months else {}
    return {
        "case": case,
        "realized_trade_count": len(trades),
        "total_realized_pnl": total_realized,
        "total_positive_pnl": total_positive,
        "top_stock": top_stock,
        "top_stock_net_pnl_share": float(top_stock.get("pnl") or 0.0) / total_realized if total_realized else None,
        "top_stock_positive_pnl_share": float(top_stock.get("positive_pnl") or 0.0) / total_positive if total_positive else None,
        "top3_stock_net_pnl_share": top3_stock_pnl / total_realized if total_realized else None,
        "top_month": top_month,
        "top_month_return": top_month.get("month_return"),
        "top_day": sorted_days[0] if sorted_days else {},
        "worst_day": worst_days[0] if worst_days else {},
        "max_drawdown": abs(float(max_dd_row.get("drawdown") or 0.0)) if max_dd_row else None,
        "max_drawdown_peak_date": max_dd_row.get("drawdown_peak_date") if max_dd_row else None,
        "max_drawdown_trough_date": max_dd_row.get("date") if max_dd_row else None,
        "nav_start": nav_rows[0]["nav"] if nav_rows else None,
        "nav_end": nav_rows[-1]["nav"] if nav_rows else None,
    }


def main() -> None:
    all_trades: list[dict[str, Any]] = []
    all_nav_rows: list[dict[str, Any]] = []
    all_stock_rows: list[dict[str, Any]] = []
    all_month_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case, cfg in CASES.items():
        names = _name_map(cfg["signal_file"])
        trades, nav_rows = _parse_log(case, cfg["log_file"], names)
        stock_rows = _aggregate(trades, ["case", "symbol", "name"])
        month_trade_rows = _aggregate(trades, ["case", "month"])
        month_nav_rows = _month_nav(case, nav_rows)
        month_nav_by_key = {(row["case"], row["month"]): row for row in month_nav_rows}
        for row in month_trade_rows:
            nav = month_nav_by_key.get((row["case"], row["month"]), {})
            row.update(
                {
                    "start_nav": nav.get("start_nav"),
                    "end_nav": nav.get("end_nav"),
                    "month_return": nav.get("month_return"),
                    "min_drawdown_in_month": nav.get("min_drawdown_in_month"),
                    "nav_days": nav.get("days"),
                }
            )
        summaries.append(_summarize(case, trades, nav_rows, stock_rows, month_nav_rows))
        all_trades.extend(trades)
        all_nav_rows.extend(nav_rows)
        all_stock_rows.extend(stock_rows)
        all_month_rows.extend(month_trade_rows)

    for rows in (all_stock_rows, all_month_rows):
        totals: dict[str, float] = defaultdict(float)
        positives: dict[str, float] = defaultdict(float)
        for row in rows:
            case = str(row["case"])
            pnl = float(row.get("pnl") or 0.0)
            totals[case] += pnl
            if pnl > 0:
                positives[case] += pnl
        for row in rows:
            case = str(row["case"])
            pnl = float(row.get("pnl") or 0.0)
            pos = float(row.get("positive_pnl") or 0.0)
            row["net_pnl_share"] = pnl / totals[case] if totals[case] else None
            row["positive_pnl_share"] = pos / positives[case] if positives[case] else None

    for row in all_nav_rows:
        row["month"] = str(row["date"])[:6]

    _write_csv(OUT_TRADE_CSV, sorted(all_trades, key=lambda row: (row["case"], row["sell_date"], row["symbol"])))
    _write_csv(OUT_STOCK_CSV, sorted(all_stock_rows, key=lambda row: (row["case"], -float(row.get("pnl") or 0.0))))
    _write_csv(OUT_MONTH_CSV, sorted(all_month_rows, key=lambda row: (row["case"], row["month"])))
    _write_csv(OUT_DAY_CSV, sorted(all_nav_rows, key=lambda row: (row["case"], row["date"])))
    OUT_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 高收益候选收益集中度拆解 20260630",
        "",
        "## 边界",
        "",
        "本报告为 L5/L6 research-only 分析。只解析既有掘金回测日志，不修改生产策略参数，不生成正式信号，不触发交易。",
        "",
        "成交 PnL 使用日志中的 `BUY_ATTEMPT` / `SELL_ATTEMPT` 做 FIFO 近似配对；组合回撤和每日收益使用日志中的 `EXPOSURE ... nav=...` 序列。",
        "",
        "## 结论摘要",
        "",
    ]
    for summary in summaries:
        top_stock = summary.get("top_stock") or {}
        top_month = summary.get("top_month") or {}
        top_day = summary.get("top_day") or {}
        worst_day = summary.get("worst_day") or {}
        lines.extend(
            [
                f"### {summary['case']}",
                "",
                f"- 近似已实现交易数：`{summary['realized_trade_count']}`",
                f"- 近似已实现净 PnL：`{summary['total_realized_pnl']:.2f}`",
                f"- 最大回撤区间：`{summary['max_drawdown_peak_date']}` 到 `{summary['max_drawdown_trough_date']}`，回撤约 `{summary['max_drawdown']:.2%}`",
                f"- 最大贡献股票：`{top_stock.get('symbol')}` `{top_stock.get('name')}`，净 PnL `{float(top_stock.get('pnl') or 0.0):.2f}`，占净 PnL `{summary['top_stock_net_pnl_share']:.2%}`",
                f"- 前三大贡献股票合计占净 PnL：`{summary['top3_stock_net_pnl_share']:.2%}`",
                f"- 最强月份：`{top_month.get('month')}`，组合月收益 `{float(top_month.get('month_return') or 0.0):.2%}`",
                f"- 最强单日：`{top_day.get('date')}`，组合日收益 `{float(top_day.get('daily_return') or 0.0):.2%}`",
                f"- 最差单日：`{worst_day.get('date')}`，组合日收益 `{float(worst_day.get('daily_return') or 0.0):.2%}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 判断",
            "",
            "如果最大贡献股票或前三大贡献股票占净 PnL 比例过高，说明 500%+ 年化结果存在明显贡献集中风险，不能直接作为生产准入稳健性证据。",
            "",
            "本拆解不替代完整剔除回测。下一步若继续推进，应生成剔除最大贡献股票、剔除最大贡献月份、剔除最大贡献单日后的信号文件，并用掘金重新回测。",
            "",
            "## 证据路径",
            "",
            f"- `{OUT_TRADE_CSV}`",
            f"- `{OUT_STOCK_CSV}`",
            f"- `{OUT_MONTH_CSV}`",
            f"- `{OUT_DAY_CSV}`",
            f"- `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "summary_file": str(OUT_MD), "cases": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
