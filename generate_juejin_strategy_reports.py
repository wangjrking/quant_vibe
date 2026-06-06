from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from backtest_module import (
    _annualized_return,
    _annualized_volatility,
    _calmar_ratio,
    _sharpe_ratio,
    _to_float,
    calculate_net_return,
)


@dataclass
class SuiteConfig:
    start_date: str
    end_date: str
    initial_cash: float
    commission_ratio: float
    sell_tax_ratio: float
    slippage_ratio: float
    stock_pool: str


def load_config(path: str | Path) -> tuple[SuiteConfig, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    backtest = payload["juejin"]["backtest"]
    suite = SuiteConfig(
        start_date=str(backtest["start_date"]),
        end_date=str(backtest["end_date"]),
        initial_cash=float(backtest["initial_cash"]),
        commission_ratio=float(backtest["commission_ratio"]),
        sell_tax_ratio=float(backtest["sell_tax_ratio"]),
        slippage_ratio=float(backtest["slippage_ratio"]),
        stock_pool=str(backtest["stock_pool"]),
    )
    return suite, payload["juejin"]["strategies"]


def load_trade_calendar(db_path: Path, start_date: str, end_date: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT trade_date
            FROM daily_data
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
            """,
            (start_date, end_date),
        ).fetchall()
    return [row[0] for row in rows]


def load_open_prices(db_path: Path, start_date: str, end_date: str) -> dict[tuple[str, str], float]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT stock_code, trade_date, open
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= ? AND trade_date <= ?
            """,
            (start_date, end_date),
        ).fetchall()
    return {
        (str(stock_code), str(trade_date)): float(open_price)
        for stock_code, trade_date, open_price in rows
        if open_price is not None
    }


def load_signal_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["buy_date"] = str(row.get("buy_date") or "")
        row["stock_code"] = str(row.get("stock_code") or "")
        row["target_pct"] = _to_float(row.get("target_pct"))
    rows.sort(key=lambda row: (row["buy_date"], int(float(row.get("rank") or 999999))))
    return rows


def run_signal_backtest(
    signals: list[dict],
    suite: SuiteConfig,
    holding_days: int,
    max_positions: int,
    default_target_pct: float,
    db_path: Path,
) -> dict:
    trade_dates = load_trade_calendar(db_path, suite.start_date, suite.end_date)
    date_index = {date: idx for idx, date in enumerate(trade_dates)}
    open_prices = load_open_prices(db_path, suite.start_date, suite.end_date)
    grouped: dict[str, list[dict]] = {}
    for signal in signals:
        buy_date = signal["buy_date"]
        if buy_date and suite.start_date <= buy_date <= suite.end_date:
            grouped.setdefault(buy_date, []).append(signal)

    cash = 1.0
    positions: list[dict] = []
    trades: list[dict] = []
    equity_curve: list[dict] = []
    total_signals = 0
    executed_signals = 0
    missing_buy_price = 0
    missing_sell_price = 0

    for trade_date in trade_dates:
        idx = date_index[trade_date]

        remaining = []
        for pos in positions:
            if pos["exit_index"] <= idx:
                exit_value = pos["capital"] * (1.0 + pos["net_return"])
                cash += exit_value
                trade = dict(pos)
                trade["exit_value"] = exit_value
                trade["exit_trade_date"] = trade_date
                trades.append(trade)
            else:
                remaining.append(pos)
        positions = remaining

        signals_today = grouped.get(trade_date, [])
        total_signals += len(signals_today)
        open_slots = max(max_positions - len(positions), 0)
        if open_slots > 0 and signals_today:
            starting_equity = cash + sum(pos["capital"] for pos in positions)
            held_codes = {pos["stock_code"] for pos in positions}
            for signal in signals_today:
                if open_slots <= 0:
                    break
                stock_code = signal["stock_code"]
                if stock_code in held_codes:
                    continue
                buy_price = open_prices.get((stock_code, trade_date))
                if buy_price is None or buy_price <= 0:
                    missing_buy_price += 1
                    continue
                exit_idx = min(idx + holding_days, len(trade_dates) - 1)
                exit_date = trade_dates[exit_idx]
                sell_price = open_prices.get((stock_code, exit_date))
                if sell_price is None or sell_price <= 0:
                    missing_sell_price += 1
                    continue
                target_pct = signal.get("target_pct")
                if target_pct is None or target_pct <= 0:
                    target_pct = default_target_pct
                capital = min(cash, starting_equity * target_pct)
                if capital <= 0:
                    continue
                net_return = calculate_net_return(
                    buy_price,
                    sell_price,
                    commission_rate=suite.commission_ratio,
                    sell_tax_rate=suite.sell_tax_ratio,
                    slippage_rate=suite.slippage_ratio,
                )
                if net_return is None:
                    continue
                positions.append(
                    {
                        "entry_trade_date": trade_date,
                        "exit_trade_date_scheduled": exit_date,
                        "exit_index": exit_idx,
                        "stock_code": stock_code,
                        "name": signal.get("name"),
                        "rank": signal.get("rank"),
                        "target_pct": target_pct,
                        "capital": capital,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "net_return": net_return,
                    }
                )
                held_codes.add(stock_code)
                cash -= capital
                open_slots -= 1
                executed_signals += 1

        equity = cash + sum(pos["capital"] for pos in positions)
        equity_curve.append(
            {
                "trade_date": trade_date,
                "equity": equity,
                "cash": cash,
                "position_count": len(positions),
            }
        )

    for pos in positions:
        exit_value = pos["capital"] * (1.0 + pos["net_return"])
        cash += exit_value
        trade = dict(pos)
        trade["exit_value"] = exit_value
        trade["exit_trade_date"] = pos["exit_trade_date_scheduled"]
        trades.append(trade)

    if equity_curve:
        equity_curve[-1]["equity"] = cash

    running_peak = 0.0
    max_drawdown = 0.0
    daily_returns = []
    previous = 1.0
    for item in equity_curve:
        equity = float(item["equity"])
        running_peak = max(running_peak, equity)
        drawdown = equity / running_peak - 1.0 if running_peak > 0 else 0.0
        item["drawdown"] = drawdown
        max_drawdown = min(max_drawdown, drawdown)
        day_return = equity / previous - 1.0 if previous > 0 else 0.0
        daily_returns.append(day_return)
        previous = equity

    annualized = _annualized_return(cash, len(equity_curve))
    metrics = {
        "stock_pool_target": suite.stock_pool,
        "trade_day_count": len(equity_curve),
        "signal_count": total_signals,
        "executed_signal_count": executed_signals,
        "buy_price_coverage": executed_signals / total_signals if total_signals else 0.0,
        "missing_buy_price_count": missing_buy_price,
        "missing_sell_price_count": missing_sell_price,
        "trade_count": len(trades),
        "final_equity": cash,
        "cumulative_return": cash - 1.0,
        "annualized_return": annualized,
        "annualized_volatility": _annualized_volatility(daily_returns),
        "sharpe": _sharpe_ratio(daily_returns),
        "calmar": _calmar_ratio(annualized, max_drawdown),
        "max_drawdown": max_drawdown,
        "win_rate": (sum(1 for trade in trades if trade["net_return"] > 0) / len(trades)) if trades else 0.0,
        "avg_trade_return": (sum(trade["net_return"] for trade in trades) / len(trades)) if trades else 0.0,
    }
    return {"metrics": metrics, "equity_curve": equity_curve, "trades": trades}


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def write_chart(equity_curve: list[dict], title: str, path: Path) -> None:
    if not equity_curve:
        return
    dates = [row["trade_date"] for row in equity_curve]
    equity = [row["equity"] for row in equity_curve]
    drawdown = [row.get("drawdown", 0.0) for row in equity_curve]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(dates, equity, color="#1f77b4", linewidth=1.5)
    axes[0].set_title(title)
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.3)
    axes[1].fill_between(dates, drawdown, 0, color="#d62728", alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(name: str, strategy: dict, suite: SuiteConfig, result: dict, path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        f"# {name} 回测报告",
        "",
        f"- 策略目录: `{strategy['path']}`",
        f"- 信号文件: `{strategy['signal_file']}`",
        f"- 目标股票池: `{suite.stock_pool}`",
        f"- 回测区间: `{suite.start_date}` 到 `{suite.end_date}`",
        f"- 持有天数: `{strategy['holding_days']}`",
        f"- 最大持仓数: `{strategy['max_positions']}`",
        "",
        "## 指标",
        "",
        f"- 信号数: `{metrics['signal_count']}`",
        f"- 成功执行信号数: `{metrics['executed_signal_count']}`",
        f"- 买入价覆盖率: `{metrics['buy_price_coverage']:.2%}`",
        f"- 交易笔数: `{metrics['trade_count']}`",
        f"- 累计收益: `{metrics['cumulative_return']:.2%}`",
        f"- 年化收益: `{metrics['annualized_return']:.2%}`",
        f"- 最大回撤: `{metrics['max_drawdown']:.2%}`",
        f"- 夏普: `{metrics['sharpe']:.4f}`",
        f"- 卡玛: `{metrics['calmar']:.4f}`",
        f"- 胜率: `{metrics['win_rate']:.2%}`",
        f"- 平均单笔收益: `{metrics['avg_trade_return']:.2%}`",
        "",
        "## 说明",
        "",
        "- 这是基于信号文件和本地行情开盘价的近似回测，不等同于掘金服务器的官方撮合结果。",
        "- 当前配置目标股票池为全A，但实际可回测覆盖取决于本地 `STOCK_DAILY_DATA` 覆盖范围和信号文件来源。",
        "- 历史回测不代表未来一定赚钱。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def strategy_title(key: str) -> str:
    return {
        "limit_board": "打板策略",
        "long_term": "长线策略",
        "short_term": "短线策略",
    }.get(key, key)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate 2-year juejin strategy backtest reports.")
    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--data-dir", default="../data_file")
    args = parser.parse_args(argv)

    suite, strategies = load_config(args.config)
    db_path = Path(args.data_dir) / "odb.db"
    summary_rows = []

    for key, strategy in strategies.items():
        signals = load_signal_rows(strategy["signal_file"])
        result = run_signal_backtest(
            signals=signals,
            suite=suite,
            holding_days=int(strategy["holding_days"]),
            max_positions=int(strategy["max_positions"]),
            default_target_pct=float(strategy["default_target_pct"]),
            db_path=db_path,
        )
        strategy_path = Path(strategy["path"])
        write_metrics(result["metrics"], strategy_path / "backtest_metrics.json")
        write_csv(result["equity_curve"], strategy_path / "backtest_equity.csv")
        write_csv(result["trades"], strategy_path / "backtest_trades.csv")
        write_chart(result["equity_curve"], f"{strategy_title(key)} 近2年回测", strategy_path / "backtest_equity.png")
        write_markdown_report(strategy_title(key), strategy, suite, result, strategy_path / "backtest_report.md")
        summary_rows.append(
            {
                "strategy_key": key,
                "strategy_name": strategy_title(key),
                **result["metrics"],
            }
        )

    summary_rows.sort(key=lambda row: row["annualized_return"], reverse=True)
    summary_path = Path(args.data_dir) / "reports" / "juejin_strategy_suite_summary.csv"
    summary_md = Path(args.data_dir) / "reports" / "juejin_strategy_suite_summary.md"
    write_csv(summary_rows, summary_path)
    lines = [
        "# 掘金三策略近2年回测汇总",
        "",
        f"- 目标股票池: `{suite.stock_pool}`",
        f"- 回测区间: `{suite.start_date}` 到 `{suite.end_date}`",
        "",
        "| 策略 | 年化 | 累计收益 | 最大回撤 | 夏普 | 胜率 | 执行信号 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['strategy_name']} | {row['annualized_return']:.2%} | {row['cumulative_return']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['sharpe']:.4f} | {row['win_rate']:.2%} | {row['executed_signal_count']} |"
        )
    lines.extend(
        [
            "",
            "说明:",
            "- 汇总基于各策略目录里的 `backtest_report.md / backtest_metrics.json / backtest_equity.csv / backtest_trades.csv`。",
            "- 历史回测不代表未来一定赚钱。",
        ]
    )
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary_csv: {summary_path}")
    print(f"summary_md: {summary_md}")


if __name__ == "__main__":
    main()
