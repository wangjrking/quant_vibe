"""Portfolio-level backtest for executable daily stock signals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from backtest_module import (
    BacktestConfig,
    _annualized_return,
    _annualized_volatility,
    _calmar_ratio,
    _passes_filters,
    _sharpe_ratio,
    _to_float,
    _value,
    calculate_net_return,
    read_prediction_rows,
    resolve_slippage_rate,
)


@dataclass
class PortfolioBacktestConfig:
    top_k: int = 3
    max_positions: int = 10
    holding_days: int = 10
    start_date: str | None = None
    end_date: str | None = None
    pred_col: str = "pred_prob"
    buy_col: str = "post_open"
    sell_col: str = "post12_open"
    close_col: str = "close"
    min_pred_prob: float | None = 0.01
    max_atr_ratio: float | None = 0.10
    commission_rate: float = 0.0003
    sell_tax_rate: float = 0.0005
    slippage_rate: float = 0.001
    liquidity_slippage_enabled: bool = False
    liquidity_amount_col: str = "amount"
    liquidity_turnover_col: str = "turnover_rate"
    liquidity_slippage_amount_steps: tuple[tuple[float, float], ...] = (
        (3.0e5, 0.0005),
        (1.0e5, 0.0010),
        (5.0e4, 0.0020),
    )
    liquidity_slippage_turnover_steps: tuple[tuple[float, float], ...] = (
        (1.0, 0.0005),
        (0.5, 0.0010),
    )
    liquidity_slippage_cap: float = 0.004
    skip_limit_up_open: bool = True
    exclude_st: bool = True
    exclude_current_limit: bool = True
    market_filter_col: str | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    position_weight_mode: str = "equal"


def _date_allowed(date: str, config: PortfolioBacktestConfig) -> bool:
    if config.start_date and date < config.start_date:
        return False
    if config.end_date and date > config.end_date:
        return False
    return True


def _candidate_config(config: PortfolioBacktestConfig) -> BacktestConfig:
    return BacktestConfig(
        top_k=config.top_k,
        pred_col=config.pred_col,
        buy_col=config.buy_col,
        sell_col=config.sell_col,
        close_col=config.close_col,
        min_pred_prob=config.min_pred_prob,
        max_atr_ratio=config.max_atr_ratio,
        commission_rate=config.commission_rate,
        sell_tax_rate=config.sell_tax_rate,
        slippage_rate=config.slippage_rate,
        liquidity_slippage_enabled=config.liquidity_slippage_enabled,
        liquidity_amount_col=config.liquidity_amount_col,
        liquidity_turnover_col=config.liquidity_turnover_col,
        liquidity_slippage_amount_steps=config.liquidity_slippage_amount_steps,
        liquidity_slippage_turnover_steps=config.liquidity_slippage_turnover_steps,
        liquidity_slippage_cap=config.liquidity_slippage_cap,
        skip_limit_up_open=config.skip_limit_up_open,
        exclude_st=config.exclude_st,
        exclude_current_limit=config.exclude_current_limit,
    )


def _is_market_allowed(row, config: PortfolioBacktestConfig) -> bool:
    if not config.market_filter_col:
        return True
    value = _value(row, config.market_filter_col)
    if value in (None, "", "None"):
        return False
    try:
        return bool(float(value))
    except (TypeError, ValueError):
        return bool(value)


def _group_rows_by_date(rows, config: PortfolioBacktestConfig):
    grouped = {}
    for row in rows:
        date = str(_value(row, "trade_date", ""))
        if not date or not _date_allowed(date, config):
            continue
        grouped.setdefault(date, []).append(row)
    return grouped


def _apply_exit_rules(net_return: float, config: PortfolioBacktestConfig) -> float:
    adjusted = net_return
    if config.stop_loss_pct is not None:
        adjusted = max(adjusted, -abs(config.stop_loss_pct))
    if config.take_profit_pct is not None:
        adjusted = min(adjusted, abs(config.take_profit_pct))
    return adjusted


def _selection_weights(selected: list[tuple], mode: str) -> list[float]:
    if not selected:
        return []
    normalized_mode = str(mode or "equal").lower()
    if normalized_mode == "equal":
        return [1.0] * len(selected)
    if normalized_mode == "rank":
        return [float(len(selected) - idx) for idx in range(len(selected))]
    if normalized_mode == "score":
        preds = [_to_float(_value(item[0], "pred_prob")) or 0.0 for item in selected]
        floor = min(preds)
        weights = [max(pred - floor, 0.0) + 1e-6 for pred in preds]
        return weights
    raise ValueError(f"unsupported position_weight_mode: {mode}")


def run_portfolio_backtest(rows, config: PortfolioBacktestConfig | None = None) -> dict:
    config = config or PortfolioBacktestConfig()
    rows_by_date = _group_rows_by_date(rows, config)
    trade_dates = sorted(rows_by_date)
    date_index = {date: idx for idx, date in enumerate(trade_dates)}
    candidate_config = _candidate_config(config)

    cash = 1.0
    positions = []
    equity_curve = []
    trades = []
    peak = 1.0
    max_drawdown = 0.0

    for date in trade_dates:
        idx = date_index[date]

        remaining_positions = []
        for pos in positions:
            if pos["exit_index"] <= idx:
                exit_value = pos["capital"] * (1.0 + pos["net_return"])
                cash += exit_value
                trade = dict(pos)
                trade["exit_signal_date"] = date
                trade["exit_value"] = exit_value
                trades.append(trade)
            else:
                remaining_positions.append(pos)
        positions = remaining_positions

        held_codes = {pos["stock_code"] for pos in positions}
        available_slots = max(config.max_positions - len(positions), 0)
        day_rows = sorted(
            rows_by_date[date],
            key=lambda row: _to_float(_value(row, config.pred_col)) or float("-inf"),
            reverse=True,
        )
        selected = []
        for row in day_rows:
            if len(selected) >= min(config.top_k, available_slots):
                break
            stock_code = _value(row, "stock_code")
            if stock_code in held_codes:
                continue
            if not _is_market_allowed(row, config):
                continue
            if not _passes_filters(row, candidate_config):
                continue
            buy = _to_float(_value(row, config.buy_col))
            sell = _to_float(_value(row, config.sell_col))
            effective_slippage = resolve_slippage_rate(row, config)
            net_return = calculate_net_return(
                buy,
                sell,
                commission_rate=config.commission_rate,
                sell_tax_rate=config.sell_tax_rate,
                slippage_rate=effective_slippage,
            )
            if net_return is None:
                continue
            net_return = _apply_exit_rules(net_return, config)
            selected.append((row, net_return, effective_slippage))
            held_codes.add(stock_code)

        weights = _selection_weights(selected, config.position_weight_mode)
        total_weight = sum(weights) or 1.0
        starting_cash = cash
        allocated_cash = 0.0
        for selected_idx, ((row, net_return, effective_slippage), weight) in enumerate(zip(selected, weights)):
            if cash <= 0:
                break
            target_capital = starting_cash * (weight / total_weight)
            capital = cash if selected_idx == len(selected) - 1 else min(target_capital, cash)
            cash -= capital
            allocated_cash += capital
            exit_index = min(idx + config.holding_days, len(trade_dates) - 1)
            positions.append(
                {
                    "entry_signal_date": date,
                    "scheduled_exit_signal_date": trade_dates[exit_index],
                    "exit_index": exit_index,
                    "stock_code": _value(row, "stock_code"),
                    "name": _value(row, "name"),
                    "pred_prob": _to_float(_value(row, config.pred_col)),
                    "buy_price": _to_float(_value(row, config.buy_col)),
                    "sell_price": _to_float(_value(row, config.sell_col)),
                    "capital": capital,
                    "capital_weight": capital / starting_cash if starting_cash > 0 else 0.0,
                    "slippage_rate": effective_slippage,
                    "net_return": net_return,
                }
            )

        equity = cash + sum(pos["capital"] for pos in positions)
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        equity_curve.append(
            {
                "trade_date": date,
                "equity": equity,
                "cash": cash,
                "position_count": len(positions),
                "drawdown": drawdown,
            }
        )

    for pos in positions:
        exit_value = pos["capital"] * (1.0 + pos["net_return"])
        cash += exit_value
        trade = dict(pos)
        trade["exit_signal_date"] = pos["scheduled_exit_signal_date"]
        trade["exit_value"] = exit_value
        trades.append(trade)

    if equity_curve:
        final_equity = cash
        equity_curve[-1]["equity"] = final_equity
        running_peak = 0.0
        max_drawdown = 0.0
        for item in equity_curve:
            running_peak = max(running_peak, item["equity"])
            item["drawdown"] = item["equity"] / running_peak - 1.0 if running_peak > 0 else 0.0
            max_drawdown = min(max_drawdown, item["drawdown"])
    else:
        final_equity = 1.0

    daily_returns = []
    previous = 1.0
    for item in equity_curve:
        equity = item["equity"]
        daily_returns.append(equity / previous - 1.0 if previous > 0 else 0.0)
        previous = equity

    annualized = _annualized_return(final_equity, len(equity_curve))
    metrics = {
        "trade_day_count": len(equity_curve),
        "trade_count": len(trades),
        "final_equity": final_equity,
        "cumulative_return": final_equity - 1.0,
        "annualized_return": annualized,
        "annualized_volatility": _annualized_volatility(daily_returns),
        "sharpe": _sharpe_ratio(daily_returns),
        "calmar": _calmar_ratio(annualized, max_drawdown),
        "max_drawdown": max_drawdown,
        "win_rate": (sum(1 for trade in trades if trade["net_return"] > 0) / len(trades)) if trades else 0.0,
        "avg_trade_return": (sum(trade["net_return"] for trade in trades) / len(trades)) if trades else 0.0,
    }
    return {"metrics": metrics, "equity_curve": equity_curve, "trades": trades}


def write_csv(rows, output_path):
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run portfolio-level backtest.")
    parser.add_argument("--db", default="data_file/odb.db")
    parser.add_argument("--table", default="stock_predict_data_10d_yield_rate")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--holding-days", type=int, default=10)
    parser.add_argument("--min-pred", type=float, default=0.01)
    parser.add_argument("--max-atr-ratio", type=float, default=0.10)
    parser.add_argument("--buy-col", default="post_open")
    parser.add_argument("--sell-col", default="post12_open")
    parser.add_argument("--liquidity-slippage", action="store_true")
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--output-trades")
    parser.add_argument("--output-equity")
    args = parser.parse_args(argv)

    rows = read_prediction_rows(args.db, args.table, args.start, args.end)
    result = run_portfolio_backtest(
        rows,
        PortfolioBacktestConfig(
            top_k=args.top_k,
            max_positions=args.max_positions,
            holding_days=args.holding_days,
            start_date=args.start,
            end_date=args.end,
            min_pred_prob=args.min_pred,
            max_atr_ratio=args.max_atr_ratio,
            buy_col=args.buy_col,
            sell_col=args.sell_col,
            liquidity_slippage_enabled=args.liquidity_slippage,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
        ),
    )
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    if args.output_trades:
        write_csv(result["trades"], Path(args.output_trades))
    if args.output_equity:
        write_csv(result["equity_curve"], Path(args.output_equity))


if __name__ == "__main__":
    main()
