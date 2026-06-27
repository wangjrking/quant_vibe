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
    min_amount: float | None = None
    min_turnover_rate: float | None = None
    max_total_mv: float | None = None
    skip_limit_up_open: bool = True
    exclude_st: bool = True
    exclude_current_limit: bool = True
    market_filter_col: str | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    score_exit_ratio: float | None = None
    min_score_exit_holding_days: int = 1
    position_weight_mode: str = "equal"
    reuse_sell_proceeds_same_day: bool = True


def _date_allowed(date: str, config: PortfolioBacktestConfig) -> bool:
    if config.start_date and date < config.start_date:
        return False
    if config.end_date and date > config.end_date:
        return False
    return True


def _limit_down_pct(stock_code, name=None, st_type=None) -> float:
    code = str(stock_code or "")
    name_text = str(name or "")
    if st_type or name_text.startswith("ST") or name_text.startswith("*ST"):
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _is_open_limit_down(row: dict, config: PortfolioBacktestConfig) -> bool:
    close = _to_float(_value(row, config.close_col))
    open_price = _to_float(_value(row, config.buy_col))
    if close is None or open_price is None or close <= 0 or open_price <= 0:
        return False
    pct = _limit_down_pct(_value(row, "stock_code"), _value(row, "name"), _value(row, "st_type"))
    return open_price <= close * (1.0 - pct) * 1.005


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
        min_amount=config.min_amount,
        min_turnover_rate=config.min_turnover_rate,
        max_total_mv=config.max_total_mv,
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


def _build_row_lookup(rows_by_date):
    lookup = {}
    for trade_date, rows in rows_by_date.items():
        for row in rows:
            stock_code = _value(row, "stock_code")
            if stock_code:
                lookup[(trade_date, stock_code)] = row
    return lookup


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
    row_lookup = _build_row_lookup(rows_by_date)
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
            should_exit = pos["exit_index"] <= idx
            exit_reason = "scheduled"
            if not should_exit and config.score_exit_ratio is not None:
                holding_days_elapsed = idx - int(pos["entry_index"])
                if holding_days_elapsed >= int(config.min_score_exit_holding_days):
                    current_row = row_lookup.get((date, pos["stock_code"]))
                    current_score = _to_float(_value(current_row, config.pred_col)) if current_row is not None else None
                    entry_score = _to_float(pos.get("pred_prob"))
                    if current_score is not None and entry_score is not None and current_score <= entry_score * float(config.score_exit_ratio):
                        exit_price = _to_float(_value(current_row, config.buy_col))
                        net_return = calculate_net_return(
                            pos["buy_price"],
                            exit_price,
                            commission_rate=config.commission_rate,
                            sell_tax_rate=config.sell_tax_rate,
                            slippage_rate=pos["slippage_rate"],
                        )
                        if net_return is not None:
                            pos = dict(pos)
                            pos["net_return"] = _apply_exit_rules(net_return, config)
                            pos["sell_price"] = exit_price
                            should_exit = True
                            exit_reason = "score_exit"
            if should_exit:
                exit_value = pos["capital"] * (1.0 + pos["net_return"])
                cash += exit_value
                trade = dict(pos)
                trade["exit_signal_date"] = date
                trade["exit_reason"] = exit_reason
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
                    "entry_index": idx,
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
        trade["exit_reason"] = "scheduled"
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


def run_order_backtest(rows, config: PortfolioBacktestConfig | None = None) -> dict:
    """GM-like order-flow backtest using next-open execution and explicit positions.

    This is stricter than ``run_portfolio_backtest``: it uses each day's signal only
    on that day, keeps cash occupied until a real exit, and defers exits when the
    exit open is limit-down.
    """
    config = config or PortfolioBacktestConfig()
    rows_by_date: dict[str, list[dict]] = {}
    stock_rows_by_date: dict[str, dict[str, dict]] = {}
    for row in rows:
        date = str(_value(row, "trade_date", "") or "")
        stock_code = str(_value(row, "stock_code", "") or "")
        if not date:
            continue
        rows_by_date.setdefault(date, []).append(row)
        if stock_code:
            stock_rows_by_date.setdefault(date, {})[stock_code] = row

    trade_dates = sorted(date for date in rows_by_date if _date_allowed(date, config))
    if not trade_dates:
        return {"metrics": {"trade_day_count": 0, "trade_count": 0, "final_equity": 1.0, "cumulative_return": 0.0}, "equity_curve": [], "trades": []}

    candidate_config = _candidate_config(config)
    candidate_config.sell_col = config.buy_col
    cash = 1.0
    positions: list[dict] = []
    trades: list[dict] = []
    equity_curve: list[dict] = []
    peak = 1.0

    for idx, date in enumerate(trade_dates):
        pending_cash = 0.0
        remaining_positions = []
        for pos in positions:
            holding_days_elapsed = idx - pos["entry_index"]
            should_exit = holding_days_elapsed >= int(pos.get("holding_days") or config.holding_days)
            exit_reason = "scheduled" if should_exit else None
            if not should_exit and config.score_exit_ratio is not None and holding_days_elapsed >= config.min_score_exit_holding_days:
                row_for_score = stock_rows_by_date.get(date, {}).get(pos["stock_code"])
                current_score = _to_float(_value(row_for_score, config.pred_col)) if row_for_score else None
                entry_score = _to_float(pos.get("pred_prob"))
                if current_score is not None and entry_score not in (None, 0) and current_score <= entry_score * config.score_exit_ratio:
                    should_exit = True
                    exit_reason = "score_exit"
            if not should_exit:
                remaining_positions.append(pos)
                continue

            exit_row = stock_rows_by_date.get(date, {}).get(pos["stock_code"])
            if not exit_row or _is_open_limit_down(exit_row, config):
                remaining_positions.append(pos)
                continue
            sell_price = _to_float(_value(exit_row, config.buy_col))
            net_return = calculate_net_return(
                pos["buy_price"],
                sell_price,
                commission_rate=config.commission_rate,
                sell_tax_rate=config.sell_tax_rate,
                slippage_rate=pos["slippage_rate"],
            )
            if net_return is None:
                remaining_positions.append(pos)
                continue
            net_return = _apply_exit_rules(net_return, config)
            exit_value = pos["capital"] * (1.0 + net_return)
            if config.reuse_sell_proceeds_same_day:
                cash += exit_value
            else:
                pending_cash += exit_value
            trade = dict(pos)
            trade.update(
                {
                    "exit_signal_date": date,
                    "exit_reason": exit_reason,
                    "sell_price": sell_price,
                    "net_return": net_return,
                    "exit_value": exit_value,
                }
            )
            trades.append(trade)
        positions = remaining_positions

        held_codes = {pos["stock_code"] for pos in positions}
        available_slots = max(config.max_positions - len(positions), 0)
        if cash > 0 and available_slots > 0:
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
                buy_price = _to_float(_value(row, config.buy_col))
                if buy_price is None or buy_price <= 0:
                    continue
                selected.append(row)
                held_codes.add(stock_code)

            weights = _selection_weights([(row, 0.0, 0.0) for row in selected], config.position_weight_mode)
            total_weight = sum(weights) or 1.0
            starting_cash = cash
            for selected_idx, (row, weight) in enumerate(zip(selected, weights)):
                if cash <= 0:
                    break
                target_capital = starting_cash * (weight / total_weight)
                capital = cash if selected_idx == len(selected) - 1 else min(target_capital, cash)
                buy_price = _to_float(_value(row, config.buy_col))
                effective_slippage = resolve_slippage_rate(row, config)
                cash -= capital
                positions.append(
                    {
                        "entry_signal_date": date,
                        "entry_index": idx,
                        "stock_code": _value(row, "stock_code"),
                        "name": _value(row, "name"),
                        "pred_prob": _to_float(_value(row, config.pred_col)),
                        "buy_price": buy_price,
                        "capital": capital,
                        "capital_weight": capital / starting_cash if starting_cash > 0 else 0.0,
                        "slippage_rate": effective_slippage,
                        "holding_days": config.holding_days,
                    }
                )

        if pending_cash:
            cash += pending_cash

        equity = cash + sum(pos["capital"] for pos in positions)
        peak = max(peak, equity)
        equity_curve.append(
            {
                "trade_date": date,
                "cash": cash,
                "position_value": sum(pos["capital"] for pos in positions),
                "equity": equity,
                "drawdown": equity / peak - 1.0 if peak > 0 else 0.0,
                "open_positions": len(positions),
            }
        )

    if positions:
        last_date = trade_dates[-1]
        for pos in positions:
            exit_row = stock_rows_by_date.get(last_date, {}).get(pos["stock_code"])
            sell_price = _to_float(_value(exit_row, config.buy_col)) if exit_row else pos["buy_price"]
            net_return = calculate_net_return(
                pos["buy_price"],
                sell_price,
                commission_rate=config.commission_rate,
                sell_tax_rate=config.sell_tax_rate,
                slippage_rate=pos["slippage_rate"],
            ) or 0.0
            exit_value = pos["capital"] * (1.0 + net_return)
            cash += exit_value
            trade = dict(pos)
            trade.update({"exit_signal_date": last_date, "exit_reason": "final", "sell_price": sell_price, "net_return": net_return, "exit_value": exit_value})
            trades.append(trade)
        positions = []
        if equity_curve:
            equity_curve[-1]["cash"] = cash
            equity_curve[-1]["position_value"] = 0.0
            equity_curve[-1]["equity"] = cash
            equity_curve[-1]["open_positions"] = 0

    final_equity = cash
    running_peak = 0.0
    max_drawdown = 0.0
    for item in equity_curve:
        running_peak = max(running_peak, item["equity"])
        item["drawdown"] = item["equity"] / running_peak - 1.0 if running_peak > 0 else 0.0
        max_drawdown = min(max_drawdown, item["drawdown"])

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
    parser.add_argument("--score-exit-ratio", type=float)
    parser.add_argument("--min-score-exit-holding-days", type=int, default=1)
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
            score_exit_ratio=args.score_exit_ratio,
            min_score_exit_holding_days=args.min_score_exit_holding_days,
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
