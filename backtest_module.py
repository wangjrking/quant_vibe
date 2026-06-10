"""Practical backtesting helpers for stock prediction output."""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from stock_pool_module import load_stock_pool


@dataclass
class BacktestConfig:
    top_k: int = 10
    start_date: str | None = None
    end_date: str | None = None
    pred_col: str = "pred_prob"
    label_col: str = "10d_yield_rate"
    return_col: str | None = None
    buy_col: str = "post_open"
    sell_col: str = "post2_open"
    close_col: str = "close"
    min_pred_prob: float | None = None
    max_atr_ratio: float | None = None
    holding_period_days: int = 1
    commission_rate: float = 0.0003
    sell_tax_rate: float = 0.0005
    slippage_rate: float = 0.001
    min_amount: float | None = None
    min_turnover_rate: float | None = None
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


@dataclass
class BacktestResult:
    metrics: dict
    daily_returns: list[dict]
    trades: list[dict]


def _value(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _rank(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and values[order[end + 1]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + end) / 2.0 + 1.0
        for idx in range(pos, end + 1):
            ranks[order[idx]] = avg_rank
        pos = end + 1
    return ranks


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def limit_up_pct(stock_code, name=None, st_type=None):
    code = str(stock_code or "")
    name_text = str(name or "")
    if st_type or name_text.startswith("ST") or name_text.startswith("*ST"):
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def calculate_net_return(
    buy_price,
    sell_price,
    commission_rate=0.0003,
    sell_tax_rate=0.0005,
    slippage_rate=0.001,
):
    """Return net percentage return after buy/sell costs and slippage."""
    buy = _to_float(buy_price)
    sell = _to_float(sell_price)
    if buy is None or sell is None or buy <= 0 or sell <= 0:
        return None
    entry_cash = buy * (1.0 + commission_rate + slippage_rate)
    exit_cash = sell * (1.0 - commission_rate - sell_tax_rate - slippage_rate)
    return exit_cash / entry_cash - 1.0


def resolve_slippage_rate(row, config) -> float:
    rate = float(getattr(config, "slippage_rate", 0.0) or 0.0)
    if not getattr(config, "liquidity_slippage_enabled", False):
        return rate

    amount = _to_float(_value(row, getattr(config, "liquidity_amount_col", "amount")))
    turnover = _to_float(_value(row, getattr(config, "liquidity_turnover_col", "turnover_rate")))

    for threshold, premium in getattr(config, "liquidity_slippage_amount_steps", ()):
        if amount is not None and amount < threshold:
            rate += premium
    for threshold, premium in getattr(config, "liquidity_slippage_turnover_steps", ()):
        if turnover is not None and turnover < threshold:
            rate += premium

    cap = float(getattr(config, "liquidity_slippage_cap", rate) or rate)
    return min(rate, cap)


def calculate_net_return_from_gross(
    gross_return,
    commission_rate=0.0003,
    sell_tax_rate=0.0005,
    slippage_rate=0.001,
):
    gross = _to_float(gross_return)
    if gross is None or gross <= -1.0:
        return None
    return calculate_net_return(
        1.0,
        1.0 + gross,
        commission_rate=commission_rate,
        sell_tax_rate=sell_tax_rate,
        slippage_rate=slippage_rate,
    )


def _is_st_row(row):
    name = str(_value(row, "name", "") or "")
    st_type = _value(row, "st_type")
    return bool(st_type) or name.startswith("ST") or name.startswith("*ST")


def _is_current_limit_row(row):
    return _value(row, "limit_times") not in (None, "", "None")


def _atr_ratio(row, config):
    close = _to_float(_value(row, config.close_col))
    atr = _to_float(_value(row, "atr_qfq"))
    if close is None or atr is None or close <= 0:
        return None
    return atr / close


def _is_buyable(row, config):
    close = _to_float(_value(row, config.close_col))
    if close is None or close <= 0:
        return False
    if config.return_col:
        if _to_float(_value(row, config.return_col)) is None:
            return False
        buy = _to_float(_value(row, config.buy_col))
        if config.skip_limit_up_open and buy is not None:
            pct = limit_up_pct(_value(row, "stock_code"), _value(row, "name"), _value(row, "st_type"))
            if buy >= close * (1.0 + pct) * 0.995:
                return False
        return True
    buy = _to_float(_value(row, config.buy_col))
    sell = _to_float(_value(row, config.sell_col))
    if close is None or buy is None or sell is None:
        return False
    if close <= 0 or buy <= 0 or sell <= 0:
        return False
    if config.skip_limit_up_open:
        pct = limit_up_pct(_value(row, "stock_code"), _value(row, "name"), _value(row, "st_type"))
        if buy >= close * (1.0 + pct) * 0.995:
            return False
    return True


def _passes_filters(row, config):
    pred = _to_float(_value(row, config.pred_col))
    if config.min_pred_prob is not None and (pred is None or pred < config.min_pred_prob):
        return False
    amount = _to_float(_value(row, getattr(config, "liquidity_amount_col", "amount")))
    if getattr(config, "min_amount", None) is not None and (amount is None or amount < config.min_amount):
        return False
    turnover = _to_float(_value(row, getattr(config, "liquidity_turnover_col", "turnover_rate")))
    if getattr(config, "min_turnover_rate", None) is not None and (turnover is None or turnover < config.min_turnover_rate):
        return False
    if config.max_atr_ratio is not None:
        ratio = _atr_ratio(row, config)
        if ratio is None or ratio > config.max_atr_ratio:
            return False
    if config.exclude_st and _is_st_row(row):
        return False
    if config.exclude_current_limit and _is_current_limit_row(row):
        return False
    return _is_buyable(row, config)


def _date_in_range(date, config):
    if config.start_date and date < config.start_date:
        return False
    if config.end_date and date > config.end_date:
        return False
    return True


def _period_return_to_daily_return(period_return, holding_period_days):
    if holding_period_days <= 1:
        return period_return
    if period_return <= -1.0:
        return -1.0
    return (1.0 + period_return) ** (1.0 / holding_period_days) - 1.0


def _annualized_return(equity, day_count, annual_days=252):
    if day_count <= 0 or equity <= 0:
        return 0.0
    return equity ** (annual_days / day_count) - 1.0


def _annualized_volatility(daily_return_values, annual_days=252):
    if len(daily_return_values) < 2:
        return 0.0
    mean_return = sum(daily_return_values) / len(daily_return_values)
    variance = sum((ret - mean_return) ** 2 for ret in daily_return_values) / (len(daily_return_values) - 1)
    return math.sqrt(variance) * math.sqrt(annual_days)


def _sharpe_ratio(daily_return_values, annual_days=252):
    volatility = _annualized_volatility(daily_return_values, annual_days=annual_days)
    if volatility == 0:
        return 0.0
    mean_return = sum(daily_return_values) / len(daily_return_values) if daily_return_values else 0.0
    return mean_return / (volatility / math.sqrt(annual_days))


def _calmar_ratio(annualized_return, max_drawdown):
    if max_drawdown >= 0:
        return 0.0
    return annualized_return / abs(max_drawdown)


def run_backtest(rows, config=None):
    config = config or BacktestConfig()
    rows_by_date = {}
    for row in rows:
        trade_date = str(_value(row, "trade_date", ""))
        if not trade_date or not _date_in_range(trade_date, config):
            continue
        pred = _to_float(_value(row, config.pred_col))
        if pred is None:
            continue
        rows_by_date.setdefault(trade_date, []).append(row)

    trades = []
    daily_returns = []
    rank_ics = []
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for trade_date in sorted(rows_by_date):
        day_rows = rows_by_date[trade_date]
        scored = sorted(day_rows, key=lambda row: _to_float(_value(row, config.pred_col)) or -math.inf, reverse=True)
        selected = []
        for row in scored:
            if not _passes_filters(row, config):
                continue
            if config.return_col:
                net_return = calculate_net_return_from_gross(
                    _value(row, config.return_col),
                    config.commission_rate,
                    config.sell_tax_rate,
                    resolve_slippage_rate(row, config),
                )
            else:
                net_return = calculate_net_return(
                    _value(row, config.buy_col),
                    _value(row, config.sell_col),
                    config.commission_rate,
                    config.sell_tax_rate,
                    resolve_slippage_rate(row, config),
                )
            if net_return is None:
                continue
            selected.append((row, net_return))
            if len(selected) >= config.top_k:
                break

        if selected:
            period_return = sum(ret for _, ret in selected) / len(selected)
            day_return = _period_return_to_daily_return(period_return, config.holding_period_days)
            equity *= 1.0 + day_return
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
            daily_returns.append(
                {
                    "trade_date": trade_date,
                    "return": day_return,
                    "period_return": period_return,
                    "equity": equity,
                    "selected_count": len(selected),
                }
            )
            for rank, (row, net_return) in enumerate(selected, start=1):
                trades.append(
                    {
                        "trade_date": trade_date,
                        "rank": rank,
                        "stock_code": _value(row, "stock_code"),
                        "name": _value(row, "name"),
                        "pred_prob": _to_float(_value(row, config.pred_col)),
                        "buy_price": _to_float(_value(row, config.buy_col)),
                        "sell_price": _to_float(_value(row, config.sell_col)),
                        "gross_return": _to_float(_value(row, config.return_col)) if config.return_col else None,
                        "slippage_rate": resolve_slippage_rate(row, config),
                        "net_return": net_return,
                    }
                )

        ic_pairs = [
            (_to_float(_value(row, config.pred_col)), _to_float(_value(row, config.label_col)))
            for row in day_rows
        ]
        ic_pairs = [(pred, label) for pred, label in ic_pairs if pred is not None and label is not None]
        if len(ic_pairs) >= 2:
            ic = _spearman([pred for pred, _ in ic_pairs], [label for _, label in ic_pairs])
            if ic is not None:
                rank_ics.append(ic)

    trade_returns = [trade["net_return"] for trade in trades]
    daily_return_values = [day["return"] for day in daily_returns]
    annualized = _annualized_return(equity, len(daily_returns))
    volatility = _annualized_volatility(daily_return_values)
    metrics = {
        "trade_day_count": len(daily_returns),
        "trade_count": len(trades),
        "avg_selected_per_day": (sum(day["selected_count"] for day in daily_returns) / len(daily_returns)) if daily_returns else 0.0,
        "avg_daily_return": (sum(day["return"] for day in daily_returns) / len(daily_returns)) if daily_returns else 0.0,
        "cumulative_return": equity - 1.0,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe": _sharpe_ratio(daily_return_values),
        "calmar": _calmar_ratio(annualized, max_drawdown),
        "win_rate": (sum(1 for ret in trade_returns if ret > 0) / len(trade_returns)) if trade_returns else 0.0,
        "avg_trade_return": (sum(trade_returns) / len(trade_returns)) if trade_returns else 0.0,
        "max_drawdown": max_drawdown,
        "mean_spearman_ic": (sum(rank_ics) / len(rank_ics)) if rank_ics else None,
    }
    return BacktestResult(metrics=metrics, daily_returns=daily_returns, trades=trades)


def read_prediction_rows(db_path, table, start_date=None, end_date=None, stock_pool_path=None):
    path = Path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        table_cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        existing_tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        enrich_cols = ["amount", "turnover_rate", "turnover_rate_f", "circ_mv", "vol", "industry"]
        missing_cols = [col for col in enrich_cols if col not in table_cols]
        sql = f'SELECT p.*'
        has_stock_daily = "STOCK_DAILY_DATA" in existing_tables
        has_daily = "daily_data" in existing_tables
        has_daily_basic = "daily_index_data" in existing_tables
        has_stock_basic = "stock_basic_data" in existing_tables
        if missing_cols:
            for col in missing_cols:
                if col == "amount":
                    expr = 's."amount"' if has_stock_daily else "NULL"
                    if has_daily:
                        expr = f"COALESCE({expr}, d.\"amount\")" if expr != "NULL" else 'd."amount"'
                    sql += f', {expr} AS "amount"'
                elif col == "vol":
                    expr = 's."vol"' if has_stock_daily else "NULL"
                    if has_daily:
                        expr = f"COALESCE({expr}, d.\"vol\")" if expr != "NULL" else 'd."vol"'
                    sql += f', {expr} AS "vol"'
                elif col in {"turnover_rate", "turnover_rate_f", "circ_mv"}:
                    expr = f's."{col}"' if has_stock_daily else "NULL"
                    if has_daily_basic:
                        expr = f'COALESCE({expr}, b."{col}")' if expr != "NULL" else f'b."{col}"'
                    sql += f', {expr} AS "{col}"'
                elif col == "industry":
                    expr = 's."industry"' if has_stock_daily else "NULL"
                    if has_stock_basic:
                        expr = f'COALESCE({expr}, sb."industry")' if expr != "NULL" else 'sb."industry"'
                    sql += f', {expr} AS "industry"'
        sql += f' FROM "{table}" p'
        if missing_cols and has_stock_daily:
            sql += ' LEFT JOIN "STOCK_DAILY_DATA" s ON p.stock_code = s.stock_code AND p.trade_date = s.trade_date'
        if missing_cols and has_daily:
            sql += ' LEFT JOIN "daily_data" d ON p.stock_code = d.ts_code AND p.trade_date = d.trade_date'
        if missing_cols and has_daily_basic:
            sql += ' LEFT JOIN "daily_index_data" b ON p.stock_code = b.ts_code AND p.trade_date = b.trade_date'
        if missing_cols and has_stock_basic:
            sql += ' LEFT JOIN "stock_basic_data" sb ON p.stock_code = sb.ts_code'
        params = []
        where = []
        if start_date:
            where.append("p.trade_date >= ?")
            params.append(start_date)
        if end_date:
            where.append("p.trade_date <= ?")
            params.append(end_date)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY p.trade_date, p.pred_prob DESC"
        rows = [dict(row) for row in conn.execute(sql, params)]
        if stock_pool_path:
            stock_pool = load_stock_pool(stock_pool_path)
            rows = [row for row in rows if str(row.get("stock_code", "")).upper() in stock_pool]
        return rows
    finally:
        conn.close()


def write_trades_csv(trades, output_path):
    if not trades:
        return
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)


def _print_metrics(metrics):
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Run a practical backtest for prediction output.")
    parser.add_argument("--db", default="../data_file/odb.db")
    parser.add_argument("--table", default="stock_predict_data_10d_yield_rate")
    parser.add_argument("--start", dest="start_date")
    parser.add_argument("--end", dest="end_date")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--return-col")
    parser.add_argument("--min-pred", type=float)
    parser.add_argument("--max-atr-ratio", type=float)
    parser.add_argument("--holding-period", type=int, default=1)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--sell-tax", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--min-amount", type=float)
    parser.add_argument("--min-turnover-rate", type=float)
    parser.add_argument("--liquidity-slippage", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = BacktestConfig(
        top_k=args.top_k,
        start_date=args.start_date,
        end_date=args.end_date,
        return_col=args.return_col,
        min_pred_prob=args.min_pred,
        max_atr_ratio=args.max_atr_ratio,
        holding_period_days=args.holding_period,
        commission_rate=args.commission,
        sell_tax_rate=args.sell_tax,
        slippage_rate=args.slippage,
        min_amount=args.min_amount,
        min_turnover_rate=args.min_turnover_rate,
        liquidity_slippage_enabled=args.liquidity_slippage,
    )
    rows = read_prediction_rows(args.db, args.table, args.start_date, args.end_date)
    result = run_backtest(rows, config)
    _print_metrics(result.metrics)
    if args.output:
        write_trades_csv(result.trades, args.output)
        print(f"trades_csv: {args.output}")


if __name__ == "__main__":
    main()
