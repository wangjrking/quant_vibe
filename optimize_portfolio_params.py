from __future__ import annotations

import argparse
import csv
from pathlib import Path

from backtest_module import read_prediction_rows
from market_filter_module import build_market_filter, build_multi_index_market_filter, load_index_data, merge_market_filter
from portfolio_backtest_module import PortfolioBacktestConfig, run_portfolio_backtest


def _float_values(value: str) -> list[float | None]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item or item.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(item))
    return values


def _int_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _score(metrics: dict, min_trades: int, max_drawdown_limit: float | None) -> float:
    trades = metrics.get("trade_count", 0)
    if trades < min_trades:
        return -999.0
    max_drawdown = abs(metrics.get("max_drawdown", 0.0))
    if max_drawdown_limit is not None and max_drawdown > max_drawdown_limit:
        return -100.0 - max_drawdown
    annual = metrics.get("annualized_return", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    calmar = metrics.get("calmar", 0.0)
    return annual + 0.10 * sharpe + 0.03 * calmar


def _str_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def optimize(rows, args):
    results = []
    for top_k in _int_values(args.top_k):
        for max_positions in _int_values(args.max_positions):
            if top_k > max_positions:
                continue
            for min_pred in _float_values(args.min_pred):
                for max_atr in _float_values(args.max_atr_ratio):
                    for stop_loss in _float_values(args.stop_loss):
                        for take_profit in _float_values(args.take_profit):
                            for weight_mode in _str_values(args.position_weight_mode):
                                config = PortfolioBacktestConfig(
                                    top_k=top_k,
                                    max_positions=max_positions,
                                    holding_days=args.holding_days,
                                    start_date=args.start,
                                    end_date=args.end,
                                    min_pred_prob=min_pred,
                                    max_atr_ratio=max_atr,
                                    stop_loss_pct=stop_loss,
                                    take_profit_pct=take_profit,
                                    market_filter_col="market_ok" if args.market_filter else None,
                                    position_weight_mode=weight_mode,
                                    liquidity_slippage_enabled=args.liquidity_slippage,
                                )
                                result = run_portfolio_backtest(rows, config)
                                metrics = result["metrics"]
                                item = {
                                    "score": _score(metrics, args.min_trades, args.max_drawdown_limit),
                                    "top_k": top_k,
                                    "max_positions": max_positions,
                                    "holding_days": args.holding_days,
                                    "min_pred_prob": min_pred,
                                    "max_atr_ratio": max_atr,
                                    "stop_loss_pct": stop_loss,
                                    "take_profit_pct": take_profit,
                                    "market_filter": args.market_filter,
                                    "position_weight_mode": weight_mode,
                                    **metrics,
                                }
                                results.append(item)
    results.sort(key=lambda row: row["score"], reverse=True)
    return results


def write_csv(rows: list[dict], output_path: str | Path) -> None:
    if not rows:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Optimize portfolio-level strategy parameters.")
    parser.add_argument("--data-dir", default="../data_file")
    parser.add_argument("--table", default="stock_predict_data_10d_yield_rate_oos_2y_ic160")
    parser.add_argument("--start", default="20240604")
    parser.add_argument("--end", default="20260604")
    parser.add_argument("--stock-pool")
    parser.add_argument("--top-k", default="1,2,3,4,5")
    parser.add_argument("--max-positions", default="3,5,8,10")
    parser.add_argument("--holding-days", type=int, default=10)
    parser.add_argument("--min-pred", default="none,0.0,0.005,0.01,0.015,0.02,0.03")
    parser.add_argument("--max-atr-ratio", default="0.05,0.06,0.08,0.10,0.12")
    parser.add_argument("--stop-loss", default="none,0.08,0.10,0.12")
    parser.add_argument("--take-profit", default="none,0.15,0.20,0.25,0.30")
    parser.add_argument("--position-weight-mode", default="equal,score,rank")
    parser.add_argument("--liquidity-slippage", action="store_true")
    parser.add_argument("--market-filter", action="store_true")
    parser.add_argument("--index-code", default="932000.CSI")
    parser.add_argument("--index-mode", default="all", choices=["all", "any"])
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=250)
    parser.add_argument("--max-drawdown-limit", type=float, default=0.20)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="../data_file/optimizer_portfolio_long_2y.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    rows = read_prediction_rows(
        data_dir / "odb.db",
        args.table,
        args.start,
        args.end,
        stock_pool_path=args.stock_pool,
    )
    if args.market_filter:
        index_codes = [item.strip() for item in args.index_code.split(",") if item.strip()]
        if len(index_codes) == 1:
            market = build_market_filter(load_index_data(data_dir), index_codes[0], args.ma_window)
        else:
            market = build_multi_index_market_filter(load_index_data(data_dir), index_codes, args.ma_window, args.index_mode)
        rows = merge_market_filter(rows, market)
    results = optimize(rows, args)
    write_csv(results, args.output)
    for row in results[: args.limit]:
        print(
            "score={score:.6f} ann={annualized_return:.6f} cum={cumulative_return:.6f} "
            "dd={max_drawdown:.6f} sharpe={sharpe:.6f} trades={trade_count} "
            "top_k={top_k} max_pos={max_positions} min_pred={min_pred_prob} max_atr={max_atr_ratio} "
            "stop={stop_loss_pct} take={take_profit_pct} market={market_filter}".format(**row)
        )
    print(f"optimizer_csv: {Path(args.output)}")


if __name__ == "__main__":
    main()
