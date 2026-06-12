from __future__ import annotations

import argparse
from pathlib import Path

from backtest_module import read_prediction_rows
from market_filter_module import build_market_filter, load_index_data, merge_market_filter
from portfolio_backtest_module import PortfolioBacktestConfig, run_portfolio_backtest, write_csv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run 2-year portfolio backtest with optional market filter.")
    parser.add_argument("--data-dir", default="data_file")
    parser.add_argument("--table", default="stock_predict_data_10d_yield_rate_oos_2y_ic160")
    parser.add_argument("--start", default="20240604")
    parser.add_argument("--end", default="20260604")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--holding-days", type=int, default=10)
    parser.add_argument("--min-pred", type=float, default=0.01)
    parser.add_argument("--max-atr-ratio", type=float, default=0.10)
    parser.add_argument("--market-filter", action="store_true")
    parser.add_argument("--index-code", default="932000.CSI")
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--output-prefix", default="data_file/reports/portfolio_2y")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    rows = read_prediction_rows(data_dir / "odb.db", args.table, args.start, args.end)
    market_col = None
    if args.market_filter:
        market = build_market_filter(load_index_data(data_dir), args.index_code, args.ma_window)
        rows = merge_market_filter(rows, market)
        market_col = "market_ok"

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
            market_filter_col=market_col,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
        ),
    )
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(result["equity_curve"], prefix.with_name(prefix.name + "_equity.csv"))
    write_csv(result["trades"], prefix.with_name(prefix.name + "_trades.csv"))


if __name__ == "__main__":
    main()
