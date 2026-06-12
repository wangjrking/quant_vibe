from __future__ import annotations

import argparse
from pathlib import Path

from backtest_module import read_prediction_rows
from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


def parse_args(argv=None):
    defaults = SelectionConfig()
    parser = argparse.ArgumentParser(description="Export next-day gm.api signals from prediction rows.")
    parser.add_argument("--db", default="data_file/odb.db")
    parser.add_argument("--table", default="stock_predict_data_10d_yield_rate_oos_2y_ic160")
    parser.add_argument("--start", default="20240604")
    parser.add_argument("--end", default="20260604")
    parser.add_argument("--stock-pool")
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--min-pred", default=str(defaults.min_pred_prob))
    parser.add_argument("--max-atr-ratio", type=float, default=defaults.max_atr_ratio)
    parser.add_argument("--max-per-industry", type=int, default=defaults.max_per_industry)
    parser.add_argument("--weight-mode", default="equal", choices=["equal", "rank", "score"])
    parser.add_argument("--target-total-pct", type=float)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--holding-days", type=int)
    parser.add_argument("--liquidity-target-pct", action="store_true")
    parser.add_argument("--liquidity-min-amount", type=float)
    parser.add_argument("--liquidity-min-turnover-rate", type=float)
    parser.add_argument("--liquidity-mid-scale", type=float, default=0.8)
    parser.add_argument("--liquidity-low-scale", type=float, default=0.6)
    parser.add_argument("--output", default="data_file/gm_signals_2y_ic160.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    min_pred = None if str(args.min_pred).strip().lower() in {"none", "null", ""} else float(args.min_pred)
    rows = read_prediction_rows(args.db, args.table, args.start, args.end, stock_pool_path=args.stock_pool)
    market_rows_by_trade_date = load_market_rows_by_trade_date(args.db, args.start, args.end)
    signals = build_gm_signal_rows(
        rows,
        SelectionConfig(
            top_k=args.top_k,
            min_pred_prob=min_pred,
            max_atr_ratio=args.max_atr_ratio,
            max_per_industry=args.max_per_industry,
        ),
        market_rows_by_trade_date=market_rows_by_trade_date,
        holding_days=args.holding_days,
        max_positions=args.max_positions,
        weight_mode=args.weight_mode,
        target_total_pct=args.target_total_pct,
        liquidity_target_pct_enabled=args.liquidity_target_pct,
        liquidity_min_amount=args.liquidity_min_amount,
        liquidity_min_turnover_rate=args.liquidity_min_turnover_rate,
        liquidity_mid_scale=args.liquidity_mid_scale,
        liquidity_low_scale=args.liquidity_low_scale,
    )
    write_gm_signals_csv(signals, args.output)
    dates = sorted({row["buy_date"] for row in signals})
    print(f"signals: {len(signals)}")
    print(f"buy_days: {len(dates)}")
    if dates:
        print(f"buy_date_range: {dates[0]}-{dates[-1]}")
    print(f"output: {Path(args.output)}")


if __name__ == "__main__":
    main()
