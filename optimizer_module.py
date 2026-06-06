"""Parameter search for prediction backtests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace

from backtest_module import BacktestConfig, read_prediction_rows, run_backtest


def _score(metrics, objective="risk_adjusted_annual"):
    cumulative = metrics.get("cumulative_return") or 0.0
    annualized = metrics.get("annualized_return") or 0.0
    drawdown = metrics.get("max_drawdown") or 0.0
    sharpe = metrics.get("sharpe") or 0.0
    calmar = metrics.get("calmar") or 0.0
    trade_days = metrics.get("trade_day_count") or 0
    if trade_days == 0:
        return -999.0
    if objective == "annualized":
        return annualized
    if objective == "calmar":
        return calmar
    if objective == "sharpe":
        return sharpe
    if objective == "cumulative":
        return cumulative + drawdown
    return annualized + drawdown + 0.05 * sharpe


def evaluate_grid(
    rows,
    top_k_values,
    min_pred_values,
    max_atr_values,
    min_amount_values=None,
    min_turnover_values=None,
    return_col=None,
    start_date=None,
    end_date=None,
    commission_rate=0.0003,
    sell_tax_rate=0.0005,
    slippage_rate=0.001,
    liquidity_slippage_enabled=False,
    holding_period_days=1,
    objective="risk_adjusted_annual",
):
    results = []
    base_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        return_col=return_col,
        commission_rate=commission_rate,
        sell_tax_rate=sell_tax_rate,
        slippage_rate=slippage_rate,
        liquidity_slippage_enabled=liquidity_slippage_enabled,
        holding_period_days=holding_period_days,
    )
    min_amount_values = min_amount_values or [None]
    min_turnover_values = min_turnover_values or [None]
    for top_k in top_k_values:
        for min_pred in min_pred_values:
            for max_atr in max_atr_values:
                for min_amount in min_amount_values:
                    for min_turnover in min_turnover_values:
                        config = replace(
                            base_config,
                            top_k=int(top_k),
                            min_pred_prob=min_pred,
                            max_atr_ratio=max_atr,
                            min_amount=min_amount,
                            min_turnover_rate=min_turnover,
                        )
                        backtest = run_backtest(rows, config)
                        item = {
                            "top_k": config.top_k,
                            "min_pred_prob": config.min_pred_prob,
                            "max_atr_ratio": config.max_atr_ratio,
                            "min_amount": config.min_amount,
                            "min_turnover_rate": config.min_turnover_rate,
                            "return_col": config.return_col,
                            "objective": objective,
                            "score": _score(backtest.metrics, objective=objective),
                            **backtest.metrics,
                        }
                        results.append(item)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _parse_optional_float_list(value):
    result = []
    for item in value.split(","):
        item = item.strip()
        if item.lower() in {"none", "null", ""}:
            result.append(None)
        else:
            result.append(float(item))
    return result


def _parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_results_csv(results, output_path):
    if not results:
        return
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Search backtest parameters.")
    parser.add_argument("--db", default="../data_file/odb.db")
    parser.add_argument("--table", default="stock_predict_data_10d_yield_rate")
    parser.add_argument("--start", dest="start_date")
    parser.add_argument("--end", dest="end_date")
    parser.add_argument("--return-col")
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--min-pred", default="none,0.01,0.02,0.03")
    parser.add_argument("--max-atr-ratio", default="none,0.06,0.08,0.10,0.12")
    parser.add_argument("--min-amount", default="none")
    parser.add_argument("--min-turnover-rate", default="none")
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--sell-tax", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--liquidity-slippage", action="store_true")
    parser.add_argument("--holding-period", type=int, default=1)
    parser.add_argument(
        "--objective",
        default="risk_adjusted_annual",
        choices=["risk_adjusted_annual", "annualized", "calmar", "sharpe", "cumulative"],
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = read_prediction_rows(args.db, args.table, args.start_date, args.end_date)
    results = evaluate_grid(
        rows,
        top_k_values=_parse_int_list(args.top_k),
        min_pred_values=_parse_optional_float_list(args.min_pred),
        max_atr_values=_parse_optional_float_list(args.max_atr_ratio),
        min_amount_values=_parse_optional_float_list(args.min_amount),
        min_turnover_values=_parse_optional_float_list(args.min_turnover_rate),
        return_col=args.return_col,
        start_date=args.start_date,
        end_date=args.end_date,
        commission_rate=args.commission,
        sell_tax_rate=args.sell_tax,
        slippage_rate=args.slippage,
        liquidity_slippage_enabled=args.liquidity_slippage,
        holding_period_days=args.holding_period,
        objective=args.objective,
    )

    for row in results[: args.limit]:
        print(
            "top_k={top_k} min_pred={min_pred_prob} max_atr={max_atr_ratio} "
            "min_amount={min_amount} min_turn={min_turnover_rate} "
            "score={score:.6f} ann={annualized_return:.6f} cum={cumulative_return:.6f} dd={max_drawdown:.6f} "
            "avg_day={avg_daily_return:.6f} win={win_rate:.6f} days={trade_day_count}".format(**row)
        )
    if args.output:
        write_results_csv(results, args.output)
        print(f"optimizer_csv: {args.output}")


if __name__ == "__main__":
    main()
