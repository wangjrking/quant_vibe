"""Strict walk-forward training and parameter selection for long-horizon strategy evaluation."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_module import get_factor_data, model_assess
from market_filter_module import build_market_filter, build_multi_index_market_filter, load_index_data, merge_market_filter
from portfolio_backtest_module import PortfolioBacktestConfig, run_portfolio_backtest, write_csv


DATE_FMT = "%Y%m%d"


@dataclass(frozen=True)
class StrictWalkForwardWindow:
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FMT).date()


def _format_date(value: date) -> str:
    return value.strftime(DATE_FMT)


def _month_end(value: date) -> date:
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _month_end(date(year, month, 1)).day)
    return date(year, month, day)


def _next_day(value: str) -> str:
    return _format_date(_parse_date(value) + timedelta(days=1))


def _filter_by_trade_date(obj, start: str, end: str):
    if obj is None:
        return None
    if hasattr(obj, "index"):
        names = list(getattr(obj.index, "names", []) or [])
        if "trade_date" in names:
            dates = obj.index.get_level_values("trade_date")
            mask = [(start <= str(item) <= end) for item in dates]
            return obj.loc[mask]
    if hasattr(obj, "columns") and "trade_date" in obj.columns:
        dates = obj["trade_date"].astype(str)
        return obj[(dates >= start) & (dates <= end)]
    return obj


def _rows_to_records(rows):
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        try:
            return rows.to_dict("records")
        except TypeError:
            pass
    return list(rows)


def _int_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _str_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_values(value: str) -> list[float | None]:
    values = []
    for item in value.split(","):
        text = item.strip()
        if not text or text.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(text))
    return values


def build_strict_walkforward_windows(
    data_start: str,
    first_test: str,
    final_test: str,
    train_years: int = 5,
    validation_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    embargo_days: int = 10,
    train_mode: str = "expanding",
) -> list[StrictWalkForwardWindow]:
    if train_years <= 0:
        raise ValueError("train_years must be positive")
    if validation_months <= 0:
        raise ValueError("validation_months must be positive")
    if test_months <= 0:
        raise ValueError("test_months must be positive")
    if step_months <= 0:
        raise ValueError("step_months must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    if train_mode not in {"fixed", "expanding"}:
        raise ValueError("train_mode must be 'fixed' or 'expanding'")

    data_start_dt = _parse_date(data_start)
    current_test_start = _parse_date(first_test)
    final_test_dt = _parse_date(final_test)
    if final_test_dt < current_test_start:
        raise ValueError("final_test must be on or after first_test")

    windows: list[StrictWalkForwardWindow] = []
    fold = 1
    while current_test_start <= final_test_dt:
        test_end_dt = min(_add_months(current_test_start, test_months) - timedelta(days=1), final_test_dt)
        validation_end_dt = current_test_start - timedelta(days=embargo_days + 1)
        validation_start_dt = _add_months(current_test_start, -validation_months)
        train_end_dt = validation_start_dt - timedelta(days=embargo_days + 1)
        if train_mode == "expanding":
            train_start_dt = data_start_dt
        else:
            train_start_dt = _add_months(validation_start_dt, -train_years * 12)
            if train_start_dt < data_start_dt:
                train_start_dt = data_start_dt

        if train_start_dt <= train_end_dt and validation_start_dt <= validation_end_dt:
            windows.append(
                StrictWalkForwardWindow(
                    fold=fold,
                    train_start=_format_date(train_start_dt),
                    train_end=_format_date(train_end_dt),
                    validation_start=_format_date(validation_start_dt),
                    validation_end=_format_date(validation_end_dt),
                    test_start=_format_date(current_test_start),
                    test_end=_format_date(test_end_dt),
                )
            )
            fold += 1

        current_test_start = _add_months(current_test_start, step_months)
    return windows


def train_predict_slice(
    *,
    train_start: str,
    train_end: str,
    predict_start: str,
    predict_end: str,
    data_file_url: str,
    label: str,
    model_type: str = "reg",
    stock_pool_path: str | None = None,
):
    split_date = _next_day(train_end)
    train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(
        train_start,
        split_date,
        label,
        data_file_url,
        stock_pool_path=stock_pool_path,
    )
    test_x = _filter_by_trade_date(test_x, predict_start, predict_end)
    test_y = _filter_by_trade_date(test_y, predict_start, predict_end)
    test_data = _filter_by_trade_date(test_data, predict_start, predict_end)
    if hasattr(test_data, "copy"):
        test_data = test_data.copy()

    if len(train_x) == 0 or len(test_x) == 0:
        return None
    return model_assess(
        train_x,
        train_y,
        test_x,
        test_y,
        train_data,
        test_data,
        model_type,
        data_file_url,
        save_shap=False,
    )


def _score(metrics: dict[str, Any], min_trades: int = 10, max_drawdown_limit: float | None = None) -> float:
    trades = int(metrics.get("trade_count", 0) or 0)
    if trades < min_trades:
        return -999.0
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    if max_drawdown_limit is not None and max_drawdown > max_drawdown_limit:
        return -100.0 - max_drawdown
    annual = float(metrics.get("annualized_return", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    calmar = float(metrics.get("calmar", 0.0) or 0.0)
    return annual + 0.10 * sharpe + 0.03 * calmar


def _prepare_market_filter_rows(rows, enabled: bool, data_dir: Path, index_codes: list[str], ma_window: int, index_mode: str):
    if not enabled:
        return rows
    index_data = load_index_data(data_dir)
    if len(index_codes) == 1:
        market = build_market_filter(index_data, index_codes[0], ma_window)
    else:
        market = build_multi_index_market_filter(index_data, index_codes, ma_window, index_mode)
    return merge_market_filter(rows, market)


def search_validation_params(
    rows,
    *,
    data_dir: Path,
    top_k_values: list[int],
    max_positions_values: list[int],
    holding_days_values: list[int],
    min_pred_values: list[float | None],
    max_atr_values: list[float | None],
    stop_loss_values: list[float | None],
    take_profit_values: list[float | None],
    weight_modes: list[str],
    market_filter_values: list[bool],
    index_codes: list[str],
    ma_window: int,
    index_mode: str,
    min_trades: int,
    max_drawdown_limit: float | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base_rows = _rows_to_records(rows)
    market_rows = _prepare_market_filter_rows(base_rows, True, data_dir, index_codes, ma_window, index_mode)

    for top_k in top_k_values:
        for max_positions in max_positions_values:
            if top_k > max_positions:
                continue
            for holding_days in holding_days_values:
                for min_pred in min_pred_values:
                    for max_atr in max_atr_values:
                        for stop_loss in stop_loss_values:
                            for take_profit in take_profit_values:
                                for weight_mode in weight_modes:
                                    for market_filter_enabled in market_filter_values:
                                        config = PortfolioBacktestConfig(
                                            top_k=top_k,
                                            max_positions=max_positions,
                                            holding_days=holding_days,
                                            min_pred_prob=min_pred,
                                            max_atr_ratio=max_atr,
                                            stop_loss_pct=stop_loss,
                                            take_profit_pct=take_profit,
                                            position_weight_mode=weight_mode,
                                            market_filter_col="market_ok" if market_filter_enabled else None,
                                        )
                                        active_rows = market_rows if market_filter_enabled else base_rows
                                        result = run_portfolio_backtest(active_rows, config)
                                        metrics = result["metrics"]
                                        results.append(
                                            {
                                                "score": _score(metrics, min_trades=min_trades, max_drawdown_limit=max_drawdown_limit),
                                                "top_k": top_k,
                                                "max_positions": max_positions,
                                                "holding_days": holding_days,
                                                "min_pred_prob": min_pred,
                                                "max_atr_ratio": max_atr,
                                                "stop_loss_pct": stop_loss,
                                                "take_profit_pct": take_profit,
                                                "position_weight_mode": weight_mode,
                                                "market_filter_enabled": market_filter_enabled,
                                                **metrics,
                                            }
                                        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _scale_equity_curve(equity_curve: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    scaled = []
    for row in equity_curve:
        scaled.append(
            {
                "trade_date": row["trade_date"],
                "equity": float(row["equity"]) * scale,
                "cash": float(row["cash"]) * scale,
                "position_count": row["position_count"],
                "drawdown": row.get("drawdown", 0.0),
            }
        )
    return scaled


def _scale_trades(trades: list[dict[str, Any]], scale: float, fold: int) -> list[dict[str, Any]]:
    scaled = []
    for row in trades:
        item = dict(row)
        item["capital"] = float(item["capital"]) * scale
        item["exit_value"] = float(item["exit_value"]) * scale
        item["fold"] = fold
        scaled.append(item)
    return scaled


def combine_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    combined_equity: list[dict[str, Any]] = []
    combined_trades: list[dict[str, Any]] = []
    capital = 1.0
    for fold_result in fold_results:
        result = fold_result["test_result"]
        combined_equity.extend(_scale_equity_curve(result["equity_curve"], capital))
        combined_trades.extend(_scale_trades(result["trades"], capital, fold_result["fold"]))
        capital *= float(result["metrics"]["final_equity"])

    running_peak = 0.0
    max_drawdown = 0.0
    daily_returns = []
    previous = 1.0
    for item in combined_equity:
        running_peak = max(running_peak, item["equity"])
        item["drawdown"] = item["equity"] / running_peak - 1.0 if running_peak > 0 else 0.0
        max_drawdown = min(max_drawdown, item["drawdown"])
        daily_returns.append(item["equity"] / previous - 1.0 if previous > 0 else 0.0)
        previous = item["equity"]

    from backtest_module import _annualized_return, _annualized_volatility, _calmar_ratio, _sharpe_ratio

    annualized = _annualized_return(capital, len(combined_equity))
    metrics = {
        "trade_day_count": len(combined_equity),
        "trade_count": len(combined_trades),
        "final_equity": capital,
        "cumulative_return": capital - 1.0,
        "annualized_return": annualized,
        "annualized_volatility": _annualized_volatility(daily_returns),
        "sharpe": _sharpe_ratio(daily_returns),
        "calmar": _calmar_ratio(annualized, max_drawdown),
        "max_drawdown": max_drawdown,
        "win_rate": (sum(1 for row in combined_trades if float(row["net_return"]) > 0) / len(combined_trades)) if combined_trades else 0.0,
        "avg_trade_return": (sum(float(row["net_return"]) for row in combined_trades) / len(combined_trades)) if combined_trades else 0.0,
        "fold_count": len(fold_results),
    }
    return {"metrics": metrics, "equity_curve": combined_equity, "trades": combined_trades}


def write_rows(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    if not rows:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_strict_walkforward(args) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    windows = build_strict_walkforward_windows(
        data_start=args.data_start,
        first_test=args.first_test,
        final_test=args.final_test,
        train_years=args.train_years,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        embargo_days=args.embargo_days,
        train_mode=args.train_mode,
    )
    fold_summaries: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    index_codes = [item.strip() for item in args.index_code.split(",") if item.strip()]
    market_filter_values = [value.lower() == "true" for value in _str_values(args.market_filter_values)]

    for window in windows:
        validation_predictions = train_predict_slice(
            train_start=window.train_start,
            train_end=window.train_end,
            predict_start=window.validation_start,
            predict_end=window.validation_end,
            data_file_url=str(data_dir),
            label=args.label,
            model_type=args.model_type,
            stock_pool_path=args.stock_pool,
        )
        if validation_predictions is None or len(validation_predictions) == 0:
            fold_summaries.append({**window.to_dict(), "skipped": "empty_validation_predictions"})
            continue

        validation_results = search_validation_params(
            validation_predictions,
            data_dir=data_dir,
            top_k_values=_int_values(args.top_k),
            max_positions_values=_int_values(args.max_positions),
            holding_days_values=_int_values(args.holding_days),
            min_pred_values=_float_values(args.min_pred),
            max_atr_values=_float_values(args.max_atr_ratio),
            stop_loss_values=_float_values(args.stop_loss),
            take_profit_values=_float_values(args.take_profit),
            weight_modes=_str_values(args.position_weight_mode),
            market_filter_values=market_filter_values,
            index_codes=index_codes,
            ma_window=args.ma_window,
            index_mode=args.index_mode,
            min_trades=args.min_trades,
            max_drawdown_limit=args.max_drawdown_limit,
        )
        if not validation_results:
            fold_summaries.append({**window.to_dict(), "skipped": "no_validation_results"})
            continue
        best = validation_results[0]

        test_predictions = train_predict_slice(
            train_start=window.train_start,
            train_end=window.validation_end,
            predict_start=window.test_start,
            predict_end=window.test_end,
            data_file_url=str(data_dir),
            label=args.label,
            model_type=args.model_type,
            stock_pool_path=args.stock_pool,
        )
        if test_predictions is None or len(test_predictions) == 0:
            fold_summaries.append({**window.to_dict(), "skipped": "empty_test_predictions", **best})
            continue

        test_rows = _prepare_market_filter_rows(
            _rows_to_records(test_predictions),
            bool(best["market_filter_enabled"]),
            data_dir,
            index_codes,
            args.ma_window,
            args.index_mode,
        )
        test_result = run_portfolio_backtest(
            test_rows,
            PortfolioBacktestConfig(
                top_k=int(best["top_k"]),
                max_positions=int(best["max_positions"]),
                holding_days=int(best["holding_days"]),
                min_pred_prob=best["min_pred_prob"],
                max_atr_ratio=best["max_atr_ratio"],
                stop_loss_pct=best["stop_loss_pct"],
                take_profit_pct=best["take_profit_pct"],
                position_weight_mode=str(best["position_weight_mode"]),
                market_filter_col="market_ok" if best["market_filter_enabled"] else None,
            ),
        )

        summary = {
            **window.to_dict(),
            "validation_score": best["score"],
            "validation_annualized_return": best["annualized_return"],
            "validation_max_drawdown": best["max_drawdown"],
            "selected_top_k": best["top_k"],
            "selected_max_positions": best["max_positions"],
            "selected_holding_days": best["holding_days"],
            "selected_min_pred_prob": best["min_pred_prob"],
            "selected_max_atr_ratio": best["max_atr_ratio"],
            "selected_stop_loss_pct": best["stop_loss_pct"],
            "selected_take_profit_pct": best["take_profit_pct"],
            "selected_position_weight_mode": best["position_weight_mode"],
            "selected_market_filter_enabled": best["market_filter_enabled"],
            "test_annualized_return": test_result["metrics"]["annualized_return"],
            "test_cumulative_return": test_result["metrics"]["cumulative_return"],
            "test_max_drawdown": test_result["metrics"]["max_drawdown"],
            "test_sharpe": test_result["metrics"]["sharpe"],
            "test_trade_count": test_result["metrics"]["trade_count"],
        }
        fold_summaries.append(summary)
        fold_results.append({"fold": window.fold, "window": window, "best": best, "test_result": test_result})

    aggregate = combine_fold_results(fold_results) if fold_results else {"metrics": {}, "equity_curve": [], "trades": []}
    return {"windows": windows, "fold_summaries": fold_summaries, "fold_results": fold_results, "aggregate": aggregate}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Strict walk-forward training with validation-only parameter search.")
    parser.add_argument("--data-dir", default="../data_file")
    parser.add_argument("--stock-pool", default=None)
    parser.add_argument("--label", default="10d_yield_rate")
    parser.add_argument("--model-type", default="reg", choices=["reg", "class"])
    parser.add_argument("--data-start", required=True)
    parser.add_argument("--first-test", required=True)
    parser.add_argument("--final-test", required=True)
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--train-mode", default="expanding", choices=["fixed", "expanding"])
    parser.add_argument("--top-k", default="1,2,3")
    parser.add_argument("--max-positions", default="5,8")
    parser.add_argument("--holding-days", default="5,10")
    parser.add_argument("--min-pred", default="none,0.005,0.01")
    parser.add_argument("--max-atr-ratio", default="0.05,0.06,0.07")
    parser.add_argument("--stop-loss", default="none,0.08")
    parser.add_argument("--take-profit", default="none")
    parser.add_argument("--position-weight-mode", default="equal,rank")
    parser.add_argument("--market-filter-values", default="false,true")
    parser.add_argument("--index-code", default="000300.SH,000905.SH")
    parser.add_argument("--index-mode", default="any", choices=["all", "any"])
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--max-drawdown-limit", type=float, default=0.40)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--equity-output", default=None)
    parser.add_argument("--trades-output", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        windows = build_strict_walkforward_windows(
            data_start=args.data_start,
            first_test=args.first_test,
            final_test=args.final_test,
            train_years=args.train_years,
            validation_months=args.validation_months,
            test_months=args.test_months,
            step_months=args.step_months,
            embargo_days=args.embargo_days,
            train_mode=args.train_mode,
        )
        for window in windows:
            print(window.to_dict())
        return windows

    result = run_strict_walkforward(args)
    if args.summary_output:
        write_rows(result["fold_summaries"], args.summary_output)
    if args.equity_output:
        write_csv(result["aggregate"]["equity_curve"], args.equity_output)
    if args.trades_output:
        write_csv(result["aggregate"]["trades"], args.trades_output)
    if args.metrics_output:
        Path(args.metrics_output).write_text(json.dumps(result["aggregate"]["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["aggregate"]["metrics"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
