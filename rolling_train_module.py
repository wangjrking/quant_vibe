"""Walk-forward rolling training orchestration for the stock model."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


DATE_FMT = "%Y%m%d"


@dataclass(frozen=True)
class RollingWindow:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FoldFeatureSelectionConfig:
    label: str = "10d_yield_rate"
    top_n: int = 160
    min_abs_ic: float = 0.005
    max_missing_ratio: float = 0.35
    folds: int = 8
    score_output_dir: str | None = None


@dataclass(frozen=True)
class RollingValidationWindow:
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def to_dict(self):
        return asdict(self)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FMT).date()


def _format_date(value: date) -> str:
    return value.strftime(DATE_FMT)


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _month_end(date(year, month, 1)).day)
    return date(year, month, day)


def _month_end(value: date) -> date:
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def build_validation_window(
    window: RollingWindow,
    *,
    validation_months: int = 12,
    embargo_days: int = 10,
) -> RollingValidationWindow:
    if validation_months <= 0:
        raise ValueError("validation_months must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")

    validation_start_dt = _add_months(_parse_date(window.test_start), -validation_months)
    validation_end_dt = _parse_date(window.train_end)
    train_end_dt = validation_start_dt - timedelta(days=embargo_days + 1)
    if _parse_date(window.train_start) > train_end_dt:
        raise ValueError("validation window leaves no training data")
    if validation_start_dt > validation_end_dt:
        raise ValueError("validation window is empty")

    return RollingValidationWindow(
        fold=window.fold,
        train_start=window.train_start,
        train_end=_format_date(train_end_dt),
        validation_start=_format_date(validation_start_dt),
        validation_end=window.train_end,
        test_start=window.test_start,
        test_end=window.test_end,
    )


def build_rolling_windows(
    data_start: str,
    first_test: str,
    final_test: str,
    train_years: int = 5,
    test_months: int = 1,
    step_months: int = 1,
    embargo_days: int = 10,
    train_mode: str = "fixed",
) -> list[RollingWindow]:
    """Build deterministic walk-forward windows.

    `embargo_days` leaves a calendar-day gap between the last training label
    row and the first test row. For a 10-day forward-return label, the default
    keeps rows dated inside the following test window out of the training set.
    """
    if train_years <= 0:
        raise ValueError("train_years must be positive")
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

    windows: list[RollingWindow] = []
    fold = 1
    while current_test_start <= final_test_dt:
        test_end_dt = min(_add_months(current_test_start, test_months) - timedelta(days=1), final_test_dt)
        train_end_dt = current_test_start - timedelta(days=embargo_days + 1)
        if train_mode == "expanding":
            train_start_dt = data_start_dt
        else:
            train_start_dt = _add_months(current_test_start, -train_years * 12)
            if train_start_dt < data_start_dt:
                train_start_dt = data_start_dt

        if train_start_dt <= train_end_dt:
            windows.append(
                RollingWindow(
                    fold=fold,
                    train_start=_format_date(train_start_dt),
                    train_end=_format_date(train_end_dt),
                    test_start=_format_date(current_test_start),
                    test_end=_format_date(test_end_dt),
                )
            )
            fold += 1

        current_test_start = _add_months(current_test_start, step_months)

    return windows


def write_summary_csv(rows: Iterable[dict], output_path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_rolling_training(
    windows: Iterable[RollingWindow],
    train_fn: Callable[[RollingWindow], dict | None],
    summary_output: str | Path | None = None,
) -> list[dict]:
    summary: list[dict] = []
    for window in windows:
        metrics = train_fn(window) or {}
        row = window.to_dict()
        row.update(metrics)
        summary.append(row)
    if summary_output:
        write_summary_csv(summary, summary_output)
    return summary


def _next_day(value: str) -> str:
    return _format_date(_parse_date(value) + timedelta(days=1))


def _filter_by_trade_date(obj, start: str, end: str):
    if obj is None:
        return None
    if hasattr(obj, "index"):
        names = list(getattr(obj.index, "names", []) or [])
        if "trade_date" in names:
            dates = obj.index.get_level_values("trade_date")
            mask = [(start <= str(value) <= end) for value in dates]
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


def _row_count(obj) -> int:
    if obj is None:
        return 0
    try:
        return int(len(obj))
    except TypeError:
        return 0


def train_predict_slice(
    *,
    train_start: str,
    train_end: str,
    predict_start: str,
    predict_end: str,
    data_file_url: str,
    label: str,
    model_type: str = "reg",
    selected_features: list[str] | None = None,
    use_light_factor_data: bool = False,
):
    from ai_module import get_factor_data, model_assess

    split_date = _next_day(train_end)
    train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(
        train_start,
        split_date,
        label,
        data_file_url,
        selected_features=selected_features,
        use_light_factor_data=use_light_factor_data,
    )
    test_x = _filter_by_trade_date(test_x, predict_start, predict_end)
    test_y = _filter_by_trade_date(test_y, predict_start, predict_end)
    test_data = _filter_by_trade_date(test_data, predict_start, predict_end)
    if hasattr(test_data, "copy"):
        test_data = test_data.copy()
    if _row_count(train_x) == 0 or _row_count(test_x) == 0:
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


def build_fold_feature_selection_fn(
    *,
    data_file_url: str,
    config: FoldFeatureSelectionConfig,
    stock_pool_path: str | None = None,
    use_light_factor_data: bool = False,
) -> Callable[[RollingWindow], list[str]]:
    from fast_feature_selection import score_features_fast, write_score_csv
    if use_light_factor_data:
        from light_factor_module import build_light_factor_frame, read_raw_frame
        from select_light_long_features import DEFAULT_CANDIDATES

        candidate_features = list(DEFAULT_CANDIDATES)
        for path in (
            Path(data_file_url) / f"selected_features_{config.label}.json",
            Path(data_file_url) / "selected_features.json",
        ):
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                loaded = payload.get("features", payload if isinstance(payload, list) else [])
                for feature in loaded:
                    feature = str(feature)
                    if feature not in candidate_features:
                        candidate_features.append(feature)
                break
        needed_columns = list(
            dict.fromkeys(
                candidate_features
                + [
                    "trade_date",
                    "stock_code",
                    "name",
                    "industry",
                    "st_type",
                    "limit_times",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "atr_qfq",
                ]
            )
        )
        raw = read_raw_frame(
            Path(data_file_url) / "odb.db",
            start="20100101",
            end=None,
            needed_columns=needed_columns,
            stock_pool_path=stock_pool_path,
        )
        frame = build_light_factor_frame(raw, candidate_features, config.label)
    else:
        from stock_pool_module import filter_frame_by_stock_pool, load_stock_pool

        data_path = Path(data_file_url) / "stock_factor_data.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"stock_factor_data.parquet not found: {data_path}")
        frame = pd.read_parquet(data_path)
        if stock_pool_path:
            stock_pool = load_stock_pool(stock_pool_path)
            frame = filter_frame_by_stock_pool(frame, stock_pool)

    def select_for_window(window: RollingWindow) -> list[str]:
        rows, selected = score_features_fast(
            frame,
            label=config.label,
            start=window.train_start,
            end=window.train_end,
            top_n=config.top_n,
            min_abs_ic=config.min_abs_ic,
            max_missing_ratio=config.max_missing_ratio,
            folds=config.folds,
        )
        if config.score_output_dir:
            output_dir = Path(config.score_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_score_csv(rows, output_dir / f"feature_ic_scores_{config.label}_rolling_fold{window.fold}.csv")
            payload = {"label": config.label, "features": selected}
            (output_dir / f"selected_features_{config.label}_rolling_fold{window.fold}.json").write_text(
                __import__("json").dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return selected

    return select_for_window


def train_one_fold_with_ai(
    window: RollingWindow,
    data_file_url: str,
    label: str = "10d_yield_rate",
    model_type: str = "reg",
    output_table: str | None = None,
    if_exists: str = "append",
    stock_pool_path: str | None = None,
    selected_features: list[str] | None = None,
    use_light_factor_data: bool = False,
) -> dict:
    """Train one fold with the existing ai_module and persist predictions.

    The existing get_factor_data split point is reused with a shifted date:
    train data ends at `window.train_end`, then the returned test data is
    sliced down to `window.test_start` through `window.test_end`.
    """
    from ai_module import get_factor_data, model_assess

    split_date = _next_day(window.train_end)
    train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(
        window.train_start,
        split_date,
        label,
        data_file_url,
        stock_pool_path=stock_pool_path,
        selected_features=selected_features,
        use_light_factor_data=use_light_factor_data,
    )

    test_x = _filter_by_trade_date(test_x, window.test_start, window.test_end)
    test_y = _filter_by_trade_date(test_y, window.test_start, window.test_end)
    test_data = _filter_by_trade_date(test_data, window.test_start, window.test_end)
    if hasattr(test_data, "copy"):
        test_data = test_data.copy()

    train_rows = _row_count(train_x)
    test_rows = _row_count(test_x)
    if train_rows == 0 or test_rows == 0:
        return {
            "train_rows": train_rows,
            "test_rows": test_rows,
            "prediction_rows": 0,
            "skipped": "empty_train_or_test",
        }

    predictions = model_assess(
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
    prediction_rows = _row_count(predictions)
    table_name = output_table or f"stock_predict_data_{label}_rolling"
    db_path = Path(data_file_url) / "odb.db"
    conn = sqlite3.connect(db_path)
    try:
        predictions.to_sql(table_name, con=conn, if_exists=if_exists, index=False)
        conn.commit()
    finally:
        conn.close()

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "prediction_rows": prediction_rows,
        "output_table": table_name,
        "selected_feature_count": len(selected_features or []),
    }


def execute_rolling_training(
    windows: Iterable[RollingWindow],
    data_file_url: str,
    label: str = "10d_yield_rate",
    model_type: str = "reg",
    output_table: str | None = None,
    summary_output: str | Path | None = None,
    stock_pool_path: str | None = None,
    fold_feature_selector: Callable[[RollingWindow], list[str]] | None = None,
    start_fold: int = 1,
    end_fold: int | None = None,
    resume_existing_table: bool = False,
    use_light_factor_data: bool = False,
) -> list[dict]:
    output_table = output_table or f"stock_predict_data_{label}_rolling"
    first_write = not resume_existing_table
    active_windows = [
        window
        for window in windows
        if window.fold >= start_fold and (end_fold is None or window.fold <= end_fold)
    ]

    def train_fn(window: RollingWindow) -> dict:
        nonlocal first_write
        selected_features = fold_feature_selector(window) if fold_feature_selector else None
        metrics = train_one_fold_with_ai(
            window,
            data_file_url=data_file_url,
            label=label,
            model_type=model_type,
            output_table=output_table,
            if_exists="replace" if first_write else "append",
            stock_pool_path=stock_pool_path,
            selected_features=selected_features,
            use_light_factor_data=use_light_factor_data,
        )
        first_write = False
        return metrics

    return run_rolling_training(active_windows, train_fn, summary_output=summary_output)


def search_window_validation_params(
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
    from strict_walkforward_module import search_validation_params

    return search_validation_params(
        _rows_to_records(rows),
        data_dir=data_dir,
        top_k_values=top_k_values,
        max_positions_values=max_positions_values,
        holding_days_values=holding_days_values,
        min_pred_values=min_pred_values,
        max_atr_values=max_atr_values,
        stop_loss_values=stop_loss_values,
        take_profit_values=take_profit_values,
        weight_modes=weight_modes,
        market_filter_values=market_filter_values,
        index_codes=index_codes,
        ma_window=ma_window,
        index_mode=index_mode,
        min_trades=min_trades,
        max_drawdown_limit=max_drawdown_limit,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Walk-forward rolling training for stock prediction.")
    parser.add_argument("--data-file-url", default="data_file")
    parser.add_argument("--data-start", required=True)
    parser.add_argument("--first-test", required=True)
    parser.add_argument("--final-test", required=True)
    parser.add_argument("--label", default="10d_yield_rate")
    parser.add_argument("--model-type", default="reg", choices=["reg", "class"])
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--train-mode", default="fixed", choices=["fixed", "expanding"])
    parser.add_argument("--output-table", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--stock-pool-path", default=None)
    parser.add_argument("--fold-feature-selection", action="store_true")
    parser.add_argument("--fold-feature-top-n", type=int, default=160)
    parser.add_argument("--fold-feature-min-abs-ic", type=float, default=0.005)
    parser.add_argument("--fold-feature-max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--fold-feature-folds", type=int, default=8)
    parser.add_argument("--fold-feature-output-dir", default=None)
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=None)
    parser.add_argument("--resume-existing-table", action="store_true")
    parser.add_argument("--use-light-factor-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    windows = build_rolling_windows(
        data_start=args.data_start,
        first_test=args.first_test,
        final_test=args.final_test,
        train_years=args.train_years,
        test_months=args.test_months,
        step_months=args.step_months,
        embargo_days=args.embargo_days,
        train_mode=args.train_mode,
    )

    if not args.execute or args.dry_run:
        rows = [window.to_dict() for window in windows]
        for row in rows:
            print(
                "fold={fold} train={train_start}-{train_end} test={test_start}-{test_end}".format(
                    **row
                )
            )
        if args.summary_output:
            write_summary_csv(rows, args.summary_output)
        return rows

    fold_feature_selector = None
    if args.fold_feature_selection:
        fold_feature_selector = build_fold_feature_selection_fn(
            data_file_url=args.data_file_url,
            stock_pool_path=args.stock_pool_path,
            use_light_factor_data=args.use_light_factor_data,
            config=FoldFeatureSelectionConfig(
                label=args.label,
                top_n=args.fold_feature_top_n,
                min_abs_ic=args.fold_feature_min_abs_ic,
                max_missing_ratio=args.fold_feature_max_missing_ratio,
                folds=args.fold_feature_folds,
                score_output_dir=args.fold_feature_output_dir,
            ),
        )

    summary = execute_rolling_training(
        windows,
        data_file_url=args.data_file_url,
        label=args.label,
        model_type=args.model_type,
        output_table=args.output_table,
        summary_output=args.summary_output,
        stock_pool_path=args.stock_pool_path,
        fold_feature_selector=fold_feature_selector,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
        resume_existing_table=args.resume_existing_table,
        use_light_factor_data=args.use_light_factor_data,
    )
    for row in summary:
        print(row)
    return summary


if __name__ == "__main__":
    main()
