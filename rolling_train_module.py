"""Walk-forward rolling training orchestration for the stock model."""

from __future__ import annotations

import argparse
import calendar
import csv
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable


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


def _row_count(obj) -> int:
    if obj is None:
        return 0
    try:
        return int(len(obj))
    except TypeError:
        return 0


def train_one_fold_with_ai(
    window: RollingWindow,
    data_file_url: str,
    label: str = "10d_yield_rate",
    model_type: str = "reg",
    output_table: str | None = None,
    if_exists: str = "append",
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
    }


def execute_rolling_training(
    windows: Iterable[RollingWindow],
    data_file_url: str,
    label: str = "10d_yield_rate",
    model_type: str = "reg",
    output_table: str | None = None,
    summary_output: str | Path | None = None,
) -> list[dict]:
    output_table = output_table or f"stock_predict_data_{label}_rolling"
    first_write = True

    def train_fn(window: RollingWindow) -> dict:
        nonlocal first_write
        metrics = train_one_fold_with_ai(
            window,
            data_file_url=data_file_url,
            label=label,
            model_type=model_type,
            output_table=output_table,
            if_exists="replace" if first_write else "append",
        )
        first_write = False
        return metrics

    return run_rolling_training(windows, train_fn, summary_output=summary_output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Walk-forward rolling training for stock prediction.")
    parser.add_argument("--data-file-url", default="../data_file")
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

    summary = execute_rolling_training(
        windows,
        data_file_url=args.data_file_url,
        label=args.label,
        model_type=args.model_type,
        output_table=args.output_table,
        summary_output=args.summary_output,
    )
    for row in summary:
        print(row)
    return summary


if __name__ == "__main__":
    main()
