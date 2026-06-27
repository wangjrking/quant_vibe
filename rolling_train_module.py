"""Walk-forward rolling training orchestration for the stock model."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from model_asset_route import (
    MODEL_FEATURE_MODE_LEGACY,
    MODEL_FEATURE_MODE_SPLIT,
    MODEL_PREDICTION_MODE_INDEPENDENT,
    MODEL_PREDICTION_MODE_LEGACY,
    enrich_research_prediction_manifest,
    require_legacy_model_asset_chain_opt_in,
    resolve_legacy_mixed_factor_path,
    resolve_legacy_prediction_db_path,
    resolve_model_feature_path,
    resolve_model_label_path,
    resolve_model_prediction_db_path,
    resolve_prediction_run_dir,
    use_legacy_mixed_features,
    use_legacy_prediction_db,
    write_prediction_manifest,
)
from stock_daily_data_route import resolve_stock_daily_db_path


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
    exclude_prefixes: tuple[str, ...] = ()


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


def _align_frame_to_existing_table(frame: pd.DataFrame, conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Align append frames to an existing SQLite table schema.

    Different rolling folds may expose a slightly different set of diagnostic
    columns after label construction. The prediction table is a research output,
    so append using the first fold's schema and keep training from failing on
    harmless extra columns.
    """
    table_info = conn.execute(f'pragma table_info("{table_name}")').fetchall()
    if not table_info:
        return frame
    columns = [row[1] for row in table_info]
    aligned = frame.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = None
    return aligned[columns]


def _model_artifact_paths(
    *,
    data_file_url: str,
    label: str,
    window: RollingWindow,
    output_table: str | None,
    prediction_output_path: str | Path | None,
) -> tuple[Path, Path]:
    if prediction_output_path:
        prediction_path = Path(prediction_output_path)
        base_dir = prediction_path.parent.parent if prediction_path.parent.name == "fold_predictions" else prediction_path.parent
    else:
        base_dir = resolve_prediction_run_dir(data_file_url, label=label, output_table=output_table, create=True)

    model_dir = base_dir / "models"
    model_path = model_dir / f"model_fold{window.fold:02d}.json"
    metadata_path = model_dir / f"model_fold{window.fold:02d}_metadata.json"
    return model_path, metadata_path


def _prediction_storage_target(
    *,
    data_file_url: str,
    label: str,
    output_table: str | None,
    prediction_output_path: str | Path | None,
    prediction_output_mode: str | None,
) -> dict[str, Any]:
    table_name = output_table or f"stock_predict_data_{label}_rolling"
    if prediction_output_path:
        path = Path(prediction_output_path)
        base_dir = path.parent.parent if path.parent.name == "fold_predictions" else path.parent
        return {
            "prediction_mode": "parquet_files",
            "table_name": table_name,
            "db_path": None,
            "prediction_path": path,
            "run_dir": base_dir,
            "manifest_path": base_dir / "prediction_manifest.json",
        }
    if use_legacy_prediction_db(prediction_output_mode):
        require_legacy_model_asset_chain_opt_in(reason="legacy odb rolling prediction output")
        db_path = resolve_legacy_prediction_db_path(data_file_url)
        mode = MODEL_PREDICTION_MODE_LEGACY
    else:
        db_path = resolve_model_prediction_db_path(data_file_url, create_parent=True)
        mode = MODEL_PREDICTION_MODE_INDEPENDENT
    run_dir = resolve_prediction_run_dir(data_file_url, label=label, output_table=table_name, create=True)
    return {
        "prediction_mode": mode,
        "table_name": table_name,
        "db_path": db_path,
        "prediction_path": None,
        "run_dir": run_dir,
        "manifest_path": run_dir / "prediction_manifest.json",
    }


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
    feature_source: str | None = None,
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
        feature_source=feature_source,
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
    feature_source: str | None = None,
) -> Callable[[RollingWindow], list[str]]:
    from leakage_guard import find_leaky_features
    if use_light_factor_data:
        from fast_feature_selection import score_features_fast, write_score_csv
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
            resolve_stock_daily_db_path(data_file_url),
            start="20100101",
            end=None,
            needed_columns=needed_columns,
            stock_pool_path=stock_pool_path,
        )
        frame = build_light_factor_frame(raw, candidate_features, config.label)
    else:
        from fast_feature_selection import score_features_fast_parquet, score_features_fast_split, write_score_csv
        feature_path = None
        label_path = None
        data_path = None
        if use_legacy_mixed_features(feature_source):
            require_legacy_model_asset_chain_opt_in(reason="legacy mixed factor feature selection input")
            data_path = resolve_legacy_mixed_factor_path(data_file_url, require_exists=True)
        else:
            feature_path = resolve_model_feature_path(data_file_url, require_exists=True)
            label_path = resolve_model_label_path(data_file_url, require_exists=True)
        frame = None

    def select_for_window(window: RollingWindow) -> list[str]:
        if config.score_output_dir:
            output_dir = Path(config.score_output_dir)
            cached_path = output_dir / f"selected_features_{config.label}_rolling_fold{window.fold}.json"
            if cached_path.exists():
                payload = json.loads(cached_path.read_text(encoding="utf-8"))
                loaded = payload.get("features", payload if isinstance(payload, list) else [])
                cached_features = [str(feature) for feature in loaded]
                if not find_leaky_features(cached_features, label=config.label):
                    return cached_features
        if use_light_factor_data:
            rows, selected = score_features_fast(
                frame,
                label=config.label,
                start=window.train_start,
                end=window.train_end,
                top_n=config.top_n,
                min_abs_ic=config.min_abs_ic,
                max_missing_ratio=config.max_missing_ratio,
                folds=config.folds,
                exclude_prefixes=config.exclude_prefixes,
            )
        else:
            if stock_pool_path:
                raise NotImplementedError("parquet chunk feature selection does not support stock_pool_path yet")
            if use_legacy_mixed_features(feature_source):
                rows, selected = score_features_fast_parquet(
                    data_path,
                    label=config.label,
                    start=window.train_start,
                    end=window.train_end,
                    top_n=config.top_n,
                    min_abs_ic=config.min_abs_ic,
                    max_missing_ratio=config.max_missing_ratio,
                    folds=config.folds,
                    exclude_prefixes=config.exclude_prefixes,
                )
            else:
                rows, selected = score_features_fast_split(
                    feature_path,
                    label_path=label_path,
                    label=config.label,
                    start=window.train_start,
                    end=window.train_end,
                    top_n=config.top_n,
                    min_abs_ic=config.min_abs_ic,
                    max_missing_ratio=config.max_missing_ratio,
                    folds=config.folds,
                    exclude_prefixes=config.exclude_prefixes,
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
    prediction_output_path: str | Path | None = None,
    feature_source: str | None = None,
    prediction_output_mode: str | None = None,
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
        feature_source=feature_source,
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

    model_path, metadata_path = _model_artifact_paths(
        data_file_url=data_file_url,
        label=label,
        window=window,
        output_table=output_table,
        prediction_output_path=prediction_output_path,
    )
    old_model_path = os.environ.get("XGB_MODEL_SAVE_PATH")
    old_metadata_path = os.environ.get("XGB_MODEL_METADATA_PATH")
    os.environ["XGB_MODEL_SAVE_PATH"] = str(model_path)
    os.environ["XGB_MODEL_METADATA_PATH"] = str(metadata_path)
    try:
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
    finally:
        if old_model_path is None:
            os.environ.pop("XGB_MODEL_SAVE_PATH", None)
        else:
            os.environ["XGB_MODEL_SAVE_PATH"] = old_model_path
        if old_metadata_path is None:
            os.environ.pop("XGB_MODEL_METADATA_PATH", None)
        else:
            os.environ["XGB_MODEL_METADATA_PATH"] = old_metadata_path

    prediction_rows = _row_count(predictions)
    storage = _prediction_storage_target(
        data_file_url=data_file_url,
        label=label,
        output_table=output_table,
        prediction_output_path=prediction_output_path,
        prediction_output_mode=prediction_output_mode,
    )
    table_name = storage["table_name"]
    if storage["prediction_path"] is not None:
        prediction_output_path = Path(storage["prediction_path"])
        prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(prediction_output_path, index=False)
    else:
        db_path = Path(storage["db_path"])
        conn = sqlite3.connect(db_path, timeout=120)
        try:
            conn.execute("PRAGMA busy_timeout=300000")
            if if_exists == "append":
                predictions = _align_frame_to_existing_table(predictions, conn, table_name)
            predictions.to_sql(table_name, con=conn, if_exists=if_exists, index=False)
            conn.commit()
        finally:
            conn.close()
        prediction_output_path = None

    manifest = {
        "label": label,
        "feature_source": feature_source or MODEL_FEATURE_MODE_SPLIT,
        "prediction_mode": storage["prediction_mode"],
        "prediction_db_path": str(storage["db_path"]) if storage["db_path"] is not None else "",
        "prediction_table": table_name,
        "prediction_output_path": str(prediction_output_path) if prediction_output_path else "",
        "model_output_path": str(model_path),
        "model_metadata_path": str(metadata_path),
        "selected_feature_count": len(selected_features or []),
    }
    write_prediction_manifest(storage["manifest_path"], enrich_research_prediction_manifest(manifest))

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "prediction_rows": prediction_rows,
        "output_table": table_name,
        "prediction_output_path": str(prediction_output_path) if prediction_output_path else "",
        "prediction_db_path": str(storage["db_path"]) if storage["db_path"] is not None else "",
        "prediction_mode": storage["prediction_mode"],
        "model_output_path": str(model_path),
        "model_metadata_path": str(metadata_path),
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
    selected_features: list[str] | None = None,
    start_fold: int = 1,
    end_fold: int | None = None,
    resume_existing_table: bool = False,
    use_light_factor_data: bool = False,
    prediction_output_path: str | Path | None = None,
    feature_source: str | None = None,
    prediction_output_mode: str | None = None,
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
        window_selected_features = fold_feature_selector(window) if fold_feature_selector else selected_features
        metrics = train_one_fold_with_ai(
            window,
            data_file_url=data_file_url,
            label=label,
            model_type=model_type,
            output_table=output_table,
            if_exists="replace" if first_write else "append",
            stock_pool_path=stock_pool_path,
            selected_features=window_selected_features,
            use_light_factor_data=use_light_factor_data,
            prediction_output_path=prediction_output_path,
            feature_source=feature_source,
            prediction_output_mode=prediction_output_mode,
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
    parser.add_argument("--prediction-output-path", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--stock-pool-path", default=None)
    parser.add_argument("--fold-feature-selection", action="store_true")
    parser.add_argument("--fold-feature-top-n", type=int, default=160)
    parser.add_argument("--fold-feature-min-abs-ic", type=float, default=0.005)
    parser.add_argument("--fold-feature-max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--fold-feature-folds", type=int, default=8)
    parser.add_argument("--fold-feature-output-dir", default=None)
    parser.add_argument("--fold-feature-exclude-prefixes", default="")
    parser.add_argument("--selected-features-path", default=None)
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=None)
    parser.add_argument("--resume-existing-table", action="store_true")
    parser.add_argument("--use-light-factor-data", action="store_true")
    parser.add_argument(
        "--feature-source",
        default=None,
        choices=[MODEL_FEATURE_MODE_SPLIT, MODEL_FEATURE_MODE_LEGACY],
    )
    parser.add_argument(
        "--prediction-output-mode",
        default=None,
        choices=[MODEL_PREDICTION_MODE_INDEPENDENT, MODEL_PREDICTION_MODE_LEGACY],
    )
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
        exclude_prefixes = tuple(
            prefix.strip()
            for prefix in str(args.fold_feature_exclude_prefixes).split(",")
            if prefix.strip()
        )
        fold_feature_selector = build_fold_feature_selection_fn(
            data_file_url=args.data_file_url,
            stock_pool_path=args.stock_pool_path,
            use_light_factor_data=args.use_light_factor_data,
            feature_source=args.feature_source,
            config=FoldFeatureSelectionConfig(
                label=args.label,
                top_n=args.fold_feature_top_n,
                min_abs_ic=args.fold_feature_min_abs_ic,
                max_missing_ratio=args.fold_feature_max_missing_ratio,
                folds=args.fold_feature_folds,
                score_output_dir=args.fold_feature_output_dir,
                exclude_prefixes=exclude_prefixes,
            ),
        )
    fixed_selected_features = None
    if args.selected_features_path:
        payload = json.loads(Path(args.selected_features_path).read_text(encoding="utf-8"))
        fixed_selected_features = [str(feature) for feature in payload.get("features", payload if isinstance(payload, list) else [])]

    summary = execute_rolling_training(
        windows,
        data_file_url=args.data_file_url,
        label=args.label,
        model_type=args.model_type,
        output_table=args.output_table,
        summary_output=args.summary_output,
        stock_pool_path=args.stock_pool_path,
        fold_feature_selector=fold_feature_selector,
        selected_features=fixed_selected_features,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
        resume_existing_table=args.resume_existing_table,
        use_light_factor_data=args.use_light_factor_data,
        prediction_output_path=args.prediction_output_path,
        feature_source=args.feature_source,
        prediction_output_mode=args.prediction_output_mode,
    )
    for row in summary:
        print(row)
    return summary


if __name__ == "__main__":
    main()
