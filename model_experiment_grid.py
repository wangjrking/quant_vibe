"""Run model-level walk-forward experiment grids and OOS strategy summaries."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from backtest_module import read_prediction_rows
from portfolio_backtest_module import PortfolioBacktestConfig, run_portfolio_backtest
from rolling_train_module import (
    FoldFeatureSelectionConfig,
    build_fold_feature_selection_fn,
    build_rolling_windows,
    execute_rolling_training,
)


XGB_ENV_KEYS = {
    "n_estimators": "XGB_N_ESTIMATORS",
    "learning_rate": "XGB_LEARNING_RATE",
    "max_depth": "XGB_MAX_DEPTH",
    "subsample": "XGB_SUBSAMPLE",
    "colsample_bytree": "XGB_COLSAMPLE_BYTREE",
    "reg_alpha": "XGB_REG_ALPHA",
    "reg_lambda": "XGB_REG_LAMBDA",
}


@dataclass(frozen=True)
class ExperimentSpec:
    labels: list[str]
    train_windows: list[str]
    feature_top_ns: list[int]
    xgb_param_sets: list[dict[str, Any]]


@dataclass(frozen=True)
class ModelExperiment:
    name: str
    label: str
    train_window: str
    train_years: int
    train_mode: str
    feature_top_n: int
    xgb_params: dict[str, Any]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in _parse_csv(value)]


def _parse_float_csv(value: str) -> list[float | None]:
    values: list[float | None] = []
    for item in _parse_csv(value):
        values.append(None if item.lower() in {"none", "null"} else float(item))
    return values


def _parse_xgb_grid(value: str) -> list[dict[str, Any]]:
    """Parse compact JSON xgb grid.

    Accepted formats:
    - JSON list of objects: [{"max_depth":2},{"max_depth":3,"reg_lambda":5}]
    - JSON object of lists: {"max_depth":[2,3],"reg_lambda":[1,5]}
    """
    payload = json.loads(value)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if not isinstance(payload, dict):
        raise ValueError("xgb grid must be a JSON object or a JSON list of objects")
    keys = list(payload)
    values = [items if isinstance(items, list) else [items] for items in payload.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _load_xgb_grid(value: str, file_path: str | None = None) -> list[dict[str, Any]]:
    if file_path:
        value = Path(file_path).read_text(encoding="utf-8")
    return _parse_xgb_grid(value)


def _window_to_mode(value: str) -> tuple[int, str]:
    text = str(value).strip().lower()
    if text == "expanding":
        return 5, "expanding"
    if text.endswith("y") and text[:-1].isdigit():
        return int(text[:-1]), "fixed"
    raise ValueError(f"unsupported train window: {value}")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _experiment_name(label: str, train_window: str, feature_top_n: int, xgb_params: dict[str, Any]) -> str:
    param_part = "_".join(f"{_safe_name(key)}{_safe_name(str(value))}" for key, value in sorted(xgb_params.items()))
    return f"{_safe_name(label)}_{_safe_name(train_window)}_fs{feature_top_n}_{param_part or 'xgbdefault'}"


def expand_experiments(spec: ExperimentSpec) -> list[ModelExperiment]:
    experiments: list[ModelExperiment] = []
    for label, train_window, feature_top_n, xgb_params in itertools.product(
        spec.labels,
        spec.train_windows,
        spec.feature_top_ns,
        spec.xgb_param_sets,
    ):
        train_years, train_mode = _window_to_mode(train_window)
        experiments.append(
            ModelExperiment(
                name=_experiment_name(label, train_window, feature_top_n, xgb_params),
                label=label,
                train_window=train_window,
                train_years=train_years,
                train_mode=train_mode,
                feature_top_n=int(feature_top_n),
                xgb_params=dict(xgb_params),
            )
        )
    return experiments


@contextmanager
def temporary_xgb_env(params: dict[str, Any]):
    previous = {}
    changed_keys = []
    for key, value in params.items():
        env_key = XGB_ENV_KEYS.get(key)
        if env_key is None:
            raise ValueError(f"unsupported xgb parameter: {key}")
        previous[env_key] = os.environ.get(env_key)
        os.environ[env_key] = str(value)
        changed_keys.append(env_key)
    try:
        yield
    finally:
        for env_key in changed_keys:
            if previous[env_key] is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous[env_key]


def build_year_slices(start: str, end: str) -> list[tuple[str, str, str]]:
    start_dt = datetime.strptime(start, "%Y%m%d").date()
    end_dt = datetime.strptime(end, "%Y%m%d").date()
    slices = []
    for year in range(start_dt.year, end_dt.year + 1):
        year_start = max(start_dt, datetime.strptime(f"{year}0101", "%Y%m%d").date())
        year_end = min(end_dt, datetime.strptime(f"{year}1231", "%Y%m%d").date())
        if year_start <= year_end:
            slices.append((str(year), year_start.strftime("%Y%m%d"), year_end.strftime("%Y%m%d")))
    return slices


def write_rows(rows: Iterable[dict[str, Any]], output_path: str | Path) -> None:
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


def _table_exists(db_path: Path, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("select 1 from sqlite_master where type='table' and name=?", (table,)).fetchone()
        return row is not None
    finally:
        conn.close()


def _sell_col_for_holding_days(days: int) -> str:
    mapping = {1: "post2_open", 3: "post4_open", 5: "post6_open", 10: "post12_open"}
    return mapping.get(days, f"post{days + 1}_open")


def _score_metrics(metrics: dict[str, Any], objective: str) -> float:
    if objective == "annual":
        return float(metrics.get("annualized_return", 0.0) or 0.0)
    if objective == "sharpe":
        return float(metrics.get("sharpe", 0.0) or 0.0)
    if objective == "rank_return":
        return float(metrics.get("annualized_return", 0.0) or 0.0) + 0.20 * float(metrics.get("sharpe", 0.0) or 0.0)
    raise ValueError(f"unsupported objective: {objective}")


def backtest_experiment_table(table: str, args) -> list[dict[str, Any]]:
    db_path = Path(args.data_dir) / "odb.db"
    base_rows = read_prediction_rows(db_path, table, args.first_test, args.final_test, stock_pool_path=args.stock_pool)
    results: list[dict[str, Any]] = []
    year_slices = [("ALL", args.first_test, args.final_test)] + build_year_slices(args.first_test, args.final_test)
    for top_k in _parse_int_csv(args.top_k):
        for hold in _parse_int_csv(args.holding_days):
            for min_pred in _parse_float_csv(args.min_pred):
                for max_atr in _parse_float_csv(args.max_atr_ratio):
                    for score_exit_ratio in _parse_float_csv(args.score_exit_ratio):
                        for min_amount in _parse_float_csv(args.min_amount):
                            for min_turnover in _parse_float_csv(args.min_turnover):
                                for max_total_mv in _parse_float_csv(args.max_total_mv):
                                    for slice_name, start, end in year_slices:
                                        config = PortfolioBacktestConfig(
                                            top_k=top_k,
                                            max_positions=max(top_k, int(args.max_positions)),
                                            holding_days=hold,
                                            start_date=start,
                                            end_date=end,
                                            min_pred_prob=min_pred,
                                            max_atr_ratio=max_atr,
                                            sell_col=_sell_col_for_holding_days(hold),
                                            slippage_rate=args.slippage,
                                            liquidity_slippage_enabled=args.liquidity_slippage,
                                            min_amount=min_amount,
                                            min_turnover_rate=min_turnover,
                                            max_total_mv=max_total_mv,
                                            score_exit_ratio=score_exit_ratio,
                                            min_score_exit_holding_days=args.min_score_exit_holding_days,
                                        )
                                        result = run_portfolio_backtest(base_rows, config)
                                        metrics = result["metrics"]
                                        results.append(
                                            {
                                                "slice": slice_name,
                                                "score": _score_metrics(metrics, args.objective),
                                                "table": table,
                                                "top_k": top_k,
                                                "holding_days": hold,
                                                "min_pred_prob": min_pred,
                                                "max_atr_ratio": max_atr,
                                                "score_exit_ratio": score_exit_ratio,
                                                "min_amount": min_amount,
                                                "min_turnover_rate": min_turnover,
                                                "max_total_mv": max_total_mv,
                                                **metrics,
                                            }
                                        )
    return results


def run_one_experiment(experiment: ModelExperiment, args) -> dict[str, Any]:
    output_table = f"{args.table_prefix}_{experiment.name}"
    db_path = Path(args.data_dir) / "odb.db"
    summary_output = Path(args.output_dir) / f"{experiment.name}_folds.csv"
    feature_cache_name = f"{_safe_name(experiment.label)}_{_safe_name(experiment.train_window)}_fs{experiment.feature_top_n}"
    feature_dir = Path(args.output_dir) / "feature_scores" / feature_cache_name
    if args.skip_existing and _table_exists(db_path, output_table):
        training_summary = [{"output_table": output_table, "skipped": "existing_table"}]
    else:
        windows = build_rolling_windows(
            data_start=args.data_start,
            first_test=args.first_test,
            final_test=args.final_test,
            train_years=experiment.train_years,
            test_months=args.test_months,
            step_months=args.step_months,
            embargo_days=args.embargo_days,
            train_mode=experiment.train_mode,
        )
        selector = build_fold_feature_selection_fn(
            data_file_url=args.data_dir,
            config=FoldFeatureSelectionConfig(
                label=experiment.label,
                top_n=experiment.feature_top_n,
                min_abs_ic=args.feature_min_abs_ic,
                max_missing_ratio=args.feature_max_missing_ratio,
                folds=args.feature_folds,
                score_output_dir=str(feature_dir),
            ),
            stock_pool_path=args.stock_pool,
            use_light_factor_data=args.use_light_factor_data,
        )
        with temporary_xgb_env(experiment.xgb_params):
            training_summary = execute_rolling_training(
                windows,
                data_file_url=args.data_dir,
                label=experiment.label,
                model_type=args.model_type,
                output_table=output_table,
                summary_output=summary_output,
                stock_pool_path=args.stock_pool,
                fold_feature_selector=selector,
                use_light_factor_data=args.use_light_factor_data,
            )
    backtest_rows = backtest_experiment_table(output_table, args)
    best_all = max((row for row in backtest_rows if row["slice"] == "ALL"), key=lambda row: row["score"], default={})
    write_rows(backtest_rows, Path(args.output_dir) / f"{experiment.name}_backtests.csv")
    return {
        "experiment": experiment.name,
        "label": experiment.label,
        "train_window": experiment.train_window,
        "train_mode": experiment.train_mode,
        "train_years": experiment.train_years,
        "feature_top_n": experiment.feature_top_n,
        "xgb_params": json.dumps(experiment.xgb_params, ensure_ascii=False, sort_keys=True),
        "output_table": output_table,
        "fold_count": len(training_summary),
        **{f"best_{key}": value for key, value in best_all.items() if key not in {"table"}},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run rolling model and trading-rule experiment grid.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data_file"))
    parser.add_argument("--stock-pool")
    parser.add_argument("--data-start", default="20100101")
    parser.add_argument("--first-test", default="20220606")
    parser.add_argument("--final-test", default="20260613")
    parser.add_argument("--labels", default="executable_1d_open_return,executable_3d_open_return,executable_5d_open_return,executable_10d_open_return")
    parser.add_argument("--train-windows", default="2y,3y,5y,expanding")
    parser.add_argument("--feature-top-ns", default="80,120,160,220")
    parser.add_argument("--xgb-grid", default='{"max_depth":[2,3,4],"reg_lambda":[1,3,5],"subsample":[0.8,1.0],"colsample_bytree":[0.8,1.0]}')
    parser.add_argument("--xgb-grid-file")
    parser.add_argument("--model-type", default="reg", choices=["reg", "class"])
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--feature-min-abs-ic", type=float, default=0.005)
    parser.add_argument("--feature-max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--feature-folds", type=int, default=8)
    parser.add_argument("--use-light-factor-data", action="store_true")
    parser.add_argument("--table-prefix", default="stock_predict_data_model_grid")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "data_file" / "reports" / "model_grid"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", default="1")
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--holding-days", default="1,3,5,10")
    parser.add_argument("--min-pred", default="none")
    parser.add_argument("--max-atr-ratio", default="none")
    parser.add_argument("--score-exit-ratio", default="none,0.8,1.0,1.2")
    parser.add_argument("--min-score-exit-holding-days", type=int, default=1)
    parser.add_argument("--min-amount", default="none,800000")
    parser.add_argument("--min-turnover", default="none,2")
    parser.add_argument("--max-total-mv", default="none,1200000")
    parser.add_argument("--slippage", type=float, default=0.0015)
    parser.add_argument("--liquidity-slippage", action="store_true")
    parser.add_argument("--objective", default="rank_return", choices=["annual", "sharpe", "rank_return"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spec = ExperimentSpec(
        labels=_parse_csv(args.labels),
        train_windows=_parse_csv(args.train_windows),
        feature_top_ns=_parse_int_csv(args.feature_top_ns),
        xgb_param_sets=_load_xgb_grid(args.xgb_grid, args.xgb_grid_file),
    )
    experiments = expand_experiments(spec)
    if args.limit is not None:
        experiments = experiments[: args.limit]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_rows([experiment.__dict__ for experiment in experiments], Path(args.output_dir) / "experiment_plan.csv")
    if not args.execute:
        for experiment in experiments:
            print(experiment)
        print(f"experiment_count: {len(experiments)}")
        print(f"plan_csv: {Path(args.output_dir) / 'experiment_plan.csv'}")
        return experiments

    summaries = []
    for experiment in experiments:
        print(f"RUN {experiment.name}")
        summaries.append(run_one_experiment(experiment, args))
        write_rows(summaries, Path(args.output_dir) / "experiment_summary.csv")
    summaries.sort(key=lambda row: float(row.get("best_score", 0.0) or 0.0), reverse=True)
    write_rows(summaries, Path(args.output_dir) / "experiment_summary_sorted.csv")
    print(json.dumps(summaries[:10], ensure_ascii=False, indent=2))
    return summaries


if __name__ == "__main__":
    main()
