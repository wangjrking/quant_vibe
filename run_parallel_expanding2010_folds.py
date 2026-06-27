from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from model_asset_route import (
    MODEL_FEATURE_MODE_LEGACY,
    MODEL_FEATURE_MODE_SPLIT,
    MODEL_PREDICTION_MODE_INDEPENDENT,
    MODEL_PREDICTION_MODE_LEGACY,
    enrich_research_prediction_manifest,
    require_legacy_model_asset_chain_opt_in,
    resolve_legacy_prediction_db_path,
    resolve_model_prediction_db_path,
    use_legacy_prediction_db,
    write_prediction_manifest,
)
from rolling_train_module import build_rolling_windows


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _table_exists(db_path: Path, table: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        return bool(conn.execute("select count(*) from sqlite_master where type='table' and name=?", (table,)).fetchone()[0])


def _table_meta(db_path: Path, table: str) -> dict[str, Any]:
    if not _table_exists(db_path, table):
        return {"exists": False, "rows": 0}
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), count(distinct stock_code) from {_quote(table)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select stock_code, trade_date, count(*) c from {_quote(table)} group by 1,2 having c>1)"
        ).fetchone()[0]
    return {
        "exists": True,
        "rows": int(row[0] or 0),
        "min_trade_date": row[1],
        "max_trade_date": row[2],
        "trade_dates": int(row[3] or 0),
        "stocks": int(row[4] or 0),
        "duplicate_keys": int(dup or 0),
    }


def _prediction_file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0}
    frame = pd.read_parquet(path, columns=["stock_code", "trade_date"])
    if frame.empty:
        return {"exists": True, "rows": 0}
    dup = int(frame.groupby(["stock_code", "trade_date"], sort=False).size().gt(1).sum())
    return {
        "exists": True,
        "rows": int(len(frame)),
        "min_trade_date": str(frame["trade_date"].min()),
        "max_trade_date": str(frame["trade_date"].max()),
        "trade_dates": int(frame["trade_date"].nunique()),
        "stocks": int(frame["stock_code"].nunique()),
        "duplicate_keys": dup,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _merge_result_rows(existing_rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in existing_rows:
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError):
            continue
        merged[fold] = row
    merged[int(new_row["fold"])] = new_row
    return [merged[fold] for fold in sorted(merged)]


def _run_fold(args, window) -> dict[str, Any]:
    fold_table = f"{args.output_table}__fold{window.fold:02d}"
    prediction_path = Path(args.output_dir) / "fold_predictions" / f"fold{window.fold:02d}.parquet"
    meta = _prediction_file_meta(prediction_path)
    if args.skip_existing and meta.get("exists") and meta.get("rows", 0) > 0 and meta.get("max_trade_date") == window.test_end:
        return {
            "fold": window.fold,
            "fold_table": fold_table,
            "prediction_path": str(prediction_path),
            "status": "skipped_existing",
            **meta,
        }

    summary_path = Path(args.output_dir) / "fold_summaries" / f"fold{window.fold:02d}.csv"
    log_path = Path(args.output_dir) / "fold_logs" / f"fold{window.fold:02d}.log"
    feature_dir = Path(args.output_dir) / "feature_scores" / args.experiment_name
    command = [
        sys.executable,
        str(Path(__file__).with_name("rolling_train_module.py")),
        "--data-file-url",
        str(args.data_file_url),
        "--data-start",
        args.data_start,
        "--first-test",
        args.first_test,
        "--final-test",
        args.final_test,
        "--label",
        args.label,
        "--model-type",
        args.model_type,
        "--train-years",
        str(args.train_years),
        "--test-months",
        str(args.test_months),
        "--step-months",
        str(args.step_months),
        "--embargo-days",
        str(args.embargo_days),
        "--train-mode",
        str(args.train_mode),
        "--output-table",
        fold_table,
        "--prediction-output-path",
        str(prediction_path),
        "--summary-output",
        str(summary_path),
        "--start-fold",
        str(window.fold),
        "--end-fold",
        str(window.fold),
        "--execute",
    ]
    if args.use_light_factor_data:
        command.append("--use-light-factor-data")
    if args.feature_source:
        command.extend(["--feature-source", str(args.feature_source)])
    if args.prediction_output_mode:
        command.extend(["--prediction-output-mode", str(args.prediction_output_mode)])
    if args.selected_features_path:
        command.extend(["--selected-features-path", str(args.selected_features_path)])
    else:
        command.extend(
            [
                "--fold-feature-selection",
                "--fold-feature-top-n",
                str(args.fold_feature_top_n),
                "--fold-feature-min-abs-ic",
                str(args.fold_feature_min_abs_ic),
                "--fold-feature-max-missing-ratio",
                str(args.fold_feature_max_missing_ratio),
                "--fold-feature-folds",
                str(args.fold_feature_folds),
                "--fold-feature-output-dir",
                str(feature_dir),
            ]
        )
        if args.fold_feature_exclude_prefixes:
            command.extend(["--fold-feature-exclude-prefixes", str(args.fold_feature_exclude_prefixes)])
    env = os.environ.copy()
    for key, value in args.xgb_env.items():
        if value is not None:
            env[key] = str(value)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent), env=env, text=True)
    meta = _prediction_file_meta(prediction_path)
    return {
        "fold": window.fold,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "fold_table": fold_table,
        "prediction_path": str(prediction_path),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "log_path": str(log_path),
        **meta,
    }


def _merge_tables(db_path: Path, output_table: str, fold_tables: list[str]) -> dict[str, Any]:
    available = [table for table in fold_tables if _table_exists(db_path, table)]
    if not available:
        raise RuntimeError("No fold tables are available to merge.")
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"drop table if exists {_quote(output_table)}")
        conn.execute(f"create table {_quote(output_table)} as select * from {_quote(available[0])} where 0")
        for table in available:
            conn.execute(f"insert into {_quote(output_table)} select * from {_quote(table)}")
        conn.commit()
    return _table_meta(db_path, output_table)


def _merge_prediction_files(db_path: Path, output_table: str, prediction_paths: list[Path]) -> dict[str, Any]:
    available = [path for path in prediction_paths if path.exists()]
    if not available:
        raise RuntimeError("No fold prediction files are available to merge.")
    with sqlite3.connect(db_path, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {_quote(output_table)}")
        first = True
        for path in available:
            frame = pd.read_parquet(path)
            frame.to_sql(output_table, con=conn, if_exists="replace" if first else "append", index=False)
            first = False
        conn.commit()
    return _table_meta(db_path, output_table)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run expanding2010 rolling folds in parallel and merge prediction tables.")
    parser.add_argument("--data-file-url", required=True)
    parser.add_argument("--data-start", default="20100101")
    parser.add_argument("--first-test", default="20240604")
    parser.add_argument("--final-test", default="20260612")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-type", choices=["reg", "class"], required=True)
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--train-mode", choices=["fixed", "expanding"], default="expanding")
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--fold-feature-top-n", type=int, default=80)
    parser.add_argument("--fold-feature-min-abs-ic", type=float, default=0.003)
    parser.add_argument("--fold-feature-max-missing-ratio", type=float, default=0.45)
    parser.add_argument("--fold-feature-folds", type=int, default=8)
    parser.add_argument("--fold-feature-exclude-prefixes", default="")
    parser.add_argument("--selected-features-path")
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--xgb-n-estimators")
    parser.add_argument("--xgb-learning-rate")
    parser.add_argument("--xgb-max-depth")
    parser.add_argument("--xgb-reg-lambda")
    parser.add_argument("--xgb-reg-alpha")
    parser.add_argument("--xgb-subsample")
    parser.add_argument("--xgb-colsample-bytree")
    parser.add_argument("--xgb-early-stopping-rounds")
    parser.add_argument("--xgb-device", default="cuda")
    parser.add_argument("--xgb-n-jobs", default="0")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
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
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.xgb_env = {
        "XGB_N_ESTIMATORS": args.xgb_n_estimators,
        "XGB_LEARNING_RATE": args.xgb_learning_rate,
        "XGB_MAX_DEPTH": args.xgb_max_depth,
        "XGB_REG_LAMBDA": args.xgb_reg_lambda,
        "XGB_REG_ALPHA": args.xgb_reg_alpha,
        "XGB_SUBSAMPLE": args.xgb_subsample,
        "XGB_COLSAMPLE_BYTREE": args.xgb_colsample_bytree,
        "XGB_EARLY_STOPPING_ROUNDS": args.xgb_early_stopping_rounds,
        "XGB_DEVICE": args.xgb_device,
        "XGB_N_JOBS": args.xgb_n_jobs,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    windows = [w for w in windows if w.fold >= args.start_fold and (args.end_fold is None or w.fold <= args.end_fold)]
    plan_rows = [w.to_dict() for w in windows]
    _write_rows(output_dir / "fold_plan.csv", plan_rows)
    if use_legacy_prediction_db(args.prediction_output_mode):
        require_legacy_model_asset_chain_opt_in(reason="legacy odb expanding2010 merged prediction output")
        db_path = resolve_legacy_prediction_db_path(args.data_file_url)
    else:
        db_path = resolve_model_prediction_db_path(args.data_file_url, create_parent=True)
    fold_tables = [f"{args.output_table}__fold{w.fold:02d}" for w in windows]
    prediction_paths = [output_dir / "fold_predictions" / f"fold{w.fold:02d}.parquet" for w in windows]

    fold_results_path = output_dir / "fold_results.csv"
    results: list[dict[str, Any]] = _read_rows(fold_results_path)
    if args.merge_only and not any(path.exists() for path in prediction_paths):
        print(json.dumps({"status": "planned_only", "folds": len(windows), "plan_csv": str(output_dir / "fold_plan.csv")}, ensure_ascii=False))
        return 0

    if not args.merge_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {executor.submit(_run_fold, args, window): window for window in windows}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                results = _merge_result_rows(results, row)
                _write_rows(fold_results_path, results)
                print(json.dumps(row, ensure_ascii=False))
                if row.get("status") == "failed":
                    raise RuntimeError(f"fold {row.get('fold')} failed; see {row.get('log_path')}")

    if args.no_merge:
        payload = {
            "status": "prediction_files_ready",
            "output_table": args.output_table,
            "prediction_files": [str(path) for path in prediction_paths if path.exists()],
        }
        (output_dir / "merge_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    merge_meta = _merge_prediction_files(db_path, args.output_table, prediction_paths)
    payload = {
        "output_table": args.output_table,
        "prediction_db_path": str(db_path),
        "prediction_mode": args.prediction_output_mode or MODEL_PREDICTION_MODE_INDEPENDENT,
        **merge_meta,
    }
    (output_dir / "merge_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_prediction_manifest(output_dir / "prediction_manifest.json", enrich_research_prediction_manifest(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
