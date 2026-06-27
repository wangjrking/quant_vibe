"""Resume the 2010-expanding label tuning grid one experiment at a time."""

from __future__ import annotations

import ast
import json
import sqlite3
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from model_experiment_grid import ModelExperiment, run_one_experiment, write_rows


DATA_DIR = r"D:\work\quant\quant_mcp\quant\data_file"
OUTPUT_DIR = Path(r"D:\work\quant\quant_mcp\quant\data_file\reports\label_tune_fast800_expanding2010_qtr_20260613")
PLAN_CSV = OUTPUT_DIR / "experiment_plan.csv"
SUMMARY_CSV = OUTPUT_DIR / "experiment_summary.csv"
SUMMARY_SORTED_CSV = OUTPUT_DIR / "experiment_summary_sorted.csv"
STATUS_JSON = OUTPUT_DIR / "resume_status.json"
TABLE_PREFIX = "stock_predict_data_label_tune_fast800_expanding2010_qtr"


def _load_completed_names() -> set[str]:
    if not SUMMARY_CSV.exists():
        return set()
    frame = pd.read_csv(SUMMARY_CSV)
    if "experiment" not in frame.columns:
        return set()
    return set(frame["experiment"].dropna().astype(str))


def _load_summary_rows() -> list[dict]:
    if not SUMMARY_CSV.exists():
        return []
    return pd.read_csv(SUMMARY_CSV).to_dict("records")


def _experiment_from_row(row) -> ModelExperiment:
    return ModelExperiment(
        name=str(row["name"]),
        label=str(row["label"]),
        train_window=str(row["train_window"]),
        train_years=int(row["train_years"]),
        train_mode=str(row["train_mode"]),
        feature_top_n=int(row["feature_top_n"]),
        xgb_params=dict(ast.literal_eval(str(row["xgb_params"]))),
    )


def _table_name(experiment: ModelExperiment) -> str:
    return f"{TABLE_PREFIX}_{experiment.name}"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone() is not None


def _drop_table_if_exists(table: str) -> None:
    db_path = Path(DATA_DIR) / "odb.db"
    conn = sqlite3.connect(db_path)
    try:
        if _table_exists(conn, table):
            conn.execute(f'drop table "{table}"')
            conn.commit()
    finally:
        conn.close()


def _write_status(status: dict) -> None:
    STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_args() -> Namespace:
    return Namespace(
        data_dir=DATA_DIR,
        stock_pool=None,
        data_start="20100101",
        first_test="20240605",
        final_test="20260613",
        test_months=3,
        step_months=3,
        embargo_days=10,
        feature_min_abs_ic=0.005,
        feature_max_missing_ratio=0.35,
        feature_folds=8,
        use_light_factor_data=True,
        model_type="reg",
        table_prefix=TABLE_PREFIX,
        output_dir=str(OUTPUT_DIR),
        skip_existing=False,
        top_k="1",
        max_positions=1,
        holding_days="1,3,5,10",
        min_pred="none",
        max_atr_ratio="none",
        score_exit_ratio="none,0.8,1.0,1.2",
        min_score_exit_holding_days=1,
        min_amount="none,800000",
        min_turnover="none,2",
        max_total_mv="none,1200000",
        slippage=0.0015,
        liquidity_slippage=False,
        objective="rank_return",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(PLAN_CSV)
    experiments = [_experiment_from_row(row) for _, row in plan.iterrows()]
    completed = _load_completed_names()
    summary_rows = _load_summary_rows()
    args = _build_args()

    for index, experiment in enumerate(experiments, start=1):
        status = {
            "total": len(experiments),
            "index": index,
            "experiment": experiment.name,
            "completed_count": len(completed),
            "state": "skipped" if experiment.name in completed else "running",
        }
        _write_status(status)
        if experiment.name in completed:
            print(f"SKIP {index}/{len(experiments)} {experiment.name}", flush=True)
            continue

        table = _table_name(experiment)
        print(f"RUN {index}/{len(experiments)} {experiment.name}", flush=True)
        _drop_table_if_exists(table)
        row = run_one_experiment(experiment, args)
        summary_rows = [item for item in summary_rows if item.get("experiment") != experiment.name]
        summary_rows.append(row)
        write_rows(summary_rows, SUMMARY_CSV)
        sorted_rows = sorted(summary_rows, key=lambda item: float(item.get("best_score", 0.0) or 0.0), reverse=True)
        write_rows(sorted_rows, SUMMARY_SORTED_CSV)
        completed.add(experiment.name)
        _write_status(
            {
                **status,
                "completed_count": len(completed),
                "state": "completed",
                "best_annualized_return": row.get("best_annualized_return"),
                "best_sharpe": row.get("best_sharpe"),
                "best_max_drawdown": row.get("best_max_drawdown"),
            }
        )
        print(f"DONE {index}/{len(experiments)} {experiment.name}", flush=True)

    _write_status({"total": len(experiments), "completed_count": len(completed), "state": "all_completed"})
    print("ALL_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
