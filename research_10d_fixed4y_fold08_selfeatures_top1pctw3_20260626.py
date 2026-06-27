from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
ROLLING_TRAIN = MAIN_DIR / "rolling_train_module.py"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
FEATURES_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_10d_tune_20260620"
    / "xgb_reg10d_d4_l4_fs160_lr003_n6000_topk_gate2_20260620"
    / "feature_scores"
    / "xgb_reg10d_d4_l4_fs160_lr003_n6000_topk_gate2_20260620"
    / "selected_features_executable_10d_open_return_rolling_fold8.json"
)
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_10d_fixed4y_fold08_selfeatures_top1pctw3_20260626"
LABEL = "executable_10d_open_return"
TARGET_NAME = "self_top1pctw3"
BASELINE_TABLES = {
    "research10d_current": {
        "kind": "sqlite_table",
        "value": "stock_predict_data_model_agent_10d_risk_balanced_v5_shrink_20260625_executable_10d_open_return_research",
    },
    "formal10d_current": {
        "kind": "sqlite_table",
        "value": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
    },
    "fixed4y_self_ref": {
        "kind": "parquet",
        "value": str(
            DATA_DIR
            / "reports"
            / "model_agent_research_10d_fixed4y_fold08_selfeatures_scan_20260626"
            / "self_ref"
            / "fold08"
            / "fold_predictions"
            / "fold08.parquet"
        ),
    },
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_labels() -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def evaluate_frame(frame: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    merged = frame.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    rows = []
    for trade_date, group in merged.groupby("trade_date", sort=True):
        rank = group["pred_prob"].rank(method="average", pct=True)
        y_rank = group[LABEL].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "rank_ic": float(rank.corr(y_rank)),
            "pearson_ic": float(group["pred_prob"].corr(group[LABEL])),
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = float(group.nlargest(min(n, len(group)), "pred_prob")[LABEL].mean())
        row["top_bottom"] = float(
            group.nlargest(min(50, len(group)), "pred_prob")[LABEL].mean()
            - group.nsmallest(min(50, len(group)), "pred_prob")[LABEL].mean()
        )
        rows.append(row)
    daily = pd.DataFrame(rows)
    summary = {
        "trade_days": int(len(daily)),
        "rank_ic": float(daily["rank_ic"].mean()),
        "rank_ic_positive_ratio": float((daily["rank_ic"] > 0).mean()),
        "pearson_ic": float(daily["pearson_ic"].mean()),
        "top_bottom": float(daily["top_bottom"].mean()),
    }
    for n in [1, 3, 5, 10, 20, 50]:
        summary[f"top{n}"] = float(daily[f"top{n}"].mean())
    return summary, daily


def load_asset(asset: dict[str, str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if asset["kind"] == "sqlite_table":
        table = asset["value"]
        query = f"select trade_date, stock_code, pred_prob from '{table}'"
        params: list[str] = []
        if start and end:
            query += " where trade_date >= ? and trade_date <= ?"
            params.extend([start, end])
        query += " order by trade_date, stock_code"
        with sqlite3.connect(MODEL_DB) as conn:
            frame = pd.read_sql_query(query, conn, params=params)
    elif asset["kind"] == "parquet":
        frame = pd.read_parquet(asset["value"], columns=["trade_date", "stock_code", "pred_prob"])
        if start and end:
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)]
    else:
        raise ValueError(f"unsupported asset kind: {asset['kind']}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def objective(summary: dict) -> float:
    return (
        1.8 * summary["top1"]
        + 1.1 * summary["top3"]
        + 0.9 * summary["top5"]
        + 0.35 * summary["top10"]
        + 0.45 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
    )


def prediction_path() -> Path:
    return REPORT_DIR / "fold08" / "fold_predictions" / "fold08.parquet"


def model_path() -> Path:
    return REPORT_DIR / "fold08" / "models" / "model_fold08.json"


def run_candidate() -> dict:
    env = os.environ.copy()
    env.update(
        {
            "XGB_REG_EVAL_METRIC": "top_return_loss",
            "XGB_TOP_RETURN_EVAL_K": "5",
            "XGB_VALIDATION_MODE": "train_tail_days",
            "XGB_VALIDATION_TAIL_DAYS": "126",
            "XGB_SAMPLE_WEIGHT_MODE": "daily_top_quantile",
            "XGB_SAMPLE_WEIGHT_TOP_PCT": "0.01",
            "XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER": "3.0",
            "XGB_DEVICE": "cpu",
            "XGB_N_JOBS": "8",
            "XGB_N_ESTIMATORS": "6000",
            "XGB_LEARNING_RATE": "0.003",
            "XGB_MAX_DEPTH": "4",
            "XGB_REG_LAMBDA": "4.0",
            "XGB_REG_ALPHA": "0.0",
            "XGB_EARLY_STOPPING_ROUNDS": "300",
        }
    )
    command = [
        str(PYTHON),
        str(ROLLING_TRAIN),
        "--data-file-url",
        "../data_file",
        "--data-start",
        "20100101",
        "--first-test",
        "20240604",
        "--final-test",
        "20260624",
        "--label",
        LABEL,
        "--model-type",
        "reg",
        "--train-years",
        "4",
        "--test-months",
        "3",
        "--step-months",
        "3",
        "--embargo-days",
        "10",
        "--train-mode",
        "fixed",
        "--selected-features-path",
        os.path.relpath(FEATURES_PATH, MAIN_DIR),
        "--output-table",
        f"stock_predict_data_model_agent_{TARGET_NAME}_fold08_20260626",
        "--prediction-output-path",
        os.path.relpath(prediction_path(), MAIN_DIR),
        "--summary-output",
        os.path.relpath(REPORT_DIR / "fold08" / "fold_results.csv", MAIN_DIR),
        "--start-fold",
        "8",
        "--end-fold",
        "8",
        "--execute",
    ]
    completed = subprocess.run(command, cwd=MAIN_DIR, env=env, capture_output=True, text=True, check=True)
    return {"stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    logs = run_candidate()
    pred = pd.read_parquet(prediction_path(), columns=["trade_date", "stock_code", "pred_prob"])
    pred["trade_date"] = pred["trade_date"].astype(str)
    pred["stock_code"] = pred["stock_code"].astype(str)
    start = str(pred["trade_date"].min())
    end = str(pred["trade_date"].max())

    rows = []
    target_summary, _ = evaluate_frame(pred, labels)
    target_summary.update(
        {
            "asset": TARGET_NAME,
            "date_min": start,
            "date_max": end,
            "objective": objective(target_summary),
            "prediction_path": str(prediction_path()),
            "model_path": str(model_path()),
        }
    )
    rows.append(target_summary)

    for baseline_name, asset in BASELINE_TABLES.items():
        frame = load_asset(asset, start, end)
        summary, _ = evaluate_frame(frame, labels)
        summary.update({"asset": baseline_name, "date_min": start, "date_max": end, "objective": objective(summary)})
        rows.append(summary)

    summary_df = pd.DataFrame(rows).sort_values(["objective", "top1", "rank_ic"], ascending=False).reset_index(drop=True)
    delta_rows = []
    target = next(row for row in rows if row["asset"] == TARGET_NAME)
    for baseline_name in BASELINE_TABLES:
        base = next(row for row in rows if row["asset"] == baseline_name)
        delta = {"asset": TARGET_NAME, "baseline": baseline_name}
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom", "objective"]:
            delta[f"delta_{metric}"] = target[metric] - base[metric]
        delta_rows.append(delta)
    delta_df = pd.DataFrame(delta_rows)

    summary_path = REPORT_DIR / "top1pctw3_summary.csv"
    delta_path = REPORT_DIR / "top1pctw3_deltas.csv"
    packet_path = REPORT_DIR / "top1pctw3_packet.json"
    logs_path = REPORT_DIR / "top1pctw3_run_logs.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
    logs_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    packet = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_fixed4y_fold08_selfeatures_top1pctw3",
        "label": LABEL,
        "features_path": str(FEATURES_PATH),
        "report_dir": str(REPORT_DIR),
        "summary_csv": str(summary_path),
        "delta_csv": str(delta_path),
        "baselines": BASELINE_TABLES,
    }
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
