from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
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
    / "model_agent_research_3d_fixed2y_fold08_20260623"
    / "feature_scores"
    / "selected_features_executable_3d_open_return_rolling_fold8.json"
)
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_3d_fixed4y_fold09_targeted_scan_20260626"

LABEL = "executable_3d_open_return"
BASELINE_TABLES = {
    "research3d_current": "stock_predict_data_model_agent_3d_rankic_balanced_v5_20260625_executable_3d_open_return_research",
    "formal3d_current": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    eval_k: int
    validation_tail_days: int
    top_pct: float
    top_multiplier: float
    n_estimators: int
    learning_rate: float
    max_depth: int
    reg_lambda: float
    reg_alpha: float


CANDIDATES = [
    Candidate(
        name="ref_v2_k5_pct005_m10_l12a02",
        eval_k=5,
        validation_tail_days=126,
        top_pct=0.005,
        top_multiplier=10.0,
        n_estimators=3000,
        learning_rate=0.006,
        max_depth=2,
        reg_lambda=12.0,
        reg_alpha=0.2,
    ),
    Candidate(
        name="k5_pct005_m8_l12a02",
        eval_k=5,
        validation_tail_days=126,
        top_pct=0.005,
        top_multiplier=8.0,
        n_estimators=3000,
        learning_rate=0.006,
        max_depth=2,
        reg_lambda=12.0,
        reg_alpha=0.2,
    ),
    Candidate(
        name="k5_pct01_m6_l12a02",
        eval_k=5,
        validation_tail_days=126,
        top_pct=0.01,
        top_multiplier=6.0,
        n_estimators=3000,
        learning_rate=0.006,
        max_depth=2,
        reg_lambda=12.0,
        reg_alpha=0.2,
    ),
    Candidate(
        name="k3_pct01_m6_l12a02",
        eval_k=3,
        validation_tail_days=126,
        top_pct=0.01,
        top_multiplier=6.0,
        n_estimators=3000,
        learning_rate=0.006,
        max_depth=2,
        reg_lambda=12.0,
        reg_alpha=0.2,
    ),
    Candidate(
        name="k5_pct005_m10_l16a03",
        eval_k=5,
        validation_tail_days=126,
        top_pct=0.005,
        top_multiplier=10.0,
        n_estimators=3200,
        learning_rate=0.006,
        max_depth=2,
        reg_lambda=16.0,
        reg_alpha=0.3,
    ),
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_labels() -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        chunks.append(part)
    labels = pd.concat(chunks, ignore_index=True)
    return labels.dropna(subset=[LABEL])


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


def load_baseline_table(table: str, start: str, end: str) -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from '{table}' where trade_date >= ? and trade_date <= ? order by trade_date, stock_code",
            conn,
            params=(start, end),
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def candidate_output_dir(candidate: Candidate) -> Path:
    return REPORT_DIR / candidate.name


def candidate_prediction_path(candidate: Candidate) -> Path:
    return candidate_output_dir(candidate) / "fold09" / "fold_predictions" / "fold09.parquet"


def candidate_summary_path(candidate: Candidate) -> Path:
    return candidate_output_dir(candidate) / "fold09" / "fold_results.csv"


def run_candidate(candidate: Candidate) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "XGB_REG_EVAL_METRIC": "top_return_loss",
            "XGB_TOP_RETURN_EVAL_K": str(candidate.eval_k),
            "XGB_VALIDATION_MODE": "train_tail_days",
            "XGB_VALIDATION_TAIL_DAYS": str(candidate.validation_tail_days),
            "XGB_SAMPLE_WEIGHT_MODE": "daily_top_quantile",
            "XGB_SAMPLE_WEIGHT_TOP_PCT": str(candidate.top_pct),
            "XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER": str(candidate.top_multiplier),
            "XGB_DEVICE": "cuda",
            "XGB_N_JOBS": "0",
            "XGB_N_ESTIMATORS": str(candidate.n_estimators),
            "XGB_LEARNING_RATE": str(candidate.learning_rate),
            "XGB_MAX_DEPTH": str(candidate.max_depth),
            "XGB_REG_LAMBDA": str(candidate.reg_lambda),
            "XGB_REG_ALPHA": str(candidate.reg_alpha),
            "XGB_EARLY_STOPPING_ROUNDS": "300",
        }
    )
    out_dir = candidate_output_dir(candidate)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "20260622",
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
        f"stock_predict_data_model_agent_{candidate.name}_fold09_20260626",
        "--prediction-output-path",
        os.path.relpath(candidate_prediction_path(candidate), MAIN_DIR),
        "--summary-output",
        os.path.relpath(candidate_summary_path(candidate), MAIN_DIR),
        "--start-fold",
        "9",
        "--end-fold",
        "9",
        "--execute",
    ]
    completed = subprocess.run(
        command,
        cwd=MAIN_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def objective(summary: dict) -> float:
    return (
        1.6 * summary["top1"]
        + 1.0 * summary["top3"]
        + 0.8 * summary["top5"]
        + 0.3 * summary["top10"]
        + 0.5 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
    )


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    results = []
    daily_frames = []
    run_logs = {}

    for candidate in CANDIDATES:
        run_logs[candidate.name] = run_candidate(candidate)
        pred = pd.read_parquet(candidate_prediction_path(candidate), columns=["trade_date", "stock_code", "pred_prob"])
        pred["trade_date"] = pred["trade_date"].astype(str)
        pred["stock_code"] = pred["stock_code"].astype(str)
        start = str(pred["trade_date"].min())
        end = str(pred["trade_date"].max())
        summary, daily = evaluate_frame(pred, labels)
        summary.update(
            {
                "asset": candidate.name,
                "date_min": start,
                "date_max": end,
                "objective": objective(summary),
                "eval_k": candidate.eval_k,
                "validation_tail_days": candidate.validation_tail_days,
                "top_pct": candidate.top_pct,
                "top_multiplier": candidate.top_multiplier,
                "n_estimators": candidate.n_estimators,
                "learning_rate": candidate.learning_rate,
                "max_depth": candidate.max_depth,
                "reg_lambda": candidate.reg_lambda,
                "reg_alpha": candidate.reg_alpha,
                "prediction_path": str(candidate_prediction_path(candidate)),
                "model_path": str(candidate_output_dir(candidate) / "fold09" / "models" / "model_fold09.json"),
            }
        )
        results.append(summary)
        daily["asset"] = candidate.name
        daily_frames.append(daily)

        for baseline_name, table in BASELINE_TABLES.items():
            baseline = load_baseline_table(table, start, end)
            base_summary, base_daily = evaluate_frame(baseline, labels)
            base_row = base_summary.copy()
            base_row.update({"asset": baseline_name, "date_min": start, "date_max": end, "objective": objective(base_summary)})
            if not any(row["asset"] == baseline_name for row in results):
                results.append(base_row)
                base_daily["asset"] = baseline_name
                daily_frames.append(base_daily)

    result_df = pd.DataFrame(results).sort_values(["objective", "top1", "rank_ic"], ascending=False).reset_index(drop=True)
    daily_df = pd.concat(daily_frames, ignore_index=True)

    baseline_metrics = {row["asset"]: row for row in results if row["asset"] in BASELINE_TABLES}
    delta_rows = []
    for candidate in CANDIDATES:
        cand_row = next(row for row in results if row["asset"] == candidate.name)
        for baseline_name in BASELINE_TABLES:
            base_row = baseline_metrics[baseline_name]
            delta = {"asset": candidate.name, "baseline": baseline_name}
            for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom", "objective"]:
                delta[f"delta_{metric}"] = cand_row[metric] - base_row[metric]
            delta_rows.append(delta)
    delta_df = pd.DataFrame(delta_rows)

    result_df.to_csv(REPORT_DIR / "fold09_targeted_scan_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(REPORT_DIR / "fold09_targeted_scan_daily.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(REPORT_DIR / "fold09_targeted_scan_deltas.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "run_logs.json").write_text(json.dumps(run_logs, ensure_ascii=False, indent=2), encoding="utf-8")

    best = result_df.iloc[0].to_dict()
    packet = {
        "generated_at": now_iso(),
        "scope": "research_only_3d_fixed4y_fold09_targeted_scan",
        "summary_csv": str(REPORT_DIR / "fold09_targeted_scan_summary.csv"),
        "daily_csv": str(REPORT_DIR / "fold09_targeted_scan_daily.csv"),
        "delta_csv": str(REPORT_DIR / "fold09_targeted_scan_deltas.csv"),
        "run_logs": str(REPORT_DIR / "run_logs.json"),
        "best_candidate": best,
        "feature_source": str(FEATURES_PATH),
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "scan_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
