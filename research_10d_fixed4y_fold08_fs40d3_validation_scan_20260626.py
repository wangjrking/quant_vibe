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
FS40_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_10d_fixed4y_fold08_featurecount_scan_20260626"
    / "selected_features"
    / "selected_features_top40.json"
)
BASE_PREDICTION = (
    DATA_DIR
    / "reports"
    / "model_agent_research_10d_fixed4y_fold08_fs40_structure_scan_20260626"
    / "fs40_d3_l8"
    / "fold08"
    / "fold_predictions"
    / "fold08.parquet"
)
BASE_MODEL = (
    DATA_DIR
    / "reports"
    / "model_agent_research_10d_fixed4y_fold08_fs40_structure_scan_20260626"
    / "fs40_d3_l8"
    / "fold08"
    / "models"
    / "model_fold08.json"
)
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_10d_fixed4y_fold08_fs40d3_validation_scan_20260626"
LABEL = "executable_10d_open_return"
BASELINE_TABLES = {
    "research10d_current": "stock_predict_data_model_agent_10d_risk_balanced_v5_shrink_20260625_executable_10d_open_return_research",
    "formal10d_current": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    eval_k: int
    validation_tail_days: int
    subsample: float
    colsample_bytree: float
    retrain: bool


CANDIDATES = [
    Candidate("base_fs40_d3_l8", 5, 126, 1.0, 1.0, False),
    Candidate("k10_t126", 10, 126, 1.0, 1.0, True),
    Candidate("k10_t84", 10, 84, 1.0, 1.0, True),
    Candidate("k10_t84_ss08_cs08", 10, 84, 0.8, 0.8, True),
    Candidate("k20_t84_ss08_cs08", 20, 84, 0.8, 0.8, True),
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
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def evaluate_frame(frame: pd.DataFrame, labels: pd.DataFrame) -> dict:
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
    return summary


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


def objective(summary: dict) -> float:
    return (
        1.8 * summary["top1"]
        + 1.1 * summary["top3"]
        + 0.9 * summary["top5"]
        + 0.35 * summary["top10"]
        + 0.45 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
    )


def out_dir(candidate: Candidate) -> Path:
    return REPORT_DIR / candidate.name


def prediction_path(candidate: Candidate) -> Path:
    if candidate.name == "base_fs40_d3_l8":
        return BASE_PREDICTION
    return out_dir(candidate) / "fold08" / "fold_predictions" / "fold08.parquet"


def model_path(candidate: Candidate) -> Path:
    if candidate.name == "base_fs40_d3_l8":
        return BASE_MODEL
    return out_dir(candidate) / "fold08" / "models" / "model_fold08.json"


def run_candidate(candidate: Candidate) -> dict:
    if not candidate.retrain:
        return {"stdout": "reused existing fs40_d3_l8 artifact", "stderr": "", "returncode": 0, "reused": True}
    if prediction_path(candidate).exists() and model_path(candidate).exists():
        return {"stdout": "skipped existing artifact", "stderr": "", "returncode": 0, "skipped_existing": True}

    env = os.environ.copy()
    env.update(
        {
            "XGB_REG_EVAL_METRIC": "top_return_loss",
            "XGB_TOP_RETURN_EVAL_K": str(candidate.eval_k),
            "XGB_VALIDATION_MODE": "train_tail_days",
            "XGB_VALIDATION_TAIL_DAYS": str(candidate.validation_tail_days),
            "XGB_SAMPLE_WEIGHT_MODE": "daily_top_quantile",
            "XGB_SAMPLE_WEIGHT_TOP_PCT": "0.005",
            "XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER": "8.0",
            "XGB_DEVICE": "cpu",
            "XGB_N_JOBS": "8",
            "XGB_N_ESTIMATORS": "6000",
            "XGB_LEARNING_RATE": "0.003",
            "XGB_MAX_DEPTH": "3",
            "XGB_REG_LAMBDA": "8.0",
            "XGB_REG_ALPHA": "0.1",
            "XGB_SUBSAMPLE": str(candidate.subsample),
            "XGB_COLSAMPLE_BYTREE": str(candidate.colsample_bytree),
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
        os.path.relpath(FS40_PATH, MAIN_DIR),
        "--output-table",
        f"stock_predict_data_model_agent_{candidate.name}_20260626",
        "--prediction-output-path",
        os.path.relpath(prediction_path(candidate), MAIN_DIR),
        "--summary-output",
        os.path.relpath(out_dir(candidate) / "fold08" / "fold_results.csv", MAIN_DIR),
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
    rows = []
    logs = {}
    cached_baselines: dict[str, dict] = {}

    for candidate in CANDIDATES:
        logs[candidate.name] = run_candidate(candidate)
        pred = pd.read_parquet(prediction_path(candidate), columns=["trade_date", "stock_code", "pred_prob"])
        pred["trade_date"] = pred["trade_date"].astype(str)
        pred["stock_code"] = pred["stock_code"].astype(str)
        start = str(pred["trade_date"].min())
        end = str(pred["trade_date"].max())
        summary = evaluate_frame(pred, labels)
        summary.update(
            {
                "asset": candidate.name,
                "date_min": start,
                "date_max": end,
                "objective": objective(summary),
                "prediction_path": str(prediction_path(candidate)),
                "model_path": str(model_path(candidate)),
                "eval_k": candidate.eval_k,
                "validation_tail_days": candidate.validation_tail_days,
                "subsample": candidate.subsample,
                "colsample_bytree": candidate.colsample_bytree,
            }
        )
        rows.append(summary)
        for baseline_name, table in BASELINE_TABLES.items():
            if baseline_name not in cached_baselines:
                baseline = load_baseline_table(table, start, end)
                base_summary = evaluate_frame(baseline, labels)
                base_summary.update(
                    {"asset": baseline_name, "date_min": start, "date_max": end, "objective": objective(base_summary)}
                )
                cached_baselines[baseline_name] = base_summary
                rows.append(base_summary)

    summary_df = pd.DataFrame(rows).sort_values(["objective", "top1", "rank_ic"], ascending=False).reset_index(drop=True)
    delta_rows = []
    for candidate in CANDIDATES:
        cand = next(row for row in rows if row["asset"] == candidate.name)
        for baseline_name in ["research10d_current", "formal10d_current", "base_fs40_d3_l8"]:
            if baseline_name == "base_fs40_d3_l8":
                base = next(row for row in rows if row["asset"] == "base_fs40_d3_l8")
            else:
                base = cached_baselines[baseline_name]
            delta = {"asset": candidate.name, "baseline": baseline_name}
            for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom", "objective"]:
                delta[f"delta_{metric}"] = cand[metric] - base[metric]
            delta_rows.append(delta)
    delta_df = pd.DataFrame(delta_rows)

    summary_path = REPORT_DIR / "validation_scan_summary.csv"
    delta_path = REPORT_DIR / "validation_scan_deltas.csv"
    logs_path = REPORT_DIR / "validation_scan_run_logs.json"
    packet_path = REPORT_DIR / "validation_scan_packet.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
    logs_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    packet = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_fixed4y_fold08_fs40d3_validation_scan",
        "label": LABEL,
        "selected_features_path": str(FS40_PATH),
        "report_dir": str(REPORT_DIR),
        "summary_csv": str(summary_path),
        "delta_csv": str(delta_path),
        "candidate_count": len(CANDIDATES),
        "candidates": [candidate.__dict__ for candidate in CANDIDATES],
        "baselines": {**BASELINE_TABLES, "base_fs40_d3_l8": str(BASE_PREDICTION)},
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")

    best = summary_df.iloc[0]
    lines = [
        "# 10D fs40_d3_l8 验证机制扫描结论",
        "",
        "## 实验目的",
        "",
        "在 `fs40_d3_l8` 已经成为当前最优近四年训练候选的基础上，继续验证验证指标与验证尾窗长度的调整是否能带来新的真实增益。",
        "",
        "本轮固定训练底座不变，只扫描：",
        "- `top_return_loss` 的 `k`",
        "- `train_tail_days` 的尾窗长度",
        "- 一档轻量 `subsample / colsample_bytree`",
        "",
        "## 结果摘要",
        "",
        f"- 当前排序第一：`{best['asset']}`",
        f"- `objective = {best['objective']:.6f}`",
        f"- `top1 = {best['top1']:.6f}`",
        f"- `top5 = {best['top5']:.6f}`",
        f"- `top10 = {best['top10']:.6f}`",
        f"- `rank_ic = {best['rank_ic']:.6f}`",
        "",
        "## 产物",
        "",
        "- 汇总表：`validation_scan_summary.csv`",
        "- 差异表：`validation_scan_deltas.csv`",
        "- 运行日志：`validation_scan_run_logs.json`",
        "- 实验清单：`validation_scan_packet.json`",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
