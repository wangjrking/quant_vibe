from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_1d_fixed4y_mixed_objective_smoke_20260627"
LABEL = "executable_1d_open_return"

FEATURES_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_1d_fixed4y_fold08_featurecount_scan_20260626"
    / "selected_features"
    / "selected_features_top40.json"
)

BASELINE_TABLES = {
    "research1d_current": "stock_predict_data_model_agent_1d_daygate_best_20260627_executable_1d_open_return_research",
    "formal1d_current": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
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
        name="mixed_top3_rankic_proxy_w005_m8_l12",
        eval_k=3,
        validation_tail_days=126,
        top_pct=0.005,
        top_multiplier=8.0,
        n_estimators=3200,
        learning_rate=0.006,
        max_depth=2,
        reg_lambda=12.0,
        reg_alpha=0.2,
    ),
    Candidate(
        name="mixed_top5_recent_w010_m6_l16",
        eval_k=5,
        validation_tail_days=84,
        top_pct=0.01,
        top_multiplier=6.0,
        n_estimators=3600,
        learning_rate=0.005,
        max_depth=2,
        reg_lambda=16.0,
        reg_alpha=0.2,
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


def objective(summary: dict) -> float:
    return (
        2.0 * summary["top1"]
        + 1.4 * summary["top3"]
        + 1.0 * summary["top5"]
        + 0.35 * summary["rank_ic"]
        + 0.15 * summary["top_bottom"]
    )


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
    return candidate_output_dir(candidate) / "fold_predictions" / "fold08_fold09.parquet"


def candidate_summary_path(candidate: Candidate) -> Path:
    return candidate_output_dir(candidate) / "fold_results.csv"


def build_command(candidate: Candidate) -> list[str]:
    return [
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
        f"stock_predict_data_model_agent_1d_{candidate.name}_20260627_research_smoke",
        "--prediction-output-path",
        os.path.relpath(candidate_prediction_path(candidate), MAIN_DIR),
        "--summary-output",
        os.path.relpath(candidate_summary_path(candidate), MAIN_DIR),
        "--prediction-output-mode",
        "independent",
        "--start-fold",
        "8",
        "--end-fold",
        "9",
        "--execute",
    ]


def candidate_env(candidate: Candidate) -> dict[str, str]:
    return {
        "XGB_REG_EVAL_METRIC": "top_return_loss",
        "XGB_TOP_RETURN_EVAL_K": str(candidate.eval_k),
        "XGB_VALIDATION_MODE": "train_tail_days",
        "XGB_VALIDATION_TAIL_DAYS": str(candidate.validation_tail_days),
        "XGB_SAMPLE_WEIGHT_MODE": "daily_top_quantile",
        "XGB_SAMPLE_WEIGHT_TOP_PCT": str(candidate.top_pct),
        "XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER": str(candidate.top_multiplier),
        "XGB_DEVICE": "cpu",
        "XGB_N_JOBS": "8",
        "XGB_N_ESTIMATORS": str(candidate.n_estimators),
        "XGB_LEARNING_RATE": str(candidate.learning_rate),
        "XGB_MAX_DEPTH": str(candidate.max_depth),
        "XGB_REG_LAMBDA": str(candidate.reg_lambda),
        "XGB_REG_ALPHA": str(candidate.reg_alpha),
        "XGB_EARLY_STOPPING_ROUNDS": "300",
    }


def run_candidate(candidate: Candidate, execute: bool) -> dict:
    out_dir = candidate_output_dir(candidate)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(candidate)
    env_delta = candidate_env(candidate)
    if not execute:
        return {
            "candidate": asdict(candidate),
            "command": command,
            "env": env_delta,
            "status": "planned_only",
            "prediction_path": str(candidate_prediction_path(candidate)),
            "summary_path": str(candidate_summary_path(candidate)),
        }

    env = os.environ.copy()
    env.update(env_delta)
    completed = subprocess.run(command, cwd=MAIN_DIR, env=env, capture_output=True, text=True, check=False)
    run_log = {
        "candidate": asdict(candidate),
        "command": command,
        "env": env_delta,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "prediction_path": str(candidate_prediction_path(candidate)),
        "summary_path": str(candidate_summary_path(candidate)),
    }
    (out_dir / "run_log.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        return {**run_log, "status": "failed"}
    return {**run_log, "status": "completed"}


def build_report(results: list[dict], execute: bool) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_fixed4y_mixed_objective_smoke",
        "execute": execute,
        "label": LABEL,
        "feature_input": str(DATA_DIR / "production_factor_parts"),
        "label_input": str(LABEL_DIR),
        "selected_features_path": str(FEATURES_PATH),
        "folds": [8, 9],
        "boundary": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
        "results": results,
    }

    if execute:
        labels = load_labels()
        summary_rows = []
        for result in results:
            if result.get("status") != "completed":
                continue
            pred_path = Path(result["prediction_path"])
            pred = pd.read_parquet(pred_path, columns=["trade_date", "stock_code", "pred_prob"])
            pred["trade_date"] = pred["trade_date"].astype(str)
            pred["stock_code"] = pred["stock_code"].astype(str)
            start = str(pred["trade_date"].min())
            end = str(pred["trade_date"].max())
            summary = evaluate_frame(pred, labels)
            summary.update(
                {
                    "asset": result["candidate"]["name"],
                    "date_min": start,
                    "date_max": end,
                    "objective": objective(summary),
                    "prediction_path": str(pred_path),
                    "summary_path": result["summary_path"],
                }
            )
            summary_rows.append(summary)
        for name, table in BASELINE_TABLES.items():
            if not summary_rows:
                continue
            start = min(row["date_min"] for row in summary_rows)
            end = max(row["date_max"] for row in summary_rows)
            frame = load_baseline_table(table, start, end)
            summary = evaluate_frame(frame, labels)
            summary.update({"asset": name, "date_min": start, "date_max": end, "objective": objective(summary)})
            summary_rows.append(summary)
        if summary_rows:
            summary_frame = pd.DataFrame(summary_rows).sort_values("objective", ascending=False)
            summary_frame.to_csv(REPORT_DIR / "smoke_eval_summary.csv", index=False, encoding="utf-8-sig")
            payload["summary_csv"] = str(REPORT_DIR / "smoke_eval_summary.csv")

    (REPORT_DIR / "smoke_plan_or_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 1D 固定四年 mixed-objective smoke 实验",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 边界",
        "",
        "本实验只属于 research-only。未修改 formal manifest，未修改 production manifest，未生成交易信号，未运行回测。",
        "",
        "## 实验设计",
        "",
        "- 标签：`executable_1d_open_return`",
        "- 窗口：固定四年",
        "- 折数：fold08 / fold09",
        "- 特征：复用 `selected_features_top40.json`",
        "- 目标：用 top-return loss 与日内顶部样本权重逼近 `Top1 + Top3 + RankIC` 混合目标",
        "",
        "## 状态",
        "",
        f"- execute：`{execute}`",
        f"- 候选数：`{len(CANDIDATES)}`",
        "",
        "## 证据",
        "",
        "- `smoke_plan_or_result.json`",
        "- 若执行训练，会生成 `smoke_eval_summary.csv`、各候选 `run_log.json`、`model_fold08.json`、`model_fold09.json`",
        "",
    ]
    (REPORT_DIR / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D fixed-4y mixed-objective research-only smoke experiment.")
    parser.add_argument("--execute", action="store_true", help="Actually run fold08/fold09 training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)
    results = [run_candidate(candidate, execute=args.execute) for candidate in CANDIDATES]
    build_report(results, execute=args.execute)
    failed = [result for result in results if result.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
