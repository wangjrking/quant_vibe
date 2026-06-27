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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_feature_separation_20260627"
LABEL = "executable_1d_open_return"

BASELINE_TABLES = {
    "research1d_current": "stock_predict_data_model_agent_1d_daygate_best_20260627_executable_1d_open_return_research",
    "formal1d_current": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    selected_features_path: Path
    eval_k: int = 3
    validation_tail_days: int = 84
    top_pct: float = 0.005
    top_multiplier: float = 10.0
    n_estimators: int = 3600
    learning_rate: float = 0.005
    max_depth: int = 2
    reg_lambda: float = 16.0
    reg_alpha: float = 0.0


def candidate_specs() -> list[Candidate]:
    return [
        Candidate("self_top24", REPORT_DIR / "selected_features_1d_self_top24.json"),
        Candidate("self_top40", REPORT_DIR / "selected_features_1d_self_top40.json"),
        Candidate("cross_confirmed_27", REPORT_DIR / "selected_features_1d_cross_confirmed_40.json"),
        Candidate("self_top56", REPORT_DIR / "selected_features_1d_self_top56.json"),
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
        1.8 * summary["top1"]
        + 1.3 * summary["top3"]
        + 1.1 * summary["top5"]
        + 0.6 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
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


def out_dir(candidate: Candidate) -> Path:
    return REPORT_DIR / candidate.name


def prediction_path(candidate: Candidate) -> Path:
    return out_dir(candidate) / "fold_predictions" / "fold08_fold09.parquet"


def summary_path(candidate: Candidate) -> Path:
    return out_dir(candidate) / "fold_results.csv"


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
        os.path.relpath(candidate.selected_features_path, MAIN_DIR),
        "--output-table",
        f"stock_predict_data_model_agent_1d_feature_sep_{candidate.name}_20260627_research_smoke",
        "--prediction-output-path",
        os.path.relpath(prediction_path(candidate), MAIN_DIR),
        "--summary-output",
        os.path.relpath(summary_path(candidate), MAIN_DIR),
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


def candidate_payload(candidate: Candidate) -> dict:
    payload = asdict(candidate)
    payload["selected_features_path"] = str(candidate.selected_features_path)
    return payload


def run_candidate(candidate: Candidate, execute: bool) -> dict:
    out_dir(candidate).mkdir(parents=True, exist_ok=True)
    command = build_command(candidate)
    env_delta = candidate_env(candidate)
    feature_payload = json.loads(candidate.selected_features_path.read_text(encoding="utf-8"))
    features = feature_payload.get("features", feature_payload if isinstance(feature_payload, list) else [])
    if not execute:
        return {
            "candidate": candidate_payload(candidate),
            "feature_count": len(features),
            "command": command,
            "env": env_delta,
            "status": "planned_only",
            "prediction_path": str(prediction_path(candidate)),
            "summary_path": str(summary_path(candidate)),
        }

    env = os.environ.copy()
    env.update(env_delta)
    completed = subprocess.run(command, cwd=MAIN_DIR, env=env, capture_output=True, text=True, check=False)
    run_log = {
        "candidate": candidate_payload(candidate),
        "feature_count": len(features),
        "command": command,
        "env": env_delta,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "prediction_path": str(prediction_path(candidate)),
        "summary_path": str(summary_path(candidate)),
    }
    (out_dir(candidate) / "run_log.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**run_log, "status": "completed" if completed.returncode == 0 else "failed"}


def build_report(results: list[dict], execute: bool) -> None:
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_fixed4y_feature_separation_smoke",
        "execute": execute,
        "label": LABEL,
        "feature_input": str(DATA_DIR / "production_factor_parts"),
        "label_input": str(LABEL_DIR),
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
            pred = pd.read_parquet(result["prediction_path"], columns=["trade_date", "stock_code", "pred_prob"])
            pred["trade_date"] = pred["trade_date"].astype(str)
            pred["stock_code"] = pred["stock_code"].astype(str)
            start = str(pred["trade_date"].min())
            end = str(pred["trade_date"].max())
            summary = evaluate_frame(pred, labels)
            summary.update(
                {
                    "asset": result["candidate"]["name"],
                    "feature_count": result["feature_count"],
                    "date_min": start,
                    "date_max": end,
                    "objective": objective(summary),
                    "prediction_path": result["prediction_path"],
                    "summary_path": result["summary_path"],
                }
            )
            summary_rows.append(summary)
        if summary_rows:
            start = min(row["date_min"] for row in summary_rows)
            end = max(row["date_max"] for row in summary_rows)
            for name, table in BASELINE_TABLES.items():
                frame = load_baseline_table(table, start, end)
                summary = evaluate_frame(frame, labels)
                summary.update({"asset": name, "date_min": start, "date_max": end, "objective": objective(summary)})
                summary_rows.append(summary)
            summary_frame = pd.DataFrame(summary_rows).sort_values("objective", ascending=False)
            summary_frame.to_csv(REPORT_DIR / "feature_separation_eval_summary.csv", index=False, encoding="utf-8-sig")
            payload["summary_csv"] = str(REPORT_DIR / "feature_separation_eval_summary.csv")

    (REPORT_DIR / "feature_separation_plan_or_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 1D 固定四年特征分离 smoke 实验",
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
        "- 变量：只改变特征集，训练目标和权重保持相对保守",
        "- 候选：self top24 / self top40 / cross-confirmed / self top56",
        "",
        "## 状态",
        "",
        f"- execute：`{execute}`",
        f"- 候选数：`{len(results)}`",
        "",
        "## 证据",
        "",
        "- `feature_separation_plan_or_result.json`",
        "- 执行后生成 `feature_separation_eval_summary.csv`、各候选 `run_log.json`、`model_fold08.json`、`model_fold09.json`",
    ]
    (REPORT_DIR / "feature_separation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D fixed-4y feature separation research-only smoke experiment.")
    parser.add_argument("--execute", action="store_true", help="Actually run fold08/fold09 training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = candidate_specs()
    missing = [str(candidate.selected_features_path) for candidate in candidates if not candidate.selected_features_path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    results = [run_candidate(candidate, execute=args.execute) for candidate in candidates]
    build_report(results, execute=args.execute)
    return 1 if any(result.get("status") == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
