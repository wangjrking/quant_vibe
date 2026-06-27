from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

import research_10d_exante_score_state_gate_v65_20260627 as gate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_10d_secondgate_exante_scan_v71_20260627"
LABEL = "executable_10d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_10d_top5_safe_gate_v58_20260627_executable_10d_open_return_research"
CAND_TABLE = "stock_predict_data_model_agent_10d_topzone_second_gate_20260627_executable_10d_open_return_research"
MIN_FULL_RANK_IC_DELTA = -0.0015


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_scores(table: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"select trade_date, stock_code, pred_prob from {table}", conn)
    df["trade_date"] = df["trade_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    return df


def read_labels(min_date: str, max_date: str) -> pd.DataFrame:
    df = pd.read_parquet(LABEL_DIR, columns=["trade_date", "stock_code", LABEL])
    df["trade_date"] = df["trade_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df = df[(df["trade_date"] >= min_date) & (df["trade_date"] <= max_date)]
    return df[df[LABEL].notna()]


def daily_eval(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group.dropna(subset=["pred_prob", LABEL])
        if len(group) < 50:
            continue
        ordered = group.sort_values("pred_prob", ascending=False)
        rows.append(
            {
                "trade_date": str(trade_date),
                "rank_ic": float(group["pred_prob"].corr(group[LABEL], method="spearman")),
                "top1": float(ordered.head(1)[LABEL].mean()),
                "top3": float(ordered.head(3)[LABEL].mean()),
                "top5": float(ordered.head(5)[LABEL].mean()),
                "top10": float(ordered.head(10)[LABEL].mean()),
                "top20": float(ordered.head(20)[LABEL].mean()),
                "top50": float(ordered.head(50)[LABEL].mean()),
            }
        )
    return pd.DataFrame(rows)


def better(row: dict[str, object], best: dict[str, object] | None) -> bool:
    if best is None:
        return True
    return (bool(row["pass_hard"]), float(row["objective"])) > (bool(best["pass_hard"]), float(best["objective"]))


def evaluate_mask(
    name: str,
    mask: pd.Series,
    base_daily: pd.DataFrame,
    cand_diff: pd.DataFrame,
    base_summary: dict[str, dict[str, float]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    row, daily, month_df = gate.evaluate_mask(name, mask, base_daily, cand_diff, base_summary)
    row["rank_ic_gate_ok"] = bool(float(row["full_rank_ic_delta"]) >= MIN_FULL_RANK_IC_DELTA)
    row["pass_hard"] = bool(row["pass_hard"] and row["rank_ic_gate_ok"])
    return row, daily, month_df


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base_scores = read_scores(BASE_TABLE)
    cand_scores = read_scores(CAND_TABLE)
    min_date = max(str(base_scores["trade_date"].min()), str(cand_scores["trade_date"].min()), "20240604")
    max_date = min(str(base_scores["trade_date"].max()), str(cand_scores["trade_date"].max()))
    labels = read_labels(min_date, max_date)

    base_daily = daily_eval(base_scores, labels)
    cand_daily = daily_eval(cand_scores, labels)
    for df in (base_daily, cand_daily):
        df["trade_date"] = df["trade_date"].astype(str)
        df.sort_values("trade_date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    common_dates = sorted(set(base_daily["trade_date"]) & set(cand_daily["trade_date"]))
    base_daily = base_daily[base_daily["trade_date"].isin(common_dates)].reset_index(drop=True)
    cand_daily = cand_daily[cand_daily["trade_date"].isin(common_dates)].reset_index(drop=True)
    base_daily.to_csv(REPORT_DIR / "base_daily_eval.csv", index=False)
    cand_daily.to_csv(REPORT_DIR / "candidate_daily_eval.csv", index=False)

    base_summary = gate.summarize(base_daily)
    raw_delta = gate.delta(gate.summarize(cand_daily), base_summary)

    feature_path = REPORT_DIR / "exante_score_state_features.csv"
    if feature_path.exists():
        features = pd.read_csv(feature_path)
        features["trade_date"] = features["trade_date"].astype(str)
    else:
        features = gate.score_state_features(base_scores, cand_scores)
        features.to_csv(feature_path, index=False)
    features = features[features["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    base_daily = base_daily[base_daily["trade_date"].isin(set(features["trade_date"]))].sort_values("trade_date").reset_index(drop=True)
    cand_daily = cand_daily[cand_daily["trade_date"].isin(set(features["trade_date"]))].sort_values("trade_date").reset_index(drop=True)
    cand_diff = cand_daily[gate.METRICS] - base_daily[gate.METRICS]

    rows: list[dict[str, object]] = []
    single_results: list[tuple[dict[str, object], str, pd.Series]] = []
    best_row: dict[str, object] | None = None
    best_daily = None
    best_month = None

    for name, mask in gate.single_candidate_masks(features):
        if not bool(mask.any()):
            continue
        row, daily, month_df = evaluate_mask(name, mask, base_daily, cand_diff, base_summary)
        rows.append(row)
        single_results.append((row, name, mask))
        if better(row, best_row):
            best_row = row
            best_daily = daily
            best_month = month_df

    top_single = sorted(single_results, key=lambda x: (bool(x[0]["pass_hard"]), float(x[0]["objective"])), reverse=True)[:50]
    for (_, name_a, mask_a), (_, name_b, mask_b) in combinations(top_single, 2):
        mask = mask_a & mask_b
        if not bool(mask.any()):
            continue
        row, daily, month_df = evaluate_mask(f"({name_a}) AND ({name_b})", mask, base_daily, cand_diff, base_summary)
        rows.append(row)
        if better(row, best_row):
            best_row = row
            best_daily = daily
            best_month = month_df

    scan = pd.DataFrame(rows).sort_values(["pass_hard", "objective"], ascending=[False, False])
    scan.to_csv(REPORT_DIR / "exante_score_state_gate_scan.csv", index=False)
    if best_daily is not None:
        best_daily.to_csv(REPORT_DIR / "best_daily_eval.csv", index=False)
    if best_month is not None:
        best_month.to_csv(REPORT_DIR / "best_monthly_delta.csv", index=False)

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_secondgate_exante_scan_v71",
        "label": LABEL,
        "base_table": BASE_TABLE,
        "candidate_table": CAND_TABLE,
        "feature_source": "same-day prediction score distributions and rank differences only",
        "min_full_rank_ic_delta": MIN_FULL_RANK_IC_DELTA,
        "eval_window": {
            "days": int(len(base_daily)),
            "min_eval_date": str(base_daily["trade_date"].min()),
            "max_eval_date": str(base_daily["trade_date"].max()),
        },
        "raw_candidate_delta_vs_base": raw_delta,
        "scan_rows": int(len(scan)),
        "pass_hard_count": int(scan["pass_hard"].sum()) if not scan.empty else 0,
        "best": best_row,
        "decision": "exante_gate_found" if best_row and best_row["pass_hard"] else "continue_research_no_exante_hard_pass",
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "exante_score_state_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 10D second-gate 前视门控扫描 v71

生成时间：{summary['generated_at']}

## 结论

决策：`{summary['decision']}`。

本扫描只使用预测日已经存在的分数分布与排名差异特征，不使用标签结果选择月份。

最佳条件：`{best_row['condition'] if best_row else '无'}`

硬门槛通过数量：`{summary['pass_hard_count']}` / `{summary['scan_rows']}`。

## 边界

- 未训练模型
- 未写入新预测表
- 未修改 formal manifest
- 未修改 production manifest
- 未生成交易信号
- 未运行策略回测
"""
    (REPORT_DIR / "exante_score_state_gate_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
