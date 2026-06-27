from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_1d_current_vs_fixed4y_daygate_scan_20260626"

LABEL = "executable_1d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_1d_new5d_condition_gate_v6_20260625_executable_1d_open_return_research"
ALT_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_1d_fixed4y_fold08_fs40_structure_scan_20260626"
    / "fs40_l16_lr005_n3600"
    / "fold08"
    / "fold_predictions"
    / "fold08.parquet"
)

SCAN_FEATURES = [
    "base_std",
    "alt_std",
    "rank_corr",
    "top10_overlap",
    "top20_overlap",
    "mean_abs_rank_gap",
]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_base_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(BASE_TABLE)} where trade_date >= ? and trade_date <= ? order by trade_date, stock_code",
            conn,
            params=("20260304", "20260603"),
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": "base_score"})


def read_alt_scores() -> pd.DataFrame:
    frame = pd.read_parquet(ALT_PATH, columns=["trade_date", "stock_code", "pred_prob"])
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": "alt_score"})


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def enrich_scores(base: pd.DataFrame, alt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = base.merge(alt, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    scores["base_rank"] = scores.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    scores["alt_rank"] = scores.groupby("trade_date")["alt_score"].rank(method="average", pct=True)

    daily_rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        top10_base = set(group.nlargest(min(10, len(group)), "base_rank")["stock_code"])
        top10_alt = set(group.nlargest(min(10, len(group)), "alt_rank")["stock_code"])
        top20_base = set(group.nlargest(min(20, len(group)), "base_rank")["stock_code"])
        top20_alt = set(group.nlargest(min(20, len(group)), "alt_rank")["stock_code"])
        daily_rows.append(
            {
                "trade_date": trade_date,
                "base_std": float(group["base_score"].std()),
                "alt_std": float(group["alt_score"].std()),
                "rank_corr": float(group["base_rank"].corr(group["alt_rank"])),
                "top10_overlap": float(len(top10_base & top10_alt) / max(1, len(top10_base | top10_alt))),
                "top20_overlap": float(len(top20_base & top20_alt) / max(1, len(top20_base | top20_alt))),
                "mean_abs_rank_gap": float((group["base_rank"] - group["alt_rank"]).abs().mean()),
            }
        )
    daily_features = pd.DataFrame(daily_rows)
    scores = scores.merge(daily_features, on="trade_date", how="left", validate="many_to_one")
    return scores, daily_features


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def daily_metrics(eval_frame: pd.DataFrame, rank_col: str, source: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "source": source,
            "rank_ic": float(group[rank_col].corr(label_rank)),
            "pearson_ic": float(group[rank_col].corr(group[LABEL])),
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = top_mean(group, rank_col, n)
        row["top_bottom"] = top_mean(group, rank_col, 50) - bottom_mean(group, rank_col, 50)
        rows.append(row)
    return pd.DataFrame(rows)


def objective_from_daily(daily: pd.DataFrame) -> float:
    summary = {col: float(daily[col].mean()) for col in ["rank_ic", "top1", "top3", "top5", "top10", "top_bottom"]}
    return (
        1.8 * summary["top1"]
        + 1.2 * summary["top3"]
        + 1.0 * summary["top5"]
        + 0.4 * summary["top10"]
        + 0.45 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
    )


def evaluate_candidate(scored: pd.DataFrame, labels: pd.DataFrame, source: str) -> pd.DataFrame:
    eval_frame = scored.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    return daily_metrics(eval_frame, "pred_prob", source)


def build_candidate(scores: pd.DataFrame, feature: str, direction: str, threshold: float) -> pd.DataFrame:
    out = scores.copy()
    active = out[feature] <= threshold if direction == "le" else out[feature] >= threshold
    out["gate_active"] = active
    out["pred_prob"] = np.where(active, out["alt_rank"], out["base_rank"])
    out["score_formula"] = f"if {feature} {direction} {threshold:.12f}: alt_rank else base_rank"
    return out


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = read_base_scores()
    alt = read_alt_scores()
    labels = load_labels("20260304", "20260603")
    scores, daily_features = enrich_scores(base, alt)

    base_eval = evaluate_candidate(scores.assign(pred_prob=scores["base_rank"]), labels, "base")
    alt_eval = evaluate_candidate(scores.assign(pred_prob=scores["alt_rank"]), labels, "alt")
    base_objective = objective_from_daily(base_eval)
    alt_objective = objective_from_daily(alt_eval)

    rows = []
    for feature in SCAN_FEATURES:
        values = daily_features[feature].dropna()
        for q in QUANTILES:
            threshold = float(values.quantile(q))
            for direction in ["le", "ge"]:
                candidate_scores = build_candidate(scores, feature, direction, threshold)
                candidate_eval = evaluate_candidate(candidate_scores, labels, f"{feature}_{direction}_{q:.1f}")
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "quantile": q,
                        "threshold": threshold,
                        "active_days": int(candidate_scores.groupby("trade_date")["gate_active"].first().sum()),
                        "objective": objective_from_daily(candidate_eval),
                        "rank_ic": float(candidate_eval["rank_ic"].mean()),
                        "top1": float(candidate_eval["top1"].mean()),
                        "top3": float(candidate_eval["top3"].mean()),
                        "top5": float(candidate_eval["top5"].mean()),
                        "top10": float(candidate_eval["top10"].mean()),
                        "top_bottom": float(candidate_eval["top_bottom"].mean()),
                    }
                )

    result = pd.DataFrame(rows).sort_values(["objective", "top1", "top5"], ascending=False).reset_index(drop=True)
    best = result.iloc[0].to_dict()
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_current_vs_fixed4y_daygate_fold08",
        "label": LABEL,
        "base_asset": BASE_TABLE,
        "alt_asset_path": str(ALT_PATH),
        "date_min": "20260304",
        "date_max": "20260603",
        "trade_days": int(base_eval["trade_date"].nunique()),
        "base_objective": base_objective,
        "alt_objective": alt_objective,
        "best_candidate": best,
        "improves_over_base": bool(best["objective"] > base_objective),
    }

    result.to_csv(REPORT_DIR / "daygate_scan_results.csv", index=False)
    daily_features.to_csv(REPORT_DIR / "daygate_daily_features.csv", index=False)
    (REPORT_DIR / "daygate_scan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D 当前主线对近四年训练候选的日级门控扫描报告（20260626）",
        "",
        "## 当前结论",
        "",
        f"- 基准主线 objective：`{base_objective:.12f}`",
        f"- 近四年候选 objective：`{alt_objective:.12f}`",
        f"- 最优门控候选 objective：`{best['objective']:.12f}`",
        f"- 是否超过当前主线：`{'是' if best['objective'] > base_objective else '否'}`",
        "",
        "## 最优候选",
        "",
        f"- 特征：`{best['feature']}`",
        f"- 方向：`{best['direction']}`",
        f"- 分位点：`{best['quantile']}`",
        f"- 阈值：`{best['threshold']:.12f}`",
        f"- 触发天数：`{int(best['active_days'])}`",
        f"- Top1：`{best['top1']:.12f}`",
        f"- Top3：`{best['top3']:.12f}`",
        f"- Top5：`{best['top5']:.12f}`",
        f"- RankIC：`{best['rank_ic']:.12f}`",
        "",
        "## 判断",
        "",
        "- 这是 fold08 局部窗口的 score-level 研究，不涉及训练、formal 发布或策略回测。",
        "- 如果最优门控仍然打不过当前主线，说明 1D 在当前候选集合里仍缺乏可利用的局部切换结构。",
        "",
        "## 产物",
        "",
        f"- 汇总：`{(REPORT_DIR / 'daygate_scan_summary.json').as_posix()}`",
        f"- 扫描结果：`{(REPORT_DIR / 'daygate_scan_results.csv').as_posix()}`",
        f"- 日级特征：`{(REPORT_DIR / 'daygate_daily_features.csv').as_posix()}`",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
