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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_1d_mainline_fs40_guard_scan_20260626"

LABEL = "executable_1d_open_return"
MAINLINE_TABLE = "stock_predict_data_model_agent_1d_new5d_condition_gate_v6_20260625_executable_1d_open_return_research"
FS40_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_1d_fixed4y_fold08_fs40_structure_scan_20260626"
    / "fs40_l16_lr005_n3600"
    / "fold08"
    / "fold_predictions"
    / "fold08.parquet"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in LABEL_DIR.glob("*.parquet"):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def load_scores() -> pd.DataFrame:
    fs40 = pd.read_parquet(FS40_PATH, columns=["trade_date", "stock_code", "pred_prob"])
    fs40["trade_date"] = fs40["trade_date"].astype(str)
    fs40["stock_code"] = fs40["stock_code"].astype(str)
    fs40 = fs40.rename(columns={"pred_prob": "fs40_score"})
    min_date = str(fs40["trade_date"].min())
    max_date = str(fs40["trade_date"].max())
    with sqlite3.connect(MODEL_DB) as conn:
        mainline = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from \"{MAINLINE_TABLE}\" where trade_date >= ? and trade_date <= ? order by trade_date, stock_code",
            conn,
            params=(min_date, max_date),
        )
    mainline["trade_date"] = mainline["trade_date"].astype(str)
    mainline["stock_code"] = mainline["stock_code"].astype(str)
    mainline = mainline.rename(columns={"pred_prob": "main_score"})
    out = mainline.merge(fs40, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    out["main_rank"] = out.groupby("trade_date")["main_score"].rank(method="average", pct=True)
    out["fs40_rank"] = out.groupby("trade_date")["fs40_score"].rank(method="average", pct=True)
    return out


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def build_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        main10 = top_set(group, "main_rank", 10)
        main20 = top_set(group, "main_rank", 20)
        main50 = top_set(group, "main_rank", 50)
        fs10 = top_set(group, "fs40_rank", 10)
        fs20 = top_set(group, "fs40_rank", 20)
        fs50 = top_set(group, "fs40_rank", 50)
        rows.append(
            {
                "trade_date": trade_date,
                "main_score_std": float(group["main_score"].std()),
                "fs40_score_std": float(group["fs40_score"].std()),
                "main_top10_score_mean": float(group.nlargest(min(10, len(group)), "main_rank")["main_score"].mean()),
                "main_top50_score_mean": float(group.nlargest(min(50, len(group)), "main_rank")["main_score"].mean()),
                "fs40_top10_score_mean": float(group.nlargest(min(10, len(group)), "fs40_rank")["fs40_score"].mean()),
                "fs40_top50_score_mean": float(group.nlargest(min(50, len(group)), "fs40_rank")["fs40_score"].mean()),
                "main_top10_top50_gap": float(
                    group.nlargest(min(10, len(group)), "main_rank")["main_score"].mean()
                    - group.nlargest(min(50, len(group)), "main_rank")["main_score"].mean()
                ),
                "fs40_top10_top50_gap": float(
                    group.nlargest(min(10, len(group)), "fs40_rank")["fs40_score"].mean()
                    - group.nlargest(min(50, len(group)), "fs40_rank")["fs40_score"].mean()
                ),
                "corr_main_fs40": float(group["main_rank"].corr(group["fs40_rank"])),
                "overlap10_main_fs40": len(main10 & fs10),
                "overlap20_main_fs40": len(main20 & fs20),
                "overlap50_main_fs40": len(main50 & fs50),
            }
        )
    return pd.DataFrame(rows)


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


def summarize(daily: pd.DataFrame, dates: list[str], window: int | None) -> dict[str, float]:
    keep = set(dates if window is None else dates[-window:])
    sub = daily[daily["trade_date"].isin(keep)]
    return {
        col: float(sub[col].mean())
        for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    }


def period_stats(delta: pd.DataFrame) -> dict[str, float | int]:
    periods = {
        "2024H2": ("20240604", "20241231"),
        "2025H1": ("20250101", "20250630"),
        "2025H2": ("20250701", "20251231"),
        "2026YTD": ("20260101", "99999999"),
    }
    out: dict[str, float | int] = {}
    top5s = []
    rankics = []
    for name, (lo, hi) in periods.items():
        sub = delta[(delta["trade_date"] >= lo) & (delta["trade_date"] <= hi)]
        top5 = float(sub["top5_delta"].mean())
        rank_ic = float(sub["rank_ic_delta"].mean())
        out[f"{name}_top5_delta"] = top5
        out[f"{name}_rank_ic_delta"] = rank_ic
        top5s.append(top5)
        rankics.append(rank_ic)
    out["positive_top5_periods"] = int(sum(v > 0 for v in top5s))
    out["min_period_top5_delta"] = float(min(top5s))
    out["min_period_rank_ic_delta"] = float(min(rankics))
    out["avg_period_top5_delta"] = float(np.mean(top5s))
    out["avg_period_rank_ic_delta"] = float(np.mean(rankics))
    return out


def scan(scores: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    main_daily = daily_metrics(eval_frame, "main_rank", "mainline").set_index("trade_date")
    fs40_daily = daily_metrics(eval_frame, "fs40_rank", "fs40").set_index("trade_date")
    dates = sorted(main_daily.index.tolist())
    feat_idx = features.set_index("trade_date")
    rows = []
    pass_daily_rows = []
    for feature in feat_idx.columns:
        vals = feat_idx[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.nunique() < 3:
            continue
        thresholds = sorted(
            set(float(vals.quantile(q)) for q in [0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95])
        )
        for threshold in thresholds:
            for op in ["<=", ">="]:
                active = feat_idx[feature] <= threshold if op == "<=" else feat_idx[feature] >= threshold
                active_dates = set(active[active].index.astype(str))
                if len(active_dates) < 10 or len(active_dates) > int(0.55 * len(feat_idx)):
                    continue
                cand = main_daily.copy()
                active_eval_dates = [d for d in active_dates if d in fs40_daily.index]
                cand.loc[active_eval_dates, :] = fs40_daily.loc[active_eval_dates, cand.columns]
                cand = cand.reset_index()
                merged = cand.merge(main_daily.reset_index(), on="trade_date", suffixes=("", "_base"))
                for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
                    merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_base"]
                full = summarize(cand, dates, None)
                base_full = summarize(main_daily.reset_index(), dates, None)
                r63 = summarize(cand, dates, 63)
                b63 = summarize(main_daily.reset_index(), dates, 63)
                r20 = summarize(cand, dates, 20)
                b20 = summarize(main_daily.reset_index(), dates, 20)
                row = {
                    "feature": feature,
                    "op": op,
                    "threshold": threshold,
                    "active_days": len(active_dates),
                    "active_ratio": len(active_dates) / len(feat_idx),
                    "recent63_active_days": len(set(dates[-63:]) & active_dates),
                    "recent20_active_days": len(set(dates[-20:]) & active_dates),
                }
                for prefix, cur, base_sum in [("full", full, base_full), ("recent63", r63, b63), ("recent20", r20, b20)]:
                    for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
                        row[f"{prefix}_{metric}_delta"] = cur[metric] - base_sum[metric]
                row.update(period_stats(merged))
                row["pass_basic"] = bool(
                    row["recent63_active_days"] >= 3
                    and row["recent63_top1_delta"] >= 0
                    and row["recent63_top5_delta"] >= 0
                    and row["recent20_top5_delta"] >= -0.0001
                    and row["full_top5_delta"] >= -0.0001
                    and row["full_rank_ic_delta"] >= -0.0004
                    and row["min_period_top5_delta"] >= -0.00015
                    and row["positive_top5_periods"] >= 3
                )
                row["objective"] = (
                    3.0 * row["recent63_top1_delta"]
                    + 2.0 * row["recent63_top5_delta"]
                    + row["recent20_top5_delta"]
                    + 0.5 * row["avg_period_top5_delta"]
                    + 0.01 * row["positive_top5_periods"]
                    - 0.01 * row["active_ratio"]
                )
                rows.append(row)
                if row["pass_basic"]:
                    tmp = merged[["trade_date", "rank_ic_delta", "top1_delta", "top5_delta", "top10_delta"]].copy()
                    tmp["feature"] = feature
                    tmp["op"] = op
                    tmp["threshold"] = threshold
                    pass_daily_rows.append(tmp)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["pass_basic", "objective"], ascending=[False, False]).reset_index(drop=True)
    pass_daily = pd.concat(pass_daily_rows, ignore_index=True) if pass_daily_rows else pd.DataFrame()
    return result, pass_daily


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    min_date = str(scores["trade_date"].min())
    max_date = str(scores["trade_date"].max())
    labels = load_labels(min_date, max_date)
    features = build_features(scores)
    scan_result, pass_daily = scan(scores, labels, features)

    scan_path = REPORT_DIR / "guard_scan_results.csv"
    daily_path = REPORT_DIR / "guard_scan_pass_daily.csv"
    feature_path = REPORT_DIR / "guard_scan_features.csv"
    packet_path = REPORT_DIR / "guard_scan_packet.json"
    report_path = REPORT_DIR / "report.md"

    scan_result.to_csv(scan_path, index=False, encoding="utf-8-sig")
    pass_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    features.to_csv(feature_path, index=False, encoding="utf-8-sig")

    packet = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_mainline_fs40_guard_scan",
        "label": LABEL,
        "mainline_table": MAINLINE_TABLE,
        "fs40_prediction_path": str(FS40_PATH),
        "guard_features_csv": str(feature_path),
        "scan_results_csv": str(scan_path),
        "pass_daily_csv": str(daily_path),
        "rows": int(len(scan_result)),
        "passed_rows": int(scan_result["pass_basic"].sum()) if not scan_result.empty else 0,
        "date_min": min_date,
        "date_max": max_date,
    }
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D 主线与 fs40 候选条件切换扫描结论",
        "",
        "## 实验目的",
        "",
        "验证当前 `1D` 研究主线与四年窗训练候选 `fs40_l16_lr005_n3600` 是否存在可利用的非线性条件互补。",
        "",
        "本轮不训练新模型，只扫描“在什么日度条件下，用 fs40 候选替换主线”是否会改善模型侧评价指标。",
        "",
        "## 扫描口径",
        "",
        "- 条件特征来自主线与 fs40 的日度分数结构：",
        "  - `main_score_std` / `fs40_score_std`",
        "  - `top10_score_mean` / `top50_score_mean`",
        "  - `top10_top50_gap`",
        "  - `corr_main_fs40`",
        "  - `overlap10/20/50_main_fs40`",
        "- 阈值：按分位数扫描",
        "- 比较方式：满足条件的交易日，用 `fs40_rank` 替换 `main_rank`",
        "",
        "## 结果摘要",
        "",
    ]
    if scan_result.empty:
        lines.extend(
            [
                "本轮没有生成有效扫描结果。",
            ]
        )
    else:
        passed = scan_result[scan_result["pass_basic"]]
        if passed.empty:
            best = scan_result.iloc[0]
            lines.extend(
                [
                    "本轮没有找到满足当前 1D 近端约束的可用条件切换候选。",
                    "",
                    "最接近通过的候选：",
                    f"- 条件：`{best['feature']} {best['op']} {best['threshold']}`",
                    f"- `recent63_active_days = {int(best['recent63_active_days'])}`",
                    f"- `recent63_top1_delta = {best['recent63_top1_delta']:.6f}`",
                    f"- `recent63_top5_delta = {best['recent63_top5_delta']:.6f}`",
                    f"- `full_top5_delta = {best['full_top5_delta']:.6f}`",
                    f"- `full_rank_ic_delta = {best['full_rank_ic_delta']:.6f}`",
                    f"- `positive_top5_periods = {int(best['positive_top5_periods'])}`",
                    f"- `objective = {best['objective']:.6f}`",
                    "",
                    "这说明：当前 1D 主线与四年窗候选之间，简单条件替换也没有形成稳定增量。",
                ]
            )
        else:
            best = passed.iloc[0]
            lines.extend(
                [
                    "本轮找到了满足当前 1D 近端约束的条件切换候选。",
                    "",
                    f"- 最优条件：`{best['feature']} {best['op']} {best['threshold']}`",
                    f"- `recent63_active_days = {int(best['recent63_active_days'])}`",
                    f"- `recent63_top1_delta = {best['recent63_top1_delta']:.6f}`",
                    f"- `recent63_top5_delta = {best['recent63_top5_delta']:.6f}`",
                    f"- `full_top5_delta = {best['full_top5_delta']:.6f}`",
                    f"- `full_rank_ic_delta = {best['full_rank_ic_delta']:.6f}`",
                    f"- `positive_top5_periods = {int(best['positive_top5_periods'])}`",
                    f"- `objective = {best['objective']:.6f}`",
                ]
            )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- 扫描结果：`guard_scan_results.csv`",
            "- 通过候选日度明细：`guard_scan_pass_daily.csv`",
            "- 日度条件特征：`guard_scan_features.csv`",
            "- 运行清单：`guard_scan_packet.json`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
