from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_10d_exante_score_state_gate_v65_20260627"
BASE_DAILY = DATA_DIR / "reports" / "model_agent_current_best_promotion_review_20260627_v60" / "executable_10d_open_return_baseline_daily_eval.csv"
CAND_DAILY = DATA_DIR / "reports" / "model_agent_current_best_promotion_review_20260627_v60" / "executable_10d_open_return_candidate_daily_eval.csv"
BASE_TABLE = "stock_predict_data_model_agent_10d_regime_blend_best_20260627_executable_10d_open_return_research"
CAND_TABLE = "stock_predict_data_model_agent_10d_top5_safe_gate_v58_20260627_executable_10d_open_return_research"

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]
FEATURES = [
    "mean_abs_rank_gap",
    "p95_abs_rank_gap",
    "max_abs_rank_gap",
    "mean_abs_score_gap",
    "p95_abs_score_gap",
    "cand_score_std",
    "base_score_std",
    "cand_score_iqr",
    "base_score_iqr",
    "cand_top1_gap",
    "base_top1_gap",
    "cand_top5_spread",
    "base_top5_spread",
    "top20_overlap",
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_scores(table: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {table}",
            conn,
        )
    df["trade_date"] = df["trade_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    return df


def score_state_features(base: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    merged = cand.merge(base, on=["trade_date", "stock_code"], suffixes=("_cand", "_base"), validate="one_to_one")
    rows: list[dict[str, float | str]] = []
    for trade_date, g in merged.groupby("trade_date", sort=True):
        c = g["pred_prob_cand"].astype(float)
        b = g["pred_prob_base"].astype(float)
        cr = c.rank(method="average", pct=True)
        br = b.rank(method="average", pct=True)
        abs_rank_gap = (cr - br).abs()
        abs_score_gap = (c - b).abs()

        c_sorted = np.sort(c.to_numpy())[::-1]
        b_sorted = np.sort(b.to_numpy())[::-1]
        top_n = min(20, len(c_sorted))
        cand_top20 = set(g.loc[c.nlargest(top_n).index, "stock_code"])
        base_top20 = set(g.loc[b.nlargest(top_n).index, "stock_code"])

        rows.append(
            {
                "trade_date": str(trade_date),
                "mean_abs_rank_gap": float(abs_rank_gap.mean()),
                "p95_abs_rank_gap": float(abs_rank_gap.quantile(0.95)),
                "max_abs_rank_gap": float(abs_rank_gap.max()),
                "mean_abs_score_gap": float(abs_score_gap.mean()),
                "p95_abs_score_gap": float(abs_score_gap.quantile(0.95)),
                "cand_score_std": float(c.std(ddof=0)),
                "base_score_std": float(b.std(ddof=0)),
                "cand_score_iqr": float(c.quantile(0.75) - c.quantile(0.25)),
                "base_score_iqr": float(b.quantile(0.75) - b.quantile(0.25)),
                "cand_top1_gap": float(c_sorted[0] - c_sorted[1]) if len(c_sorted) > 1 else 0.0,
                "base_top1_gap": float(b_sorted[0] - b_sorted[1]) if len(b_sorted) > 1 else 0.0,
                "cand_top5_spread": float(c_sorted[0] - c_sorted[min(4, len(c_sorted) - 1)]),
                "base_top5_spread": float(b_sorted[0] - b_sorted[min(4, len(b_sorted) - 1)]),
                "top20_overlap": float(len(cand_top20 & base_top20) / top_n) if top_n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "full": {m: float(daily[m].mean()) for m in METRICS},
        "recent126": {m: float(daily.tail(126)[m].mean()) for m in METRICS},
        "recent63": {m: float(daily.tail(63)[m].mean()) for m in METRICS},
        "recent20": {m: float(daily.tail(20)[m].mean()) for m in METRICS},
    }


def delta(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        scope: {metric: float(left[scope][metric] - right[scope][metric]) for metric in METRICS}
        for scope in ["full", "recent126", "recent63", "recent20"]
    }


def monthly_stats(candidate_daily: pd.DataFrame, base_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = candidate_daily.merge(base_daily, on="trade_date", suffixes=("", "_base"), validate="one_to_one")
    merged["month"] = pd.to_datetime(merged["trade_date"].astype(str), format="%Y%m%d").dt.to_period("M").astype(str)
    rows = []
    for month, g in merged.groupby("month", sort=True):
        rows.append(
            {
                "month": month,
                "rank_ic_delta": float(g["rank_ic"].mean() - g["rank_ic_base"].mean()),
                "top1_delta": float(g["top1"].mean() - g["top1_base"].mean()),
                "top5_delta": float(g["top5"].mean() - g["top5_base"].mean()),
            }
        )
    df = pd.DataFrame(rows)
    stats = {
        "month_count": int(len(df)),
        "positive_rank_ic_months": int((df["rank_ic_delta"] > 0).sum()),
        "positive_top1_months": int((df["top1_delta"] > 0).sum()),
        "positive_top5_months": int((df["top5_delta"] > 0).sum()),
        "nonnegative_top5_months": int((df["top5_delta"] >= 0).sum()),
        "positive_top1_and_nonnegative_top5_months": int(((df["top1_delta"] > 0) & (df["top5_delta"] >= 0)).sum()),
        "min_top5_delta": float(df["top5_delta"].min()),
    }
    return df, stats


def make_daily(base: pd.DataFrame, cand: pd.DataFrame, use_candidate_dates: set[str]) -> pd.DataFrame:
    out = base.copy()
    mask = out["trade_date"].isin(use_candidate_dates)
    cand_aligned = cand.set_index("trade_date").loc[out.loc[mask, "trade_date"], METRICS].reset_index(drop=True)
    out.loc[mask, METRICS] = cand_aligned.to_numpy()
    return out


def make_daily_from_mask(base: pd.DataFrame, diff: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = base.copy()
    mask_values = mask.to_numpy(dtype=bool)
    for metric in METRICS:
        out[metric] = base[metric].to_numpy() + mask_values * diff[metric].to_numpy()
    return out


def pass_hard(d: dict[str, dict[str, float]], mstats: dict[str, object]) -> bool:
    return (
        d["full"]["top1"] > 0
        and d["full"]["top5"] >= 0
        and d["recent63"]["top1"] >= 0
        and d["recent63"]["top5"] >= 0
        and d["recent20"]["top1"] >= 0
        and d["recent20"]["top5"] >= 0
        and int(mstats["positive_top5_months"]) >= 3
        and int(mstats["nonnegative_top5_months"]) == int(mstats["month_count"])
    )


def single_candidate_masks(features: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    masks: list[tuple[str, pd.Series]] = []
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for feature in FEATURES:
        values = features[feature]
        for q in quantiles:
            threshold = float(values.quantile(q))
            masks.append((f"{feature} <= {threshold:.12g}", values <= threshold))
            masks.append((f"{feature} >= {threshold:.12g}", values >= threshold))
    return masks


def evaluate_mask(
    name: str,
    mask: pd.Series,
    base_daily: pd.DataFrame,
    cand_diff: pd.DataFrame,
    base_summary: dict[str, dict[str, float]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    daily = make_daily_from_mask(base_daily, cand_diff, mask)
    candidate_summary = summarize(daily)
    d = delta(candidate_summary, base_summary)
    month_df, mstats = monthly_stats(daily, base_daily)
    hard = pass_hard(d, mstats)
    objective = (
        80.0 * d["recent20"]["top1"]
        + 40.0 * d["recent63"]["top1"]
        + 20.0 * d["full"]["top1"]
        + 25.0 * d["full"]["top5"]
        + 10.0 * d["recent63"]["top5"]
        + 5.0 * d["recent20"]["top5"]
        + 0.001 * int(mstats["positive_top5_months"])
        - 100.0 * max(0.0, -float(mstats["min_top5_delta"]))
    )
    row = {
        "condition": name,
        "active_days": int(mask.sum()),
        "objective": float(objective),
        "full_rank_ic_delta": d["full"]["rank_ic"],
        "full_top1_delta": d["full"]["top1"],
        "full_top5_delta": d["full"]["top5"],
        "recent126_top5_delta": d["recent126"]["top5"],
        "recent63_top1_delta": d["recent63"]["top1"],
        "recent63_top5_delta": d["recent63"]["top5"],
        "recent20_top1_delta": d["recent20"]["top1"],
        "recent20_top5_delta": d["recent20"]["top5"],
        "positive_top5_months": int(mstats["positive_top5_months"]),
        "nonnegative_top5_months": int(mstats["nonnegative_top5_months"]),
        "min_month_top5_delta": float(mstats["min_top5_delta"]),
        "pass_hard": bool(hard),
    }
    return row, daily, month_df


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base_daily = pd.read_csv(BASE_DAILY)
    cand_daily = pd.read_csv(CAND_DAILY)
    for df in (base_daily, cand_daily):
        df["trade_date"] = df["trade_date"].astype(str)
        df.sort_values("trade_date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    base_summary = summarize(base_daily)
    raw_cand_summary = summarize(cand_daily)
    raw_delta = delta(raw_cand_summary, base_summary)

    feature_path = REPORT_DIR / "exante_score_state_features.csv"
    if feature_path.exists():
        features = pd.read_csv(feature_path)
        features["trade_date"] = features["trade_date"].astype(str)
    else:
        features = score_state_features(read_scores(BASE_TABLE), read_scores(CAND_TABLE))
        features.to_csv(feature_path, index=False)
    eval_dates = set(base_daily["trade_date"])
    features = features[features["trade_date"].isin(eval_dates)].sort_values("trade_date").reset_index(drop=True)
    base_daily = base_daily[base_daily["trade_date"].isin(set(features["trade_date"]))].sort_values("trade_date").reset_index(drop=True)
    cand_daily = cand_daily[cand_daily["trade_date"].isin(set(features["trade_date"]))].sort_values("trade_date").reset_index(drop=True)
    cand_diff = cand_daily[METRICS] - base_daily[METRICS]

    rows: list[dict[str, object]] = []
    best_daily = None
    best_month = None
    best_row = None
    single_results: list[tuple[dict[str, object], str, pd.Series]] = []
    for name, mask in single_candidate_masks(features):
        if not bool(mask.any()):
            continue
        row, daily, month_df = evaluate_mask(name, mask, base_daily, cand_diff, base_summary)
        rows.append(row)
        single_results.append((row, name, mask))
        if best_row is None or (bool(row["pass_hard"]), float(row["objective"])) > (bool(best_row["pass_hard"]), float(best_row["objective"])):
            best_row = row
            best_daily = daily
            best_month = month_df

    top_single = sorted(single_results, key=lambda x: (bool(x[0]["pass_hard"]), float(x[0]["objective"])), reverse=True)[:40]
    for (row_a, name_a, mask_a), (row_b, name_b, mask_b) in combinations(top_single, 2):
        mask = mask_a & mask_b
        if not bool(mask.any()):
            continue
        name = f"({name_a}) AND ({name_b})"
        row, daily, month_df = evaluate_mask(name, mask, base_daily, cand_diff, base_summary)
        rows.append(row)
        if best_row is None or (bool(row["pass_hard"]), float(row["objective"])) > (bool(best_row["pass_hard"]), float(best_row["objective"])):
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
        "scope": "research_only_10d_exante_score_state_gate_v65",
        "label": "executable_10d_open_return",
        "base_table": BASE_TABLE,
        "candidate_table": CAND_TABLE,
        "feature_source": "same-day prediction score distributions and rank differences only",
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

    report = f"""# 10D 前视分数状态门控扫描 v65

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
