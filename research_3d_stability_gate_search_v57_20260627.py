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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_stability_gate_search_v57_20260627"

LABEL = "executable_3d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_3d_recent_top_daygate_best_20260627_executable_3d_open_return_research"
CANDIDATES = {
    "balanced": "stock_predict_data_model_agent_3d_balanced_daygate_latest_20260627_executable_3d_open_return_research",
    "second_gate": "stock_predict_data_model_agent_3d_active_balanced_second_gate_20260627_executable_3d_open_return_research",
    "proxy_bestmix": "stock_predict_data_model_agent_3d_proxy10d5d_bestmix_20260623_executable_3d_open_return_research",
    "guarded": "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_research",
}
TARGET_TABLE = "stock_predict_data_model_agent_3d_stability_gate_v57_20260627_executable_3d_open_return_research"

FEATURES = [
    "base_score_std",
    "cand_score_std",
    "base_score_iqr",
    "cand_score_iqr",
    "base_top20_gap",
    "cand_top20_gap",
    "rank_corr",
    "top20_overlap",
    "mean_abs_rank_gap",
]
QUANTILES = [0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        out = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(BASE_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "base_score"})
        for alias, table in CANDIDATES.items():
            part = pd.read_sql_query(
                f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
                conn,
            ).rename(columns={"pred_prob": f"{alias}_score"})
            out = out.merge(part, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    out["trade_date"] = out["trade_date"].astype(str)
    out["stock_code"] = out["stock_code"].astype(str)
    out["base_rank"] = out.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    for alias in CANDIDATES:
        out[f"{alias}_rank"] = out.groupby("trade_date")[f"{alias}_score"].rank(method="average", pct=True)
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[LABEL])
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else 0.0


def daily_metrics(eval_frame: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    valid = eval_frame.dropna(subset=[rank_col])
    rows = []
    for trade_date, group in valid.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(group[rank_col].corr(label_rank)),
                "top1": top_mean(group, rank_col, 1),
                "top5": top_mean(group, rank_col, 5),
                "top10": top_mean(group, rank_col, 10),
                "top20": top_mean(group, rank_col, 20),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summarize(daily: pd.DataFrame, window: int | None = None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {col: float(sub[col].mean()) for col in ["rank_ic", "top1", "top5", "top10", "top20"]}


def daily_features(scores: pd.DataFrame, alias: str) -> pd.DataFrame:
    rows = []
    rank_col = f"{alias}_rank"
    score_col = f"{alias}_score"
    for trade_date, group in scores.dropna(subset=[rank_col]).groupby("trade_date", sort=True):
        top20_base = set(group.nlargest(min(20, len(group)), "base_rank")["stock_code"])
        top20_cand = set(group.nlargest(min(20, len(group)), rank_col)["stock_code"])
        rows.append(
            {
                "trade_date": trade_date,
                "base_score_std": float(group["base_score"].std()),
                "cand_score_std": float(group[score_col].std()),
                "base_score_iqr": float(group["base_score"].quantile(0.75) - group["base_score"].quantile(0.25)),
                "cand_score_iqr": float(group[score_col].quantile(0.75) - group[score_col].quantile(0.25)),
                "base_top20_gap": float(group["base_rank"].nlargest(20).mean() - group["base_rank"].nlargest(50).mean()),
                "cand_top20_gap": float(group[rank_col].nlargest(20).mean() - group[rank_col].nlargest(50).mean()),
                "rank_corr": float(group["base_rank"].corr(group[rank_col])),
                "top20_overlap": float(len(top20_base & top20_cand) / max(1, len(top20_base | top20_cand))),
                "mean_abs_rank_gap": float((group["base_rank"] - group[rank_col]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def merge_daily(base_daily: pd.DataFrame, cand_daily: pd.DataFrame, active_dates: set[str]) -> pd.DataFrame:
    merged = base_daily.merge(cand_daily, on="trade_date", how="left", suffixes=("_base", "_cand"), validate="one_to_one")
    use_cand = merged["trade_date"].isin(active_dates) & merged["rank_ic_cand"].notna()
    out = pd.DataFrame({"trade_date": merged["trade_date"]})
    for col in ["rank_ic", "top1", "top5", "top10", "top20"]:
        out[col] = np.where(use_cand, merged[f"{col}_cand"], merged[f"{col}_base"])
    return out


def period_delta(daily: pd.DataFrame, base_daily: pd.DataFrame, freq: str) -> dict[str, float | int]:
    merged = daily.merge(base_daily, on="trade_date", suffixes=("", "_base"), validate="one_to_one")
    dates = pd.to_datetime(merged["trade_date"], format="%Y%m%d")
    if freq == "month":
        key = dates.dt.to_period("M").astype(str)
    elif freq == "quarter":
        key = dates.dt.to_period("Q").astype(str)
    else:
        key = pd.Series(np.arange(len(merged)) // 55, index=merged.index).astype(str)
    rows = []
    for _, group in merged.groupby(key, sort=True):
        if len(group) < 10:
            continue
        rows.append(
            {
                "top1_delta": float(group["top1"].mean() - group["top1_base"].mean()),
                "top5_delta": float(group["top5"].mean() - group["top5_base"].mean()),
                "rank_ic_delta": float(group["rank_ic"].mean() - group["rank_ic_base"].mean()),
            }
        )
    if not rows:
        return {"count": 0, "positive_top1_top5": 0, "positive_rank_ic": 0, "worst_top1": 0.0, "worst_top5": 0.0}
    df = pd.DataFrame(rows)
    return {
        "count": int(len(df)),
        "positive_top1_top5": int(((df["top1_delta"] > 0) & (df["top5_delta"] > 0)).sum()),
        "positive_rank_ic": int((df["rank_ic_delta"] > 0).sum()),
        "worst_top1": float(df["top1_delta"].min()),
        "worst_top5": float(df["top5_delta"].min()),
    }


def rolling63_delta(daily: pd.DataFrame, base_daily: pd.DataFrame) -> dict[str, float | int]:
    merged = daily.merge(base_daily, on="trade_date", suffixes=("", "_base"), validate="one_to_one")
    rows = []
    for start in range(0, max(0, len(merged) - 62)):
        sub = merged.iloc[start : start + 63]
        rows.append(
            {
                "objective_delta": float(
                    2.0 * (sub["top1"].mean() - sub["top1_base"].mean())
                    + 1.5 * (sub["top5"].mean() - sub["top5_base"].mean())
                    + 0.4 * (sub["rank_ic"].mean() - sub["rank_ic_base"].mean())
                ),
                "top1_delta": float(sub["top1"].mean() - sub["top1_base"].mean()),
                "top5_delta": float(sub["top5"].mean() - sub["top5_base"].mean()),
            }
        )
    if not rows:
        return {"windows": 0, "positive_objective": 0, "positive_top1_top5": 0, "worst_objective": 0.0}
    df = pd.DataFrame(rows)
    return {
        "windows": int(len(df)),
        "positive_objective": int((df["objective_delta"] > 0).sum()),
        "positive_top1_top5": int(((df["top1_delta"] > 0) & (df["top5_delta"] > 0)).sum()),
        "worst_objective": float(df["objective_delta"].min()),
    }


def objective(base: dict[str, dict[str, float]], cur: dict[str, dict[str, float]], stability: dict[str, object]) -> float:
    metric_gain = (
        2.0 * (cur["recent63"]["top1"] - base["recent63"]["top1"])
        + 2.0 * (cur["recent63"]["top5"] - base["recent63"]["top5"])
        + 1.2 * (cur["full"]["top1"] - base["full"]["top1"])
        + 1.5 * (cur["full"]["top5"] - base["full"]["top5"])
        + 0.8 * (cur["full"]["rank_ic"] - base["full"]["rank_ic"])
    )
    month = stability["month"]
    quarter = stability["quarter"]
    foldlike = stability["foldlike"]
    roll = stability["rolling63"]
    stability_bonus = (
        0.0004 * month["positive_top1_top5"]
        + 0.0008 * quarter["positive_top1_top5"]
        + 0.0008 * foldlike["positive_top1_top5"]
        + 0.00004 * roll["positive_objective"]
    )
    drawdown_penalty = 0.25 * max(0.0, -float(month["worst_top5"])) + 0.20 * max(0.0, -float(roll["worst_objective"]))
    return float(metric_gain + stability_bonus - drawdown_penalty)


def build_asset(scores: pd.DataFrame, feature_frame: pd.DataFrame, best: pd.Series) -> dict[str, object]:
    alias = str(best["candidate"])
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    active_dates = set(
        feature_frame.loc[feature_frame[feature] <= threshold, "trade_date"]
        if op == "<="
        else feature_frame.loc[feature_frame[feature] >= threshold, "trade_date"]
    )
    out = scores[["trade_date", "stock_code", "base_rank", f"{alias}_rank"]].copy()
    use_cand = out["trade_date"].isin(active_dates) & out[f"{alias}_rank"].notna()
    out["pred_prob"] = np.where(use_cand, out[f"{alias}_rank"], out["base_rank"])
    out["score_formula"] = np.where(use_cand, f"{alias}_rank when {feature} {op} {threshold:.12g}", "base_rank")
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[["trade_date", "stock_code", "pred_prob", "base_rank", f"{alias}_rank", "score_formula"]].to_sql(
            TARGET_TABLE, conn, if_exists="replace", index=False
        )
        conn.execute(f"create index if not exists idx_3d_stability_gate_v57_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 5"
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    base_daily = daily_metrics(eval_frame, "base_rank")
    base_summary = {"full": summarize(base_daily), "recent63": summarize(base_daily, 63), "recent20": summarize(base_daily, 20)}
    rows = []
    feature_frames = {}
    candidate_daily = {}
    for alias in CANDIDATES:
        ff = daily_features(scores, alias)
        feature_frames[alias] = ff
        ff.to_csv(REPORT_DIR / f"{alias}_daily_features.csv", index=False, encoding="utf-8-sig")
        cand_daily = daily_metrics(eval_frame, f"{alias}_rank")
        candidate_daily[alias] = cand_daily
        for feature in FEATURES:
            values = ff[feature].dropna()
            thresholds = sorted(set(float(values.quantile(q)) for q in QUANTILES))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active_dates = set(ff.loc[ff[feature] <= threshold, "trade_date"] if op == "<=" else ff.loc[ff[feature] >= threshold, "trade_date"])
                    if len(active_dates) < 40 or len(active_dates) > 330:
                        continue
                    daily = merge_daily(base_daily, cand_daily, active_dates)
                    cur_summary = {"full": summarize(daily), "recent63": summarize(daily, 63), "recent20": summarize(daily, 20)}
                    stability = {
                        "month": period_delta(daily, base_daily, "month"),
                        "quarter": period_delta(daily, base_daily, "quarter"),
                        "foldlike": period_delta(daily, base_daily, "foldlike"),
                        "rolling63": rolling63_delta(daily, base_daily),
                    }
                    row = {
                        "candidate": alias,
                        "feature": feature,
                        "op": op,
                        "threshold": threshold,
                        "active_days": len(active_dates),
                        "full_rank_ic_delta": cur_summary["full"]["rank_ic"] - base_summary["full"]["rank_ic"],
                        "full_top1_delta": cur_summary["full"]["top1"] - base_summary["full"]["top1"],
                        "full_top5_delta": cur_summary["full"]["top5"] - base_summary["full"]["top5"],
                        "recent63_rank_ic_delta": cur_summary["recent63"]["rank_ic"] - base_summary["recent63"]["rank_ic"],
                        "recent63_top1_delta": cur_summary["recent63"]["top1"] - base_summary["recent63"]["top1"],
                        "recent63_top5_delta": cur_summary["recent63"]["top5"] - base_summary["recent63"]["top5"],
                        "recent20_rank_ic_delta": cur_summary["recent20"]["rank_ic"] - base_summary["recent20"]["rank_ic"],
                        "recent20_top1_delta": cur_summary["recent20"]["top1"] - base_summary["recent20"]["top1"],
                        "recent20_top5_delta": cur_summary["recent20"]["top5"] - base_summary["recent20"]["top5"],
                        "month_positive_top1_top5": stability["month"]["positive_top1_top5"],
                        "quarter_positive_top1_top5": stability["quarter"]["positive_top1_top5"],
                        "foldlike_positive_top1_top5": stability["foldlike"]["positive_top1_top5"],
                        "rolling63_positive_objective": stability["rolling63"]["positive_objective"],
                        "worst_month_top5_delta": stability["month"]["worst_top5"],
                        "worst_rolling63_objective_delta": stability["rolling63"]["worst_objective"],
                    }
                    row["objective"] = objective(base_summary, cur_summary, stability)
                    row["pass_stability"] = bool(
                        row["full_top1_delta"] > 0
                        and row["full_top5_delta"] > 0
                        and row["recent63_top1_delta"] >= 0
                        and row["recent63_top5_delta"] >= 0
                        and row["month_positive_top1_top5"] >= 5
                        and row["quarter_positive_top1_top5"] >= 3
                        and row["foldlike_positive_top1_top5"] >= 3
                        and row["rolling63_positive_objective"] >= 90
                    )
                    rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        [
            "pass_stability",
            "objective",
            "month_positive_top1_top5",
            "quarter_positive_top1_top5",
            "foldlike_positive_top1_top5",
            "recent63_top5_delta",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    results.to_csv(REPORT_DIR / "stability_gate_results.csv", index=False, encoding="utf-8-sig")
    best = results.iloc[0].to_dict() if len(results) else None
    asset_stats = None
    if best and best["pass_stability"] and best["objective"] > 0:
        asset_stats = build_asset(scores, feature_frames[str(best["candidate"])], results.iloc[0])

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_3d_stability_gate_search_v57",
        "base_table": BASE_TABLE,
        "candidate_tables": CANDIDATES,
        "target_table": TARGET_TABLE,
        "base_metrics": base_summary,
        "candidate_count": int(len(results)),
        "pass_stability_count": int(results["pass_stability"].sum()) if len(results) else 0,
        "best": best,
        "asset_stats": asset_stats,
        "decision": "promote_v57_candidate" if asset_stats else "no_stability_gate_promoted",
        "promotion_rule": "requires full/recent63 Top1+Top5 nonnegative and minimum positive month/quarter/foldlike/rolling coverage",
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "research_prediction_table_write_only_if_gate_passed": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "stability_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 3D 稳定门控搜索 v57",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        f"决策：`{summary['decision']}`。",
        "",
        "本轮只在 research-only 范围内扫描已有 3D 候选评分表的日期门控，目标是提高月度、季度、类 fold 和滚动 63 日窗口的稳定覆盖。",
        "",
        "## 边界",
        "",
        "- 未训练模型",
        "- 未改 formal manifest",
        "- 未改 production manifest",
        "- 未生成交易信号",
        "- 未运行策略回测",
        "",
        "## 证据",
        "",
        "- `stability_gate_summary.json`",
        "- `stability_gate_results.csv`",
    ]
    (REPORT_DIR / "stability_gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "pass_stability_count": summary["pass_stability_count"], "best": best, "asset_stats": asset_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
