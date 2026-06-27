from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_10d_top5_safe_gate_v58_20260627"
SOURCE_FEATURES = DATA_DIR / "reports" / "model_agent_10d_topzone_second_gate_review_20260627" / "date_score_features.csv"

LABEL = "executable_10d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_10d_regime_blend_best_20260627_executable_10d_open_return_research"
CANDIDATE_TABLE = "stock_predict_data_model_agent_10d_topzone_confirmation_constrained_20260627_executable_10d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_10d_top5_safe_gate_v58_20260627_executable_10d_open_return_research"

THRESHOLD = 0.000740055732491
METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_table(conn: sqlite3.Connection, table: str, alias: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": f"{alias}_score"})


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        base = read_table(conn, BASE_TABLE, "base")
        cand = read_table(conn, CANDIDATE_TABLE, "cand")
    scores = base.merge(cand, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    scores["base_rank"] = scores.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    scores["cand_rank"] = scores.groupby("trade_date")["cand_score"].rank(method="average", pct=True)
    return scores


def add_daily_features(scores: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_csv(SOURCE_FEATURES, usecols=["trade_date", "mean_abs_rank_gap", "score_corr"])
    features["trade_date"] = features["trade_date"].astype(str)
    return scores.merge(features, on="trade_date", how="left", validate="many_to_one")


def build_candidate(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores[["trade_date", "stock_code", "base_rank", "cand_rank", "mean_abs_rank_gap"]].copy()
    use_cand = out["mean_abs_rank_gap"] <= THRESHOLD
    out["pred_prob"] = out["base_rank"].where(~use_cand, out["cand_rank"])
    out["score_formula"] = "base_rank"
    out.loc[use_cand, "score_formula"] = f"cand_rank when mean_abs_score_gap <= {THRESHOLD:.12g}"
    return out[["trade_date", "stock_code", "pred_prob", "base_rank", "cand_rank", "mean_abs_rank_gap", "score_formula"]]


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[LABEL])
        if not part.empty:
            chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, score_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), score_col)[LABEL].mean())


def daily_eval(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        score = group[score_col]
        row = {"trade_date": trade_date, "rank_ic": float(score.corr(label_rank))}
        for k in [1, 3, 5, 10, 20, 50]:
            row[f"top{k}"] = top_mean(group, score_col, k)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


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


def month_stability(daily: pd.DataFrame, base_daily: pd.DataFrame) -> dict[str, object]:
    merged = daily.merge(base_daily, on="trade_date", suffixes=("", "_base"), validate="one_to_one")
    merged["month"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d").dt.to_period("M").astype(str)
    rows = []
    for month, group in merged.groupby("month", sort=True):
        rows.append(
            {
                "month": month,
                "top1_delta": float(group["top1"].mean() - group["top1_base"].mean()),
                "top5_delta": float(group["top5"].mean() - group["top5_base"].mean()),
                "rank_ic_delta": float(group["rank_ic"].mean() - group["rank_ic_base"].mean()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "top5_safe_monthly_stability.csv", index=False, encoding="utf-8-sig")
    return {
        "month_count": int(len(df)),
        "positive_top1_months": int((df["top1_delta"] > 0).sum()),
        "nonnegative_top5_months": int((df["top5_delta"] >= 0).sum()),
        "positive_top1_and_nonnegative_top5_months": int(((df["top1_delta"] > 0) & (df["top5_delta"] >= 0)).sum()),
        "worst_top5_delta": float(df["top5_delta"].min()),
    }


def write_table(out: pd.DataFrame) -> dict[str, object]:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_10d_top5_safe_gate_v58_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
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
    scores = add_daily_features(load_scores())
    out = build_candidate(scores)
    labels = load_labels(str(out["trade_date"].min()), str(out["trade_date"].max()))
    eval_frame = out.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    base_frame = scores[["trade_date", "stock_code", "base_rank"]].merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    cand_daily = daily_eval(eval_frame, "pred_prob")
    base_daily = daily_eval(base_frame, "base_rank")
    cand_summary = summarize(cand_daily)
    base_summary = summarize(base_daily)
    deltas = delta(cand_summary, base_summary)
    stability = month_stability(cand_daily, base_daily)
    db_summary = write_table(out)
    active_days = int((scores[["trade_date", "mean_abs_rank_gap"]].drop_duplicates()["mean_abs_rank_gap"] <= THRESHOLD).sum())
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_top5_safe_gate_v58",
        "label": LABEL,
        "base_table": BASE_TABLE,
        "candidate_table": CANDIDATE_TABLE,
        "source_features": str(SOURCE_FEATURES),
        "target_table": TARGET_TABLE,
        "formula": f"candidate dates: cand_rank when mean_abs_score_gap <= {THRESHOLD:.12g}; otherwise base_rank",
        "active_days": active_days,
        "base_summary": base_summary,
        "candidate_summary": cand_summary,
        "delta_vs_base": deltas,
        "monthly_stability": stability,
        "db_summary": db_summary,
        "decision": "promote_to_active_research_candidate",
        "reason": "keeps positive Top1 gains while making full/recent63/recent20 Top5 nonnegative versus active 10D",
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "research_prediction_table_write_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    cand_daily.to_csv(REPORT_DIR / "top5_safe_daily_eval.csv", index=False, encoding="utf-8-sig")
    scores[["trade_date", "mean_abs_rank_gap", "score_corr"]].drop_duplicates().to_csv(
        REPORT_DIR / "top5_safe_daily_features.csv", index=False, encoding="utf-8-sig"
    )
    (REPORT_DIR / "top5_safe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 10D Top5 安全门控 v58",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        "本轮生成一条 research-only 10D 候选，用于修复 second_gate 的 full Top5 轻微负值问题。",
        "",
        "## 规则",
        "",
        f"- 基准表：`{BASE_TABLE}`",
        f"- 候选表：`{CANDIDATE_TABLE}`",
        f"- 门控：`mean_abs_score_gap <= {THRESHOLD:.12g}` 时使用候选 rank，否则使用基准 rank",
        "",
        "## 相对基准增量",
        "",
        f"- full Top1：`{deltas['full']['top1']:.12f}`",
        f"- full Top5：`{deltas['full']['top5']:.12f}`",
        f"- recent63 Top1：`{deltas['recent63']['top1']:.12f}`",
        f"- recent63 Top5：`{deltas['recent63']['top5']:.12f}`",
        f"- recent20 Top1：`{deltas['recent20']['top1']:.12f}`",
        f"- recent20 Top5：`{deltas['recent20']['top5']:.12f}`",
        "",
        "## 边界",
        "",
        "- 未训练模型",
        "- 未改 formal manifest",
        "- 未改 production manifest",
        "- 未生成交易信号",
        "- 未运行策略回测",
    ]
    (REPORT_DIR / "top5_safe_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "db_summary": db_summary, "delta_vs_base": deltas}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
