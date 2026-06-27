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
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_10d_monthly_top5_double_veto_v62_20260627"
FEATURE_PATH = DATA_DIR / "reports" / "model_agent_10d_topzone_second_gate_review_20260627" / "date_score_features.csv"
V61_SCAN = DATA_DIR / "reports" / "model_agent_10d_monthly_top5_veto_v61_20260627" / "monthly_top5_veto_scan.csv"

LABEL = "executable_10d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_10d_regime_blend_best_20260627_executable_10d_open_return_research"
CANDIDATE_TABLE = "stock_predict_data_model_agent_10d_topzone_confirmation_constrained_20260627_executable_10d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_10d_monthly_top5_double_veto_v62_20260627_executable_10d_open_return_research"

BASE_GATE_FEATURE = "mean_abs_rank_gap"
BASE_GATE_THRESHOLD = 0.000740055732491
FEATURES = [
    "row_count",
    "base_score_std",
    "base_score_iqr",
    "cand_score_std",
    "cand_score_iqr",
    "score_corr",
    "mean_abs_rank_gap",
    "top20_overlap",
    "same_top1",
    "base_top5_gap",
    "base_top20_gap",
    "cand_top5_gap",
    "cand_top20_gap",
    "cand_top1_base_rank",
    "base_top1_cand_rank",
]
QUANTILES = [0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90]
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
    features = pd.read_csv(FEATURE_PATH)
    features["trade_date"] = features["trade_date"].astype(str)
    return scores.merge(features, on="trade_date", how="left", validate="many_to_one")


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
        score = group[score_col].rank(method="average", pct=True)
        label_rank = group[LABEL].rank(method="average", pct=True)
        ranked = group.assign(_score_rank=score)
        row = {"trade_date": trade_date, "rank_ic": float(score.corr(label_rank))}
        for k in [1, 3, 5, 10, 20, 50]:
            row[f"top{k}"] = top_mean(ranked, "_score_rank", k)
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


def monthly_delta(candidate_daily: pd.DataFrame, base_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = candidate_daily.merge(base_daily, on="trade_date", suffixes=("", "_base"), validate="one_to_one")
    merged["month"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d").dt.to_period("M").astype(str)
    rows = []
    for month, group in merged.groupby("month", sort=True):
        rows.append(
            {
                "month": month,
                "rank_ic_delta": float(group["rank_ic"].mean() - group["rank_ic_base"].mean()),
                "top1_delta": float(group["top1"].mean() - group["top1_base"].mean()),
                "top5_delta": float(group["top5"].mean() - group["top5_base"].mean()),
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
        "min_rank_ic_delta": float(df["rank_ic_delta"].min()),
        "min_top5_delta": float(df["top5_delta"].min()),
    }
    return df, stats


def merge_daily(base_daily: pd.DataFrame, cand_daily: pd.DataFrame, active_dates: set[str]) -> pd.DataFrame:
    merged = base_daily.merge(cand_daily, on="trade_date", suffixes=("_base", "_cand"), validate="one_to_one")
    use_cand = merged["trade_date"].isin(active_dates)
    out = pd.DataFrame({"trade_date": merged["trade_date"]})
    for metric in METRICS:
        out[metric] = np.where(use_cand, merged[f"{metric}_cand"], merged[f"{metric}_base"])
    return out


def make_conditions(features: pd.DataFrame, base_gate_dates: set[str]) -> list[dict[str, object]]:
    conditions = []
    scan = pd.read_csv(V61_SCAN)
    selected = pd.concat(
        [
            scan.sort_values(["objective", "full_top1_delta"], ascending=False).head(15),
            scan.sort_values(["positive_top5_months", "min_month_top5_delta", "objective"], ascending=False).head(15),
            scan[scan["min_month_top5_delta"] >= 0].sort_values(["positive_top5_months", "objective"], ascending=False).head(15),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["feature", "op", "threshold"])
    for _, item in selected.iterrows():
        feature = str(item["feature"])
        op = str(item["op"])
        threshold = float(item["threshold"])
        dates = set(features.loc[features[feature] <= threshold, "trade_date"] if op == "<=" else features.loc[features[feature] >= threshold, "trade_date"])
        active_dates = base_gate_dates & dates
        if 5 <= len(active_dates) <= len(base_gate_dates):
            conditions.append(
                {
                    "name": f"{feature}_{op}_{threshold:.12g}",
                    "feature": feature,
                    "op": op,
                    "threshold": threshold,
                    "active_dates": active_dates,
                }
            )
    return conditions


def evaluate_candidate(
    active_dates: set[str],
    base_daily: pd.DataFrame,
    cand_daily: pd.DataFrame,
    base_summary: dict[str, dict[str, float]],
) -> tuple[dict[str, object], pd.DataFrame, dict[str, dict[str, float]], dict[str, object]]:
    daily = merge_daily(base_daily, cand_daily, active_dates)
    cand_summary = summarize(daily)
    deltas = delta(cand_summary, base_summary)
    month_df, month_stats = monthly_delta(daily, base_daily)
    objective = (
        3.0 * deltas["recent20"]["top1"]
        + 2.0 * deltas["recent63"]["top1"]
        + 1.2 * deltas["full"]["top1"]
        + 3.5 * deltas["full"]["top5"]
        + 2.2 * deltas["recent63"]["top5"]
        + 1.7 * deltas["recent20"]["top5"]
        + 0.08 * month_stats["positive_top5_months"]
        + 0.03 * month_stats["nonnegative_top5_months"]
        - 5.0 * max(0.0, -month_stats["min_top5_delta"])
    )
    row = {
        "active_days": len(active_dates),
        "objective": float(objective),
        "full_rank_ic_delta": deltas["full"]["rank_ic"],
        "full_top1_delta": deltas["full"]["top1"],
        "full_top5_delta": deltas["full"]["top5"],
        "recent126_top5_delta": deltas["recent126"]["top5"],
        "recent63_top1_delta": deltas["recent63"]["top1"],
        "recent63_top5_delta": deltas["recent63"]["top5"],
        "recent20_top1_delta": deltas["recent20"]["top1"],
        "recent20_top5_delta": deltas["recent20"]["top5"],
        "positive_top5_months": month_stats["positive_top5_months"],
        "nonnegative_top5_months": month_stats["nonnegative_top5_months"],
        "min_month_top5_delta": month_stats["min_top5_delta"],
        "pass_hard": bool(
            deltas["full"]["top5"] >= 0
            and deltas["recent63"]["top5"] >= 0
            and deltas["recent20"]["top5"] >= 0
            and month_stats["positive_top5_months"] >= 3
            and month_stats["min_top5_delta"] >= 0
            and deltas["full"]["rank_ic"] >= -0.0005
        ),
    }
    return row, daily, deltas, month_stats


def scan(base_daily: pd.DataFrame, cand_daily: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object] | None, pd.DataFrame | None]:
    base_gate_dates = set(features.loc[features[BASE_GATE_FEATURE] <= BASE_GATE_THRESHOLD, "trade_date"])
    base_summary = summarize(base_daily)
    conditions = make_conditions(features, base_gate_dates)
    rows = []
    best_payload = None
    best_daily = None
    for left, right in combinations(conditions, 2):
        for combiner in ["and"]:
            active_dates = (left["active_dates"] & right["active_dates"]) if combiner == "and" else (left["active_dates"] | right["active_dates"])
            active_dates = base_gate_dates & active_dates
            if len(active_dates) < 5 or len(active_dates) > len(base_gate_dates):
                continue
            row, daily, deltas, month_stats = evaluate_candidate(active_dates, base_daily, cand_daily, base_summary)
            row.update(
                {
                    "condition_left": left["name"],
                    "condition_right": right["name"],
                    "combiner": combiner,
                    "left_feature": left["feature"],
                    "left_op": left["op"],
                    "left_threshold": left["threshold"],
                    "right_feature": right["feature"],
                    "right_op": right["op"],
                    "right_threshold": right["threshold"],
                }
            )
            rows.append(row)
            if row["pass_hard"] and (best_payload is None or row["objective"] > best_payload["row"]["objective"]):
                best_payload = {
                    "row": row,
                    "active_dates": sorted(active_dates),
                    "delta_vs_base": deltas,
                    "monthly_stats": month_stats,
                    "monthly_delta": monthly_delta(daily, base_daily)[0],
                }
                best_daily = daily
    result = pd.DataFrame(rows).sort_values(
        ["pass_hard", "objective", "positive_top5_months", "full_top1_delta"],
        ascending=[False, False, False, False],
    )
    return result, best_payload, best_daily


def build_scores(scores: pd.DataFrame, active_dates: set[str]) -> pd.DataFrame:
    out = scores[["trade_date", "stock_code", "base_rank", "cand_rank"]].copy()
    use_cand = out["trade_date"].isin(active_dates)
    out["pred_prob"] = np.where(use_cand, out["cand_rank"], out["base_rank"])
    out["score_formula"] = "base_rank"
    out.loc[use_cand, "score_formula"] = "cand_rank after monthly_top5_double_veto_v62"
    return out


def write_table(scores: pd.DataFrame, active_dates: set[str]) -> dict[str, object]:
    out = build_scores(scores, active_dates)
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[["trade_date", "stock_code", "pred_prob", "base_rank", "cand_rank", "score_formula"]].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_10d_monthly_top5_double_veto_v62_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
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
    eval_base = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    base_daily = daily_eval(eval_base, "base_rank")
    cand_daily = daily_eval(eval_base, "cand_rank")
    features = scores[["trade_date", *FEATURES]].drop_duplicates().sort_values("trade_date")
    result, best, best_daily = scan(base_daily, cand_daily, features)
    result.to_csv(REPORT_DIR / "monthly_top5_double_veto_scan.csv", index=False, encoding="utf-8-sig")
    asset_stats = None
    if best is not None:
        asset_stats = write_table(scores, set(best["active_dates"]))
        best["monthly_delta"].to_csv(REPORT_DIR / "best_monthly_delta.csv", index=False, encoding="utf-8-sig")
        best_daily.to_csv(REPORT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_monthly_top5_double_veto_v62",
        "label": LABEL,
        "base_table": BASE_TABLE,
        "candidate_table": CANDIDATE_TABLE,
        "target_table": TARGET_TABLE if asset_stats else None,
        "base_gate": f"{BASE_GATE_FEATURE} <= {BASE_GATE_THRESHOLD}",
        "scan_rows": int(len(result)),
        "pass_hard_count": int(result["pass_hard"].sum()) if len(result) else 0,
        "best": best["row"] if best is not None else (result.iloc[0].to_dict() if len(result) else None),
        "best_delta_vs_base": best["delta_vs_base"] if best is not None else None,
        "best_monthly_stats": best["monthly_stats"] if best is not None else None,
        "asset_stats": asset_stats,
        "decision": "promote_v62_research_candidate" if asset_stats else "no_double_veto_candidate_passed",
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
    (REPORT_DIR / "monthly_top5_double_veto_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 10D 月度 Top5 双条件 veto 搜索 v62",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        f"决策：`{summary['decision']}`。",
        "",
        "本轮只在 research-only 范围内扫描双条件日期级 veto，用于修复 v60 暴露的 10D 最差月份 Top5 负增量。",
        "",
        "## 边界",
        "",
        "- 未训练模型",
        "- 未改 formal manifest",
        "- 未改 production manifest",
        "- 未生成交易信号",
        "- 未运行策略回测",
    ]
    (REPORT_DIR / "monthly_top5_double_veto_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "pass_hard_count": summary["pass_hard_count"], "best": summary["best"], "asset_stats": asset_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
