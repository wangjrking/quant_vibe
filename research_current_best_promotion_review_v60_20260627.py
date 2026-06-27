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
SNAPSHOT = DATA_DIR / "reports" / "model_agent_current_best_research_snapshot_20260627_v59" / "current_best_research_snapshot_v59.json"
CONSTRAINTS = ROOT / "quant" / "main" / "config" / "model_promotion_constraints_v1_20260625.json"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_current_best_promotion_review_20260627_v60"

BASELINES = {
    "executable_1d_open_return": "stock_predict_data_model_agent_1d_daygate_best_20260627_executable_1d_open_return_research",
    "executable_3d_open_return": "stock_predict_data_model_agent_3d_recent_top_daygate_best_20260627_executable_3d_open_return_research",
    "executable_5d_open_return": "stock_predict_data_model_agent_5d_recent_top_condblend_latest_20260626_executable_5d_open_return_research",
    "executable_10d_open_return": "stock_predict_data_model_agent_10d_regime_blend_best_20260627_executable_10d_open_return_research",
}
METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_score_table(conn: sqlite3.Connection, table: str, score_col: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": score_col})


def table_summary(conn: sqlite3.Connection, table: str) -> dict[str, object]:
    row = conn.execute(
        f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(table)}"
    ).fetchone()
    dup = conn.execute(
        f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(table)} group by trade_date, stock_code having c > 1)"
    ).fetchone()[0]
    latest = conn.execute(
        f"select trade_date, count(*), count(distinct stock_code) from {quote(table)} group by trade_date order by trade_date desc limit 5"
    ).fetchall()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def load_labels(label: str, min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", label])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[label])
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise RuntimeError(f"no labels loaded for {label}")
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, score_col: str, label: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), score_col)[label].mean())


def daily_eval(frame: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        label_rank = group[label].rank(method="average", pct=True)
        score = group[score_col].rank(method="average", pct=True)
        row = {"trade_date": trade_date, "rank_ic": float(score.corr(label_rank))}
        ranked = group.assign(_score_rank=score)
        for k in [1, 3, 5, 10, 20, 50]:
            row[f"top{k}"] = top_mean(ranked, "_score_rank", label, k)
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


def monthly_stats(candidate_daily: pd.DataFrame, base_daily: pd.DataFrame) -> dict[str, object]:
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
    return {
        "month_count": int(len(df)),
        "positive_rank_ic_months": int((df["rank_ic_delta"] > 0).sum()),
        "positive_top1_months": int((df["top1_delta"] > 0).sum()),
        "positive_top5_months": int((df["top5_delta"] > 0).sum()),
        "nonnegative_top5_months": int((df["top5_delta"] >= 0).sum()),
        "positive_top1_and_nonnegative_top5_months": int(((df["top1_delta"] > 0) & (df["top5_delta"] >= 0)).sum()),
        "min_rank_ic_delta": float(df["rank_ic_delta"].min()),
        "min_top5_delta": float(df["top5_delta"].min()),
    }


def check_hard(label: str, deltas: dict[str, dict[str, float]], monthly: dict[str, object], summary: dict[str, object], constraints: dict[str, object]) -> tuple[bool, list[str], list[str]]:
    hard = constraints["hard_constraints"]
    common = hard["quality"]["common"]
    by_label = hard["quality"]["by_label"][label]
    failures = []
    reminders = []
    if summary["min_trade_date"] > hard["coverage"]["required_min_trade_date"]:
        failures.append(f"coverage_min_trade_date_after_required:{summary['min_trade_date']}")
    if summary["null_pred_prob"] != hard["coverage"]["null_pred_prob_must_equal"]:
        failures.append(f"null_pred_prob={summary['null_pred_prob']}")
    if summary["duplicate_key_groups"] != hard["coverage"]["duplicate_key_groups_must_equal"]:
        failures.append(f"duplicate_key_groups={summary['duplicate_key_groups']}")
    if deltas["full"]["top5"] < common["full_top5_delta_floor"]:
        failures.append(f"full_top5_delta={deltas['full']['top5']:.12g}")
    if deltas["recent63"]["top5"] < common["recent63_top5_delta_floor"]:
        failures.append(f"recent63_top5_delta={deltas['recent63']['top5']:.12g}")
    if deltas["recent20"]["top5"] < common["recent20_top5_delta_floor"]:
        failures.append(f"recent20_top5_delta={deltas['recent20']['top5']:.12g}")
    if monthly["positive_top5_months"] < common["positive_top5_periods_floor"]:
        failures.append(f"positive_top5_months={monthly['positive_top5_months']}")
    if deltas["full"]["rank_ic"] < by_label["full_rank_ic_delta_floor"]:
        failures.append(f"full_rank_ic_delta={deltas['full']['rank_ic']:.12g}")
    if monthly["min_top5_delta"] < by_label["min_period_top5_delta_floor"]:
        failures.append(f"min_month_top5_delta={monthly['min_top5_delta']:.12g}")
    if deltas["recent63"]["top1"] <= 0:
        reminders.append(f"recent63_top1_delta_nonpositive:{deltas['recent63']['top1']:.12g}")
    if deltas["recent20"]["top1"] <= 0:
        reminders.append(f"recent20_top1_delta_nonpositive:{deltas['recent20']['top1']:.12g}")
    return (len(failures) == 0), failures, reminders


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    constraints = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    active = {item["label"]: item for item in snapshot["current_best_mainlines"]}
    rows = []
    packets = {}
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        for label, item in active.items():
            table = item["table"]
            baseline = BASELINES[label]
            cand_summary_db = table_summary(conn, table)
            base_summary_db = table_summary(conn, baseline)
            scores = read_score_table(conn, table, "candidate_score")
            base_scores = read_score_table(conn, baseline, "base_score")
            labels = load_labels(label, max(cand_summary_db["min_trade_date"], base_summary_db["min_trade_date"]), min(cand_summary_db["max_trade_date"], base_summary_db["max_trade_date"]))
            frame = scores.merge(base_scores, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            frame = frame.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            cand_daily = daily_eval(frame, "candidate_score", label)
            base_daily = daily_eval(frame, "base_score", label)
            cand_daily.to_csv(REPORT_DIR / f"{label}_candidate_daily_eval.csv", index=False, encoding="utf-8-sig")
            base_daily.to_csv(REPORT_DIR / f"{label}_baseline_daily_eval.csv", index=False, encoding="utf-8-sig")
            cand_metrics = summarize(cand_daily)
            base_metrics = summarize(base_daily)
            deltas = delta(cand_metrics, base_metrics)
            monthly = monthly_stats(cand_daily, base_daily)
            pass_hard, failures, reminders = check_hard(label, deltas, monthly, cand_summary_db, constraints)
            decision = "ready_for_l4_candidate_review" if pass_hard and table != baseline else "continue_research"
            if table == baseline:
                failures = failures + ["no_new_candidate_vs_baseline"]
            packet = {
                "label": label,
                "asset": item.get("asset"),
                "table": table,
                "baseline_table": baseline,
                "status": item.get("status"),
                "coverage": cand_summary_db,
                "eval_window": {
                    "days": int(len(cand_daily)),
                    "min_eval_date": str(cand_daily["trade_date"].min()),
                    "max_eval_date": str(cand_daily["trade_date"].max()),
                    "matched_rows": int(len(frame)),
                },
                "candidate_metrics": cand_metrics,
                "baseline_metrics": base_metrics,
                "delta_vs_baseline": deltas,
                "monthly_delta_stats": monthly,
                "hard_gate_pass": bool(pass_hard and table != baseline),
                "hard_gate_failures": failures,
                "soft_reminders": reminders,
                "decision": decision,
            }
            packets[label] = packet
            rows.append(
                {
                    "label": label,
                    "table": table,
                    "baseline_table": baseline,
                    "hard_gate_pass": packet["hard_gate_pass"],
                    "decision": decision,
                    "full_rank_ic_delta": deltas["full"]["rank_ic"],
                    "full_top1_delta": deltas["full"]["top1"],
                    "full_top5_delta": deltas["full"]["top5"],
                    "recent63_top1_delta": deltas["recent63"]["top1"],
                    "recent63_top5_delta": deltas["recent63"]["top5"],
                    "recent20_top1_delta": deltas["recent20"]["top1"],
                    "recent20_top5_delta": deltas["recent20"]["top5"],
                    "positive_top5_months": monthly["positive_top5_months"],
                    "min_month_top5_delta": monthly["min_top5_delta"],
                    "failures": ";".join(failures),
                    "reminders": ";".join(reminders),
                }
            )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(REPORT_DIR / "current_best_promotion_matrix_v60.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_current_best_promotion_review_v60",
        "snapshot": str(SNAPSHOT),
        "constraints": str(CONSTRAINTS),
        "reviewed_labels": list(active.keys()),
        "ready_for_l4_candidate_review": [row["label"] for row in rows if row["hard_gate_pass"]],
        "continue_research": [row["label"] for row in rows if not row["hard_gate_pass"]],
        "packets": packets,
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
    (REPORT_DIR / "current_best_promotion_review_v60.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 当前最优研究模型统一复核 v60",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        f"- 可进入 L4 candidate 送审准备：`{', '.join(summary['ready_for_l4_candidate_review']) or '无'}`",
        f"- 继续研究：`{', '.join(summary['continue_research']) or '无'}`",
        "",
        "## 边界",
        "",
        "- 未训练模型",
        "- 未写入新预测表",
        "- 未改 formal manifest",
        "- 未改 production manifest",
        "- 未生成交易信号",
        "- 未运行策略回测",
        "",
        "## 证据",
        "",
        "- `current_best_promotion_review_v60.json`",
        "- `current_best_promotion_matrix_v60.csv`",
        "- 各标签 `*_candidate_daily_eval.csv` 与 `*_baseline_daily_eval.csv`",
    ]
    (REPORT_DIR / "current_best_promotion_review_v60.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ready": summary["ready_for_l4_candidate_review"], "continue": summary["continue_research"], "report_dir": str(REPORT_DIR)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
