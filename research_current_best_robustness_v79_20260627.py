from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
SNAPSHOT = DATA_DIR / "reports" / "model_agent_current_best_research_snapshot_20260627_v78" / "current_best_research_snapshot_v78.json"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_current_best_robustness_20260627_v79"


FORMAL_TABLES = {
    "executable_1d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
    "executable_3d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "executable_5d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
    "executable_10d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
}
METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_scores(conn: sqlite3.Connection, table: str, alias: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    ).rename(columns={"pred_prob": alias})
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def load_labels(label: str, min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        try:
            part = pd.read_parquet(path, columns=["trade_date", "stock_code", label])
        except Exception:
            continue
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[label])
        if not part.empty:
            chunks.append(part)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["trade_date", "stock_code", label])


def top_mean(group: pd.DataFrame, score_col: str, label: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), score_col)[label].mean()) if len(group) else 0.0


def daily_eval(frame: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in frame.dropna(subset=[score_col, label]).groupby("trade_date", sort=True):
        if len(group) < 100:
            continue
        score_rank = group[score_col].rank(method="average", pct=True)
        label_rank = group[label].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "rank_ic": float(score_rank.corr(label_rank)),
        }
        for n in [1, 3, 5, 10, 20]:
            row[f"top{n}"] = top_mean(group, score_col, label, n)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summarize(daily: pd.DataFrame) -> dict[str, float]:
    return {metric: float(daily[metric].mean()) for metric in METRICS}


def delta_frame(candidate_daily: pd.DataFrame, formal_daily: pd.DataFrame) -> pd.DataFrame:
    merged = candidate_daily.merge(formal_daily, on="trade_date", suffixes=("_candidate", "_formal"), validate="one_to_one")
    out = pd.DataFrame({"trade_date": merged["trade_date"]})
    for metric in METRICS:
        out[f"{metric}_delta"] = merged[f"{metric}_candidate"] - merged[f"{metric}_formal"]
    out["month"] = out["trade_date"].str.slice(0, 6)
    out["quarter"] = out["trade_date"].str.slice(0, 4) + "Q" + (((out["trade_date"].str.slice(4, 6).astype(int) - 1) // 3) + 1).astype(str)
    return out


def period_summary(delta: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for period, group in delta.groupby(by, sort=True):
        row = {by: period, "days": int(len(group))}
        for metric in METRICS:
            row[f"{metric}_delta_mean"] = float(group[f"{metric}_delta"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def table_summary(conn: sqlite3.Connection, table: str) -> dict[str, object]:
    row = conn.execute(
        f"""
        select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
               sum(case when pred_prob is null then 1 else 0 end)
        from {quote(table)}
        """
    ).fetchone()
    latest = conn.execute(
        f"""
        select trade_date, count(*), count(distinct stock_code)
        from {quote(table)}
        group by trade_date
        order by trade_date desc
        limit 1
        """
    ).fetchone()
    dup = conn.execute(
        f"""
        select count(*) from (
          select trade_date, stock_code, count(*) c
          from {quote(table)}
          group by trade_date, stock_code
          having c > 1
        )
        """
    ).fetchone()[0]
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_trade_date": str(latest[0]),
        "latest_day_rows": int(latest[1]),
        "latest_day_stocks": int(latest[2]),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    rows = []
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_current_best_robustness_v79",
        "snapshot": str(SNAPSHOT),
        "results": {},
        "boundaries": {
            "research_only": True,
            "read_only_evaluation": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    with sqlite3.connect(DB_PATH, timeout=300) as conn:
        for label, item in snapshot["current_best"].items():
            candidate_table = str(item["table"])
            formal_table = FORMAL_TABLES[label]
            candidate_summary = table_summary(conn, candidate_table)
            formal_summary = table_summary(conn, formal_table)
            min_date = max(candidate_summary["min_trade_date"], formal_summary["min_trade_date"])
            max_date = min(candidate_summary["max_trade_date"], formal_summary["max_trade_date"])
            candidate = read_scores(conn, candidate_table, "candidate_score")
            formal = read_scores(conn, formal_table, "formal_score")
            labels = load_labels(label, min_date, max_date)
            eval_frame = candidate.merge(formal, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            eval_frame = eval_frame.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            candidate_daily = daily_eval(eval_frame, "candidate_score", label)
            formal_daily = daily_eval(eval_frame, "formal_score", label)
            delta = delta_frame(candidate_daily, formal_daily)
            month = period_summary(delta, "month")
            quarter = period_summary(delta, "quarter")
            recent = {
                "full": {metric: float(delta[f"{metric}_delta"].mean()) for metric in METRICS},
                "recent126": {metric: float(delta.tail(126)[f"{metric}_delta"].mean()) for metric in METRICS},
                "recent63": {metric: float(delta.tail(63)[f"{metric}_delta"].mean()) for metric in METRICS},
                "recent20": {metric: float(delta.tail(20)[f"{metric}_delta"].mean()) for metric in METRICS},
            }
            month.to_csv(REPORT_DIR / f"{label}_monthly_delta.csv", index=False, encoding="utf-8-sig")
            quarter.to_csv(REPORT_DIR / f"{label}_quarterly_delta.csv", index=False, encoding="utf-8-sig")
            candidate_daily.to_csv(REPORT_DIR / f"{label}_candidate_daily_eval.csv", index=False, encoding="utf-8-sig")
            formal_daily.to_csv(REPORT_DIR / f"{label}_formal_daily_eval.csv", index=False, encoding="utf-8-sig")
            diagnostics = {
                "asset": item["asset"],
                "candidate_table": candidate_table,
                "formal_table": formal_table,
                "eval_days": int(len(delta)),
                "eval_min_date": str(delta["trade_date"].min()),
                "eval_max_date": str(delta["trade_date"].max()),
                "candidate_summary": candidate_summary,
                "formal_summary": formal_summary,
                "recent": recent,
                "month_top5_positive": int((month["top5_delta_mean"] > 0).sum()),
                "month_top5_nonnegative": int((month["top5_delta_mean"] >= 0).sum()),
                "month_top5_min": float(month["top5_delta_mean"].min()),
                "quarter_top5_positive": int((quarter["top5_delta_mean"] > 0).sum()),
                "quarter_top5_nonnegative": int((quarter["top5_delta_mean"] >= 0).sum()),
                "quarter_top5_min": float(quarter["top5_delta_mean"].min()),
            }
            payload["results"][label] = diagnostics
            rows.append(
                {
                    "label": label,
                    "asset": item["asset"],
                    "table": candidate_table,
                    "eval_days": diagnostics["eval_days"],
                    "eval_max_date": diagnostics["eval_max_date"],
                    "full_rank_ic_delta": recent["full"]["rank_ic"],
                    "full_top1_delta": recent["full"]["top1"],
                    "full_top5_delta": recent["full"]["top5"],
                    "recent126_top5_delta": recent["recent126"]["top5"],
                    "recent63_top5_delta": recent["recent63"]["top5"],
                    "recent20_top5_delta": recent["recent20"]["top5"],
                    "month_top5_positive": diagnostics["month_top5_positive"],
                    "month_top5_min": diagnostics["month_top5_min"],
                    "quarter_top5_positive": diagnostics["quarter_top5_positive"],
                    "quarter_top5_min": diagnostics["quarter_top5_min"],
                }
            )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(REPORT_DIR / "robustness_matrix_v79.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "robustness_summary_v79.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 当前最优研究模型稳健性复核 v79",
        "",
        "## 结论",
        "",
        "本报告只做模型侧只读评价，不训练、不写预测表、不修改 formal manifest、不生成信号、不跑回测。",
        "",
        "## 摘要",
        "",
    ]
    for row in rows:
        boundary_note = ""
        if row["month_top5_min"] < 0:
            boundary_note = "，存在负贡献月份"
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 资产：`{row['asset']}`",
                f"- 可评价截止：`{row['eval_max_date']}`，可评价日 `{row['eval_days']}`",
                f"- 全窗口 Top5 delta：`{row['full_top5_delta']:.10f}`",
                f"- 近 126 日 Top5 delta：`{row['recent126_top5_delta']:.10f}`",
                f"- 近 63 日 Top5 delta：`{row['recent63_top5_delta']:.10f}`",
                f"- 近 20 日 Top5 delta：`{row['recent20_top5_delta']:.10f}`",
                f"- 月度 Top5 正贡献 `{row['month_top5_positive']}`，最差月 `{row['month_top5_min']:.10f}`{boundary_note}",
                f"- 季度 Top5 正贡献 `{row['quarter_top5_positive']}`，最差季度 `{row['quarter_top5_min']:.10f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 证据",
            "",
            "- `robustness_summary_v79.json`",
            "- `robustness_matrix_v79.csv`",
            "- 每个标签的 `*_monthly_delta.csv`、`*_quarterly_delta.csv`、`*_candidate_daily_eval.csv`、`*_formal_daily_eval.csv`。",
        ]
    )
    (REPORT_DIR / "robustness_report_v79.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(matrix.to_string(index=False))
    print(REPORT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
