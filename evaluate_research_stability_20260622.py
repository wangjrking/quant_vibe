from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from evaluate_prediction_asset import evaluate_frame


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_stability_20260622"
LEADERBOARD_PATH = DATA_DIR / "reports" / "model_agent_research_leaderboard_20260622" / "research_leaderboard.json"

TOP_K = [1, 3, 5, 10, 20, 50]
METRICS = ["rank_ic", "top1_mean", "top3_mean", "top5_mean", "top10_mean", "top20_mean", "top50_mean"]


def u(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


def read_prediction(conn: sqlite3.Connection, table: str, label_col: str) -> pd.DataFrame:
    query = (
        f"select trade_date, stock_code, pred_prob, {label_col} "
        f"from '{table}' where trade_date >= '20240604' and trade_date <= '20260618'"
    )
    frame = pd.read_sql_query(query, conn)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def daily_eval(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    _, daily = evaluate_frame(
        frame,
        score_col="pred_prob",
        label_col=label_col,
        top_k=TOP_K,
        quantiles=10,
    )
    daily["month"] = daily["trade_date"].str.slice(0, 6)
    daily["half_year"] = (
        daily["trade_date"].str.slice(0, 4)
        + "H"
        + daily["trade_date"].str.slice(4, 6).astype(int).map(lambda month: "1" if month <= 6 else "2")
    )
    return daily


def aggregate_delta(delta_daily: pd.DataFrame, period_col: str) -> pd.DataFrame:
    grouped = delta_daily.groupby(period_col, sort=True)
    rows = []
    for period, group in grouped:
        row = {period_col: period, "trade_days": int(len(group))}
        for metric in METRICS:
            col = f"{metric}_delta"
            row[col] = float(group[col].mean())
            row[f"{metric}_win_rate"] = float((group[col] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def front_score(row: dict[str, float]) -> float:
    return (
        float(row["top1_mean_delta"]) * 3.0
        + float(row["top3_mean_delta"]) * 2.0
        + float(row["top5_mean_delta"]) * 1.5
        + float(row["top10_mean_delta"])
        + float(row["rank_ic_delta"]) * 0.10
    )


def build_markdown(summary: pd.DataFrame) -> str:
    title = u("\\u6a21\\u578b\\u7814\\u7a76\\u8d44\\u4ea7\\u7a33\\u5b9a\\u6027\\u8bca\\u65ad\\uff0820260622\\uff09")
    conclusion = u("\\u7ed3\\u8bba")
    observations = u("\\u6a21\\u578b\\u4fa7\\u89c2\\u5bdf")
    evidence = u("\\u8bc1\\u636e\\u8def\\u5f84")
    total_table = u("\\u603b\\u8868")
    monthly_table = u("\\u6708\\u5ea6")
    half_year_table = u("\\u534a\\u5e74\\u5ea6")
    header = u(
        "\\u007c \\u6807\\u7b7e \\u007c \\u5019\\u9009 \\u007c \\u65e5\\u6570 \\u007c front delta "
        "\\u007c RankIC delta \\u007c RankIC \\u80dc\\u7387 \\u007c Top1 delta \\u007c Top1 \\u80dc\\u7387 "
        "\\u007c Top3 delta \\u007c Top3 \\u80dc\\u7387 \\u007c"
    )
    lines = [
        f"# {title}",
        "",
        f"## {conclusion}",
        "",
        u(
            "\\u672c\\u8f6e\\u53ea\\u505a\\u8bc4\\u4ef7\\u7a33\\u5b9a\\u6027\\u8bca\\u65ad\\uff0c"
            "\\u672a\\u8bad\\u7ec3\\u6a21\\u578b\\uff0c\\u672a\\u751f\\u6210\\u65b0\\u9884\\u6d4b\\u8d44\\u4ea7\\uff0c"
            "\\u672a\\u6539 formal manifest\\uff0c\\u672a\\u751f\\u6210\\u4ea4\\u6613\\u4fe1\\u53f7\\u6216\\u56de\\u6d4b\\u7ed3\\u8bba\\u3002"
        ),
        "",
        header,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| `{row['label']}` | `{row['retained_name']}` | {int(row['trade_days'])} | "
            f"{row['front_delta_score']:+.6f} | {row['rank_ic_delta']:+.6f} | {row['rank_ic_win_rate']:.3f} | "
            f"{row['top1_mean_delta']:+.6f} | {row['top1_mean_win_rate']:.3f} | "
            f"{row['top3_mean_delta']:+.6f} | {row['top3_mean_win_rate']:.3f} |"
        )

    lines += [
        "",
        f"## {observations}",
        "",
        u("- `10D`\\uff1aTop1/Top3/Top5/Top10 \\u5bf9 formal \\u7684\\u5e73\\u5747\\u6539\\u5584\\u6700\\u660e\\u786e\\uff0c\\u4f46 RankIC \\u4e0e formal \\u57fa\\u672c\\u6301\\u5e73\\uff0c\\u5c5e\\u4e8e\\u524d\\u6392\\u91cd\\u6392\\u578b\\u6539\\u5584\\u3002"),
        u("- `5D`\\uff1a\\u76f8\\u5bf9 formal \\u548c 5D aux \\u5747\\u4fdd\\u6301\\u524d\\u6392\\u6539\\u5584\\uff0c\\u7a33\\u5b9a\\u6027\\u9700\\u8981\\u4f18\\u5148\\u770b\\u7b56\\u7565\\u4fa7\\u5bf9 Top1/Top3 \\u7684\\u627f\\u63a5\\u3002"),
        u("- `3D`\\uff1aRankIC \\u548c Top1 \\u63d0\\u5347\\u660e\\u663e\\uff0c\\u4f46 Top3 \\u56de\\u64a4\\u4ecd\\u662f\\u4e3b\\u8981\\u98ce\\u9669\\uff0c\\u540e\\u7eed\\u4e0d\\u5e94\\u7ee7\\u7eed\\u53ea\\u505a\\u8de8\\u5468\\u671f rank \\u5fae\\u8c03\\u3002"),
        u("- `1D`\\uff1a\\u76f8\\u5bf9 formal \\u6539\\u5584\\u663e\\u8457\\uff0c\\u4f46\\u7edd\\u5bf9\\u6536\\u76ca\\u4ecd\\u5f31\\uff1b\\u4e0b\\u4e00\\u6b65\\u5e94\\u4f18\\u5148\\u8003\\u8651\\u91cd\\u65b0\\u8bad\\u7ec3\\u6216\\u4e13\\u7528\\u76ee\\u6807\\u51fd\\u6570\\u3002"),
        "",
        f"## {evidence}",
        "",
        f"- JSON: `{(REPORT_DIR / 'stability_summary.json').as_posix()}`",
        f"- {total_table} CSV: `{(REPORT_DIR / 'stability_summary.csv').as_posix()}`",
        f"- {monthly_table} CSV: `{(REPORT_DIR / 'stability_monthly_all.csv').as_posix()}`",
        f"- {half_year_table} CSV: `{(REPORT_DIR / 'stability_half_year_all.csv').as_posix()}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))["leaderboard"]

    label_rows = []
    monthly_all = []
    half_all = []

    with sqlite3.connect(MODEL_DB) as conn:
        for item in leaderboard:
            label = item["label"]
            formal_table = item["formal_metrics"]["table"]
            research_table = item["table"]

            formal_daily = daily_eval(read_prediction(conn, formal_table, label), label)
            research_daily = daily_eval(read_prediction(conn, research_table, label), label)
            merged = research_daily.merge(
                formal_daily,
                on=["trade_date", "month", "half_year"],
                suffixes=("_research", "_formal"),
                validate="one_to_one",
            )
            delta = merged[["trade_date", "month", "half_year"]].copy()
            for metric in METRICS:
                delta[f"{metric}_delta"] = merged[f"{metric}_research"] - merged[f"{metric}_formal"]
            delta["label"] = label
            delta["retained_name"] = item["retained_name"]
            delta.to_csv(REPORT_DIR / f"{label}_daily_delta.csv", index=False, encoding="utf-8-sig")

            month = aggregate_delta(delta, "month")
            month.insert(0, "label", label)
            month.insert(1, "retained_name", item["retained_name"])
            month.to_csv(REPORT_DIR / f"{label}_monthly_delta.csv", index=False, encoding="utf-8-sig")
            monthly_all.append(month)

            half = aggregate_delta(delta, "half_year")
            half.insert(0, "label", label)
            half.insert(1, "retained_name", item["retained_name"])
            half.to_csv(REPORT_DIR / f"{label}_half_year_delta.csv", index=False, encoding="utf-8-sig")
            half_all.append(half)

            full = {f"{metric}_delta": float(delta[f"{metric}_delta"].mean()) for metric in METRICS}
            wins = {f"{metric}_win_rate": float((delta[f"{metric}_delta"] > 0).mean()) for metric in METRICS}
            label_rows.append(
                {
                    "label": label,
                    "retained_name": item["retained_name"],
                    "research_table": research_table,
                    "formal_table": formal_table,
                    "trade_days": int(len(delta)),
                    "front_delta_score": front_score(full),
                    **full,
                    **wins,
                    "caveat": item["caveat"],
                }
            )

    summary = pd.DataFrame(label_rows).sort_values("front_delta_score", ascending=False)
    monthly = pd.concat(monthly_all, ignore_index=True)
    half_year = pd.concat(half_all, ignore_index=True)
    summary.to_csv(REPORT_DIR / "stability_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(REPORT_DIR / "stability_monthly_all.csv", index=False, encoding="utf-8-sig")
    half_year.to_csv(REPORT_DIR / "stability_half_year_all.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": "2026-06-22T09:05:00+08:00",
        "scope": "research_only_vs_formal_stability",
        "summary": label_rows,
        "outputs": {
            "summary_csv": str(REPORT_DIR / "stability_summary.csv"),
            "monthly_csv": str(REPORT_DIR / "stability_monthly_all.csv"),
            "half_year_csv": str(REPORT_DIR / "stability_half_year_all.csv"),
        },
        "model_side_limits": [
            "No model trained",
            "No new prediction asset generated",
            "No formal manifest changed",
            "No trading signal generated",
            "No strategy rule or backtest conclusion generated",
        ],
    }
    (REPORT_DIR / "stability_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "stability_report.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "labels": len(label_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
