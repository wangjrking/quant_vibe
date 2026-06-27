from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_best_with_daygate5dhi_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research"
BEST_TABLE = "stock_predict_data_model_agent_1d_proxy10dnew_curgap_top1safe_20260623_executable_1d_open_return_research"
DAYGATE_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_20260623_executable_1d_open_return_research"
DAYGATE_FLAG_CSV = DATA_DIR / "reports" / "model_agent_1d_gate092_new5d040_daygate_20260623" / "daily_gate_flags.csv"
RESEARCH_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_20260623_executable_1d_open_return_research"

LABEL_COL = "executable_1d_open_return"
NEWCOMER_MEAN_5D_PCT_THRESHOLD = 0.729546
WINDOWS = {
    "full": ("20240604", "20260611"),
    "recent126": ("20251201", "20260611"),
    "recent63": ("20260304", "20260611"),
    "q1_2025": ("20250101", "20250331"),
    "q2_2025": ("20250401", "20250630"),
}


def evaluate(score_frame: pd.DataFrame, score_col: str) -> tuple[dict, pd.DataFrame]:
    data = score_frame[["trade_date", "stock_code", LABEL_COL, score_col]].dropna().copy()
    daily_rows: list[dict] = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "top1": float(ordered.head(1)[LABEL_COL].mean()),
                "top3": float(ordered.head(3)[LABEL_COL].mean()),
                "top5": float(ordered.head(5)[LABEL_COL].mean()),
                "top10": float(ordered.head(10)[LABEL_COL].mean()),
                "top20": float(ordered.head(20)[LABEL_COL].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    out = {}
    for name, (date_from, date_to) in WINDOWS.items():
        win = daily[(daily["trade_date"] >= date_from) & (daily["trade_date"] <= date_to)].copy()
        out[name] = {
            "date_from": date_from,
            "date_to": date_to,
            "trade_days": int(len(win)),
            "top1": float(win["top1"].mean()),
            "top3": float(win["top3"].mean()),
            "top5": float(win["top5"].mean()),
            "top10": float(win["top10"].mean()),
            "top20": float(win["top20"].mean()),
        }
    return out, daily


def focus_score(summary: dict) -> float:
    return (
        summary["recent63"]["top1"] * 6
        + summary["recent63"]["top3"] * 4
        + summary["recent63"]["top5"] * 5
        + summary["recent126"]["top1"] * 4
        + summary["recent126"]["top3"] * 3
        + summary["recent126"]["top5"] * 3
        + summary["full"]["top1"] * 0.5
        + summary["full"]["top3"] * 0.4
        + summary["full"]["top5"] * 0.3
    )


def weak_window_score(summary: dict) -> float:
    return (
        summary["q1_2025"]["top1"] * 2
        + summary["q1_2025"]["top5"] * 3
        + summary["q2_2025"]["top1"] * 2
        + summary["q2_2025"]["top5"] * 3
    )


def compare_windows(best_windows: dict, day_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63", "q1_2025", "q2_2025"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"best_{metric}"] = best_windows[name][metric]
            row[f"daygate_{metric}"] = day_windows[name][metric]
            row[f"new_{metric}"] = new_windows[name][metric]
            row[f"new_minus_best_{metric}"] = new_windows[name][metric] - best_windows[name][metric]
            row[f"new_minus_daygate_{metric}"] = new_windows[name][metric] - day_windows[name][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def build_period_breakdown(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = daily.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    if period == "year":
        frame["period"] = frame["trade_date"].str.slice(0, 4)
    elif period == "halfyear":
        frame["period"] = frame["trade_date"].str.slice(0, 4) + "H" + frame["trade_date"].str.slice(4, 6).astype(int).map(lambda m: "1" if m <= 6 else "2")
    elif period == "quarter":
        quarter = frame["trade_date"].str.slice(4, 6).astype(int).map(lambda m: str((m - 1) // 3 + 1))
        frame["period"] = frame["trade_date"].str.slice(0, 4) + "Q" + quarter
    else:
        raise ValueError(f"unsupported period: {period}")

    grouped = (
        frame.groupby("period", sort=True)
        .agg(
            trade_days=("trade_date", "count"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            top10=("top10", "mean"),
            top20=("top20", "mean"),
        )
        .reset_index()
    )
    return grouped


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        labels = pd.read_sql_query(
            f"select trade_date, stock_code, [{LABEL_COL}] as {LABEL_COL} from '{LABEL_TABLE}' where [{LABEL_COL}] is not null",
            conn,
        )
        best = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as best_pred_prob from '{BEST_TABLE}'",
            conn,
        )
        daygate = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as daygate_pred_prob from '{DAYGATE_TABLE}'",
            conn,
        )

    daily_flags = pd.read_csv(DAYGATE_FLAG_CSV, dtype={"trade_date": str})

    for df in (labels, best, daygate):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(best, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daygate, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daily_flags, on="trade_date", how="left")

    daily_choice_rows: list[dict] = []
    chosen_frames: list[pd.DataFrame] = []

    for trade_date, group in frame.groupby("trade_date", sort=True):
        newcomer_mean_5d_pct = float(group["newcomer_mean_5d_pct"].iloc[0])
        chosen_model = "daygate_1d" if newcomer_mean_5d_pct >= NEWCOMER_MEAN_5D_PCT_THRESHOLD else "best_1d"
        score_series = group["daygate_pred_prob"].astype(float) if chosen_model == "daygate_1d" else group["best_pred_prob"].astype(float)

        chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
        chosen["pred_prob"] = score_series
        chosen["chosen_model"] = chosen_model
        chosen["newcomer_mean_5d_pct"] = newcomer_mean_5d_pct
        chosen["used_source_gate"] = bool(group["used_source_gate"].iloc[0])
        chosen_frames.append(chosen)

        daily_choice_rows.append(
            {
                "trade_date": trade_date,
                "chosen_model": chosen_model,
                "newcomer_mean_5d_pct": newcomer_mean_5d_pct,
                "used_source_gate": bool(group["used_source_gate"].iloc[0]),
                "newcomer_count": int(group["newcomer_count"].iloc[0]),
            }
        )

    candidate = pd.concat(chosen_frames, ignore_index=True)
    daily_choice = pd.DataFrame(daily_choice_rows)
    daily_choice.to_csv(OUT_DIR / "daily_choices.csv", index=False, encoding="utf-8-sig")

    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute(f"drop table if exists '{RESEARCH_TABLE}'")
        candidate[["trade_date", "stock_code", LABEL_COL, "pred_prob"]].to_sql(RESEARCH_TABLE, conn, index=False)
        db_row = pd.read_sql_query(
            f"""
            select
              count(*) as row_count,
              min(trade_date) as min_trade_date,
              max(trade_date) as max_trade_date,
              count(distinct trade_date) as trade_days,
              count(distinct stock_code) as stock_count,
              sum(case when pred_prob is null then 1 else 0 end) as null_pred_prob
            from '{RESEARCH_TABLE}'
            """,
            conn,
        ).iloc[0].to_dict()
        dup = pd.read_sql_query(
            f"""
            select count(*) as duplicate_key_groups
            from (
              select trade_date, stock_code, count(*) as c
              from '{RESEARCH_TABLE}'
              group by trade_date, stock_code
              having count(*) > 1
            )
            """,
            conn,
        ).iloc[0].to_dict()

    db_row.update(dup)
    db_row["db_path"] = str(MODEL_DB)
    db_row["table"] = RESEARCH_TABLE

    best_eval, _ = evaluate(frame.assign(score=frame["best_pred_prob"]), "score")
    daygate_eval, _ = evaluate(frame.assign(score=frame["daygate_pred_prob"]), "score")
    selected_eval, selected_daily = evaluate(
        candidate[["trade_date", "stock_code", LABEL_COL, "pred_prob"]].rename(columns={"pred_prob": "score"}),
        "score",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_1d_best_with_daygate5dhi_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "best_table": BEST_TABLE,
        "daygate_table": DAYGATE_TABLE,
        "rule": {
            "type": "daily_gate_daygate_fallback",
            "if_newcomer_mean_5d_pct_gte": NEWCOMER_MEAN_5D_PCT_THRESHOLD,
            "choose_model": "daygate_1d",
            "default": "best_1d",
        },
        "best_focus": focus_score(best_eval),
        "daygate_focus": focus_score(daygate_eval),
        "selected_focus": focus_score(selected_eval),
        "best_weak_window_score": weak_window_score(best_eval),
        "daygate_weak_window_score": weak_window_score(daygate_eval),
        "selected_weak_window_score": weak_window_score(selected_eval),
        "selected_minus_best_focus": focus_score(selected_eval) - focus_score(best_eval),
        "selected_minus_best_weak_window_score": weak_window_score(selected_eval) - weak_window_score(best_eval),
        "best_windows": best_eval,
        "daygate_windows": daygate_eval,
        "selected_windows": selected_eval,
        "choice_counts": daily_choice["chosen_model"].value_counts().to_dict(),
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "mixes current best 1d candidate with daygate candidate",
            "uses only ex-ante daily flag newcomer_mean_5d_pct from existing daygate report",
            "designed to repair 2025Q1/Q2 while keeping high recent-window focus",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(best_eval, daygate_eval, selected_eval).to_csv(
        OUT_DIR / "window_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(selected_daily, "year").to_csv(
        OUT_DIR / "yearly_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(selected_daily, "halfyear").to_csv(
        OUT_DIR / "halfyear_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(selected_daily, "quarter").to_csv(
        OUT_DIR / "quarter_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_for_l5",
        "promotion_requires_user_confirmation": True,
        "source_type": "sqlite_table",
        "label": LABEL_COL,
        "candidate_id": "model_agent_1d_best_with_daygate5dhi_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": payload["generated_at"],
        "row_count": int(db_row["row_count"]),
        "trade_days": int(db_row["trade_days"]),
        "stock_count": int(db_row["stock_count"]),
        "min_trade_date": db_row["min_trade_date"],
        "max_trade_date": db_row["max_trade_date"],
        "duplicate_keys": int(db_row["duplicate_key_groups"]),
        "null_pred_prob": int(db_row["null_pred_prob"]),
        "best_table_ref": f"MODEL_PREDICTIONS.db::{BEST_TABLE}",
        "daygate_table_ref": f"MODEL_PREDICTIONS.db::{DAYGATE_TABLE}",
        "daygate_flag_csv": str(DAYGATE_FLAG_CSV),
        "notes": "1D 研究候选以当前最强的 10D 代理候选为默认排序，仅当 daygate 候选的现成日级特征 newcomer_mean_5d_pct 达到高阈值时切换到 daygate 候选。目标是在保持近期窗口强度的同时，修复 2025Q1/Q2 弱窗口。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 1D 当前最强候选叠加 daygate 高 5D 一致性门控",
        "",
        "## 当前结论",
        "",
        f"- 当前最强 1D 候选：`{BEST_TABLE}`",
        f"- 弱窗口修复来源：`{DAYGATE_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 门控阈值：`newcomer_mean_5d_pct >= {NEWCOMER_MEAN_5D_PCT_THRESHOLD}`",
        f"- 当前最强候选 Focus：`{payload['best_focus']:.12f}`",
        f"- 新候选 Focus：`{payload['selected_focus']:.12f}`",
        f"- Focus 增量：`{payload['selected_minus_best_focus']:.12f}`",
        f"- 当前最强候选弱窗口分：`{payload['best_weak_window_score']:.12f}`",
        f"- 新候选弱窗口分：`{payload['selected_weak_window_score']:.12f}`",
        f"- 弱窗口分增量：`{payload['selected_minus_best_weak_window_score']:.12f}`",
        "",
        "## 规则",
        "",
        "- 默认沿用当前最强 1D 候选。",
        "- 当 daygate 报告里现成的 `newcomer_mean_5d_pct` 足够高时，认为 1D 和 5D 的头部一致性较强，当日切换到 daygate 候选。",
        "- 这是一个只用现成研究资产和现成日级特征的 research 门控，不训练模型，不改 formal。",
        "",
        "## 窗口结果",
        "",
        f"- recent63：`Top1={selected_eval['recent63']['top1']:.8f}` `Top3={selected_eval['recent63']['top3']:.8f}` `Top5={selected_eval['recent63']['top5']:.8f}` `Top10={selected_eval['recent63']['top10']:.8f}`",
        f"- recent126：`Top1={selected_eval['recent126']['top1']:.8f}` `Top3={selected_eval['recent126']['top3']:.8f}` `Top5={selected_eval['recent126']['top5']:.8f}`",
        f"- 2025Q1：`Top1={selected_eval['q1_2025']['top1']:.8f}` `Top5={selected_eval['q1_2025']['top5']:.8f}`",
        f"- 2025Q2：`Top1={selected_eval['q2_2025']['top1']:.8f}` `Top5={selected_eval['q2_2025']['top5']:.8f}`",
        "",
        "## 模型选择计数",
        "",
        f"- `{json.dumps(payload['choice_counts'], ensure_ascii=False)}`",
        "",
        "## 治理说明",
        "",
        "- 本次只生成 research 候选资产。",
        "- 未训练模型。",
        "- 未发布 formal 资产。",
        "- 未生成交易信号，未制定交易规则，未跑回测。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'daily_choices.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'window_comparison.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'yearly_stability.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'quarter_stability.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
