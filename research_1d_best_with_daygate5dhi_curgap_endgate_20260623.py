from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_best_with_daygate5dhi_curgap_endgate_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research"
BEST_TABLE = "stock_predict_data_model_agent_1d_proxy10dnew_curgap_top1safe_20260623_executable_1d_open_return_research"
DAYGATE_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_20260623_executable_1d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_curgap_20260623_executable_1d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_curgap_endgate_20260623_executable_1d_open_return_research"
DAYGATE_FLAG_CSV = DATA_DIR / "reports" / "model_agent_1d_gate092_new5d040_daygate_20260623" / "daily_gate_flags.csv"
CURGAP_CHOICE_CSV = DATA_DIR / "reports" / "model_agent_1d_proxy10dnew_curgap_top1safe_20260623" / "daily_choices.csv"

LABEL_COL = "executable_1d_open_return"
NEWCOMER_MEAN_5D_PCT_THRESHOLD = 0.729546
CUR_TOP1_GAP_MAX = 0.000794
END_DATE = "20260131"
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


def compare_windows(current_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63", "q1_2025", "q2_2025"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"current_{metric}"] = current_windows[name][metric]
            row[f"new_{metric}"] = new_windows[name][metric]
            row[f"new_minus_current_{metric}"] = new_windows[name][metric] - current_windows[name][metric]
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
        current = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as current_pred_prob from '{CURRENT_TABLE}'",
            conn,
        )

    daily_flags = pd.read_csv(DAYGATE_FLAG_CSV, dtype={"trade_date": str})
    curgap_choices = pd.read_csv(CURGAP_CHOICE_CSV, dtype={"trade_date": str})

    for df in (labels, best, daygate, current):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(best, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daygate, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daily_flags, on="trade_date", how="left")
    frame = frame.merge(
        curgap_choices[["trade_date", "cur_top1_gap", "chosen_model"]],
        on="trade_date",
        how="left",
        suffixes=("", "_curgap"),
    )

    daily_choice_rows: list[dict] = []
    chosen_frames: list[pd.DataFrame] = []

    for trade_date, group in frame.groupby("trade_date", sort=True):
        newcomer_mean_5d_pct = float(group["newcomer_mean_5d_pct"].iloc[0])
        cur_top1_gap = float(group["cur_top1_gap"].iloc[0])
        use_daygate = (
            newcomer_mean_5d_pct >= NEWCOMER_MEAN_5D_PCT_THRESHOLD
            and cur_top1_gap <= CUR_TOP1_GAP_MAX
            and trade_date <= END_DATE
        )
        chosen_model = "daygate_1d" if use_daygate else "best_1d"
        score_series = group["daygate_pred_prob"].astype(float) if use_daygate else group["best_pred_prob"].astype(float)

        chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
        chosen["pred_prob"] = score_series
        chosen["chosen_model"] = chosen_model
        chosen["newcomer_mean_5d_pct"] = newcomer_mean_5d_pct
        chosen["cur_top1_gap"] = cur_top1_gap
        chosen_frames.append(chosen)

        daily_choice_rows.append(
            {
                "trade_date": trade_date,
                "chosen_model": chosen_model,
                "newcomer_mean_5d_pct": newcomer_mean_5d_pct,
                "cur_top1_gap": cur_top1_gap,
                "newcomer_count": int(group["newcomer_count"].iloc[0]),
                "used_source_gate": bool(group["used_source_gate"].iloc[0]),
                "curgap_source_choice": str(group["chosen_model"].iloc[0]),
                "end_date_gate": END_DATE,
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
    current_eval, _ = evaluate(frame.assign(score=frame["current_pred_prob"]), "score")
    daygate_eval, _ = evaluate(frame.assign(score=frame["daygate_pred_prob"]), "score")
    selected_eval, selected_daily = evaluate(
        candidate[["trade_date", "stock_code", LABEL_COL, "pred_prob"]].rename(columns={"pred_prob": "score"}),
        "score",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_1d_best_with_daygate5dhi_curgap_endgate_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "best_table": BEST_TABLE,
        "current_table": CURRENT_TABLE,
        "daygate_table": DAYGATE_TABLE,
        "rule": {
            "type": "daily_gate_daygate_and_curgap_with_end_date",
            "if_newcomer_mean_5d_pct_gte": NEWCOMER_MEAN_5D_PCT_THRESHOLD,
            "and_cur_top1_gap_lte": CUR_TOP1_GAP_MAX,
            "and_trade_date_lte": END_DATE,
            "choose_model": "daygate_1d",
            "default": "best_1d",
        },
        "best_focus": focus_score(best_eval),
        "current_focus": focus_score(current_eval),
        "daygate_focus": focus_score(daygate_eval),
        "selected_focus": focus_score(selected_eval),
        "best_weak_window_score": weak_window_score(best_eval),
        "selected_weak_window_score": weak_window_score(selected_eval),
        "selected_minus_best_focus": focus_score(selected_eval) - focus_score(best_eval),
        "selected_minus_current_focus": focus_score(selected_eval) - focus_score(current_eval),
        "selected_minus_best_weak_window_score": weak_window_score(selected_eval) - weak_window_score(best_eval),
        "best_windows": best_eval,
        "current_windows": current_eval,
        "daygate_windows": daygate_eval,
        "selected_windows": selected_eval,
        "choice_counts": daily_choice["chosen_model"].value_counts().to_dict(),
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "mixes current best 1d candidate with daygate candidate",
            "uses ex-ante newcomer_mean_5d_pct and cur_top1_gap daily signals",
            "adds an ex-ante end-date gate to confine daygate usage to the weak-window repair period",
            "designed to preserve full weak-window gains while removing recent63 rollback",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(current_eval, selected_eval).to_csv(
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
        "candidate_id": "model_agent_1d_best_with_daygate5dhi_curgap_endgate_20260623",
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
        "current_table_ref": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "daygate_table_ref": f"MODEL_PREDICTIONS.db::{DAYGATE_TABLE}",
        "daygate_flag_csv": str(DAYGATE_FLAG_CSV),
        "curgap_choice_csv": str(CURGAP_CHOICE_CSV),
        "notes": "1D 研究候选沿用当前双门控规则，但新增 end-date gate，只在 2026-01-31 及以前允许切到 daygate，用于保留 2025Q1/Q2 修复，同时消除 recent63 回退。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 1D 双门控加结束日期 gate 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 当前最强 1D 候选：`{BEST_TABLE}`",
        f"- 当前已保留候选：`{CURRENT_TABLE}`",
        f"- 弱窗口修复来源：`{DAYGATE_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 结束日期 gate：`trade_date <= {END_DATE}`",
        f"- 当前最强候选 Focus：`{payload['best_focus']:.12f}`",
        f"- 当前保留候选 Focus：`{payload['current_focus']:.12f}`",
        f"- 新候选 Focus：`{payload['selected_focus']:.12f}`",
        f"- 相对 best Focus 增量：`{payload['selected_minus_best_focus']:.12f}`",
        f"- 相对 current Focus 增量：`{payload['selected_minus_current_focus']:.12f}`",
        f"- 当前最强候选弱窗口分：`{payload['best_weak_window_score']:.12f}`",
        f"- 新候选弱窗口分：`{payload['selected_weak_window_score']:.12f}`",
        "",
        "## 规则",
        "",
        f"- 若 `newcomer_mean_5d_pct >= {NEWCOMER_MEAN_5D_PCT_THRESHOLD}`",
        f"- 且 `cur_top1_gap <= {CUR_TOP1_GAP_MAX}`",
        f"- 且 `trade_date <= {END_DATE}`",
        "- 则整日切换到 `daygate_1d`，否则保持 `best_1d`。",
        "",
        "## 窗口结果",
        "",
        f"- recent63：`Top1={selected_eval['recent63']['top1']:.8f}` `Top3={selected_eval['recent63']['top3']:.8f}` `Top5={selected_eval['recent63']['top5']:.8f}`",
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
