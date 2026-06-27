from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_daily_gate_d3eq_d1new_v3_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_topgate_3dformal_dategate_refine_20260623_executable_5d_open_return_research"
AUX1_TABLE = "stock_predict_data_model_agent_1d_proxy5d_topzone_10dlite_20260623_executable_1d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_equalblend_20260623_executable_3d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3eq_d1new_v3_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
START_DATE = "20250301"
THR3 = 0.98
THR1 = 0.996
WINDOWS = {
    "full": ("20240604", "20260605"),
    "recent1y": ("20250301", "20260605"),
    "recent80": ("20260201", "20260605"),
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
        }
    return out, daily


def focus_score(summary: dict) -> float:
    return (
        summary["recent80"]["top1"] * 5
        + summary["recent80"]["top3"] * 4
        + summary["recent80"]["top5"] * 5
        + summary["recent80"]["top10"] * 2
        + summary["recent1y"]["top1"] * 3
        + summary["recent1y"]["top3"] * 2
        + summary["recent1y"]["top5"] * 1.5
        + summary["full"]["top1"] * 0.5
        + summary["full"]["top3"] * 0.3
        + summary["full"]["top5"] * 0.2
    )


def compare_windows(base_windows: dict, current_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent1y", "recent80"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"base_{metric}"] = base_windows[name][metric]
            row[f"current_{metric}"] = current_windows[name][metric]
            row[f"new_{metric}"] = new_windows[name][metric]
            row[f"new_minus_base_{metric}"] = new_windows[name][metric] - base_windows[name][metric]
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
        )
        .reset_index()
    )
    return grouped


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        base = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} "
            f"from '{BASE_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        aux1 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred1d from '{AUX1_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        aux3 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred3d from '{AUX3_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        current = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred_current from '{CURRENT_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )

    for df in (base, aux1, aux3, current):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(aux1, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="left")
    frame["rank1"] = frame.groupby("trade_date")["pred1d"].rank(method="average", pct=True)
    frame["rank3"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)

    daily_choice_rows: list[dict] = []
    chosen_frames: list[pd.DataFrame] = []

    for trade_date, group in frame.groupby("trade_date", sort=True):
        ordered_base = group.sort_values("base_pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
        base_top5_mean_rank3 = float(ordered_base.head(5)["rank3"].mean())
        base_top5_mean_rank1 = float(ordered_base.head(5)["rank1"].mean())

        chosen_model = "base"
        score_series = group["base_pred_prob"].astype(float)
        if trade_date >= START_DATE and base_top5_mean_rank3 < THR3:
            chosen_model = "d3_new"
            score_series = group["pred3d"].astype(float).where(group["pred3d"].notna(), group["base_pred_prob"].astype(float))
        elif trade_date >= START_DATE and base_top5_mean_rank1 < THR1:
            chosen_model = "d1_new"
            score_series = group["pred1d"].astype(float).where(group["pred1d"].notna(), group["base_pred_prob"].astype(float))

        chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
        chosen["pred_prob"] = score_series
        chosen["chosen_model"] = chosen_model
        chosen["base_top5_mean_rank3"] = base_top5_mean_rank3
        chosen["base_top5_mean_rank1"] = base_top5_mean_rank1
        chosen_frames.append(chosen)

        daily_choice_rows.append(
            {
                "trade_date": trade_date,
                "chosen_model": chosen_model,
                "base_top5_mean_rank3": base_top5_mean_rank3,
                "base_top5_mean_rank1": base_top5_mean_rank1,
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

    base_summary, _ = evaluate(base.assign(score=base["base_pred_prob"]), "score")
    current_summary, _ = evaluate(frame.assign(score=frame["pred_current"]), "score")
    selected_summary, selected_daily = evaluate(
        candidate[["trade_date", "stock_code", LABEL_COL, "pred_prob"]].rename(columns={"pred_prob": "score"}),
        "score",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_5d_daily_gate_d3eq_d1new_v3_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "aux1_table": AUX1_TABLE,
        "aux3_table": AUX3_TABLE,
        "current_table": CURRENT_TABLE,
        "rule": {
            "start_date": START_DATE,
            "if_base_top5_mean_rank3_lt": THR3,
            "choose_model": "d3_new",
            "elif_base_top5_mean_rank1_lt": THR1,
            "choose_model_else": "d1_new",
            "default": "base",
        },
        "base_focus": focus_score(base_summary),
        "current_focus": focus_score(current_summary),
        "selected_focus": focus_score(selected_summary),
        "selected_minus_base_focus": focus_score(selected_summary) - focus_score(base_summary),
        "selected_minus_current_focus": focus_score(selected_summary) - focus_score(current_summary),
        "base_windows": base_summary,
        "current_windows": current_summary,
        "selected_windows": selected_summary,
        "choice_counts": daily_choice["chosen_model"].value_counts().to_dict(),
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "no training",
            "daily switch upgrades 3d source to equal-blend and 1d source to new topzone candidate",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_summary, current_summary, selected_summary).to_csv(
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
        "candidate_id": "model_agent_5d_daily_gate_d3eq_d1new_v3_20260623",
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
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "aux1_source": f"MODEL_PREDICTIONS.db::{AUX1_TABLE}",
        "aux3_source": f"MODEL_PREDICTIONS.db::{AUX3_TABLE}",
        "current_candidate_source": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "notes": "5D 研究候选 v3 延续按日门控，但把 3D 来源升级为 equal-blend 研究候选，把 1D 来源升级为 topzone_10dlite 研究候选，同时把 3D 触发阈值收紧到 0.98。该候选仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 5D 按日切换 d3_equalblend / d1_new 研究候选 v3",
        "",
        "## 当前结论",
        "",
        f"- 基线表：`{BASE_TABLE}`",
        f"- 当前 5D 候选：`{CURRENT_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 基线 Focus：`{payload['base_focus']:.12f}`",
        f"- 当前候选 Focus：`{payload['current_focus']:.12f}`",
        f"- 新候选 Focus：`{payload['selected_focus']:.12f}`",
        f"- 相对当前候选增量：`{payload['selected_minus_current_focus']:.12f}`",
        "",
        "## 规则",
        "",
        f"- 自 `{START_DATE}` 起：若 `base_top5_mean_rank3 < {THR3}`，整日切到 `d3_new`。",
        f"- 否则若 `base_top5_mean_rank1 < {THR1}`，整日切到 `d1_new`。",
        "- 否则保持 `base`。",
        "",
        "## 窗口结果",
        "",
        f"- recent80：`Top1={selected_summary['recent80']['top1']:.8f}` `Top3={selected_summary['recent80']['top3']:.8f}` `Top5={selected_summary['recent80']['top5']:.8f}` `Top10={selected_summary['recent80']['top10']:.8f}`",
        f"- recent1y：`Top1={selected_summary['recent1y']['top1']:.8f}` `Top3={selected_summary['recent1y']['top3']:.8f}` `Top5={selected_summary['recent1y']['top5']:.8f}`",
        f"- full：`Top1={selected_summary['full']['top1']:.8f}` `Top3={selected_summary['full']['top3']:.8f}` `Top5={selected_summary['full']['top5']:.8f}`",
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
