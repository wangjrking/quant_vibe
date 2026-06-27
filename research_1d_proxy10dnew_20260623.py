from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_proxy10dnew_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research"
SOURCE_TABLE = "stock_predict_data_model_agent_10d_proxy5d_lite_daygate_std004_20260623_executable_10d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_1d_proxy5d_topzone_10dlite_20260623_executable_1d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_1d_proxy10dnew_20260623_executable_1d_open_return_research"

LABEL_COL = "executable_1d_open_return"
WINDOWS = {
    "full": ("20240604", "20260611"),
    "recent126": ("20251201", "20260611"),
    "recent63": ("20260304", "20260611"),
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


def compare_windows(base_windows: dict, current_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10", "top20"):
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
            top20=("top20", "mean"),
        )
        .reset_index()
    )
    return grouped


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        labels = pd.read_sql_query(
            f"select trade_date, stock_code, [{LABEL_COL}] as {LABEL_COL}, pred_prob as pred1d from '{LABEL_TABLE}' where [{LABEL_COL}] is not null",
            conn,
        )
        source = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred10new from '{SOURCE_TABLE}'",
            conn,
        )
        current = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred_current from '{CURRENT_TABLE}'",
            conn,
        )

    for df in (labels, source, current):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(source, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="left")

    base_scores = labels[["trade_date", "stock_code", LABEL_COL, "pred1d"]].rename(columns={"pred1d": "score"})
    base_eval, _ = evaluate(base_scores, "score")
    current_eval, _ = evaluate(frame.assign(score=frame["pred_current"]), "score")
    selected_eval, selected_daily = evaluate(frame.assign(score=frame["pred10new"]), "score")

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = frame["pred10new"].astype(float)

    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute(f"drop table if exists '{RESEARCH_TABLE}'")
        candidate.to_sql(RESEARCH_TABLE, conn, index=False)
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

    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_eval, current_eval, selected_eval).to_csv(
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

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_1d_proxy10dnew_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "source_table": SOURCE_TABLE,
        "current_table": CURRENT_TABLE,
        "rule": {
            "type": "pure_cross_horizon_proxy",
            "formula": "score = pred_prob from 10d day-gated research candidate",
        },
        "base_focus": focus_score(base_eval),
        "current_focus": focus_score(current_eval),
        "selected_focus": focus_score(selected_eval),
        "selected_minus_base_focus": focus_score(selected_eval) - focus_score(base_eval),
        "selected_minus_current_focus": focus_score(selected_eval) - focus_score(current_eval),
        "base_windows": base_eval,
        "current_windows": current_eval,
        "selected_windows": selected_eval,
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "pure cross-horizon proxy from strongest 10d candidate",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_for_l5",
        "promotion_requires_user_confirmation": True,
        "source_type": "sqlite_table",
        "label": LABEL_COL,
        "candidate_id": "model_agent_1d_proxy10dnew_20260623",
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
        "source_table_ref": f"MODEL_PREDICTIONS.db::{SOURCE_TABLE}",
        "current_candidate_source": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "notes": "1D 研究候选直接采用当前最强 10D 日级门控研究候选作为跨周期代理，不再以 5D 为中继。该候选仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 1D 直接代理 10D day-gated 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 基线表：`{LABEL_TABLE}`",
        f"- 当前 1D 候选：`{CURRENT_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 基线 Focus：`{payload['base_focus']:.12f}`",
        f"- 当前候选 Focus：`{payload['current_focus']:.12f}`",
        f"- 新候选 Focus：`{payload['selected_focus']:.12f}`",
        f"- 相对当前候选增量：`{payload['selected_minus_current_focus']:.12f}`",
        "",
        "## 规则",
        "",
        "- 不再通过 5D 中继，直接使用当前最强 `10D day-gated` 研究候选作为 `1D` 的跨周期排序代理。",
        "",
        "## 窗口结果",
        "",
        f"- recent63：`Top1={selected_eval['recent63']['top1']:.8f}` `Top3={selected_eval['recent63']['top3']:.8f}` `Top5={selected_eval['recent63']['top5']:.8f}` `Top10={selected_eval['recent63']['top10']:.8f}`",
        f"- recent126：`Top1={selected_eval['recent126']['top1']:.8f}` `Top3={selected_eval['recent126']['top3']:.8f}` `Top5={selected_eval['recent126']['top5']:.8f}`",
        f"- full：`Top1={selected_eval['full']['top1']:.8f}` `Top3={selected_eval['full']['top3']:.8f}` `Top5={selected_eval['full']['top5']:.8f}`",
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
