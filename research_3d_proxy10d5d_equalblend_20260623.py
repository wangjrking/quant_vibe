from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_3d_proxy10d5d_equalblend_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
BASE10_TABLE = "stock_predict_data_model_agent_10d_proxy5d_topzone_blend_20260623_executable_10d_open_return_research"
BASE5_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_blend_20260623_executable_3d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_equalblend_20260623_executable_3d_open_return_research"

LABEL_COL = "executable_3d_open_return"
W10 = 0.50
W5 = 0.50
WINDOWS = {
    "full": ("20240604", "20260609"),
    "recent126": ("20251201", "20260609"),
    "recent63": ("20260304", "20260609"),
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
        summary["recent63"]["top1"] * 5
        + summary["recent63"]["top3"] * 4
        + summary["recent63"]["top5"] * 5
        + summary["recent63"]["top10"] * 2
        + summary["recent126"]["top1"] * 3
        + summary["recent126"]["top3"] * 2
        + summary["recent126"]["top5"] * 1.5
        + summary["full"]["top1"] * 0.5
        + summary["full"]["top3"] * 0.3
        + summary["full"]["top5"] * 0.2
    )


def compare_windows(base_windows: dict, current_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63"):
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
        labels = pd.read_sql_query(
            f"select trade_date, stock_code, [{LABEL_COL}] as {LABEL_COL} from '{LABEL_TABLE}' where [{LABEL_COL}] is not null",
            conn,
        )
        pred10 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred10 from '{BASE10_TABLE}'",
            conn,
        )
        pred5 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred5 from '{BASE5_TABLE}'",
            conn,
        )
        current = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred_current from '{CURRENT_TABLE}'",
            conn,
        )

    for df in (labels, pred10, pred5, current):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(pred10, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(pred5, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="left")
    frame["rank10"] = frame.groupby("trade_date")["pred10"].rank(method="average", pct=True)
    frame["rank5"] = frame.groupby("trade_date")["pred5"].rank(method="average", pct=True)
    frame["blend_score"] = W10 * frame["rank10"] + W5 * frame["rank5"]

    base_eval, _ = evaluate(frame.assign(score=frame["rank10"]), "score")
    current_eval, _ = evaluate(frame.assign(score=frame["pred_current"]), "score")
    selected_eval, selected_daily = evaluate(frame.assign(score=frame["blend_score"]), "score")

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = frame["blend_score"].astype(float)

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
        "candidate_id": "model_agent_3d_proxy10d5d_equalblend_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "base10_table": BASE10_TABLE,
        "base5_table": BASE5_TABLE,
        "current_table": CURRENT_TABLE,
        "rule": {
            "type": "rank_blend",
            "w10": W10,
            "w5": W5,
            "formula": "score = 0.50 * rank10_proxy + 0.50 * rank5_best",
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
            "cross_horizon_proxy_blend",
            "equal blend replaces previous 0.60/0.40 setting",
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
        "candidate_id": "model_agent_3d_proxy10d5d_equalblend_20260623",
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
        "base10_source": f"MODEL_PREDICTIONS.db::{BASE10_TABLE}",
        "base5_source": f"MODEL_PREDICTIONS.db::{BASE5_TABLE}",
        "current_candidate_source": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "notes": "3D 研究候选继续采用跨周期 rank blend，但把 10D proxy / 5D best 的组合权重从 0.60/0.40 调整到 0.50/0.50。该调整来自同口径网格扫描，目标是在不改变模型来源的情况下提升近期与全窗口的综合评价。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 3D 等权跨周期 rank blend 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 基线 10D 代理表：`{BASE10_TABLE}`",
        f"- 5D 辅助表：`{BASE5_TABLE}`",
        f"- 当前 3D 候选：`{CURRENT_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 基线 Focus：`{payload['base_focus']:.12f}`",
        f"- 当前候选 Focus：`{payload['current_focus']:.12f}`",
        f"- 新候选 Focus：`{payload['selected_focus']:.12f}`",
        f"- 相对当前候选增量：`{payload['selected_minus_current_focus']:.12f}`",
        "",
        "## 规则",
        "",
        f"- `score = {W10:.2f} * rank10_proxy + {W5:.2f} * rank5_best`",
        "- 只调整组合权重，不改模型来源，不引入训练。",
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
        f"- `{(OUT_DIR / 'halfyear_stability.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'quarter_stability.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
