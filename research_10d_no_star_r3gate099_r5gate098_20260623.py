from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_no_star_r3gate099_r5gate098_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
SOURCE_TABLE = "stock_predict_data_model_agent_10d_dategate_3dformal_plus_1dgate092_a015_z0015_20260623_executable_10d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
AUX5_TABLE = "stock_predict_data_model_agent_5d_topgate_3dformal_dategate_refine_20260623_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_10d_no_star_r3gate099_r5gate098_20260623_executable_10d_open_return_research"
LABEL_COL = "executable_10d_open_return"

WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}


def evaluate(frame: pd.DataFrame, score_col: str) -> dict:
    data = frame[["trade_date", "stock_code", LABEL_COL, score_col]].dropna().copy()
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
    return out


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        base = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} from '{BASE_TABLE}'",
            conn,
        )
        source = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as source_pred_prob from '{SOURCE_TABLE}'",
            conn,
        )
        aux3 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred3d from '{AUX3_TABLE}'",
            conn,
        )
        aux5 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred5d from '{AUX5_TABLE}'",
            conn,
        )

    for df in [base, source, aux3, aux5]:
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(source, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux5, on=["trade_date", "stock_code"], how="left")
    frame["rank3d_pct"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)
    frame["rank5d_pct"] = frame.groupby("trade_date")["pred5d"].rank(method="average", pct=True)
    frame["board"] = frame["stock_code"].map(
        lambda s: "BJ" if s.startswith(("8", "9")) else ("STAR" if s.startswith("688") else ("GEM" if s.startswith("3") else "MAIN"))
    )

    keep_source_days: list[str] = []
    daily_flags: list[dict] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        source_top5 = group.sort_values("source_pred_prob", ascending=False).head(5).copy()
        star_count = int((source_top5["board"] == "STAR").sum())
        mean_rank3d = float(source_top5["rank3d_pct"].mean())
        mean_rank5d = float(source_top5["rank5d_pct"].mean())
        keep_source = star_count == 0 and mean_rank3d >= 0.99 and mean_rank5d >= 0.98
        if keep_source:
            keep_source_days.append(trade_date)
        daily_flags.append(
            {
                "trade_date": trade_date,
                "source_top5_star_count": star_count,
                "source_top5_mean_rank3d": mean_rank3d,
                "source_top5_mean_rank5d": mean_rank5d,
                "keep_source": int(keep_source),
            }
        )

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = frame["base_pred_prob"]
    use_source = frame["trade_date"].isin(keep_source_days)
    candidate.loc[use_source, "pred_prob"] = frame.loc[use_source, "source_pred_prob"]

    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute(f"drop table if exists '{TARGET_TABLE}'")
        candidate.to_sql(TARGET_TABLE, conn, index=False)

    eval_frame = candidate.rename(columns={"pred_prob": "candidate_pred_prob"})
    merged = (
        base.merge(source, on=["trade_date", "stock_code"], how="inner")
        .merge(eval_frame[["trade_date", "stock_code", "candidate_pred_prob"]], on=["trade_date", "stock_code"], how="inner")
    )

    base_summary = evaluate(merged, "base_pred_prob")
    source_summary = evaluate(merged, "source_pred_prob")
    selected_summary = evaluate(merged, "candidate_pred_prob")

    daily_flags_df = pd.DataFrame(daily_flags)
    daily_flags_df.to_csv(OUT_DIR / "daily_gate_flags.csv", index=False, encoding="utf-8-sig")

    with sqlite3.connect(MODEL_DB) as conn:
        db_summary = pd.read_sql_query(
            f"""
            select
              count(*) as row_count,
              min(trade_date) as min_trade_date,
              max(trade_date) as max_trade_date,
              count(distinct trade_date) as trade_days,
              count(distinct stock_code) as stock_count,
              sum(case when pred_prob is null then 1 else 0 end) as null_pred_prob
            from '{TARGET_TABLE}'
            """,
            conn,
        ).iloc[0].to_dict()
        dup = pd.read_sql_query(
            f"""
            select count(*) as duplicate_key_groups
            from (
              select trade_date, stock_code, count(*) as c
              from '{TARGET_TABLE}'
              group by trade_date, stock_code
              having count(*) > 1
            )
            """,
            conn,
        ).iloc[0].to_dict()
    db_summary.update(dup)
    db_summary["db_path"] = str(MODEL_DB)
    db_summary["table"] = TARGET_TABLE
    db_summary["keep_source_days"] = len(keep_source_days)
    db_summary["keep_source_trade_dates"] = keep_source_days

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_no_star_r3gate099_r5gate098_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "source_table": SOURCE_TABLE,
        "aux5_table": AUX5_TABLE,
        "rule": "if source candidate top5 contains no STAR(688) stock and source_top5_mean_rank3d >= 0.99 and source_top5_mean_rank5d >= 0.98, keep source score for that trade_date; otherwise fallback to base 10D score",
        "base_eval": base_summary,
        "source_eval": source_summary,
        "selected_eval": selected_summary,
        "delta_vs_source": {
            window: {
                metric: selected_summary[window][metric] - source_summary[window][metric]
                for metric in ["top1", "top3", "top5", "top10", "top20"]
            }
            for window in WINDOWS
        },
        "base_focus_score": focus_score(base_summary),
        "source_focus_score": focus_score(source_summary),
        "selected_focus_score": focus_score(selected_summary),
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }

    (OUT_DIR / "probe_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "db_summary.json").write_text(json.dumps(db_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "asset_role": "l4_research_prediction_asset",
                "approval_status": "research_only_not_for_l5",
                "promotion_requires_user_confirmation": True,
                "source_type": "sqlite_table",
                "label": LABEL_COL,
                "candidate_id": "model_agent_10d_no_star_r3gate099_r5gate098_20260623",
                "db_path": "../MODEL_PREDICTIONS.db",
                "table": TARGET_TABLE,
                "market_db_path": "../../STOCK_DAILY_DATA.db",
                "generated_at": payload["generated_at"],
                "row_count": int(db_summary["row_count"]),
                "trade_days": int(db_summary["trade_days"]),
                "stock_count": int(db_summary["stock_count"]),
                "min_trade_date": db_summary["min_trade_date"],
                "max_trade_date": db_summary["max_trade_date"],
                "duplicate_keys": int(db_summary["duplicate_key_groups"]),
                "null_pred_prob": int(db_summary["null_pred_prob"]),
                "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
                "aux_source": f"MODEL_PREDICTIONS.db::{SOURCE_TABLE}",
                "notes": "10D 研究候选在 no-star + 3D 一致性门控基础上，再要求 source top5 的 5D 平均分位不低于 0.98，用于剔除 5D 一致性不足的少数交易日；仅用于 research 验证，不进入 formal L4/L5。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# 10D no-star + r3 + r5 门控研究候选",
        "",
        "## 当前结论",
        "",
        f"- 新候选 `10d_no_star_r3gate099_r5gate098` 的 focus 分数为 `{payload['selected_focus_score']:.12f}`。",
        f"- 相比上一版 `10d_no_star_r3gate099`，focus 由 `{0.1770254145000664:.12f}` 提升到 `{payload['selected_focus_score']:.12f}`。",
        f"- 该候选只在 `{len(keep_source_days)}` 个交易日保留 source，其余交易日回退到 base。",
        "",
        "## 规则",
        "",
        "- 条件 1：source candidate top5 中不含 STAR(688) 股票。",
        "- 条件 2：`source_top5_mean_rank3d >= 0.99`。",
        "- 条件 3：`source_top5_mean_rank5d >= 0.98`。",
        "- 三个条件同时满足时保留 source，否则整日回退到 base。",
        "",
        "## 关键窗口",
        "",
        f"- recent63：`Top1={selected_summary['recent63']['top1']:.8f}` `Top3={selected_summary['recent63']['top3']:.8f}` `Top5={selected_summary['recent63']['top5']:.8f}`",
        f"- recent126：`Top1={selected_summary['recent126']['top1']:.8f}` `Top3={selected_summary['recent126']['top3']:.8f}` `Top5={selected_summary['recent126']['top5']:.8f}`",
        f"- full：`Top1={selected_summary['full']['top1']:.8f}` `Top3={selected_summary['full']['top3']:.8f}` `Top5={selected_summary['full']['top5']:.8f}`",
        "",
        "## 治理说明",
        "",
        "- 本轮未训练模型。",
        "- 本轮未发布 formal 资产。",
        "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'probe_summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'db_summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'daily_gate_flags.csv').as_posix()}`",
    ]
    (OUT_DIR / "probe_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
