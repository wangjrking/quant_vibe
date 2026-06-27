from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_daily_gate_d3d1_v2_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_topgate_3dformal_dategate_refine_20260623_executable_5d_open_return_research"
AUX1_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
START_DATE = "20250301"
THR3 = 0.985
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

    for df in (base, aux1, aux3):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(aux1, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")
    frame["rank1"] = frame.groupby("trade_date")["pred1d"].rank(method="average", pct=True)
    frame["rank3"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)

    daily_choice_rows: list[dict] = []
    chosen_frames: list[pd.DataFrame] = []

    for trade_date, group in frame.groupby("trade_date", sort=True):
        ordered_base = group.sort_values("base_pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
        base_top5_mean_rank3 = float(ordered_base.head(5)["rank3"].mean())
        base_top5_mean_rank1 = float(ordered_base.head(5)["rank1"].mean())

        chosen_model = "base"
        score_col = "base_pred_prob"
        if trade_date >= START_DATE and base_top5_mean_rank3 < THR3:
            chosen_model = "d3"
            score_col = "pred3d"
        elif trade_date >= START_DATE and base_top5_mean_rank1 < THR1:
            chosen_model = "d1"
            score_col = "pred1d"

        chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
        chosen["pred_prob"] = group[score_col].astype(float)
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
    selected_summary, selected_daily = evaluate(
        candidate[["trade_date", "stock_code", LABEL_COL, "pred_prob"]].rename(columns={"pred_prob": "score"}),
        "score",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_5d_daily_gate_d3d1_v2_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "aux1_table": AUX1_TABLE,
        "aux3_table": AUX3_TABLE,
        "rule": {
            "start_date": START_DATE,
            "if_base_top5_mean_rank3_lt": THR3,
            "choose_model": "d3",
            "elif_base_top5_mean_rank1_lt": THR1,
            "choose_model_else": "d1",
            "default": "base",
        },
        "base_focus": focus_score(base_summary),
        "selected_focus": focus_score(selected_summary),
        "focus_delta": focus_score(selected_summary) - focus_score(base_summary),
        "base_windows": base_summary,
        "selected_windows": selected_summary,
        "choice_counts": daily_choice["chosen_model"].value_counts().to_dict(),
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_for_l5",
        "promotion_requires_user_confirmation": True,
        "source_type": "sqlite_table",
        "label": LABEL_COL,
        "candidate_id": "model_agent_5d_daily_gate_d3d1_v2_20260623",
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
        "notes": "5D 研究候选 v2 自 20250301 起按日门控：若当前 5D base top5 在 3D 排序中的平均分位低于 0.985，则整日切到 3D；否则若在 1D 排序中的平均分位低于 0.996，则整日切到 1D；其余日期保持 base。仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 5D 按日门控切换 d3/d1 研究候选 v2",
        "",
        "## 当前结论",
        "",
        f"- 基线表：`{BASE_TABLE}`",
        f"- 新研究表：`{RESEARCH_TABLE}`",
        f"- Focus：`{payload['base_focus']:.12f}` -> `{payload['selected_focus']:.12f}`",
        f"- Focus 增量：`{payload['focus_delta']:.12f}`",
        "",
        "## 规则",
        "",
        f"- 自 `{START_DATE}` 起：若 `base_top5_mean_rank3 < {THR3}`，整日使用 3D 排序。",
        f"- 否则若 `base_top5_mean_rank1 < {THR1}`，整日使用 1D 排序。",
        "- 否则维持原 5D base 排序。",
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
        "- 本轮仅生成 research 候选资产。",
        "- 未训练模型。",
        "- 未发布 formal 资产。",
        "- 未生成交易信号、未制定交易规则、未跑回测。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'daily_choices.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
