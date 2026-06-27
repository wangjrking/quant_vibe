from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_proxy5d_topzone_blend_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
BASE_PROXY_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research"
AUX10_TABLE = "stock_predict_data_model_agent_10d_no_star_r3gate991_r5gate98_20260623_executable_10d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_proxy5d_topzone_blend_20260623_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"
START_DATE = "20260201"
ZONE = 0.02
ALPHA10 = 0.01
ALPHA3 = 0.005
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
        }
    out["daily"] = daily
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
        labels = pd.read_sql_query(
            f"select trade_date, stock_code, [{LABEL_COL}] as {LABEL_COL} from '{LABEL_TABLE}'",
            conn,
        )
        proxy = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred5d from '{BASE_PROXY_TABLE}'",
            conn,
        )
        aux10 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred10d from '{AUX10_TABLE}'",
            conn,
        )
        aux3 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred3d from '{AUX3_TABLE}'",
            conn,
        )

    for df in (labels, proxy, aux10, aux3):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(proxy, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(aux10, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")

    frame["rank5d"] = frame.groupby("trade_date")["pred5d"].rank(method="average", pct=True)
    frame["rank10d"] = frame.groupby("trade_date")["pred10d"].rank(method="average", pct=True)
    frame["rank3d"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)
    frame["rank10d"] = frame["rank10d"].fillna(frame["rank5d"])
    frame["rank3d"] = frame["rank3d"].fillna(frame["rank5d"])

    base_score = frame["rank5d"].copy()
    mask = (frame["trade_date"] >= START_DATE) & (frame["rank5d"] >= 1.0 - ZONE)
    blend_score = base_score.copy()
    blend_score.loc[mask] = (
        frame.loc[mask, "rank5d"]
        + ALPHA10 * frame.loc[mask, "rank10d"]
        + ALPHA3 * frame.loc[mask, "rank3d"]
    )

    base_eval = evaluate(frame.assign(score=base_score), "score")
    selected_eval = evaluate(frame.assign(score=blend_score), "score")

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = blend_score.astype(float)

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

    daily_activity = (
        frame.loc[mask, ["trade_date", "stock_code"]]
        .groupby("trade_date", as_index=False)
        .size()
        .rename(columns={"size": "active_rows"})
    )
    daily_activity.to_csv(OUT_DIR / "active_trade_dates.csv", index=False, encoding="utf-8-sig")
    selected_eval["daily"].to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_proxy5d_topzone_blend_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "base_proxy_table": BASE_PROXY_TABLE,
        "aux10_table": AUX10_TABLE,
        "aux3_table": AUX3_TABLE,
        "rule": {
            "base_rank_source": "5d_best_rank",
            "start_date": START_DATE,
            "zone": ZONE,
            "alpha10": ALPHA10,
            "alpha3": ALPHA3,
            "formula": "if trade_date >= start_date and rank5d >= 1-zone: score = rank5d + alpha10 * rank10d + alpha3 * rank3d; else score = rank5d",
        },
        "base_focus": focus_score(base_eval),
        "selected_focus": focus_score(selected_eval),
        "focus_delta": focus_score(selected_eval) - focus_score(base_eval),
        "base_windows": {k: v for k, v in base_eval.items() if k != "daily"},
        "selected_windows": {k: v for k, v in selected_eval.items() if k != "daily"},
        "active_rows": int(mask.sum()),
        "active_trade_days": int(mask.groupby(frame["trade_date"]).any().sum()),
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "cross_horizon_proxy_5d_for_10d_label",
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
        "candidate_id": "model_agent_10d_proxy5d_topzone_blend_20260623",
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
        "base_proxy_source": f"MODEL_PREDICTIONS.db::{BASE_PROXY_TABLE}",
        "aux10_source": f"MODEL_PREDICTIONS.db::{AUX10_TABLE}",
        "aux3_source": f"MODEL_PREDICTIONS.db::{AUX3_TABLE}",
        "notes": "10D 研究候选采用跨周期代理：以当前最优 5D 研究表作为主排序，仅在 20260201 之后且 5D 排名进入前 2% 的股票上，叠加少量 10D 与 3D 排序分用于局部重排。仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 10D 跨周期 5D 代理局部重排研究候选",
        "",
        "## 当前结论",
        "",
        f"- 基线代理表：`{BASE_PROXY_TABLE}`",
        f"- 新研究表：`{RESEARCH_TABLE}`",
        f"- Focus：`{payload['base_focus']:.12f}` -> `{payload['selected_focus']:.12f}`",
        f"- Focus 增量：`{payload['focus_delta']:.12f}`",
        "",
        "## 规则",
        "",
        f"- 自 `{START_DATE}` 起，若 `rank5d >= {1.0 - ZONE:.2f}`，则使用：`rank5d + {ALPHA10} * rank10d + {ALPHA3} * rank3d`。",
        "- 其余股票保持 `rank5d` 原排序。",
        "",
        "## 窗口结果",
        "",
        f"- recent63：`Top1={selected_eval['recent63']['top1']:.8f}` `Top3={selected_eval['recent63']['top3']:.8f}` `Top5={selected_eval['recent63']['top5']:.8f}`",
        f"- recent126：`Top1={selected_eval['recent126']['top1']:.8f}` `Top3={selected_eval['recent126']['top3']:.8f}` `Top5={selected_eval['recent126']['top5']:.8f}`",
        f"- full：`Top1={selected_eval['full']['top1']:.8f}` `Top3={selected_eval['full']['top3']:.8f}` `Top5={selected_eval['full']['top5']:.8f}`",
        "",
        "## 活跃范围",
        "",
        f"- 活跃行数：`{payload['active_rows']}`",
        f"- 活跃交易日：`{payload['active_trade_days']}`",
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
        f"- `{(OUT_DIR / 'active_trade_dates.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
