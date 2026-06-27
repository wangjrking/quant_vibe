from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_proxy5d_topzone_lite_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
BASE_PROXY_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research"
AUX10_TABLE = "stock_predict_data_model_agent_10d_no_star_r3gate991_r5gate98_20260623_executable_10d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
CURRENT_TABLE = "stock_predict_data_model_agent_10d_proxy5d_topzone_blend_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_proxy5d_topzone_lite_20260623_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"
START_DATE = "20260201"
ZONE = 0.02
ALPHA10 = 0.005
ALPHA3 = 0.0025
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


def compare_windows(base_windows: dict, current_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"base_{metric}"] = base_windows[name][metric]
            row[f"current_{metric}"] = current_windows[name][metric]
            row[f"lite_{metric}"] = new_windows[name][metric]
            row[f"lite_minus_base_{metric}"] = new_windows[name][metric] - base_windows[name][metric]
            row[f"lite_minus_current_{metric}"] = new_windows[name][metric] - current_windows[name][metric]
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
        current = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred_current from '{CURRENT_TABLE}'",
            conn,
        )

    for df in (labels, proxy, aux10, aux3, current):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(proxy, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(aux10, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="left")

    frame["rank5d"] = frame.groupby("trade_date")["pred5d"].rank(method="average", pct=True)
    frame["rank10d"] = frame.groupby("trade_date")["pred10d"].rank(method="average", pct=True)
    frame["rank3d"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)
    frame["rank10d"] = frame["rank10d"].fillna(frame["rank5d"])
    frame["rank3d"] = frame["rank3d"].fillna(frame["rank5d"])

    base_score = frame["rank5d"].copy()
    mask = (frame["trade_date"] >= START_DATE) & (frame["rank5d"] >= 1.0 - ZONE)
    lite_score = base_score.copy()
    lite_score.loc[mask] = (
        frame.loc[mask, "rank5d"]
        + ALPHA10 * frame.loc[mask, "rank10d"]
        + ALPHA3 * frame.loc[mask, "rank3d"]
    )

    base_eval = evaluate(frame.assign(score=base_score), "score")
    current_eval = evaluate(frame.assign(score=frame["pred_current"]), "score")
    lite_eval = evaluate(frame.assign(score=lite_score), "score")

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = lite_score.astype(float)

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

    compare_df = compare_windows(base_eval, current_eval, lite_eval)
    compare_df.to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")
    lite_eval["daily"].to_csv(OUT_DIR / "lite_daily_eval.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(lite_eval["daily"], "year").to_csv(
        OUT_DIR / "lite_yearly_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(lite_eval["daily"], "halfyear").to_csv(
        OUT_DIR / "lite_halfyear_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(lite_eval["daily"], "quarter").to_csv(
        OUT_DIR / "lite_quarter_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_proxy5d_topzone_lite_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "base_proxy_table": BASE_PROXY_TABLE,
        "aux10_table": AUX10_TABLE,
        "aux3_table": AUX3_TABLE,
        "current_table": CURRENT_TABLE,
        "rule": {
            "base_rank_source": "5d_best_rank",
            "start_date": START_DATE,
            "zone": ZONE,
            "alpha10": ALPHA10,
            "alpha3": ALPHA3,
            "formula": "if trade_date >= start_date and rank5d >= 1-zone: score = rank5d + alpha10 * rank10d + alpha3 * rank3d; else score = rank5d",
        },
        "base_focus": focus_score(base_eval),
        "current_focus": focus_score(current_eval),
        "lite_focus": focus_score(lite_eval),
        "lite_minus_base_focus": focus_score(lite_eval) - focus_score(base_eval),
        "lite_minus_current_focus": focus_score(lite_eval) - focus_score(current_eval),
        "base_windows": {k: v for k, v in base_eval.items() if k != "daily"},
        "current_windows": {k: v for k, v in current_eval.items() if k != "daily"},
        "lite_windows": {k: v for k, v in lite_eval.items() if k != "daily"},
        "active_rows": int(mask.sum()),
        "active_trade_days": int(mask.groupby(frame["trade_date"]).any().sum()),
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "cross_horizon_proxy_5d_for_10d_label",
            "lighter top-zone rerank than current 10d candidate",
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
        "candidate_id": "model_agent_10d_proxy5d_topzone_lite_20260623",
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
        "current_candidate_source": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "notes": "10D 研究候选的轻量版局部重排。保持 5D 代理为主排序，只在 20260201 之后且 5D 排名进入顶部 2% 的股票上，叠加更轻的 10D/3D 排名权重，用于验证近端前排收益是否能在减小重排力度后继续改善。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 10D 轻量顶部分区重排研究候选",
        "",
        "## 当前结论",
        "",
        f"- 基线代理表：`{BASE_PROXY_TABLE}`",
        f"- 当前 10D 研究表：`{CURRENT_TABLE}`",
        f"- 新轻量候选表：`{RESEARCH_TABLE}`",
        f"- 基线 Focus：`{payload['base_focus']:.12f}`",
        f"- 当前候选 Focus：`{payload['current_focus']:.12f}`",
        f"- 轻量候选 Focus：`{payload['lite_focus']:.12f}`",
        f"- 相对基线增量：`{payload['lite_minus_base_focus']:.12f}`",
        f"- 相对当前候选增量：`{payload['lite_minus_current_focus']:.12f}`",
        "",
        "## 规则",
        "",
        f"- 起始日期：`{START_DATE}`",
        f"- 触发区间：`rank5d >= {1.0 - ZONE:.2f}`",
        f"- 局部重排公式：`rank5d + {ALPHA10} * rank10d + {ALPHA3} * rank3d`",
        f"- 相比当前候选，仅降低顶部分区内的辅助排序力度，不改训练、不改特征来源。",
        "",
        "## 评价摘要",
        "",
        f"- recent63：`Top1={lite_eval['recent63']['top1']:.8f}` `Top3={lite_eval['recent63']['top3']:.8f}` `Top5={lite_eval['recent63']['top5']:.8f}`",
        f"- recent126：`Top1={lite_eval['recent126']['top1']:.8f}` `Top3={lite_eval['recent126']['top3']:.8f}` `Top5={lite_eval['recent126']['top5']:.8f}`",
        f"- full：`Top1={lite_eval['full']['top1']:.8f}` `Top3={lite_eval['full']['top3']:.8f}` `Top5={lite_eval['full']['top5']:.8f}`",
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
        f"- `{(OUT_DIR / 'lite_yearly_stability.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'lite_halfyear_stability.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'lite_quarter_stability.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
