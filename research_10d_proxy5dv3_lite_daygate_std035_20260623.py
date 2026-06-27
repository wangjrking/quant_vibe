from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_proxy5dv3_lite_daygate_std035_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
BASE5_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3eq_d1new_v3_20260623_executable_5d_open_return_research"
AUX10_TABLE = "stock_predict_data_model_agent_10d_no_star_r3gate991_r5gate98_20260623_executable_10d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
CURRENT_TABLE = "stock_predict_data_model_agent_10d_std035_local_reblend_1dbest_20260623_executable_10d_open_return_research"
PREVIOUS_BEST_TABLE = "stock_predict_data_model_agent_10d_proxy5d_lite_daygate_std035_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_proxy5dv3_lite_daygate_std035_20260623_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"
START_DATE = "20260201"
ZONE = 0.02
ALPHA10 = 0.005
ALPHA3 = 0.0025
STD_GATE = 0.35
WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}


def evaluate(frame: pd.DataFrame, score_col: str) -> tuple[dict, pd.DataFrame]:
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


def compare_windows(base_windows: dict, previous_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"base_{metric}"] = base_windows[name][metric]
            row[f"previous_{metric}"] = previous_windows[name][metric]
            row[f"new_{metric}"] = new_windows[name][metric]
            row[f"new_minus_base_{metric}"] = new_windows[name][metric] - base_windows[name][metric]
            row[f"new_minus_previous_{metric}"] = new_windows[name][metric] - previous_windows[name][metric]
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
            f"select trade_date, stock_code, pred_prob as pred5d from '{BASE5_TABLE}'",
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
        previous = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred_previous from '{PREVIOUS_BEST_TABLE}'",
            conn,
        )

    for df in (labels, proxy, aux10, aux3, current, previous):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(proxy, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(aux10, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(previous, on=["trade_date", "stock_code"], how="left")

    frame["rank5d"] = frame.groupby("trade_date")["pred5d"].rank(method="average", pct=True)
    frame["rank10d"] = frame.groupby("trade_date")["pred10d"].rank(method="average", pct=True)
    frame["rank3d"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)
    frame["rank10d"] = frame["rank10d"].fillna(frame["rank5d"])
    frame["rank3d"] = frame["rank3d"].fillna(frame["rank5d"])

    base_score = frame["rank5d"].copy()
    lite_score = base_score.copy()
    top_zone_mask = (frame["trade_date"] >= START_DATE) & (frame["rank5d"] >= 1.0 - ZONE)
    lite_score.loc[top_zone_mask] = (
        frame.loc[top_zone_mask, "rank5d"]
        + ALPHA10 * frame.loc[top_zone_mask, "rank10d"]
        + ALPHA3 * frame.loc[top_zone_mask, "rank3d"]
    )

    day_gate_rows: list[dict] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        top5_base = group.nlargest(5, "rank5d")
        std_r10_top5 = float(top5_base["rank10d"].std())
        use_lite = bool(std_r10_top5 < STD_GATE)
        day_gate_rows.append(
            {
                "trade_date": trade_date,
                "std_r10_top5_base": std_r10_top5,
                "use_lite_gate": use_lite,
            }
        )
    day_gate = pd.DataFrame(day_gate_rows)
    frame = frame.merge(day_gate, on="trade_date", how="left")

    selected_score = base_score.copy()
    selected_score.loc[frame["use_lite_gate"]] = lite_score.loc[frame["use_lite_gate"]]

    base_eval, _ = evaluate(frame.assign(score=base_score), "score")
    current_eval, _ = evaluate(frame.assign(score=frame["pred_current"]), "score")
    previous_eval, _ = evaluate(frame.assign(score=frame["pred_previous"]), "score")
    selected_eval, selected_daily = evaluate(frame.assign(score=selected_score), "score")

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = selected_score.astype(float)

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

    gate_summary = {
        "gate_rule": f"use lite when std(rank10d of base top5) < {STD_GATE}",
        "gate_true_days": int(day_gate["use_lite_gate"].sum()),
        "gate_false_days": int((~day_gate["use_lite_gate"]).sum()),
        "gate_true_recent63_days": int(
            day_gate.loc[
                (day_gate["trade_date"] >= WINDOWS["recent63"][0]) & (day_gate["trade_date"] <= WINDOWS["recent63"][1]),
                "use_lite_gate",
            ].sum()
        ),
    }

    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    day_gate.to_csv(OUT_DIR / "day_gate.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_eval, previous_eval, selected_eval).to_csv(
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
        "candidate_id": "model_agent_10d_proxy5dv3_lite_daygate_std035_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "base_proxy_table": BASE5_TABLE,
        "aux10_table": AUX10_TABLE,
        "aux3_table": AUX3_TABLE,
        "current_table": CURRENT_TABLE,
        "previous_best_table": PREVIOUS_BEST_TABLE,
        "lite_rule": {
            "start_date": START_DATE,
            "zone": ZONE,
            "alpha10": ALPHA10,
            "alpha3": ALPHA3,
        },
        "day_gate_rule": gate_summary,
        "base_focus": focus_score(base_eval),
        "current_focus": focus_score(current_eval),
        "previous_focus": focus_score(previous_eval),
        "selected_focus": focus_score(selected_eval),
        "selected_minus_base_focus": focus_score(selected_eval) - focus_score(base_eval),
        "selected_minus_previous_focus": focus_score(selected_eval) - focus_score(previous_eval),
        "base_windows": base_eval,
        "current_windows": current_eval,
        "previous_windows": previous_eval,
        "selected_windows": selected_eval,
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "cross_horizon_proxy_5d_for_10d_label",
            "replaces old 5d proxy with current 5d v3 research candidate",
            "day-level switch between base5 and lite10 candidate",
            "gate uses ex-ante dispersion of rank10 inside base top5",
            "std gate widened from 0.04 to 0.35",
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
        "candidate_id": "model_agent_10d_proxy5dv3_lite_daygate_std035_20260623",
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
        "base_proxy_source": f"MODEL_PREDICTIONS.db::{BASE5_TABLE}",
        "current_candidate_ref": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "previous_best_table_ref": f"MODEL_PREDICTIONS.db::{PREVIOUS_BEST_TABLE}",
        "notes": "10D research 候选沿用当前 std035 lite/day-gate 结构，但把 5D proxy 从旧 d3d1_v2 替换成当前更强的 5D v3 候选，用于验证 10D 是否被旧 5D 底座限制。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 10D 使用 5D v3 作为 proxy 的 std035 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 当前对比候选：`{CURRENT_TABLE}`",
        f"- 旧 std035 候选：`{PREVIOUS_BEST_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 5D proxy：`{BASE5_TABLE}`",
        f"- 相对当前候选 Focus 增量：`{payload['selected_minus_base_focus']:.12f}`",
        f"- 相对旧 std035 候选 Focus 增量：`{payload['selected_minus_previous_focus']:.12f}`",
        "",
        "## 规则",
        "",
        "- lite 结构不变，只替换 5D proxy 为当前更强的 5D v3。",
        f"- top-zone：`rank5d >= {1.0 - ZONE:.2f}`",
        f"- 局部重排：`rank5d + {ALPHA10} * rank10d + {ALPHA3} * rank3d`",
        f"- 日级 gate：`std(rank10d of base top5) < {STD_GATE}`",
        "",
        "## 关键窗口",
        "",
        f"- recent63：`Top1={selected_eval['recent63']['top1']:.8f}` `Top3={selected_eval['recent63']['top3']:.8f}` `Top5={selected_eval['recent63']['top5']:.8f}`",
        f"- recent126：`Top1={selected_eval['recent126']['top1']:.8f}` `Top3={selected_eval['recent126']['top3']:.8f}` `Top5={selected_eval['recent126']['top5']:.8f}`",
        f"- full：`Top1={selected_eval['full']['top1']:.8f}` `Top3={selected_eval['full']['top3']:.8f}` `Top5={selected_eval['full']['top5']:.8f}`",
        "",
        "## 治理说明",
        "",
        "- 本次只生成 research 候选资产。",
        "- 未训练模型。",
        "- 未发布 formal 资产。",
        "- 未生成交易信号，未跑回测。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'day_gate.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'window_comparison.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
