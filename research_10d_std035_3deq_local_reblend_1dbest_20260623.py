from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_std035_3deq_local_reblend_1dbest_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
BASE5_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research"
AUX10_TABLE = "stock_predict_data_model_agent_10d_no_star_r3gate991_r5gate98_20260623_executable_10d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_equalblend_20260623_executable_3d_open_return_research"
AUX1_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_20260623_executable_1d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_10d_std035_local_reblend_1dbest_20260623_executable_10d_open_return_research"
PREVIOUS_BEST_TABLE = "stock_predict_data_model_agent_10d_proxy5d_lite_daygate_std035_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_std035_3deq_local_reblend_1dbest_20260623_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
LITE_START_DATE = "20260201"
LITE_ZONE = 0.02
ALPHA10 = 0.005
ALPHA3 = 0.0025
STD_GATE = 0.35
WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}
RERANK_STARTS = ["20260225", "20260201", "20260115"]
RERANK_ZONES = [0.001, 0.0015, 0.002, 0.003]
RERANK_ALPHAS = [0.002, 0.005, 0.01, 0.015]


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
        aux1 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred1d from '{AUX1_TABLE}'",
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

    for df in (labels, proxy, aux10, aux3, aux1, current, previous):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(proxy, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(aux10, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux1, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(previous, on=["trade_date", "stock_code"], how="left")

    frame["rank5d"] = frame.groupby("trade_date")["pred5d"].rank(method="average", pct=True)
    frame["rank10d"] = frame.groupby("trade_date")["pred10d"].rank(method="average", pct=True)
    frame["rank3d"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)
    frame["rank1d"] = frame.groupby("trade_date")["pred1d"].rank(method="average", pct=True)
    frame["rank10d"] = frame["rank10d"].fillna(frame["rank5d"])
    frame["rank3d"] = frame["rank3d"].fillna(frame["rank5d"])
    frame["rank1d"] = frame["rank1d"].fillna(frame["rank5d"])

    base_score = frame["rank5d"].copy()
    lite_score = base_score.copy()
    top_zone_mask = (frame["trade_date"] >= LITE_START_DATE) & (frame["rank5d"] >= 1.0 - LITE_ZONE)
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

    std035_score = base_score.copy()
    std035_score.loc[frame["use_lite_gate"]] = lite_score.loc[frame["use_lite_gate"]]
    frame["std035_score"] = std035_score
    frame["rank_std035"] = frame.groupby("trade_date")["std035_score"].rank(method="average", pct=True)

    base_eval, _ = evaluate(frame.assign(score=std035_score), "score")
    current_eval, _ = evaluate(frame.assign(score=frame["pred_current"]), "score")
    previous_eval, _ = evaluate(frame.assign(score=frame["pred_previous"]), "score")
    base_focus = focus_score(base_eval)

    rows: list[dict] = []
    best_rule: dict | None = None
    best_focus = float("-inf")
    best_summary: dict | None = None
    best_score_series: pd.Series | None = None

    for start in RERANK_STARTS:
        for zone in RERANK_ZONES:
            for alpha in RERANK_ALPHAS:
                active_mask = (frame["trade_date"] >= start) & (frame["rank_std035"] >= 1.0 - zone)
                score = frame["rank_std035"].copy()
                score.loc[active_mask] = frame.loc[active_mask, "rank_std035"] + alpha * frame.loc[active_mask, "rank1d"]
                variant = frame[["trade_date", "stock_code", LABEL_COL]].copy()
                variant["score"] = score
                summary, _ = evaluate(variant, "score")
                focus = focus_score(summary)
                row = {
                    "start": start,
                    "zone": zone,
                    "alpha": alpha,
                    "active_rows": int(active_mask.sum()),
                    "focus": focus,
                    "delta_focus": focus - base_focus,
                    "recent63_top1": summary["recent63"]["top1"],
                    "recent63_top3": summary["recent63"]["top3"],
                    "recent63_top5": summary["recent63"]["top5"],
                    "recent63_top10": summary["recent63"]["top10"],
                    "recent126_top1": summary["recent126"]["top1"],
                    "recent126_top3": summary["recent126"]["top3"],
                    "recent126_top5": summary["recent126"]["top5"],
                    "full_top1": summary["full"]["top1"],
                    "full_top3": summary["full"]["top3"],
                    "full_top5": summary["full"]["top5"],
                }
                rows.append(row)
                if (
                    focus > best_focus + 1e-15
                    or (
                        abs(focus - best_focus) <= 1e-15
                        and best_rule is not None
                        and (start > best_rule["start"] or (start == best_rule["start"] and int(active_mask.sum()) < best_rule["active_rows"]))
                    )
                ):
                    best_focus = focus
                    best_rule = row
                    best_summary = summary
                    best_score_series = variant["score"].copy()

    result_df = pd.DataFrame(rows).sort_values(
        ["focus", "recent63_top1", "recent126_top1", "full_top1"],
        ascending=False,
    )
    result_df.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")

    if best_rule is None or best_summary is None or best_score_series is None:
        raise RuntimeError("No candidate rule evaluated")

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = best_score_series.astype(float)

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

    selected_daily = evaluate(frame.assign(score=best_score_series), "score")[1]
    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    day_gate.to_csv(OUT_DIR / "day_gate.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_eval, previous_eval, best_summary).to_csv(
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
        "candidate_id": "model_agent_10d_std035_3deq_local_reblend_1dbest_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "base_proxy_table": BASE5_TABLE,
        "aux10_table": AUX10_TABLE,
        "aux3_table": AUX3_TABLE,
        "aux1_table": AUX1_TABLE,
        "current_table": CURRENT_TABLE,
        "previous_best_table": PREVIOUS_BEST_TABLE,
        "lite_rule": {
            "start_date": LITE_START_DATE,
            "zone": LITE_ZONE,
            "alpha10": ALPHA10,
            "alpha3": ALPHA3,
        },
        "day_gate_rule": {
            "gate_rule": f"use lite when std(rank10d of base top5) < {STD_GATE}",
            "gate_true_days": int(day_gate["use_lite_gate"].sum()),
            "gate_false_days": int((~day_gate["use_lite_gate"]).sum()),
            "gate_true_recent63_days": int(
                day_gate.loc[
                    (day_gate["trade_date"] >= WINDOWS["recent63"][0]) & (day_gate["trade_date"] <= WINDOWS["recent63"][1]),
                    "use_lite_gate",
                ].sum()
            ),
        },
        "rerank_rule": best_rule,
        "base_focus": base_focus,
        "current_focus": focus_score(current_eval),
        "previous_focus": focus_score(previous_eval),
        "selected_focus": best_focus,
        "selected_minus_base_focus": best_focus - base_focus,
        "selected_minus_current_focus": best_focus - focus_score(current_eval),
        "selected_minus_previous_focus": best_focus - focus_score(previous_eval),
        "base_windows": base_eval,
        "current_windows": current_eval,
        "previous_windows": previous_eval,
        "selected_windows": best_summary,
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "cross_horizon_proxy_5d_for_10d_label",
            "replaces 3d formal helper with current 3d equalblend research candidate",
            "rebuilds std035 structure and rescans narrow 1d-best local rerank",
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
        "candidate_id": "model_agent_10d_std035_3deq_local_reblend_1dbest_20260623",
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
        "aux3_source": f"MODEL_PREDICTIONS.db::{AUX3_TABLE}",
        "aux1_source": f"MODEL_PREDICTIONS.db::{AUX1_TABLE}",
        "current_candidate_ref": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "notes": "10D research 候选重建当前 std035 + 1D best 结构，但把 3D 辅助源从 guarded formal 替换为当前 3D equalblend research 候选，用于检查 3D research 是否能继续提升 10D 头部质量。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 10D 用 3D equalblend 重建 std035 并叠加 1D best 的研究候选",
        "",
        "## 当前结论",
        "",
        f"- 当前对比候选：`{CURRENT_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 相对当前候选 Focus 增量：`{payload['selected_minus_current_focus']:.12f}`",
        f"- 相对旧 std035 候选 Focus 增量：`{payload['selected_minus_previous_focus']:.12f}`",
        "",
        "## 结构",
        "",
        "- 5D proxy 仍沿用旧 d3d1_v2。",
        "- 10D lite/day-gate std035 结构保持不变。",
        "- 3D 辅助源改为当前更强的 3D equalblend research 候选。",
        "- 再在其上重扫一轮极窄 1D best 局部重排。",
        "",
        "## 最优重排规则",
        "",
        f"- start：`{best_rule['start']}`",
        f"- zone：`{best_rule['zone']}`",
        f"- alpha：`{best_rule['alpha']}`",
        f"- active_rows：`{best_rule['active_rows']}`",
        "",
        "## 关键窗口",
        "",
        f"- recent63：`Top1={best_summary['recent63']['top1']:.8f}` `Top3={best_summary['recent63']['top3']:.8f}` `Top5={best_summary['recent63']['top5']:.8f}`",
        f"- recent126：`Top1={best_summary['recent126']['top1']:.8f}` `Top3={best_summary['recent126']['top3']:.8f}` `Top5={best_summary['recent126']['top5']:.8f}`",
        f"- full：`Top1={best_summary['full']['top1']:.8f}` `Top3={best_summary['full']['top3']:.8f}` `Top5={best_summary['full']['top5']:.8f}`",
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
        f"- `{(OUT_DIR / 'grid_results.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'day_gate.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
