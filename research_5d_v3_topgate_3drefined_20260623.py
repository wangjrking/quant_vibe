from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_v3_topgate_3drefined_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_v3_daymix_latest1d_20260623_executable_5d_open_return_research"
AUX_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_bestmix_refine_20260623_executable_3d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_v3_topgate_3drefined_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
WINDOWS = {
    "full": ("20240604", "20260605"),
    "recent1y": ("20250301", "20260605"),
    "recent80": ("20260201", "20260605"),
}

DATE_GATES = ["20260101", "20260201", "20260301"]
BASE_RANK_THRESHOLDS = [0.995, 0.996, 0.997, 0.998, 0.999]
AUX_ALPHAS = [0.005, 0.01, 0.015, 0.02]


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


def compare_windows(base_windows: dict, selected_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent1y", "recent80"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"base_{metric}"] = base_windows[name][metric]
            row[f"selected_{metric}"] = selected_windows[name][metric]
            row[f"selected_minus_base_{metric}"] = selected_windows[name][metric] - base_windows[name][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def build_period_breakdown(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = daily.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    if period == "year":
        frame["period"] = frame["trade_date"].str.slice(0, 4)
    elif period == "quarter":
        quarter = frame["trade_date"].str.slice(4, 6).astype(int).map(lambda m: str((m - 1) // 3 + 1))
        frame["period"] = frame["trade_date"].str.slice(0, 4) + "Q" + quarter
    else:
        raise ValueError(f"unsupported period: {period}")
    return (
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


def materialize(merged: pd.DataFrame, best_rule: dict, best_summary: dict, best_focus: float, base_focus: float) -> dict:
    out = merged.copy()
    out["pred_prob"] = out["selected_score"].astype(float)
    keep_cols = [
        "trade_date",
        "stock_code",
        LABEL_COL,
        "pred_prob",
        "base_pred_prob",
        "rank_base",
        "aux_pred_prob",
        "rank_aux",
        "topgate_applied",
        "topgate_reason",
    ]
    out = out[keep_cols]
    with sqlite3.connect(MODEL_DB) as conn:
        out.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{RESEARCH_TABLE}_date_code on '{RESEARCH_TABLE}'(trade_date, stock_code)")
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), count(distinct stock_code) "
            f"from '{RESEARCH_TABLE}'"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from '{RESEARCH_TABLE}' "
            "group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        null_pred = conn.execute(f"select count(*) from '{RESEARCH_TABLE}' where pred_prob is null").fetchone()[0]
    return {
        "db_path": str(MODEL_DB),
        "table": RESEARCH_TABLE,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "duplicate_key_groups": int(dup),
        "null_pred_prob": int(null_pred),
        "selected_focus": best_focus,
        "focus_delta": best_focus - base_focus,
        "selected_rule": best_rule,
        "selected_windows": best_summary,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        base_rows = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} "
            f"from '{BASE_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        aux_rows = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as aux_pred_prob "
            f"from '{AUX_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )

    for df in (base_rows, aux_rows):
        df["trade_date"] = df["trade_date"].astype(str)

    merged = base_rows.merge(aux_rows, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    merged["rank_base"] = merged.groupby("trade_date")["base_pred_prob"].rank(method="average", pct=True)
    merged["rank_aux"] = merged.groupby("trade_date")["aux_pred_prob"].rank(method="average", pct=True)
    merged["rank_aux"] = merged["rank_aux"].fillna(merged["rank_base"])

    base_summary, base_daily = evaluate(merged.rename(columns={"base_pred_prob": "score"}), "score")
    base_focus = focus_score(base_summary)
    base_recent80_top3 = base_summary["recent80"]["top3"]
    base_recent80_top5 = base_summary["recent80"]["top5"]

    grid_rows: list[dict] = []
    best_rule: dict | None = None
    best_focus = float("-inf")
    best_summary: dict | None = None
    best_daily: pd.DataFrame | None = None
    best_score: pd.Series | None = None
    best_topgate_mask: pd.Series | None = None
    best_passed = False

    for date_gate in DATE_GATES:
        for threshold in BASE_RANK_THRESHOLDS:
            for alpha in AUX_ALPHAS:
                topgate_mask = (merged["trade_date"] >= date_gate) & (merged["rank_base"] >= threshold)
                score = merged["rank_base"].copy()
                score.loc[topgate_mask] = score.loc[topgate_mask] + alpha * merged.loc[topgate_mask, "rank_aux"]
                eval_summary, daily_eval = evaluate(merged.assign(score=score), "score")
                focus = focus_score(eval_summary)
                pass_guard = (
                    eval_summary["recent80"]["top3"] >= base_recent80_top3 - 0.0005
                    and eval_summary["recent80"]["top5"] >= base_recent80_top5 - 0.0005
                )
                row = {
                    "date_gate": date_gate,
                    "base_rank_threshold": threshold,
                    "aux_add_alpha": alpha,
                    "topgate_rows": int(topgate_mask.sum()),
                    "focus": focus,
                    "delta_focus": focus - base_focus,
                    "guard_pass": pass_guard,
                    "full_top1": eval_summary["full"]["top1"],
                    "full_top3": eval_summary["full"]["top3"],
                    "full_top5": eval_summary["full"]["top5"],
                    "full_top10": eval_summary["full"]["top10"],
                    "recent1y_top1": eval_summary["recent1y"]["top1"],
                    "recent1y_top3": eval_summary["recent1y"]["top3"],
                    "recent1y_top5": eval_summary["recent1y"]["top5"],
                    "recent1y_top10": eval_summary["recent1y"]["top10"],
                    "recent80_top1": eval_summary["recent80"]["top1"],
                    "recent80_top3": eval_summary["recent80"]["top3"],
                    "recent80_top5": eval_summary["recent80"]["top5"],
                    "recent80_top10": eval_summary["recent80"]["top10"],
                }
                grid_rows.append(row)

                better = False
                if pass_guard and not best_passed:
                    better = True
                elif pass_guard == best_passed:
                    if focus > best_focus + 1e-15:
                        better = True
                    elif abs(focus - best_focus) <= 1e-15 and best_rule is not None:
                        if row["recent80_top3"] > best_rule["recent80_top3"] + 1e-15:
                            better = True
                        elif abs(row["recent80_top3"] - best_rule["recent80_top3"]) <= 1e-15 and row["recent80_top5"] > best_rule["recent80_top5"] + 1e-15:
                            better = True

                if better:
                    best_rule = row
                    best_focus = focus
                    best_summary = eval_summary
                    best_daily = daily_eval.copy()
                    best_score = score.copy()
                    best_topgate_mask = topgate_mask.copy()
                    best_passed = pass_guard

    if best_rule is None or best_summary is None or best_daily is None or best_score is None or best_topgate_mask is None:
        raise RuntimeError("No candidate rule evaluated")

    merged["selected_score"] = best_score.astype(float)
    merged["topgate_applied"] = best_topgate_mask.astype(bool)
    merged["topgate_reason"] = ""
    merged.loc[merged["topgate_applied"], "topgate_reason"] = (
        "date_gate="
        + str(best_rule["date_gate"])
        + ";threshold="
        + str(best_rule["base_rank_threshold"])
        + ";alpha="
        + str(best_rule["aux_add_alpha"])
    )

    result_df = pd.DataFrame(grid_rows).sort_values(
        ["guard_pass", "focus", "recent80_top3", "recent80_top5", "recent1y_top3"],
        ascending=[False, False, False, False, False],
    )
    result_df.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")

    db_summary = materialize(merged, best_rule, best_summary, best_focus, base_focus)
    best_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    base_daily.to_csv(OUT_DIR / "baseline_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_summary, best_summary).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_daily, "year").to_csv(OUT_DIR / "yearly_stability.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_daily, "quarter").to_csv(OUT_DIR / "quarter_stability.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_5d_v3_topgate_3drefined_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
        "search_space": {
            "date_gates": DATE_GATES,
            "base_rank_thresholds": BASE_RANK_THRESHOLDS,
            "aux_alphas": AUX_ALPHAS,
        },
        "base_focus": base_focus,
        "selected_focus": best_focus,
        "focus_delta": best_focus - base_focus,
        "selected_rule": best_rule,
        "guard_rule": {
            "recent80_top3_min": base_recent80_top3 - 0.0005,
            "recent80_top5_min": base_recent80_top5 - 0.0005,
            "selected_passed": best_passed,
        },
        "base_windows": base_summary,
        "selected_windows": best_summary,
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "5d current best receives a small 3d refined rank add-on only inside the top bucket after a date gate",
            "selection prefers candidates that preserve recent80 top3/top5",
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
        "candidate_id": "model_agent_5d_v3_topgate_3drefined_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": payload["generated_at"],
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "aux_source": f"MODEL_PREDICTIONS.db::{AUX_TABLE}",
        "selected_rule": best_rule,
        "notes": "5D 当前研究最优分数在顶部极小分位区间引入 3D refined rank 微调，仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = f"""# 5D 顶部分位接入 3D refined 研究候选

## 当前结论

- 基线 5D 资产：`{BASE_TABLE}`
- 3D refined 辅助资产：`{AUX_TABLE}`
- 新 research 候选：`{RESEARCH_TABLE}`
- Focus：`{base_focus:.12f}` -> `{best_focus:.12f}`
- Focus 增量：`{best_focus - base_focus:.12f}`
- 守卫是否通过：`{best_passed}`

## 最优规则

- `date_gate={best_rule['date_gate']}`
- `base_rank_threshold={best_rule['base_rank_threshold']}`
- `aux_add_alpha={best_rule['aux_add_alpha']}`
- `topgate_rows={best_rule['topgate_rows']}`

## 关键窗口

- recent80：`Top1={best_summary['recent80']['top1']:.8f}` `Top3={best_summary['recent80']['top3']:.8f}` `Top5={best_summary['recent80']['top5']:.8f}`
- recent1y：`Top1={best_summary['recent1y']['top1']:.8f}` `Top3={best_summary['recent1y']['top3']:.8f}` `Top5={best_summary['recent1y']['top5']:.8f}`
- full：`Top1={best_summary['full']['top1']:.8f}` `Top3={best_summary['full']['top3']:.8f}` `Top5={best_summary['full']['top5']:.8f}`

## 治理说明

- 本次只生成 research 候选资产。
- 未训练模型。
- 未切换 formal 资产。
- 未生成交易信号，未跑回测。
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
