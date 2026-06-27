from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_3d_proxy10d5d_bestmix_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
BASE10_TABLE = "stock_predict_data_model_agent_10d_daygate_joint_refine_20260623_executable_10d_open_return_research"
BASE5_TABLE = "stock_predict_data_model_agent_5d_v3_daymix_latest1d_20260623_executable_5d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_equalblend_20260623_executable_3d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_bestmix_20260623_executable_3d_open_return_research"

LABEL_COL = "executable_3d_open_return"
WEIGHT_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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

    base_eval, _ = evaluate(frame.assign(score=frame["rank10"]), "score")
    current_eval, _ = evaluate(frame.assign(score=frame["pred_current"]), "score")
    base_focus = focus_score(base_eval)
    current_focus = focus_score(current_eval)

    grid_rows: list[dict] = []
    best_rule: dict | None = None
    best_score: pd.Series | None = None
    best_eval: dict | None = None
    best_daily: pd.DataFrame | None = None
    best_focus = float("-inf")

    for w10 in WEIGHT_GRID:
        w5 = 1.0 - w10
        score = w10 * frame["rank10"] + w5 * frame["rank5"]
        eval_summary, daily_eval = evaluate(frame.assign(score=score), "score")
        focus = focus_score(eval_summary)
        row = {
            "w10": w10,
            "w5": w5,
            "focus": focus,
            "delta_vs_base": focus - base_focus,
            "delta_vs_current": focus - current_focus,
            "full_top1": eval_summary["full"]["top1"],
            "full_top3": eval_summary["full"]["top3"],
            "full_top5": eval_summary["full"]["top5"],
            "full_top10": eval_summary["full"]["top10"],
            "recent126_top1": eval_summary["recent126"]["top1"],
            "recent126_top3": eval_summary["recent126"]["top3"],
            "recent126_top5": eval_summary["recent126"]["top5"],
            "recent126_top10": eval_summary["recent126"]["top10"],
            "recent63_top1": eval_summary["recent63"]["top1"],
            "recent63_top3": eval_summary["recent63"]["top3"],
            "recent63_top5": eval_summary["recent63"]["top5"],
            "recent63_top10": eval_summary["recent63"]["top10"],
        }
        grid_rows.append(row)
        if (
            focus > best_focus + 1e-15
            or (
                abs(focus - best_focus) <= 1e-15
                and best_rule is not None
                and (
                    row["recent63_top3"] > best_rule["recent63_top3"] + 1e-15
                    or (
                        abs(row["recent63_top3"] - best_rule["recent63_top3"]) <= 1e-15
                        and row["recent63_top10"] > best_rule["recent63_top10"] + 1e-15
                    )
                )
            )
        ):
            best_focus = focus
            best_rule = row
            best_score = score.copy()
            best_eval = eval_summary
            best_daily = daily_eval.copy()

    if best_rule is None or best_score is None or best_eval is None or best_daily is None:
        raise RuntimeError("No candidate rule evaluated")

    candidate = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate["pred_prob"] = best_score.astype(float)

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

    pd.DataFrame(grid_rows).sort_values(
        ["focus", "recent63_top3", "recent63_top10", "recent126_top3"],
        ascending=False,
    ).to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")
    best_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_eval, current_eval, best_eval).to_csv(
        OUT_DIR / "window_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(best_daily, "year").to_csv(
        OUT_DIR / "yearly_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(best_daily, "halfyear").to_csv(
        OUT_DIR / "halfyear_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_period_breakdown(best_daily, "quarter").to_csv(
        OUT_DIR / "quarter_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_3d_proxy10d5d_bestmix_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "base10_table": BASE10_TABLE,
        "base5_table": BASE5_TABLE,
        "current_table": CURRENT_TABLE,
        "search_space": {
            "w10_grid": WEIGHT_GRID,
            "w5_rule": "w5 = 1 - w10",
        },
        "selected_rule": {
            "type": "rank_blend",
            "w10": best_rule["w10"],
            "w5": best_rule["w5"],
            "formula": f"score = {best_rule['w10']:.2f} * rank10 + {best_rule['w5']:.2f} * rank5",
        },
        "base_focus": base_focus,
        "current_focus": current_focus,
        "selected_focus": best_focus,
        "selected_minus_base_focus": best_focus - base_focus,
        "selected_minus_current_focus": best_focus - current_focus,
        "base_windows": base_eval,
        "current_windows": current_eval,
        "selected_windows": best_eval,
        "db_summary": db_row,
        "governance_notes": [
            "research only",
            "cross_horizon_proxy_blend",
            "uses current best 10D and current best 5D research sources",
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
        "candidate_id": "model_agent_3d_proxy10d5d_bestmix_20260623",
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
        "notes": "3D 研究候选改为使用当前最强 10D 与当前最强 5D 的 rank blend，并通过权重网格选择最优组合。本次仅用于研究验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 3D 最优源组合 rank blend 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 10D 评分源：`{BASE10_TABLE}`",
        f"- 5D 评分源：`{BASE5_TABLE}`",
        f"- 上一版 3D 候选：`{CURRENT_TABLE}`",
        f"- 新版 3D 候选：`{RESEARCH_TABLE}`",
        f"- 10D-only baseline focus：`{base_focus:.12f}`",
        f"- 上一版 3D focus：`{current_focus:.12f}`",
        f"- 新版 3D focus：`{best_focus:.12f}`",
        f"- 相对上一版提升：`{best_focus - current_focus:.12f}`",
        "",
        "## 最优组合",
        "",
        f"- `w10 = {best_rule['w10']:.2f}`",
        f"- `w5 = {best_rule['w5']:.2f}`",
        f"- `formula = {best_rule['w10']:.2f} * rank10 + {best_rule['w5']:.2f} * rank5`",
        "",
        "## 关键窗口结果",
        "",
        f"- recent63：`Top1={best_eval['recent63']['top1']:.8f}` `Top3={best_eval['recent63']['top3']:.8f}` `Top5={best_eval['recent63']['top5']:.8f}` `Top10={best_eval['recent63']['top10']:.8f}`",
        f"- recent126：`Top1={best_eval['recent126']['top1']:.8f}` `Top3={best_eval['recent126']['top3']:.8f}` `Top5={best_eval['recent126']['top5']:.8f}` `Top10={best_eval['recent126']['top10']:.8f}`",
        f"- full：`Top1={best_eval['full']['top1']:.8f}` `Top3={best_eval['full']['top3']:.8f}` `Top5={best_eval['full']['top5']:.8f}` `Top10={best_eval['full']['top10']:.8f}`",
        "",
        "## 治理说明",
        "",
        "- 本次仅生成 research 候选资产。",
        "- 未训练模型。",
        "- 未发布 formal 资产。",
        "- 未生成交易信号，未制定交易规则，未跑回测。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'grid_results.csv').as_posix()}`",
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
