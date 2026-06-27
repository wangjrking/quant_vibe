from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_v4_daily_gate_d3best_d1best_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_v3_daymix_latest1d_20260623_executable_5d_open_return_research"
AUX1_TABLE = "stock_predict_data_model_agent_1d_current10d_endgate_refine_20260623_executable_1d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_bestmix_refine_20260623_executable_3d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_5d_v3_daymix_latest1d_20260623_executable_5d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_v4_daily_gate_d3best_d1best_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
START_DATES = ["20250301", "20250601", "20251201"]
THR3_LIST = [0.97, 0.975, 0.98, 0.985, 0.99]
THR1_LIST = [0.97, 0.98, 0.99, 0.995]
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


def materialize(candidate: pd.DataFrame) -> dict:
    final = candidate[["trade_date", "stock_code", LABEL_COL, "pred_prob"]].copy()
    with sqlite3.connect(MODEL_DB) as conn:
        final.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{RESEARCH_TABLE}_date_code on '{RESEARCH_TABLE}'(trade_date, stock_code)")
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), count(distinct stock_code) from '{RESEARCH_TABLE}'"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from '{RESEARCH_TABLE}' group by trade_date, stock_code having c > 1)"
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
    }


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

    base_summary, base_daily = evaluate(base.rename(columns={"base_pred_prob": "score"}), "score")
    base_focus = focus_score(base_summary)
    base_recent80_top3 = base_summary["recent80"]["top3"]
    base_recent80_top5 = base_summary["recent80"]["top5"]

    rows: list[dict] = []
    best_result: dict | None = None

    grouped = [(trade_date, group.copy()) for trade_date, group in frame.groupby("trade_date", sort=True)]

    for start_date in START_DATES:
        for thr3 in THR3_LIST:
            for thr1 in THR1_LIST:
                chosen_frames: list[pd.DataFrame] = []
                choice_counts = {"base": 0, "d3_best": 0, "d1_best": 0}
                daily_choice_rows: list[dict] = []

                for trade_date, group in grouped:
                    ordered_base = group.sort_values("base_pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
                    base_top5_mean_rank3 = float(ordered_base.head(5)["rank3"].mean())
                    base_top5_mean_rank1 = float(ordered_base.head(5)["rank1"].mean())

                    chosen_model = "base"
                    score_series = group["base_pred_prob"].astype(float)
                    if trade_date >= start_date and base_top5_mean_rank3 < thr3:
                        chosen_model = "d3_best"
                        score_series = group["pred3d"].astype(float).where(
                            group["pred3d"].notna(),
                            group["base_pred_prob"].astype(float),
                        )
                    elif trade_date >= start_date and base_top5_mean_rank1 < thr1:
                        chosen_model = "d1_best"
                        score_series = group["pred1d"].astype(float).where(
                            group["pred1d"].notna(),
                            group["base_pred_prob"].astype(float),
                        )

                    choice_counts[chosen_model] += 1
                    chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
                    chosen["pred_prob"] = score_series
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
                selected_summary, selected_daily = evaluate(
                    candidate[["trade_date", "stock_code", LABEL_COL, "pred_prob"]].rename(columns={"pred_prob": "score"}),
                    "score",
                )
                selected_focus = focus_score(selected_summary)
                guard_pass = (
                    selected_summary["recent80"]["top3"] >= base_recent80_top3 - 0.0003
                    and selected_summary["recent80"]["top5"] >= base_recent80_top5 - 0.0003
                )

                row = {
                    "start_date": start_date,
                    "thr3": thr3,
                    "thr1": thr1,
                    "focus": selected_focus,
                    "delta_focus": selected_focus - base_focus,
                    "guard_pass": guard_pass,
                    "recent80_top1": selected_summary["recent80"]["top1"],
                    "recent80_top3": selected_summary["recent80"]["top3"],
                    "recent80_top5": selected_summary["recent80"]["top5"],
                    "recent80_top10": selected_summary["recent80"]["top10"],
                    "recent1y_top1": selected_summary["recent1y"]["top1"],
                    "recent1y_top3": selected_summary["recent1y"]["top3"],
                    "recent1y_top5": selected_summary["recent1y"]["top5"],
                    "full_top1": selected_summary["full"]["top1"],
                    "full_top3": selected_summary["full"]["top3"],
                    "full_top5": selected_summary["full"]["top5"],
                    "choice_counts": choice_counts,
                }
                rows.append(row)

                better = False
                if best_result is None:
                    better = True
                elif guard_pass and not best_result["row"]["guard_pass"]:
                    better = True
                elif guard_pass == best_result["row"]["guard_pass"]:
                    if selected_focus > best_result["row"]["focus"] + 1e-12:
                        better = True
                    elif abs(selected_focus - best_result["row"]["focus"]) <= 1e-12:
                        if row["recent80_top3"] > best_result["row"]["recent80_top3"] + 1e-12:
                            better = True
                        elif abs(row["recent80_top3"] - best_result["row"]["recent80_top3"]) <= 1e-12 and row["recent1y_top3"] > best_result["row"]["recent1y_top3"] + 1e-12:
                            better = True

                if better:
                    best_result = {
                        "row": row,
                        "candidate": candidate.copy(),
                        "daily_choice": daily_choice.copy(),
                        "summary": selected_summary,
                        "daily": selected_daily.copy(),
                    }

    if best_result is None:
        raise RuntimeError("No candidate evaluated")

    grid = pd.DataFrame(rows).sort_values(
        ["guard_pass", "focus", "recent80_top3", "recent1y_top3", "full_top3"],
        ascending=[False, False, False, False, False],
    )
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")
    best_result["daily_choice"].to_csv(OUT_DIR / "daily_choices.csv", index=False, encoding="utf-8-sig")

    db_summary = materialize(best_result["candidate"])
    best_result["daily"].to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    base_daily.to_csv(OUT_DIR / "base_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_summary, best_result["summary"]).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_result["daily"], "year").to_csv(OUT_DIR / "yearly_stability.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_result["daily"], "quarter").to_csv(OUT_DIR / "quarter_stability.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_5d_v4_daily_gate_d3best_d1best_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "aux1_table": AUX1_TABLE,
        "aux3_table": AUX3_TABLE,
        "current_table": CURRENT_TABLE,
        "search_space": {
            "start_dates": START_DATES,
            "thr3_list": THR3_LIST,
            "thr1_list": THR1_LIST,
        },
        "best_rule": best_result["row"],
        "base_focus": base_focus,
        "selected_focus": best_result["row"]["focus"],
        "selected_minus_base_focus": best_result["row"]["focus"] - base_focus,
        "base_windows": base_summary,
        "selected_windows": best_result["summary"],
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "rebuilds the daily-gate family on top of the current 5d research best",
            "upgrades the 3d and 1d sources to the current research best assets",
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
        "candidate_id": "model_agent_5d_v4_daily_gate_d3best_d1best_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": payload["generated_at"],
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "aux1_source": f"MODEL_PREDICTIONS.db::{AUX1_TABLE}",
        "aux3_source": f"MODEL_PREDICTIONS.db::{AUX3_TABLE}",
        "selected_rule": best_result["row"],
        "notes": "5D 研究候选 v4 以当前 5D 最优 research 资产为底座，按日门控少量切到当前 3D 最优或 1D 最优，主要用于验证强底座上是否仍有结构性增益。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = f"""# 5D v4 日门控研究候选

## 当前结论

- 基线 5D：`{BASE_TABLE}`
- 3D 辅助：`{AUX3_TABLE}`
- 1D 辅助：`{AUX1_TABLE}`
- 新 research 候选：`{RESEARCH_TABLE}`
- Focus：`{base_focus:.12f}` -> `{best_result['row']['focus']:.12f}`
- Focus 增量：`{best_result['row']['focus'] - base_focus:.12f}`

## 最优规则

- `start_date={best_result['row']['start_date']}`
- `thr3={best_result['row']['thr3']}`
- `thr1={best_result['row']['thr1']}`
- `choice_counts={json.dumps(best_result['row']['choice_counts'], ensure_ascii=False)}`

## 关键窗口

- recent80：`Top1={best_result['summary']['recent80']['top1']:.8f}` `Top3={best_result['summary']['recent80']['top3']:.8f}` `Top5={best_result['summary']['recent80']['top5']:.8f}`
- recent1y：`Top1={best_result['summary']['recent1y']['top1']:.8f}` `Top3={best_result['summary']['recent1y']['top3']:.8f}` `Top5={best_result['summary']['recent1y']['top5']:.8f}`
- full：`Top1={best_result['summary']['full']['top1']:.8f}` `Top3={best_result['summary']['full']['top3']:.8f}` `Top5={best_result['summary']['full']['top5']:.8f}`

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
