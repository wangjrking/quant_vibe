from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_v3_daygate_current1d_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_daygate_joint_refine_20260623_executable_10d_open_return_research"
AUX1_TABLE = "stock_predict_data_model_agent_1d_current10d_endgate_refine_20260623_executable_1d_open_return_research"
DAY_GATE_CSV = DATA_DIR / "reports" / "model_agent_10d_proxy5d_lite_daygate_std035_20260623" / "day_gate.csv"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_v3_daygate_current1d_20260623_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}

STARTS = ["20251201", "20260101", "20260115"]
STD_THRESHOLDS = [0.015, 0.02, 0.025]
ZONES = [0.00125, 0.0015, 0.00175]
ALPHAS = [0.01, 0.012, 0.015]


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


def compare_windows(base_windows: dict, selected_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"base_{metric}"] = base_windows[name][metric]
            row[f"selected_{metric}"] = selected_windows[name][metric]
            row[f"selected_minus_base_{metric}"] = selected_windows[name][metric] - base_windows[name][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def materialize(frame: pd.DataFrame, score: pd.Series) -> dict:
    out = frame[["trade_date", "stock_code", LABEL_COL, "base_pred_prob", "rank_base", "rank1_cur", "std_r10_top5_base"]].copy()
    out["pred_prob"] = score.astype(float)
    out = out[["trade_date", "stock_code", LABEL_COL, "pred_prob", "base_pred_prob", "rank_base", "rank1_cur", "std_r10_top5_base"]]
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
            f"select trade_date, stock_code, pred_prob as pred1_cur "
            f"from '{AUX1_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )

    for df in (base, aux1):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(aux1, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame["rank_base"] = frame.groupby("trade_date")["base_pred_prob"].rank(method="average", pct=True)
    frame["rank1_cur"] = frame.groupby("trade_date")["pred1_cur"].rank(method="average", pct=True)
    frame["rank1_cur"] = frame["rank1_cur"].fillna(frame["rank_base"])

    day_gate = pd.read_csv(DAY_GATE_CSV)
    day_gate["trade_date"] = day_gate["trade_date"].astype(str)
    frame = frame.merge(day_gate[["trade_date", "std_r10_top5_base"]], on="trade_date", how="left")

    base_summary, base_daily = evaluate(frame.assign(score=frame["base_pred_prob"]), "score")
    base_focus = focus_score(base_summary)
    base_recent63_top3 = base_summary["recent63"]["top3"]
    base_recent63_top5 = base_summary["recent63"]["top5"]

    rows: list[dict] = []
    best_rule: dict | None = None
    best_focus = float("-inf")
    best_summary: dict | None = None
    best_daily: pd.DataFrame | None = None
    best_score: pd.Series | None = None
    best_passed = False

    for start in STARTS:
        for threshold in STD_THRESHOLDS:
            for zone in ZONES:
                for alpha in ALPHAS:
                    active_mask = (
                        (frame["trade_date"] >= start)
                        & (frame["rank_base"] >= 1.0 - zone)
                        & (frame["std_r10_top5_base"] < threshold)
                    )
                    score = frame["rank_base"].copy()
                    score.loc[active_mask] = frame.loc[active_mask, "rank_base"] + alpha * frame.loc[active_mask, "rank1_cur"]
                    summary, daily_eval = evaluate(frame.assign(score=score), "score")
                    focus = focus_score(summary)
                    pass_guard = (
                        summary["recent63"]["top3"] >= base_recent63_top3 - 0.0001
                        and summary["recent63"]["top5"] >= base_recent63_top5 - 0.0001
                    )
                    row = {
                        "start": start,
                        "std_threshold": threshold,
                        "zone": zone,
                        "alpha": alpha,
                        "active_rows": int(active_mask.sum()),
                        "active_days": int(frame.loc[active_mask, "trade_date"].nunique()),
                        "focus": focus,
                        "delta_focus": focus - base_focus,
                        "guard_pass": pass_guard,
                        "full_top1": summary["full"]["top1"],
                        "full_top3": summary["full"]["top3"],
                        "full_top5": summary["full"]["top5"],
                        "full_top10": summary["full"]["top10"],
                        "recent126_top1": summary["recent126"]["top1"],
                        "recent126_top3": summary["recent126"]["top3"],
                        "recent126_top5": summary["recent126"]["top5"],
                        "recent126_top10": summary["recent126"]["top10"],
                        "recent63_top1": summary["recent63"]["top1"],
                        "recent63_top3": summary["recent63"]["top3"],
                        "recent63_top5": summary["recent63"]["top5"],
                        "recent63_top10": summary["recent63"]["top10"],
                    }
                    rows.append(row)

                    better = False
                    if pass_guard and not best_passed:
                        better = True
                    elif pass_guard == best_passed:
                        if focus > best_focus + 1e-15:
                            better = True
                        elif abs(focus - best_focus) <= 1e-15 and best_rule is not None:
                            if row["recent63_top1"] > best_rule["recent63_top1"] + 1e-15:
                                better = True
                            elif abs(row["recent63_top1"] - best_rule["recent63_top1"]) <= 1e-15 and row["active_days"] < best_rule["active_days"]:
                                better = True

                    if better:
                        best_rule = row
                        best_focus = focus
                        best_summary = summary
                        best_daily = daily_eval.copy()
                        best_score = score.copy()
                        best_passed = pass_guard

    if best_rule is None or best_summary is None or best_daily is None or best_score is None:
        raise RuntimeError("No candidate rule evaluated")

    result_df = pd.DataFrame(rows).sort_values(
        ["guard_pass", "focus", "recent63_top1", "recent126_top1", "full_top1"],
        ascending=[False, False, False, False, False],
    )
    result_df.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")

    db_summary = materialize(frame, best_score)
    best_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    base_daily.to_csv(OUT_DIR / "baseline_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_summary, best_summary).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_v3_daygate_current1d_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "aux1_table": AUX1_TABLE,
        "search_space": {
            "starts": STARTS,
            "std_thresholds": STD_THRESHOLDS,
            "zones": ZONES,
            "alphas": ALPHAS,
        },
        "base_focus": base_focus,
        "selected_focus": best_focus,
        "focus_delta": best_focus - base_focus,
        "best_rule": best_rule,
        "guard_rule": {
            "recent63_top3_min": base_recent63_top3 - 0.0001,
            "recent63_top5_min": base_recent63_top5 - 0.0001,
            "selected_passed": best_passed,
        },
        "base_windows": base_summary,
        "selected_windows": best_summary,
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "retests the current 10d head rerank using the latest 1d current10d research source",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# 10D v3 current1d 研究候选",
        "",
        f"- 基座表：`{BASE_TABLE}`",
        f"- 辅助 1D：`{AUX1_TABLE}`",
        f"- 最优规则：`start={best_rule['start']}`、`std_threshold={best_rule['std_threshold']}`、`zone={best_rule['zone']}`、`alpha={best_rule['alpha']}`",
        f"- Focus：`{best_focus:.12f}`，相对基座变化：`{best_focus - base_focus:+.12f}`",
        "- 仅研究用途，不进入 formal。",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
