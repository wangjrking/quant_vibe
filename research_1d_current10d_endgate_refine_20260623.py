from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_current10d_endgate_refine_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research"
DAYGATE_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_20260623_executable_1d_open_return_research"
CURRENT1D_TABLE = "stock_predict_data_model_agent_1d_proxy5d_topzone_10dlite_20260623_executable_1d_open_return_research"
SOURCE10D_TABLE = "stock_predict_data_model_agent_10d_daygate_joint_refine_20260623_executable_10d_open_return_research"
CURRENT_RETAINED_TABLE = "stock_predict_data_model_agent_1d_endgate_threshold_refine_20260623_executable_1d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_1d_current10d_endgate_refine_20260623_executable_1d_open_return_research"
DAYGATE_FLAG_CSV = DATA_DIR / "reports" / "model_agent_1d_gate092_new5d040_daygate_20260623" / "daily_gate_flags.csv"

LABEL_COL = "executable_1d_open_return"
CUR_TOP1_GAP_THRESHOLD = 0.000795
NEWCOMER_THRESHOLDS = [0.72, 0.724, 0.728, 0.729546, 0.732]
BEST_TOP1_GAP_MAX_LIST = [0.00065, 0.000795, 0.0009, 0.0011]
END_DATES = ["20251231", "20260131", "20260228"]
WINDOWS = {
    "full": ("20240604", "20260611"),
    "recent126": ("20251201", "20260611"),
    "recent63": ("20260304", "20260611"),
    "q1_2025": ("20250101", "20250331"),
    "q2_2025": ("20250401", "20250630"),
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


def weak_window_score(summary: dict) -> float:
    return (
        summary["q1_2025"]["top1"] * 2
        + summary["q1_2025"]["top5"] * 3
        + summary["q2_2025"]["top1"] * 2
        + summary["q2_2025"]["top5"] * 3
    )


def compare_windows(current_windows: dict, selected_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63", "q1_2025", "q2_2025"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"current_{metric}"] = current_windows[name][metric]
            row[f"new_{metric}"] = selected_windows[name][metric]
            row[f"new_minus_current_{metric}"] = selected_windows[name][metric] - current_windows[name][metric]
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
            top20=("top20", "mean"),
        )
        .reset_index()
    )


def materialize(final: pd.DataFrame) -> dict:
    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute(f"drop table if exists '{RESEARCH_TABLE}'")
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
        labels = pd.read_sql_query(
            f"select trade_date, stock_code, [{LABEL_COL}] as {LABEL_COL} from '{LABEL_TABLE}' where [{LABEL_COL}] is not null",
            conn,
        )
        daygate = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as daygate_pred_prob from '{DAYGATE_TABLE}'",
            conn,
        )
        current1d = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred_current from '{CURRENT1D_TABLE}'",
            conn,
        )
        source10d = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred10new from '{SOURCE10D_TABLE}'",
            conn,
        )
        current_retained = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred_current_retained from '{CURRENT_RETAINED_TABLE}'",
            conn,
        )

    for df in (labels, daygate, current1d, source10d, current_retained):
        df["trade_date"] = df["trade_date"].astype(str)

    daygate_flags = pd.read_csv(DAYGATE_FLAG_CSV, dtype={"trade_date": str})

    proxy_frame = labels.merge(source10d, on=["trade_date", "stock_code"], how="inner")
    proxy_frame = proxy_frame.merge(current1d, on=["trade_date", "stock_code"], how="left")

    default_frames: list[pd.DataFrame] = []
    default_choice_rows: list[dict] = []
    for trade_date, group in proxy_frame.groupby("trade_date", sort=True):
        ordered_current = group.sort_values("pred_current", ascending=False, kind="mergesort").reset_index(drop=True)
        cur_top1_gap_from_current1d = float(ordered_current.iloc[0]["pred_current"] - ordered_current.iloc[4]["pred_current"])
        use_current1d = cur_top1_gap_from_current1d > CUR_TOP1_GAP_THRESHOLD
        score = group["pred_current"].astype(float) if use_current1d else group["pred10new"].astype(float)

        ordered_best = group.assign(best_pred_prob=score).sort_values("best_pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
        best_top1_gap = float(ordered_best.iloc[0]["best_pred_prob"] - ordered_best.iloc[4]["best_pred_prob"])

        chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
        chosen["best_pred_prob"] = score
        default_frames.append(chosen)
        default_choice_rows.append(
            {
                "trade_date": trade_date,
                "default_source": "current_1d" if use_current1d else "source_10dnew",
                "cur_top1_gap_from_current1d": cur_top1_gap_from_current1d,
                "best_top1_gap": best_top1_gap,
            }
        )

    default_best = pd.concat(default_frames, ignore_index=True)
    default_choice_df = pd.DataFrame(default_choice_rows)
    default_choice_df.to_csv(OUT_DIR / "default_best_daily_choices.csv", index=False, encoding="utf-8-sig")

    current_retained_eval = labels.merge(
        current_retained,
        on=["trade_date", "stock_code"],
        how="inner",
    )
    current_eval, current_daily = evaluate(
        current_retained_eval.rename(columns={"pred_current_retained": "score"}),
        "score",
    )
    current_focus = focus_score(current_eval)
    current_weak = weak_window_score(current_eval)

    frame = labels.merge(default_best[["trade_date", "stock_code", "best_pred_prob"]], on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daygate, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daygate_flags, on="trade_date", how="left")
    frame = frame.merge(default_choice_df[["trade_date", "best_top1_gap"]], on="trade_date", how="left")

    rows: list[dict] = []
    best_result: dict | None = None

    for newcomer_threshold in NEWCOMER_THRESHOLDS:
        for best_top1_gap_max in BEST_TOP1_GAP_MAX_LIST:
            for end_date in END_DATES:
                chosen_frames: list[pd.DataFrame] = []
                daily_choice_rows: list[dict] = []
                for trade_date, group in frame.groupby("trade_date", sort=True):
                    newcomer_mean_5d_pct = float(group["newcomer_mean_5d_pct"].iloc[0])
                    best_top1_gap = float(group["best_top1_gap"].iloc[0])
                    use_daygate = (
                        newcomer_mean_5d_pct >= newcomer_threshold
                        and best_top1_gap <= best_top1_gap_max
                        and trade_date <= end_date
                    )
                    chosen_model = "daygate_1d" if use_daygate else "default_best_1d"
                    score = group["daygate_pred_prob"].astype(float) if use_daygate else group["best_pred_prob"].astype(float)

                    chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
                    chosen["pred_prob"] = score
                    chosen_frames.append(chosen)
                    daily_choice_rows.append(
                        {
                            "trade_date": trade_date,
                            "chosen_model": chosen_model,
                            "newcomer_mean_5d_pct": newcomer_mean_5d_pct,
                            "best_top1_gap": best_top1_gap,
                            "newcomer_count": int(group["newcomer_count"].iloc[0]),
                            "used_source_gate": bool(group["used_source_gate"].iloc[0]),
                            "newcomer_threshold": newcomer_threshold,
                            "best_top1_gap_max": best_top1_gap_max,
                            "end_date_gate": end_date,
                        }
                    )

                candidate = pd.concat(chosen_frames, ignore_index=True)
                daily_choice = pd.DataFrame(daily_choice_rows)
                selected_eval, selected_daily = evaluate(candidate.rename(columns={"pred_prob": "score"}), "score")
                selected_focus = focus_score(selected_eval)
                selected_weak = weak_window_score(selected_eval)

                guard_pass = (
                    selected_weak >= current_weak
                    and selected_eval["recent63"]["top3"] >= current_eval["recent63"]["top3"]
                    and selected_eval["recent63"]["top5"] >= current_eval["recent63"]["top5"]
                )

                row = {
                    "newcomer_threshold": newcomer_threshold,
                    "best_top1_gap_max": best_top1_gap_max,
                    "end_date": end_date,
                    "focus": selected_focus,
                    "weak_window_score": selected_weak,
                    "switch_days": int((daily_choice["chosen_model"] == "daygate_1d").sum()),
                    "guard_pass": guard_pass,
                    "full_top1": selected_eval["full"]["top1"],
                    "full_top3": selected_eval["full"]["top3"],
                    "full_top5": selected_eval["full"]["top5"],
                    "recent126_top1": selected_eval["recent126"]["top1"],
                    "recent126_top3": selected_eval["recent126"]["top3"],
                    "recent126_top5": selected_eval["recent126"]["top5"],
                    "recent63_top1": selected_eval["recent63"]["top1"],
                    "recent63_top3": selected_eval["recent63"]["top3"],
                    "recent63_top5": selected_eval["recent63"]["top5"],
                    "q2_2025_top1": selected_eval["q2_2025"]["top1"],
                    "q2_2025_top5": selected_eval["q2_2025"]["top5"],
                }
                rows.append(row)
                switch_days = row["switch_days"]

                better = False
                if best_result is None:
                    better = True
                elif guard_pass and not best_result["guard_pass"]:
                    better = True
                elif guard_pass == best_result["guard_pass"]:
                    if selected_focus > best_result["focus"] + 1e-12:
                        better = True
                    elif abs(selected_focus - best_result["focus"]) <= 1e-12 and selected_weak > best_result["weak_window_score"] + 1e-12:
                        better = True
                    elif (
                        abs(selected_focus - best_result["focus"]) <= 1e-12
                        and abs(selected_weak - best_result["weak_window_score"]) <= 1e-12
                        and switch_days < best_result["row"]["switch_days"]
                    ):
                        better = True

                if better:
                    best_result = {
                        "row": row,
                        "candidate": candidate.copy(),
                        "daily_choice": daily_choice.copy(),
                        "selected_eval": selected_eval,
                        "selected_daily": selected_daily.copy(),
                        "focus": selected_focus,
                        "weak_window_score": selected_weak,
                        "guard_pass": guard_pass,
                    }

    if best_result is None:
        raise RuntimeError("No candidate evaluated")

    grid = pd.DataFrame(rows).sort_values(
        ["guard_pass", "focus", "weak_window_score", "full_top1"],
        ascending=[False, False, False, False],
    )
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")
    best_result["daily_choice"].to_csv(OUT_DIR / "daily_choices.csv", index=False, encoding="utf-8-sig")

    final = best_result["candidate"][["trade_date", "stock_code", LABEL_COL, "pred_prob"]]
    db_summary = materialize(final)

    best_result["selected_daily"].to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(current_eval, best_result["selected_eval"]).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_result["selected_daily"], "year").to_csv(OUT_DIR / "yearly_stability.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_result["selected_daily"], "quarter").to_csv(OUT_DIR / "quarter_stability.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_1d_current10d_endgate_refine_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "daygate_table": DAYGATE_TABLE,
        "current1d_table": CURRENT1D_TABLE,
        "source10d_table": SOURCE10D_TABLE,
        "current_retained_table": CURRENT_RETAINED_TABLE,
        "default_best_rule": {
            "type": "current1d_fallback_on_10d_current_best",
            "if_cur_top1_gap_from_current1d_gt": CUR_TOP1_GAP_THRESHOLD,
            "choose_model": "current_1d",
            "default": "source_10dnew",
        },
        "search_space": {
            "newcomer_thresholds": NEWCOMER_THRESHOLDS,
            "best_top1_gap_max_list": BEST_TOP1_GAP_MAX_LIST,
            "end_dates": END_DATES,
        },
        "current_focus": current_focus,
        "current_weak_window_score": current_weak,
        "best_rule": best_result["row"],
        "selected_focus": best_result["focus"],
        "selected_weak_window_score": best_result["weak_window_score"],
        "selected_minus_current_focus": best_result["focus"] - current_focus,
        "selected_minus_current_weak_window_score": best_result["weak_window_score"] - current_weak,
        "current_windows": current_eval,
        "selected_windows": best_result["selected_eval"],
        "db_summary": db_summary,
        "guard_rule": {
            "weak_window_score_min": current_weak,
            "recent63_top3_min": current_eval["recent63"]["top3"],
            "recent63_top5_min": current_eval["recent63"]["top5"],
            "selected_passed": bool(best_result["guard_pass"]),
        },
        "governance_notes": [
            "research only",
            "default source is a current-1d fallback over current 10d best",
            "daygate repair is still used to preserve weak-window behavior",
            "selection requires weak-window score and recent63 top3/top5 to stay at or above the current retained best",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 1D current10d endgate refine 研究

## 当前结论

- 这轮研究把 `1D retained best` 的默认源改成“`10D current-best` 为主，`current_1d` 兜底”的结构。
- 同时保留 `daygate` 作为弱窗口修复手段。
- 最优规则：
  - `newcomer_mean_5d_pct >= {best_result['row']['newcomer_threshold']}`
  - `best_top1_gap <= {best_result['row']['best_top1_gap_max']}`
  - `trade_date <= {best_result['row']['end_date']}`

## 指标变化

- `focus`：`{current_focus:.12f}` -> `{best_result['focus']:.12f}`
- `weak_window_score`：`{current_weak:.12f}` -> `{best_result['weak_window_score']:.12f}`
- `recent63 Top3`：`{current_eval['recent63']['top3']:.12f}` -> `{best_result['selected_eval']['recent63']['top3']:.12f}`
- `recent63 Top5`：`{current_eval['recent63']['top5']:.12f}` -> `{best_result['selected_eval']['recent63']['top5']:.12f}`

## 说明

- 这版的关键点不是放弃弱窗口修复，而是先把默认大头来源升级到当前 `10D best`，再用 `daygate` 把 `Q1/Q2 2025` 修复拉回来。
- 本次只生成 research 资产，不进入 formal L4/L5。
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
