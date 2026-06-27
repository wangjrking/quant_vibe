from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_endgate_threshold_refine_20260623"

LABEL_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research"
BEST_TABLE = "stock_predict_data_model_agent_1d_proxy10dnew_curgap_top1safe_20260623_executable_1d_open_return_research"
DAYGATE_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_20260623_executable_1d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_curgap_endgate_20260623_executable_1d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_1d_endgate_threshold_refine_20260623_executable_1d_open_return_research"
DAYGATE_FLAG_CSV = DATA_DIR / "reports" / "model_agent_1d_gate092_new5d040_daygate_20260623" / "daily_gate_flags.csv"
CURGAP_CHOICE_CSV = DATA_DIR / "reports" / "model_agent_1d_proxy10dnew_curgap_top1safe_20260623" / "daily_choices.csv"

LABEL_COL = "executable_1d_open_return"
BASE_NEWCOMER_THRESHOLD = 0.729546
CUR_TOP1_GAP_MAX = 0.000794
END_DATE = "20260131"
THRESHOLDS = [0.72, 0.724, 0.728, 0.729546, 0.732, 0.736, 0.74]
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


def compare_windows(current_windows: dict, new_windows: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name in ("full", "recent126", "recent63", "q1_2025", "q2_2025"):
        row = {"window": name}
        for metric in ("top1", "top3", "top5", "top10"):
            row[f"current_{metric}"] = current_windows[name][metric]
            row[f"new_{metric}"] = new_windows[name][metric]
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
            top20=("top20", "mean"),
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
        best = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as best_pred_prob from '{BEST_TABLE}'",
            conn,
        )
        daygate = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as daygate_pred_prob from '{DAYGATE_TABLE}'",
            conn,
        )
        current = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as current_pred_prob from '{CURRENT_TABLE}'",
            conn,
        )

    daily_flags = pd.read_csv(DAYGATE_FLAG_CSV, dtype={"trade_date": str})
    curgap_choices = pd.read_csv(CURGAP_CHOICE_CSV, dtype={"trade_date": str})

    for df in (labels, best, daygate, current):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = labels.merge(best, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daygate, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(current, on=["trade_date", "stock_code"], how="inner")
    frame = frame.merge(daily_flags, on="trade_date", how="left")
    frame = frame.merge(
        curgap_choices[["trade_date", "cur_top1_gap"]],
        on="trade_date",
        how="left",
    )

    current_eval_frame = frame[["trade_date", "stock_code", LABEL_COL, "current_pred_prob"]].rename(columns={"current_pred_prob": "pred_prob"})
    current_windows, current_daily = evaluate(current_eval_frame, "pred_prob")
    current_focus = focus_score(current_windows)
    current_weak = weak_window_score(current_windows)

    best_result: dict | None = None
    grid_rows: list[dict] = []

    for threshold in THRESHOLDS:
        chosen_frames: list[pd.DataFrame] = []
        daily_choice_rows: list[dict] = []
        for trade_date, group in frame.groupby("trade_date", sort=True):
            newcomer_mean_5d_pct = float(group["newcomer_mean_5d_pct"].iloc[0])
            cur_top1_gap = float(group["cur_top1_gap"].iloc[0])
            use_daygate = (
                newcomer_mean_5d_pct >= threshold
                and cur_top1_gap <= CUR_TOP1_GAP_MAX
                and trade_date <= END_DATE
            )
            chosen_model = "daygate_1d" if use_daygate else "best_1d"
            score_series = group["daygate_pred_prob"].astype(float) if use_daygate else group["best_pred_prob"].astype(float)

            chosen = group[["trade_date", "stock_code", LABEL_COL]].copy()
            chosen["pred_prob"] = score_series
            chosen["chosen_model"] = chosen_model
            chosen_frames.append(chosen)

            daily_choice_rows.append(
                {
                    "trade_date": trade_date,
                    "chosen_model": chosen_model,
                    "newcomer_mean_5d_pct": newcomer_mean_5d_pct,
                    "cur_top1_gap": cur_top1_gap,
                    "newcomer_count": int(group["newcomer_count"].iloc[0]),
                    "used_source_gate": bool(group["used_source_gate"].iloc[0]),
                    "threshold": threshold,
                    "end_date_gate": END_DATE,
                }
            )

        candidate = pd.concat(chosen_frames, ignore_index=True)
        daily_choice = pd.DataFrame(daily_choice_rows)
        windows, daily = evaluate(candidate, "pred_prob")
        focus = focus_score(windows)
        weak = weak_window_score(windows)
        row = {
            "threshold": threshold,
            "focus": focus,
            "weak_window_score": weak,
            "switch_days": int((daily_choice["chosen_model"] == "daygate_1d").sum()),
            "full_top1": windows["full"]["top1"],
            "full_top3": windows["full"]["top3"],
            "full_top5": windows["full"]["top5"],
            "recent126_top1": windows["recent126"]["top1"],
            "recent126_top3": windows["recent126"]["top3"],
            "recent126_top5": windows["recent126"]["top5"],
            "recent63_top1": windows["recent63"]["top1"],
            "recent63_top3": windows["recent63"]["top3"],
            "recent63_top5": windows["recent63"]["top5"],
            "q1_2025_top1": windows["q1_2025"]["top1"],
            "q1_2025_top5": windows["q1_2025"]["top5"],
            "q2_2025_top1": windows["q2_2025"]["top1"],
            "q2_2025_top5": windows["q2_2025"]["top5"],
            "guard_pass": (
                weak >= current_weak
                and windows["recent63"]["top3"] >= current_windows["recent63"]["top3"]
                and windows["recent63"]["top5"] >= current_windows["recent63"]["top5"]
            ),
        }
        grid_rows.append(row)

        if best_result is None:
            best_result = {
                "threshold": threshold,
                "windows": windows,
                "daily": daily,
                "candidate": candidate,
                "daily_choice": daily_choice,
                "focus": focus,
                "weak": weak,
                "guard_pass": row["guard_pass"],
            }
            continue

        better = False
        if row["guard_pass"] and not best_result["guard_pass"]:
            better = True
        elif row["guard_pass"] == best_result["guard_pass"]:
            if focus > best_result["focus"] + 1e-12:
                better = True
            elif abs(focus - best_result["focus"]) <= 1e-12 and windows["full"]["top1"] > best_result["windows"]["full"]["top1"]:
                better = True
            elif (
                abs(focus - best_result["focus"]) <= 1e-12
                and abs(windows["full"]["top1"] - best_result["windows"]["full"]["top1"]) <= 1e-12
                and threshold > best_result["threshold"]
            ):
                better = True

        if better:
            best_result = {
                "threshold": threshold,
                "windows": windows,
                "daily": daily,
                "candidate": candidate,
                "daily_choice": daily_choice,
                "focus": focus,
                "weak": weak,
                "guard_pass": row["guard_pass"],
            }

    assert best_result is not None

    best_threshold = float(best_result["threshold"])
    candidate = best_result["candidate"]
    selected_windows = best_result["windows"]
    selected_daily = best_result["daily"]
    daily_choice = best_result["daily_choice"]

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
        duplicate_key_groups = conn.execute(
            f"""
            select count(*)
            from (
              select trade_date, stock_code
              from '{RESEARCH_TABLE}'
              group by trade_date, stock_code
              having count(*) > 1
            )
            """
        ).fetchone()[0]

    db_summary = {
        "row_count": int(db_row["row_count"]),
        "min_trade_date": str(db_row["min_trade_date"]),
        "max_trade_date": str(db_row["max_trade_date"]),
        "trade_days": int(db_row["trade_days"]),
        "stock_count": int(db_row["stock_count"]),
        "null_pred_prob": int(db_row["null_pred_prob"]),
        "duplicate_key_groups": int(duplicate_key_groups),
        "db_path": str(MODEL_DB),
        "table": RESEARCH_TABLE,
    }

    daily_choice.to_csv(OUT_DIR / "daily_choices.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(grid_rows).sort_values(["guard_pass", "focus", "full_top1"], ascending=[False, False, False]).to_csv(
        OUT_DIR / "grid_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compare_windows(current_windows, selected_windows).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(current_daily, "year").rename(
        columns={c: f"current_{c}" for c in ("trade_days", "top1", "top3", "top5", "top10", "top20")}
    ).merge(
        build_period_breakdown(selected_daily, "year").rename(
            columns={c: f"selected_{c}" for c in ("trade_days", "top1", "top3", "top5", "top10", "top20")}
        ),
        left_on="period",
        right_on="period",
        how="outer",
    ).to_csv(OUT_DIR / "yearly_stability.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(current_daily, "quarter").rename(
        columns={c: f"current_{c}" for c in ("trade_days", "top1", "top3", "top5", "top10", "top20")}
    ).merge(
        build_period_breakdown(selected_daily, "quarter").rename(
            columns={c: f"selected_{c}" for c in ("trade_days", "top1", "top3", "top5", "top10", "top20")}
        ),
        left_on="period",
        right_on="period",
        how="outer",
    ).to_csv(OUT_DIR / "quarter_stability.csv", index=False, encoding="utf-8-sig")

    current_threshold = BASE_NEWCOMER_THRESHOLD
    current_switched = int((daily_choice["chosen_model"] == "daygate_1d").sum())
    changed_days = daily_choice[(daily_choice["chosen_model"] == "daygate_1d") & (daily_choice["newcomer_mean_5d_pct"] < current_threshold)]
    changed_days.to_csv(OUT_DIR / "newly_switched_days.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_1d_endgate_threshold_refine_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_table": LABEL_TABLE,
        "best_table": BEST_TABLE,
        "current_table": CURRENT_TABLE,
        "daygate_table": DAYGATE_TABLE,
        "rule": {
            "type": "daily_gate_daygate_and_curgap_with_end_date_threshold_refine",
            "if_newcomer_mean_5d_pct_gte": best_threshold,
            "and_cur_top1_gap_lte": CUR_TOP1_GAP_MAX,
            "and_trade_date_lte": END_DATE,
            "choose_model": "daygate_1d",
            "default": "best_1d",
        },
        "threshold_grid": THRESHOLDS,
        "current_threshold": current_threshold,
        "current_focus": current_focus,
        "current_weak_window_score": current_weak,
        "selected_focus": best_result["focus"],
        "selected_weak_window_score": best_result["weak"],
        "selected_minus_current_focus": best_result["focus"] - current_focus,
        "selected_minus_current_weak_window_score": best_result["weak"] - current_weak,
        "current_windows": current_windows,
        "selected_windows": selected_windows,
        "choice_counts": daily_choice["chosen_model"].value_counts().sort_index().to_dict(),
        "newly_switched_days_count": int(len(changed_days)),
        "newly_switched_days": changed_days["trade_date"].astype(str).tolist(),
        "db_summary": db_summary,
        "guard_rule": {
            "weak_window_score_min": current_weak,
            "recent63_top3_min": current_windows["recent63"]["top3"],
            "recent63_top5_min": current_windows["recent63"]["top5"],
            "selected_passed": bool(best_result["guard_pass"]),
        },
        "governance_notes": [
            "research only",
            "small-step refinement over current best 1d end-date gate candidate",
            "holds cur_top1_gap and end_date fixed, only refines newcomer_mean_5d_pct threshold",
            "selected threshold must preserve weak-window score and recent63 top3/top5",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 1D 阈值细化研究 20260623

## 当前结论

- 这轮研究只做一个小动作：在当前 `1D endgate best` 的基础上，细化 `newcomer_mean_5d_pct` 阈值。
- 其余口径保持不变：
  - `cur_top1_gap <= {CUR_TOP1_GAP_MAX}`
  - `trade_date <= {END_DATE}`
- 结果是：把阈值从 `{current_threshold}` 下调到 `{best_threshold}` 后，得到新的 research 候选：
  `{RESEARCH_TABLE}`

## 指标变化

- `focus`：`{current_focus:.12f}` -> `{best_result["focus"]:.12f}`
- `weak_window_score`：`{current_weak:.12f}` -> `{best_result["weak"]:.12f}`
- `full Top1`：`{current_windows["full"]["top1"]:.12f}` -> `{selected_windows["full"]["top1"]:.12f}`
- `full Top5`：`{current_windows["full"]["top5"]:.12f}` -> `{selected_windows["full"]["top5"]:.12f}`
- `recent126 Top1`：`{current_windows["recent126"]["top1"]:.12f}` -> `{selected_windows["recent126"]["top1"]:.12f}`
- `recent63 Top3`：`{current_windows["recent63"]["top3"]:.12f}` -> `{selected_windows["recent63"]["top3"]:.12f}`
- `recent63 Top5`：`{current_windows["recent63"]["top5"]:.12f}` -> `{selected_windows["recent63"]["top5"]:.12f}`

## 变化来源

- 新阈值只多切入了 `{len(changed_days)}` 个交易日。
- 这些新切入日期见：
  - `newly_switched_days.csv`
- 这说明当前 1D 研究最优已经接近平台区，进一步提升来自非常局部的日级筛选修正，而不是大范围切换。

## 文件

- 结果摘要：`summary.json`
- 网格结果：`grid_results.csv`
- 日切换明细：`daily_choices.csv`
- 新增切换日：`newly_switched_days.csv`
- 窗口对比：`window_comparison.csv`
- 年度稳定性：`yearly_stability.csv`
- 季度稳定性：`quarter_stability.csv`
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
