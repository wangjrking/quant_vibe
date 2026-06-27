from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_v4_proxy_switch_by_std_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_daygate_joint_refine_20260623_executable_10d_open_return_research"
PROXY_TABLE = "stock_predict_data_model_agent_10d_proxy5dbest_lite_daygate_std035_20260623_executable_10d_open_return_research"
DAY_GATE_CSV = DATA_DIR / "reports" / "model_agent_10d_proxy5d_lite_daygate_std035_20260623" / "day_gate.csv"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_v4_proxy_switch_by_std_20260623_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}

STARTS = ["20251118", "20251201", "20260101", "20260201"]
STD_LOS = [0.0, 0.02, 0.05, 0.08, 0.1]
STD_HIS = [0.08, 0.12, 0.18, 0.25, 0.35]


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


def materialize(frame: pd.DataFrame) -> dict:
    out = frame[["trade_date", "stock_code", LABEL_COL, "pred_prob", "base_pred_prob", "proxy_pred_prob", "std_r10_top5_base", "use_lite_gate", "selected_model"]].copy()
    with sqlite3.connect(MODEL_DB) as conn:
        out.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
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
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} from '{BASE_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        proxy = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as proxy_pred_prob from '{PROXY_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )

    base["trade_date"] = base["trade_date"].astype(str)
    proxy["trade_date"] = proxy["trade_date"].astype(str)
    frame = base.merge(proxy, on=["trade_date", "stock_code"], how="left", validate="one_to_one")

    gate = pd.read_csv(DAY_GATE_CSV)
    gate["trade_date"] = gate["trade_date"].astype(str)
    frame = frame.merge(gate[["trade_date", "std_r10_top5_base", "use_lite_gate"]], on="trade_date", how="left")

    base_summary, base_daily = evaluate(frame.rename(columns={"base_pred_prob": "score"}), "score")
    base_focus = focus_score(base_summary)
    base_recent63_top3 = base_summary["recent63"]["top3"]
    base_recent63_top5 = base_summary["recent63"]["top5"]

    rows: list[dict] = []
    best_rule: dict | None = None
    best_focus = float("-inf")
    best_summary: dict | None = None
    best_daily: pd.DataFrame | None = None
    best_frame: pd.DataFrame | None = None
    best_passed = False

    for start in STARTS:
        for std_lo in STD_LOS:
            for std_hi in STD_HIS:
                if std_hi <= std_lo:
                    continue
                switch_day_mask = (
                    (frame["trade_date"] >= start)
                    & (frame["use_lite_gate"].fillna(False).astype(bool))
                    & (frame["std_r10_top5_base"] >= std_lo)
                    & (frame["std_r10_top5_base"] < std_hi)
                )
                candidate = frame.copy()
                candidate["pred_prob"] = candidate["base_pred_prob"]
                candidate["selected_model"] = "base"
                candidate.loc[switch_day_mask, "pred_prob"] = candidate.loc[switch_day_mask, "proxy_pred_prob"].where(
                    candidate.loc[switch_day_mask, "proxy_pred_prob"].notna(),
                    candidate.loc[switch_day_mask, "base_pred_prob"],
                )
                candidate.loc[switch_day_mask, "selected_model"] = "proxy"

                summary, daily_eval = evaluate(candidate.rename(columns={"pred_prob": "score"}), "score")
                focus = focus_score(summary)
                pass_guard = (
                    summary["recent63"]["top3"] >= base_recent63_top3 - 0.0001
                    and summary["recent63"]["top5"] >= base_recent63_top5 - 0.0001
                )
                row = {
                    "start": start,
                    "std_lo": std_lo,
                    "std_hi": std_hi,
                    "switch_days": int(candidate.loc[candidate["selected_model"] == "proxy", "trade_date"].nunique()),
                    "focus": focus,
                    "delta_focus": focus - base_focus,
                    "guard_pass": pass_guard,
                    "full_top1": summary["full"]["top1"],
                    "full_top3": summary["full"]["top3"],
                    "full_top5": summary["full"]["top5"],
                    "recent126_top1": summary["recent126"]["top1"],
                    "recent126_top3": summary["recent126"]["top3"],
                    "recent126_top5": summary["recent126"]["top5"],
                    "recent63_top1": summary["recent63"]["top1"],
                    "recent63_top3": summary["recent63"]["top3"],
                    "recent63_top5": summary["recent63"]["top5"],
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
                        elif abs(row["recent63_top1"] - best_rule["recent63_top1"]) <= 1e-15 and row["recent126_top3"] > best_rule["recent126_top3"] + 1e-15:
                            better = True

                if better:
                    best_rule = row
                    best_focus = focus
                    best_summary = summary
                    best_daily = daily_eval.copy()
                    best_frame = candidate.copy()
                    best_passed = pass_guard

    if best_rule is None or best_summary is None or best_daily is None or best_frame is None:
        raise RuntimeError("No candidate evaluated")

    grid = pd.DataFrame(rows).sort_values(
        ["guard_pass", "focus", "recent63_top1", "recent126_top3", "full_top1"],
        ascending=[False, False, False, False, False],
    )
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")

    db_summary = materialize(best_frame)
    best_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    base_daily.to_csv(OUT_DIR / "base_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_summary, best_summary).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")

    switch_days = (
        best_frame.loc[best_frame["selected_model"] == "proxy", ["trade_date", "std_r10_top5_base"]]
        .drop_duplicates()
        .sort_values("trade_date")
    )
    switch_days.to_csv(OUT_DIR / "selected_proxy_days.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_v4_proxy_switch_by_std_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "proxy_table": PROXY_TABLE,
        "search_space": {
            "starts": STARTS,
            "std_los": STD_LOS,
            "std_his": STD_HIS,
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
            "tries a day-level switch from current 10d best to proxy5dbest only on lite-gate days inside a narrow std band",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# 10D v4 按 std 区间切 proxy 研究候选",
        "",
        f"- 基座：`{BASE_TABLE}`",
        f"- 备选：`{PROXY_TABLE}`",
        f"- 最优规则：`start={best_rule['start']}`，`std_lo={best_rule['std_lo']}`，`std_hi={best_rule['std_hi']}`",
        f"- Focus：`{best_focus:.12f}`，相对基座变化：`{best_focus - base_focus:+.12f}`",
        f"- 切换天数：`{best_rule['switch_days']}`",
        "- 仅研究用途，不进入 formal。",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
