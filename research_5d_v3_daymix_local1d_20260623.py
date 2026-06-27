from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_v3_daymix_local1d_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3eq_d1new_v3_20260623_executable_5d_open_return_research"
LOCAL_TABLE = "stock_predict_data_model_agent_5d_v3_local_reblend_1dbest_10dstd035_20260623_executable_5d_open_return_research"
ONE_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_curgap_endgate_20260623_executable_1d_open_return_research"
TEN_SIGNAL_TABLE = "stock_predict_data_model_agent_10d_daygate_joint_refine_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_v3_daymix_local1d_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
WINDOWS = {
    "full": ("20240604", "20260605"),
    "recent1y": ("20250301", "20260605"),
    "recent80": ("20260201", "20260605"),
}

START_DATES = ["20250601", "20251201", "20260201"]
THR10_LIST = [0.995, 0.997, 0.998, 0.999, 0.9995]
THR1_LIST = [0.97, 0.975, 0.98, 0.985]
PRIORITIES = ["local_first", "one_first"]


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


def build_choice_df(signal_df: pd.DataFrame, start_date: str, thr10: float, thr1: float, priority: str) -> pd.DataFrame:
    out = signal_df[["trade_date", "base_top5_mean_rank10", "base_top5_mean_rank1"]].copy()
    out["chosen_model"] = "base"
    ten_mask = (out["trade_date"] >= start_date) & (out["base_top5_mean_rank10"] < thr10)
    one_mask = (out["trade_date"] >= start_date) & (out["base_top5_mean_rank1"] < thr1)

    if priority == "local_first":
        out.loc[ten_mask, "chosen_model"] = "local"
        out.loc[~ten_mask & one_mask, "chosen_model"] = "one"
    elif priority == "one_first":
        out.loc[one_mask, "chosen_model"] = "one"
        out.loc[~one_mask & ten_mask, "chosen_model"] = "local"
    else:
        raise ValueError(priority)
    return out


def choose_daily_metrics(base_daily: pd.DataFrame, local_daily: pd.DataFrame, one_daily: pd.DataFrame, choice_df: pd.DataFrame) -> pd.DataFrame:
    out = base_daily.copy()
    out = out.merge(local_daily.rename(columns={m: f"local_{m}" for m in ("top1", "top3", "top5", "top10")}), on="trade_date", how="left")
    out = out.merge(one_daily.rename(columns={m: f"one_{m}" for m in ("top1", "top3", "top5", "top10")}), on="trade_date", how="left")
    out = out.merge(choice_df[["trade_date", "chosen_model"]], on="trade_date", how="left")
    for metric in ("top1", "top3", "top5", "top10"):
        local_mask = out["chosen_model"] == "local"
        one_mask = out["chosen_model"] == "one"
        out.loc[local_mask, metric] = out.loc[local_mask, f"local_{metric}"]
        out.loc[one_mask, metric] = out.loc[one_mask, f"one_{metric}"]
    return out[["trade_date", "top1", "top3", "top5", "top10", "chosen_model"]]


def summary_from_daily(daily: pd.DataFrame) -> dict:
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
    return out


def materialize(base_rows: pd.DataFrame, local_rows: pd.DataFrame, one_rows: pd.DataFrame, choice_df: pd.DataFrame) -> dict:
    out = base_rows.merge(choice_df[["trade_date", "chosen_model"]], on="trade_date", how="left")
    local_pick = local_rows.rename(columns={"pred_prob": "local_pred_prob"})[["trade_date", "stock_code", "local_pred_prob"]]
    one_pick = one_rows.rename(columns={"pred_prob": "one_pred_prob"})[["trade_date", "stock_code", "one_pred_prob"]]
    out = out.merge(local_pick, on=["trade_date", "stock_code"], how="left")
    out = out.merge(one_pick, on=["trade_date", "stock_code"], how="left")
    out["pred_prob"] = out["base_pred_prob"].astype(float)
    local_mask = out["chosen_model"] == "local"
    one_mask = out["chosen_model"] == "one"
    out.loc[local_mask, "pred_prob"] = out.loc[local_mask, "local_pred_prob"].where(
        out.loc[local_mask, "local_pred_prob"].notna(),
        out.loc[local_mask, "base_pred_prob"],
    )
    out.loc[one_mask, "pred_prob"] = out.loc[one_mask, "one_pred_prob"].where(
        out.loc[one_mask, "one_pred_prob"].notna(),
        out.loc[one_mask, "base_pred_prob"],
    )
    final = out[["trade_date", "stock_code", LABEL_COL, "pred_prob"]]

    with sqlite3.connect(MODEL_DB) as conn:
        final.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
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
        base_rows = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} "
            f"from '{BASE_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        local_rows = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from '{LOCAL_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        one_rows = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from '{ONE_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        ten_signal_rows = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred10d from '{TEN_SIGNAL_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )

    for df in (base_rows, local_rows, one_rows, ten_signal_rows):
        df["trade_date"] = df["trade_date"].astype(str)

    signal_frame = base_rows.merge(one_rows.rename(columns={"pred_prob": "pred1d"}), on=["trade_date", "stock_code"], how="left")
    signal_frame = signal_frame.merge(ten_signal_rows, on=["trade_date", "stock_code"], how="left")
    signal_frame["rank1"] = signal_frame.groupby("trade_date")["pred1d"].rank(method="average", pct=True)
    signal_frame["rank10"] = signal_frame.groupby("trade_date")["pred10d"].rank(method="average", pct=True)
    signal_frame["rank_base"] = signal_frame.groupby("trade_date")["base_pred_prob"].rank(method="average", pct=True)
    signal_frame["rank1"] = signal_frame["rank1"].fillna(signal_frame["rank_base"])
    signal_frame["rank10"] = signal_frame["rank10"].fillna(signal_frame["rank_base"])

    daily_signal_rows = []
    for trade_date, group in signal_frame.groupby("trade_date", sort=True):
        ordered_base = group.sort_values("base_pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
        daily_signal_rows.append(
            {
                "trade_date": trade_date,
                "base_top5_mean_rank10": float(ordered_base.head(5)["rank10"].mean()),
                "base_top5_mean_rank1": float(ordered_base.head(5)["rank1"].mean()),
            }
        )
    signal_df = pd.DataFrame(daily_signal_rows)

    base_summary, base_daily = evaluate(base_rows.rename(columns={"base_pred_prob": "score"}), "score")
    local_summary, local_daily = evaluate(local_rows.merge(base_rows[["trade_date", "stock_code", LABEL_COL]], on=["trade_date", "stock_code"], how="left").rename(columns={"pred_prob": "score"}), "score")
    one_summary, one_daily = evaluate(one_rows.merge(base_rows[["trade_date", "stock_code", LABEL_COL]], on=["trade_date", "stock_code"], how="left").rename(columns={"pred_prob": "score"}), "score")
    base_focus = focus_score(base_summary)

    rows: list[dict] = []
    best_rule: dict | None = None
    best_summary: dict | None = None
    best_choice_df: pd.DataFrame | None = None
    best_daily: pd.DataFrame | None = None
    best_focus = float("-inf")
    best_passed = False

    base_recent80_top3 = base_summary["recent80"]["top3"]
    base_recent80_top5 = base_summary["recent80"]["top5"]

    for start_date in START_DATES:
        for thr10 in THR10_LIST:
            for thr1 in THR1_LIST:
                for priority in PRIORITIES:
                    choice_df = build_choice_df(signal_df, start_date, thr10, thr1, priority)
                    mixed_daily = choose_daily_metrics(base_daily, local_daily, one_daily, choice_df)
                    summary = summary_from_daily(mixed_daily)
                    focus = focus_score(summary)
                    pass_guard = (
                        summary["recent80"]["top3"] >= base_recent80_top3 - 0.0002
                        and summary["recent80"]["top5"] >= base_recent80_top5
                    )
                    row = {
                        "start_date": start_date,
                        "thr10": thr10,
                        "thr1": thr1,
                        "priority": priority,
                        "focus": focus,
                        "delta_focus": focus - base_focus,
                        "guard_pass": pass_guard,
                        "recent80_top1": summary["recent80"]["top1"],
                        "recent80_top3": summary["recent80"]["top3"],
                        "recent80_top5": summary["recent80"]["top5"],
                        "recent80_top10": summary["recent80"]["top10"],
                        "recent1y_top1": summary["recent1y"]["top1"],
                        "recent1y_top3": summary["recent1y"]["top3"],
                        "recent1y_top5": summary["recent1y"]["top5"],
                        "full_top1": summary["full"]["top1"],
                        "full_top3": summary["full"]["top3"],
                        "full_top5": summary["full"]["top5"],
                        "switch_local_days": int((choice_df["chosen_model"] == "local").sum()),
                        "switch_one_days": int((choice_df["chosen_model"] == "one").sum()),
                    }
                    rows.append(row)

                    better = False
                    if pass_guard and not best_passed:
                        better = True
                    elif pass_guard == best_passed:
                        if focus > best_focus + 1e-15:
                            better = True
                        elif abs(focus - best_focus) <= 1e-15 and best_rule is not None:
                            if row["recent80_top5"] > best_rule["recent80_top5"] + 1e-15:
                                better = True
                            elif abs(row["recent80_top5"] - best_rule["recent80_top5"]) <= 1e-15 and row["switch_local_days"] + row["switch_one_days"] < best_rule["switch_local_days"] + best_rule["switch_one_days"]:
                                better = True
                    if better:
                        best_focus = focus
                        best_rule = row
                        best_summary = summary
                        best_choice_df = choice_df.copy()
                        best_daily = mixed_daily.copy()
                        best_passed = pass_guard

    if best_rule is None or best_summary is None or best_choice_df is None or best_daily is None:
        raise RuntimeError("No candidate rule evaluated")

    result_df = pd.DataFrame(rows).sort_values(
        ["guard_pass", "focus", "recent80_top5", "recent80_top3", "full_top1"],
        ascending=[False, False, False, False, False],
    )
    result_df.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")
    best_choice_df.to_csv(OUT_DIR / "daily_choices.csv", index=False, encoding="utf-8-sig")

    db_summary = materialize(base_rows, local_rows, one_rows, best_choice_df)
    best_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_summary, best_summary).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_daily, "year").to_csv(OUT_DIR / "yearly_stability.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_daily, "quarter").to_csv(OUT_DIR / "quarter_stability.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_5d_v3_daymix_local1d_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "local_table": LOCAL_TABLE,
        "one_table": ONE_TABLE,
        "ten_signal_table": TEN_SIGNAL_TABLE,
        "search_space": {
            "start_dates": START_DATES,
            "thr10_list": THR10_LIST,
            "thr1_list": THR1_LIST,
            "priorities": PRIORITIES,
        },
        "base_focus": base_focus,
        "best_rule": best_rule,
        "selected_focus": best_focus,
        "focus_delta": best_focus - base_focus,
        "guard_rule": {
            "recent80_top3_min": base_recent80_top3 - 0.0002,
            "recent80_top5_min": base_recent80_top5,
            "selected_passed": best_passed,
        },
        "base_windows": base_summary,
        "local_windows": local_summary,
        "one_windows": one_summary,
        "selected_windows": best_summary,
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "daily switch mixes current 5d best with local reblend and current 1d best candidate",
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
        "candidate_id": "model_agent_5d_v3_daymix_local1d_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": payload["generated_at"],
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "local_source": f"MODEL_PREDICTIONS.db::{LOCAL_TABLE}",
        "one_source": f"MODEL_PREDICTIONS.db::{ONE_TABLE}",
        "ten_signal_source": f"MODEL_PREDICTIONS.db::{TEN_SIGNAL_TABLE}",
        "selected_rule": best_rule,
        "notes": "5D 研究候选以当前 v3 为底座，按天切换到 local reblend 或当前 1D best，仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 5D v3 按天混合 local reblend 与 1D best 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 当前 5D best：`{BASE_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- Focus：`{base_focus:.12f}` -> `{best_focus:.12f}`",
        f"- Focus 增量：`{best_focus - base_focus:.12f}`",
        f"- 守卫是否通过：`{best_passed}`",
        "",
        "## 最优规则",
        "",
        f"- start_date：`{best_rule['start_date']}`",
        f"- thr10：`{best_rule['thr10']}`",
        f"- thr1：`{best_rule['thr1']}`",
        f"- priority：`{best_rule['priority']}`",
        f"- switch_local_days：`{best_rule['switch_local_days']}`",
        f"- switch_one_days：`{best_rule['switch_one_days']}`",
        "",
        "## 关键窗口",
        "",
        f"- recent80：`Top1={best_summary['recent80']['top1']:.8f}` `Top3={best_summary['recent80']['top3']:.8f}` `Top5={best_summary['recent80']['top5']:.8f}`",
        f"- recent1y：`Top1={best_summary['recent1y']['top1']:.8f}` `Top3={best_summary['recent1y']['top3']:.8f}` `Top5={best_summary['recent1y']['top5']:.8f}`",
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
        f"- `{(OUT_DIR / 'daily_choices.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
