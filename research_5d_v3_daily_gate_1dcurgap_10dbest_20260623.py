from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_v3_daily_gate_1dcurgap_10dbest_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3eq_d1new_v3_20260623_executable_5d_open_return_research"
AUX1_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_curgap_20260623_executable_1d_open_return_research"
AUX10_TABLE = "stock_predict_data_model_agent_10d_std035_local_reblend_1dbest_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_v3_daily_gate_1dcurgap_10dbest_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
WINDOWS = {
    "full": ("20240604", "20260605"),
    "recent1y": ("20250301", "20260605"),
    "recent80": ("20260201", "20260605"),
}

START_DATES = ["20250301", "20250601", "20251201", "20260201"]
THR10_LIST = [0.975, 0.98, 0.985, 0.99, 0.995]
THR1_LIST = [0.97, 0.975, 0.98, 0.985, 0.99]


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


def materialize(out: pd.DataFrame) -> dict:
    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute(f"drop table if exists '{RESEARCH_TABLE}'")
        out.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_code on '{RESEARCH_TABLE}'(trade_date, stock_code)"
        )
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
            f"select trade_date, stock_code, pred_prob as pred1d from '{AUX1_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
        aux10 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred10d from '{AUX10_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )

    for df in (base, aux1, aux10):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(aux1, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux10, on=["trade_date", "stock_code"], how="left")
    base_rank = frame.groupby("trade_date")["base_pred_prob"].rank(method="average", pct=True)
    frame["rank1"] = frame.groupby("trade_date")["pred1d"].rank(method="average", pct=True)
    frame["rank10"] = frame.groupby("trade_date")["pred10d"].rank(method="average", pct=True)
    frame["rank1"] = frame["rank1"].fillna(base_rank)
    frame["rank10"] = frame["rank10"].fillna(base_rank)

    base_summary, base_daily = evaluate(frame.assign(score=frame["base_pred_prob"]), "score")
    aux1_summary, aux1_daily = evaluate(
        frame.assign(score=frame["pred1d"].where(frame["pred1d"].notna(), frame["base_pred_prob"])),
        "score",
    )
    aux10_summary, aux10_daily = evaluate(
        frame.assign(score=frame["pred10d"].where(frame["pred10d"].notna(), frame["base_pred_prob"])),
        "score",
    )
    base_focus = focus_score(base_summary)

    daily_source = (
        base_daily.rename(columns={"top1": "base_top1", "top3": "base_top3", "top5": "base_top5", "top10": "base_top10"})
        .merge(
            aux1_daily.rename(columns={"top1": "one_top1", "top3": "one_top3", "top5": "one_top5", "top10": "one_top10"}),
            on="trade_date",
            how="left",
        )
        .merge(
            aux10_daily.rename(columns={"top1": "ten_top1", "top3": "ten_top3", "top5": "ten_top5", "top10": "ten_top10"}),
            on="trade_date",
            how="left",
        )
    )

    daily_gate = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        ordered_base = group.sort_values("base_pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
        daily_gate.append(
            {
                "trade_date": trade_date,
                "base_top5_mean_rank10": float(ordered_base.head(5)["rank10"].mean()),
                "base_top5_mean_rank1": float(ordered_base.head(5)["rank1"].mean()),
            }
        )
    gate_df = pd.DataFrame(daily_gate)
    daily_source = daily_source.merge(gate_df, on="trade_date", how="left")

    rows: list[dict] = []
    best_rule: dict | None = None
    best_focus = float("-inf")
    best_choice_df: pd.DataFrame | None = None
    best_summary: dict | None = None
    best_daily: pd.DataFrame | None = None

    for start_date in START_DATES:
        for thr10 in THR10_LIST:
            for thr1 in THR1_LIST:
                choice_df = daily_source[["trade_date", "base_top5_mean_rank10", "base_top5_mean_rank1"]].copy()
                choice_df["chosen_model"] = "base"
                mask10 = (choice_df["trade_date"] >= start_date) & (choice_df["base_top5_mean_rank10"] < thr10)
                choice_df.loc[mask10, "chosen_model"] = "ten_best"
                mask1 = (
                    (choice_df["trade_date"] >= start_date)
                    & ~mask10
                    & (choice_df["base_top5_mean_rank1"] < thr1)
                )
                choice_df.loc[mask1, "chosen_model"] = "one_curgap"

                chosen_daily = daily_source.merge(choice_df[["trade_date", "chosen_model"]], on="trade_date", how="left")
                chosen_daily["top1"] = chosen_daily["base_top1"]
                chosen_daily["top3"] = chosen_daily["base_top3"]
                chosen_daily["top5"] = chosen_daily["base_top5"]
                chosen_daily["top10"] = chosen_daily["base_top10"]
                ten_mask = chosen_daily["chosen_model"] == "ten_best"
                one_mask = chosen_daily["chosen_model"] == "one_curgap"
                for metric in ("top1", "top3", "top5", "top10"):
                    chosen_daily.loc[ten_mask, metric] = chosen_daily.loc[ten_mask, f"ten_{metric}"]
                    chosen_daily.loc[one_mask, metric] = chosen_daily.loc[one_mask, f"one_{metric}"]

                summary = {}
                for name, (date_from, date_to) in WINDOWS.items():
                    win = chosen_daily[(chosen_daily["trade_date"] >= date_from) & (chosen_daily["trade_date"] <= date_to)].copy()
                    summary[name] = {
                        "date_from": date_from,
                        "date_to": date_to,
                        "trade_days": int(len(win)),
                        "top1": float(win["top1"].mean()),
                        "top3": float(win["top3"].mean()),
                        "top5": float(win["top5"].mean()),
                        "top10": float(win["top10"].mean()),
                    }
                focus = focus_score(summary)
                row = {
                    "start_date": start_date,
                    "thr10": thr10,
                    "thr1": thr1,
                    "focus": focus,
                    "delta_focus": focus - base_focus,
                    "full_top1": summary["full"]["top1"],
                    "full_top3": summary["full"]["top3"],
                    "full_top5": summary["full"]["top5"],
                    "full_top10": summary["full"]["top10"],
                    "recent1y_top1": summary["recent1y"]["top1"],
                    "recent1y_top3": summary["recent1y"]["top3"],
                    "recent1y_top5": summary["recent1y"]["top5"],
                    "recent1y_top10": summary["recent1y"]["top10"],
                    "recent80_top1": summary["recent80"]["top1"],
                    "recent80_top3": summary["recent80"]["top3"],
                    "recent80_top5": summary["recent80"]["top5"],
                    "recent80_top10": summary["recent80"]["top10"],
                    "switch_to_10d_days": int((choice_df["chosen_model"] == "ten_best").sum()),
                    "switch_to_1d_days": int((choice_df["chosen_model"] == "one_curgap").sum()),
                }
                rows.append(row)

                if (
                    focus > best_focus + 1e-15
                    or (
                        abs(focus - best_focus) <= 1e-15
                        and best_rule is not None
                        and (
                            row["recent80_top1"] > best_rule["recent80_top1"] + 1e-15
                            or (
                                abs(row["recent80_top1"] - best_rule["recent80_top1"]) <= 1e-15
                                and row["switch_to_10d_days"] + row["switch_to_1d_days"]
                                < best_rule["switch_to_10d_days"] + best_rule["switch_to_1d_days"]
                            )
                        )
                    )
                ):
                    best_focus = focus
                    best_rule = row
                    best_choice_df = choice_df.copy()
                    best_summary = summary
                    best_daily = chosen_daily[["trade_date", "top1", "top3", "top5", "top10"]].copy()

    if best_rule is None or best_choice_df is None or best_summary is None or best_daily is None:
        raise RuntimeError("No candidate rule evaluated")

    result_df = pd.DataFrame(rows).sort_values(
        ["focus", "recent80_top1", "recent1y_top1", "full_top1"],
        ascending=False,
    )
    result_df.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")
    best_choice_df.to_csv(OUT_DIR / "daily_choices.csv", index=False, encoding="utf-8-sig")

    frame_out = frame.merge(best_choice_df[["trade_date", "chosen_model"]], on="trade_date", how="left")
    frame_out["pred_prob"] = frame_out["base_pred_prob"].astype(float)
    mask10 = frame_out["chosen_model"] == "ten_best"
    mask1 = frame_out["chosen_model"] == "one_curgap"
    frame_out.loc[mask10, "pred_prob"] = frame_out.loc[mask10, "pred10d"].where(
        frame_out.loc[mask10, "pred10d"].notna(),
        frame_out.loc[mask10, "base_pred_prob"],
    )
    frame_out.loc[mask1, "pred_prob"] = frame_out.loc[mask1, "pred1d"].where(
        frame_out.loc[mask1, "pred1d"].notna(),
        frame_out.loc[mask1, "base_pred_prob"],
    )

    db_summary = materialize(frame_out[["trade_date", "stock_code", LABEL_COL, "pred_prob"]])
    best_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    compare_windows(base_summary, best_summary).to_csv(OUT_DIR / "window_comparison.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_daily, "year").to_csv(OUT_DIR / "yearly_stability.csv", index=False, encoding="utf-8-sig")
    build_period_breakdown(best_daily, "quarter").to_csv(OUT_DIR / "quarter_stability.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_5d_v3_daily_gate_1dcurgap_10dbest_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "aux1_table": AUX1_TABLE,
        "aux10_table": AUX10_TABLE,
        "base_focus": base_focus,
        "best_rule": best_rule,
        "selected_focus": best_focus,
        "focus_delta": best_focus - base_focus,
        "base_windows": base_summary,
        "selected_windows": best_summary,
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "daily gate switches only between current 5d best and current 1d/10d best research assets",
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
        "candidate_id": "model_agent_5d_v3_daily_gate_1dcurgap_10dbest_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": payload["generated_at"],
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "aux1_source": f"MODEL_PREDICTIONS.db::{AUX1_TABLE}",
        "aux10_source": f"MODEL_PREDICTIONS.db::{AUX10_TABLE}",
        "selected_rule": {
            "start_date": best_rule["start_date"],
            "if_base_top5_mean_rank10_lt": best_rule["thr10"],
            "choose_model": "ten_best",
            "elif_base_top5_mean_rank1_lt": best_rule["thr1"],
            "choose_model_else": "one_curgap",
            "default": "base",
        },
        "base_focus": base_focus,
        "selected_focus": best_focus,
        "focus_delta": best_focus - base_focus,
        "row_count": db_summary["row_count"],
        "trade_days": db_summary["trade_days"],
        "stock_count": db_summary["stock_count"],
        "min_trade_date": db_summary["min_trade_date"],
        "max_trade_date": db_summary["max_trade_date"],
        "duplicate_keys": db_summary["duplicate_key_groups"],
        "null_pred_prob": db_summary["null_pred_prob"],
        "notes": "5D 研究候选以当前 v3 最优候选为基座，只在日级触发条件满足时切换到当前最优 10D 或当前最优 1D research 资产，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 5D v3 按日门控 1D curgap / 10D best 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 当前 5D 最优候选：`{BASE_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- Focus：`{base_focus:.12f}` -> `{best_focus:.12f}`",
        f"- Focus 增量：`{best_focus - base_focus:.12f}`",
        "",
        "## 最优规则",
        "",
        f"- start_date：`{best_rule['start_date']}`",
        f"- 若 `base_top5_mean_rank10 < {best_rule['thr10']}`，整日切换到 `ten_best`",
        f"- 否则若 `base_top5_mean_rank1 < {best_rule['thr1']}`，整日切换到 `one_curgap`",
        "- 否则保持 `base`",
        f"- 切到 10D 的天数：`{best_rule['switch_to_10d_days']}`",
        f"- 切到 1D 的天数：`{best_rule['switch_to_1d_days']}`",
        "",
        "## 关键窗口",
        "",
        f"- recent80：`Top1={best_summary['recent80']['top1']:.8f}` `Top3={best_summary['recent80']['top3']:.8f}` `Top5={best_summary['recent80']['top5']:.8f}` `Top10={best_summary['recent80']['top10']:.8f}`",
        f"- recent1y：`Top1={best_summary['recent1y']['top1']:.8f}` `Top3={best_summary['recent1y']['top3']:.8f}` `Top5={best_summary['recent1y']['top5']:.8f}`",
        f"- full：`Top1={best_summary['full']['top1']:.8f}` `Top3={best_summary['full']['top3']:.8f}` `Top5={best_summary['full']['top5']:.8f}`",
        "",
        "## 治理说明",
        "",
        "- 本次只生成 research 候选资产。",
        "- 未训练模型。",
        "- 未发布 formal 资产。",
        "- 未生成交易信号，未制定交易规则，未跑回测。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'grid_results.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'daily_choices.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'window_comparison.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
