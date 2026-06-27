from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_daygate_joint_refine_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_proxy5d_lite_daygate_std035_20260623_executable_10d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_10d_std035_local_reblend_1dbest_daygate_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_daygate_joint_refine_20260623_executable_10d_open_return_research"
DAY_GATE_CSV = DATA_DIR / "reports" / "model_agent_10d_proxy5d_lite_daygate_std035_20260623" / "day_gate.csv"

LABEL_COL = "executable_10d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}

CURRENT_START = "20251201"
CURRENT_STD = 0.02
CURRENT_ZONE = 0.0015
CURRENT_ALPHA = 0.01

STARTS = ["20251118", "20251201", "20260101", "20260115"]
STD_THRESHOLDS = [0.015, 0.02, 0.025, 0.03]
ZONES = [0.00125, 0.0015, 0.00175, 0.002]
ALPHAS = [0.008, 0.01, 0.012]


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


def materialize(frame: pd.DataFrame, score: pd.Series) -> dict:
    out = frame[["trade_date", "stock_code", LABEL_COL, "base_pred_prob", "rank_base", "rank1", "std_r10_top5_base"]].copy()
    out["pred_prob"] = score.astype(float)
    out = out[["trade_date", "stock_code", LABEL_COL, "pred_prob", "base_pred_prob", "rank_base", "rank1", "std_r10_top5_base"]]
    with sqlite3.connect(MODEL_DB) as conn:
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


def build_score(frame: pd.DataFrame, start: str, std_threshold: float, zone: float, alpha: float) -> tuple[pd.Series, pd.Series]:
    active_mask = (
        (frame["trade_date"] >= start)
        & (frame["rank_base"] >= 1.0 - zone)
        & (frame["std_r10_top5_base"] < std_threshold)
    )
    score = frame["rank_base"].copy()
    score.loc[active_mask] = frame.loc[active_mask, "rank_base"] + alpha * frame.loc[active_mask, "rank1"]
    return score, active_mask


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, [{LABEL_COL}] as {LABEL_COL}, base_pred_prob, rank_base, rank1 "
            f"from '{CURRENT_TABLE}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )

    frame["trade_date"] = frame["trade_date"].astype(str)
    day_gate = pd.read_csv(DAY_GATE_CSV)
    day_gate["trade_date"] = day_gate["trade_date"].astype(str)
    frame = frame.merge(day_gate[["trade_date", "std_r10_top5_base"]], on="trade_date", how="left")

    base_summary, _ = evaluate(frame.assign(score=frame["base_pred_prob"]), "score")
    current_score, current_mask = build_score(frame, CURRENT_START, CURRENT_STD, CURRENT_ZONE, CURRENT_ALPHA)
    current_summary, _ = evaluate(frame.assign(score=current_score), "score")
    base_focus = focus_score(base_summary)
    current_focus = focus_score(current_summary)

    rows: list[dict] = []
    best_rule: dict | None = None
    best_focus = float("-inf")
    best_summary: dict | None = None
    best_score_series: pd.Series | None = None

    for start in STARTS:
        for threshold in STD_THRESHOLDS:
            for zone in ZONES:
                for alpha in ALPHAS:
                    score, active_mask = build_score(frame, start, threshold, zone, alpha)
                    summary, _ = evaluate(frame.assign(score=score), "score")
                    focus = focus_score(summary)
                    row = {
                        "start": start,
                        "std_threshold": threshold,
                        "zone": zone,
                        "alpha": alpha,
                        "active_rows": int(active_mask.sum()),
                        "active_days": int(frame.loc[active_mask, "trade_date"].nunique()),
                        "focus": focus,
                        "delta_vs_base": focus - base_focus,
                        "delta_vs_current_best": focus - current_focus,
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
                            and (
                                row["recent63_top1"] > best_rule["recent63_top1"] + 1e-15
                                or (
                                    abs(row["recent63_top1"] - best_rule["recent63_top1"]) <= 1e-15
                                    and row["active_days"] < best_rule["active_days"]
                                )
                            )
                        )
                    ):
                        best_focus = focus
                        best_rule = row
                        best_summary = summary
                        best_score_series = score.copy()

    result_df = pd.DataFrame(rows).sort_values(
        ["focus", "recent63_top1", "recent126_top1", "full_top1"],
        ascending=False,
    )
    result_df.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")

    if best_rule is None or best_summary is None or best_score_series is None:
        raise RuntimeError("No candidate rule evaluated")

    db_summary = materialize(frame, best_score_series)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_daygate_joint_refine_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "current_table": CURRENT_TABLE,
        "search_space": {
            "starts": STARTS,
            "std_thresholds": STD_THRESHOLDS,
            "zones": ZONES,
            "alphas": ALPHAS,
        },
        "current_rule": {
            "start": CURRENT_START,
            "std_threshold": CURRENT_STD,
            "zone": CURRENT_ZONE,
            "alpha": CURRENT_ALPHA,
            "active_rows": int(current_mask.sum()),
            "active_days": int(frame.loc[current_mask, "trade_date"].nunique()),
        },
        "best_rule": best_rule,
        "base_focus": base_focus,
        "current_focus": current_focus,
        "selected_focus": best_focus,
        "selected_minus_base_focus": best_focus - base_focus,
        "selected_minus_current_focus": best_focus - current_focus,
        "base_windows": base_summary,
        "current_windows": current_summary,
        "selected_windows": best_summary,
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "refines the current 10d best by scanning a narrow neighborhood around the existing day gate and local rerank rule",
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
        "candidate_id": "model_agent_10d_daygate_joint_refine_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": payload["generated_at"],
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "current_best_source": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "day_gate_source": str(DAY_GATE_CSV),
        "selected_rule": best_rule,
        "notes": "10D 研究候选围绕当前 best 的 day gate 与局部重排参数做窄邻域精扫，仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 10D day gate 窄邻域精扫研究候选",
        "",
        "## 当前结论",
        "",
        f"- 当前 10D best：`{CURRENT_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- 当前 Focus：`{current_focus:.12f}`",
        f"- 新候选 Focus：`{best_focus:.12f}`",
        f"- 相对当前 Focus 增量：`{best_focus - current_focus:.12f}`",
        "",
        "## 最优规则",
        "",
        f"- start：`{best_rule['start']}`",
        f"- std_threshold：`{best_rule['std_threshold']}`",
        f"- zone：`{best_rule['zone']}`",
        f"- alpha：`{best_rule['alpha']}`",
        f"- active_days：`{best_rule['active_days']}`",
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
        f"- `{(OUT_DIR / 'grid_results.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
