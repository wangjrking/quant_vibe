from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_std035_local_reblend_1dcurgap_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_proxy5d_lite_daygate_std035_20260623_executable_10d_open_return_research"
AUX1_TABLE = "stock_predict_data_model_agent_1d_best_with_daygate5dhi_curgap_20260623_executable_1d_open_return_research"
CURRENT_TABLE = "stock_predict_data_model_agent_10d_std035_local_reblend_1dbest_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_std035_local_reblend_1dcurgap_20260623_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}

STARTS = ["20260225", "20260201", "20260115"]
ZONES = [0.001, 0.0015, 0.002, 0.003]
ALPHAS = [0.002, 0.005, 0.01, 0.015]


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
    out = frame[["trade_date", "stock_code", LABEL_COL, "base_pred_prob", "rank_base", "rank1"]].copy()
    out["pred_prob"] = score.astype(float)
    out = out[["trade_date", "stock_code", LABEL_COL, "pred_prob", "base_pred_prob", "rank_base", "rank1"]]
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

    for df in (base, aux1):
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(aux1, on=["trade_date", "stock_code"], how="left")
    frame["rank_base"] = frame.groupby("trade_date")["base_pred_prob"].rank(method="average", pct=True)
    frame["rank1"] = frame.groupby("trade_date")["pred1d"].rank(method="average", pct=True)
    frame["rank1"] = frame["rank1"].fillna(frame["rank_base"])

    base_summary, _ = evaluate(frame.assign(score=frame["base_pred_prob"]), "score")
    base_focus = focus_score(base_summary)

    rows: list[dict] = []
    best_rule: dict | None = None
    best_focus = float("-inf")
    best_summary: dict | None = None
    best_score_series: pd.Series | None = None

    for start in STARTS:
        for zone in ZONES:
            for alpha in ALPHAS:
                active_mask = (frame["trade_date"] >= start) & (frame["rank_base"] >= 1.0 - zone)
                score = frame["rank_base"].copy()
                score.loc[active_mask] = frame.loc[active_mask, "rank_base"] + alpha * frame.loc[active_mask, "rank1"]
                variant = frame[["trade_date", "stock_code", LABEL_COL]].copy()
                variant["score"] = score
                summary, _ = evaluate(variant, "score")
                focus = focus_score(summary)
                row = {
                    "start": start,
                    "zone": zone,
                    "alpha": alpha,
                    "active_rows": int(active_mask.sum()),
                    "focus": focus,
                    "delta_focus": focus - base_focus,
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
                        and (start > best_rule["start"] or (start == best_rule["start"] and int(active_mask.sum()) < best_rule["active_rows"]))
                    )
                ):
                    best_focus = focus
                    best_rule = row
                    best_summary = summary
                    best_score_series = variant["score"].copy()

    result_df = pd.DataFrame(rows).sort_values(
        ["focus", "recent63_top1", "recent126_top1", "full_top1"],
        ascending=False,
    )
    result_df.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")

    if best_rule is None or best_summary is None or best_score_series is None:
        raise RuntimeError("No candidate rule evaluated")

    db_summary = materialize(frame, best_score_series)

    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_for_l5",
        "promotion_requires_user_confirmation": True,
        "source_type": "sqlite_table",
        "label": LABEL_COL,
        "candidate_id": "model_agent_10d_std035_local_reblend_1dcurgap_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "aux1_source": f"MODEL_PREDICTIONS.db::{AUX1_TABLE}",
        "current_best_source": f"MODEL_PREDICTIONS.db::{CURRENT_TABLE}",
        "selected_rule": {
            "start": best_rule["start"],
            "zone": best_rule["zone"],
            "alpha": best_rule["alpha"],
            "formula": "if trade_date >= start and base_rank >= 1-zone: score = base_rank + alpha * rank1d; else score = base_rank",
        },
        "base_focus": base_focus,
        "selected_focus": best_focus,
        "focus_delta": best_focus - base_focus,
        "base_windows": base_summary,
        "selected_windows": best_summary,
        "row_count": db_summary["row_count"],
        "trade_days": db_summary["trade_days"],
        "stock_count": db_summary["stock_count"],
        "min_trade_date": db_summary["min_trade_date"],
        "max_trade_date": db_summary["max_trade_date"],
        "duplicate_keys": db_summary["duplicate_key_groups"],
        "null_pred_prob": db_summary["null_pred_prob"],
        "notes": "10D research 候选以当前 std035 最优候选为基座，只在极窄头部区域叠加当前 1D curgap 排序，用于检查最新 1D retained research 是否能进一步抬升 10D 前排质量。不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_table": BASE_TABLE,
        "aux1_table": AUX1_TABLE,
        "current_table": CURRENT_TABLE,
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
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 10D std035 局部叠加 1D curgap 研究候选",
        "",
        "## 当前结论",
        "",
        f"- 当前 10D 基座：`{BASE_TABLE}`",
        f"- 当前 10D best：`{CURRENT_TABLE}`",
        f"- 新研究候选：`{RESEARCH_TABLE}`",
        f"- Focus：`{base_focus:.12f}` -> `{best_focus:.12f}`",
        f"- Focus 增量：`{best_focus - base_focus:.12f}`",
        "",
        "## 最优规则",
        "",
        f"- start：`{best_rule['start']}`",
        f"- zone：`{best_rule['zone']}`",
        f"- alpha：`{best_rule['alpha']}`",
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
