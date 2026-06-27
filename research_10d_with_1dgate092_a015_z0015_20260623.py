from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_with_1dgate092_a015_z0015_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
AUX_TABLE = "stock_predict_data_model_agent_1d_guard_micro_vol_gate092_20250101_20260623_executable_1d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_dategate_3dformal_plus_1dgate092_a015_z0015_20260623_executable_10d_open_return_research"
LABEL_COL = "executable_10d_open_return"

START_DATE = "20260301"
ZONE_PCT = 0.0015
ALPHA = 0.015

WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}


def evaluate(frame: pd.DataFrame, score_col: str) -> dict:
    data = frame[["trade_date", "stock_code", LABEL_COL, score_col]].dropna().copy()
    daily_rows: list[dict] = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(ordered[score_col].corr(ordered[LABEL_COL], method="spearman")),
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
            "trade_days": int(win["trade_date"].nunique()),
            "rank_ic": float(win["rank_ic"].mean()),
            "top1": float(win["top1"].mean()),
            "top3": float(win["top3"].mean()),
            "top5": float(win["top5"].mean()),
            "top10": float(win["top10"].mean()),
            "top20": float(win["top20"].mean()),
        }
    return out


def recent_focus_score(summary: dict) -> float:
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        base = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} from '{BASE_TABLE}'",
            conn,
        )
        aux = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as aux_pred_prob from '{AUX_TABLE}'",
            conn,
        )

    base["trade_date"] = base["trade_date"].astype(str)
    aux["trade_date"] = aux["trade_date"].astype(str)
    frame = base.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame["base_rank"] = frame.groupby("trade_date")["base_pred_prob"].rank(method="average", pct=True)
    frame["aux_rank"] = frame.groupby("trade_date")["aux_pred_prob"].rank(method="average", pct=True)

    threshold = 1.0 - ZONE_PCT
    gate_mask = (
        frame["trade_date"].ge(START_DATE)
        & frame["base_rank"].ge(threshold)
        & frame["aux_rank"].notna()
    )
    frame["pred_prob"] = frame["base_rank"].where(~gate_mask, frame["base_rank"] + ALPHA * frame["aux_rank"])
    frame["gate_applied"] = gate_mask

    base_eval = evaluate(frame.rename(columns={"base_rank": "score"}), "score")
    selected_eval = evaluate(frame.rename(columns={"pred_prob": "score"}), "score")

    output = frame[
        [
            "trade_date",
            "stock_code",
            "pred_prob",
            "base_pred_prob",
            "aux_pred_prob",
            "base_rank",
            "aux_rank",
            "gate_applied",
            LABEL_COL,
        ]
    ]

    with sqlite3.connect(MODEL_DB) as conn:
        output.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_code "
            f"on '{RESEARCH_TABLE}'(trade_date, stock_code)"
        )
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_pred "
            f"on '{RESEARCH_TABLE}'(trade_date, pred_prob desc)"
        )
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   count(distinct stock_code),
                   sum(case when pred_prob is null then 1 else 0 end)
            from '{RESEARCH_TABLE}'
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*) from (
              select trade_date, stock_code, count(*) c
              from '{RESEARCH_TABLE}'
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]

    db_summary = {
        "db_path": str(MODEL_DB),
        "table": RESEARCH_TABLE,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(dup),
        "gate_start": START_DATE,
        "zone_pct": ZONE_PCT,
        "alpha": ALPHA,
        "gate_rows": int(frame["gate_applied"].sum()),
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
    }
    (OUT_DIR / "db_summary.json").write_text(json.dumps(db_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    delta = {}
    for window in WINDOWS:
        delta[window] = {
            metric: selected_eval[window][metric] - base_eval[window][metric]
            for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_dategate_3dformal_plus_1dgate092_a015_z0015_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
        "score_formula": f"if trade_date >= {START_DATE} and base_rank >= {threshold:.4f}: base_rank + {ALPHA:.3f} * aux_rank else base_rank",
        "base_eval": base_eval,
        "selected_eval": selected_eval,
        "delta_vs_base": delta,
        "base_recent_focus_score": recent_focus_score(base_eval),
        "selected_recent_focus_score": recent_focus_score(selected_eval),
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_for_l5",
        "promotion_requires_user_confirmation": True,
        "source_type": "sqlite_table",
        "label": LABEL_COL,
        "candidate_id": "model_agent_10d_dategate_3dformal_plus_1dgate092_a015_z0015_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": db_summary["row_count"],
        "trade_days": db_summary["trade_days"],
        "stock_count": db_summary["stock_count"],
        "min_trade_date": db_summary["min_trade_date"],
        "max_trade_date": db_summary["max_trade_date"],
        "duplicate_keys": db_summary["duplicate_key_groups"],
        "null_pred_prob": db_summary["null_pred_prob"],
        "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
        "aux_source": f"MODEL_PREDICTIONS.db::{AUX_TABLE}",
        "notes": "10D 候选在现有 dategate+3D formal 基线之上，仅对最近窗口的极头部区域叠加 1D gate092 研究分数，作为 research-only 候选，不进入 formal L4 或 L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 10D 叠加 1D Gate092 窄头部候选",
        "",
        "## 当前结论",
        "",
        "本轮在现有 `10D dategate + 3D formal` 候选之上，只对最近窗口的极头部区域叠加 `1D gate092` 分数。",
        "相比前一个 `a012 / z0.5%` 版本，这版把作用区间收窄到更小的头部范围，并把叠加强度提高到 `0.015`，目的是保住 Top1/Top3 改善的同时，减少 Top5 被错误换入的问题。",
        "",
        "## 参数",
        "",
        f"- 起始日期：`{START_DATE}`",
        f"- 头部区域：`top {ZONE_PCT:.2%}`",
        f"- 叠加系数：`{ALPHA}`",
        f"- 触发行数：`{db_summary['gate_rows']}`",
        "",
        "## 相对 base 候选的变化",
        "",
    ]
    for window in WINDOWS:
        lines.extend(
            [
                f"### {window}",
                f"- RankIC：`{base_eval[window]['rank_ic']:.8f} -> {selected_eval[window]['rank_ic']:.8f}`",
                f"- Top1：`{base_eval[window]['top1']:.8f} -> {selected_eval[window]['top1']:.8f}`",
                f"- Top3：`{base_eval[window]['top3']:.8f} -> {selected_eval[window]['top3']:.8f}`",
                f"- Top5：`{base_eval[window]['top5']:.8f} -> {selected_eval[window]['top5']:.8f}`",
                f"- Top10：`{base_eval[window]['top10']:.8f} -> {selected_eval[window]['top10']:.8f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 治理说明",
            "",
            "- 本轮未训练模型。",
            "- 本轮未修改任何 formal manifest。",
            "- 本轮未生成交易信号，未制定交易规则，未跑回测。",
            "- 新资产仅为 `research_only_not_for_l5`。",
            "",
            "## 证据路径",
            "",
            f"- DB 摘要：`{(OUT_DIR / 'db_summary.json').as_posix()}`",
            f"- 汇总 JSON：`{(OUT_DIR / 'probe_summary.json').as_posix()}`",
            f"- research manifest：`{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
            f"- research DB 表：`{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`",
        ]
    )
    (OUT_DIR / "probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
