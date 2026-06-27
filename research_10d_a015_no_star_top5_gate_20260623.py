from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_a015_no_star_top5_gate_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
SOURCE_TABLE = "stock_predict_data_model_agent_10d_dategate_3dformal_plus_1dgate092_a015_z0015_20260623_executable_10d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_10d_a015_no_star_top5_gate_20260623_executable_10d_open_return_research"
LABEL_COL = "executable_10d_open_return"

FULL_WINDOWS = {
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
    for name, (date_from, date_to) in FULL_WINDOWS.items():
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


def focus_score(summary: dict) -> float:
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
        source = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as source_pred_prob from '{SOURCE_TABLE}'",
            conn,
        )

    base["trade_date"] = base["trade_date"].astype(str)
    source["trade_date"] = source["trade_date"].astype(str)
    frame = base.merge(source, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame["board"] = frame["stock_code"].map(
        lambda s: "BJ" if s.startswith(("8", "9")) else ("STAR" if s.startswith("688") else ("GEM" if s.startswith("3") else "MAIN"))
    )

    output_parts: list[pd.DataFrame] = []
    daily_flags: list[dict] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        g = group.copy()
        source_top5 = g.sort_values("source_pred_prob", ascending=False).head(5)
        has_star_in_top5 = bool((source_top5["board"] == "STAR").any())
        g["pred_prob"] = g["base_pred_prob"] if has_star_in_top5 else g["source_pred_prob"]
        g["used_base_fallback"] = has_star_in_top5
        output_parts.append(g)
        daily_flags.append(
            {
                "trade_date": trade_date,
                "used_base_fallback": has_star_in_top5,
                "source_top5_star_count": int((source_top5["board"] == "STAR").sum()),
            }
        )

    output = pd.concat(output_parts, ignore_index=True)
    daily_flag_df = pd.DataFrame(daily_flags)
    daily_flag_df.to_csv(OUT_DIR / "daily_gate_flags.csv", index=False, encoding="utf-8-sig")

    base_eval = evaluate(output.rename(columns={"base_pred_prob": "score"}), "score")
    source_eval = evaluate(output.rename(columns={"source_pred_prob": "score"}), "score")
    selected_eval = evaluate(output.rename(columns={"pred_prob": "score"}), "score")

    db_output = output[
        [
            "trade_date",
            "stock_code",
            "pred_prob",
            "base_pred_prob",
            "source_pred_prob",
            "used_base_fallback",
            LABEL_COL,
        ]
    ]

    with sqlite3.connect(MODEL_DB) as conn:
        db_output.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
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
        "base_table": BASE_TABLE,
        "source_table": SOURCE_TABLE,
        "fallback_days": int(daily_flag_df["used_base_fallback"].sum()),
        "fallback_trade_dates": daily_flag_df.loc[daily_flag_df["used_base_fallback"], "trade_date"].tolist(),
    }
    (OUT_DIR / "db_summary.json").write_text(json.dumps(db_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    delta_vs_source = {}
    for window in FULL_WINDOWS:
        delta_vs_source[window] = {
            metric: selected_eval[window][metric] - source_eval[window][metric]
            for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_10d_a015_no_star_top5_gate_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "source_table": SOURCE_TABLE,
        "rule": "if source candidate top5 contains any STAR(688) stock, fallback to base 10D score for that trade_date; otherwise keep a015 score",
        "base_eval": base_eval,
        "source_eval": source_eval,
        "selected_eval": selected_eval,
        "delta_vs_source": delta_vs_source,
        "base_focus_score": focus_score(base_eval),
        "source_focus_score": focus_score(source_eval),
        "selected_focus_score": focus_score(selected_eval),
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
        "candidate_id": "model_agent_10d_a015_no_star_top5_gate_20260623",
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
        "aux_source": f"MODEL_PREDICTIONS.db::{SOURCE_TABLE}",
        "notes": "10D 候选在 a015_z0015 基础上增加日级门控：若当日候选 Top5 中包含任意 STAR(688) 股票，则整日退回 base 10D 分数。该资产仅用于 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 10D a015 无 STAR Top5 门控候选",
        "",
        "## 当前结论",
        "",
        "本轮继续沿用 `a015_z0015` 的头部增强思路，但加入更强的日级门控。",
        "若当日 `a015_z0015` 的 Top5 中出现任意 STAR(688) 股票，则整日退回 `base 10D`；否则保留 `a015_z0015`。",
        "这个规则的目标，是保住 `a015` 在近期 `Top1/Top3` 上的优势，同时减少 STAR 主导日对整体 focus 的拖累。",
        "",
        "## 关键结果",
        "",
        f"- base focus：`{summary['base_focus_score']:.12f}`",
        f"- a015 focus：`{summary['source_focus_score']:.12f}`",
        f"- 新候选 focus：`{summary['selected_focus_score']:.12f}`",
        f"- fallback 天数：`{db_summary['fallback_days']}`",
        "",
        "## 相对 a015 的变化",
        "",
    ]
    for window in FULL_WINDOWS:
        lines.extend(
            [
                f"### {window}",
                f"- RankIC：`{source_eval[window]['rank_ic']:.8f} -> {selected_eval[window]['rank_ic']:.8f}`",
                f"- Top1：`{source_eval[window]['top1']:.8f} -> {selected_eval[window]['top1']:.8f}`",
                f"- Top3：`{source_eval[window]['top3']:.8f} -> {selected_eval[window]['top3']:.8f}`",
                f"- Top5：`{source_eval[window]['top5']:.8f} -> {selected_eval[window]['top5']:.8f}`",
                f"- Top10：`{source_eval[window]['top10']:.8f} -> {selected_eval[window]['top10']:.8f}`",
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
            f"- `{(OUT_DIR / 'db_summary.json').as_posix()}`",
            f"- `{(OUT_DIR / 'probe_summary.json').as_posix()}`",
            f"- `{(OUT_DIR / 'daily_gate_flags.csv').as_posix()}`",
            f"- `{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`",
        ]
    )
    (OUT_DIR / "probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
