from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_gate092_new5d040_bj0_20260623"

BASE_TABLE = "stock_predict_data_model_agent_1d_dense_guard_smallcap_vr_neutralfill_20260623_executable_1d_open_return_research"
SOURCE_TABLE = "stock_predict_data_model_agent_1d_guard_micro_vol_gate092_20250101_20260623_executable_1d_open_return_research"
AUX_5D_TABLE = "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research"
LABEL_COL = "executable_1d_open_return"

WINDOWS = {
    "full": ("20240604", "20260611"),
    "recent126": ("20251201", "20260611"),
    "recent63": ("20260304", "20260611"),
}


def evaluate(frame: pd.DataFrame, score_col: str) -> dict:
    data = frame[["trade_date", "stock_code", LABEL_COL, score_col]].dropna().copy()
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
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} from '{BASE_TABLE}' where trade_date <= '20260611'",
            conn,
        )
        source = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as source_pred_prob from '{SOURCE_TABLE}' where trade_date <= '20260611'",
            conn,
        )
        aux5d = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred5d from '{AUX_5D_TABLE}' where trade_date <= '20260611'",
            conn,
        )

    for df in [base, source, aux5d]:
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(source, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(aux5d, on=["trade_date", "stock_code"], how="left")
    frame["rank5d_pct"] = frame.groupby("trade_date")["pred5d"].rank(method="average", pct=True)
    frame["board"] = frame["stock_code"].map(
        lambda s: "BJ" if s.startswith(("8", "9")) else ("STAR" if s.startswith("688") else ("GEM" if s.startswith("3") else "MAIN"))
    )

    output_parts: list[pd.DataFrame] = []
    daily_flags: list[dict] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        g = group.copy()
        base_top10 = g.sort_values("base_pred_prob", ascending=False).head(10)
        source_top10 = g.sort_values("source_pred_prob", ascending=False).head(10)
        source_top5 = source_top10.head(5)
        base_codes = set(base_top10["stock_code"])
        newcomers = source_top10[~source_top10["stock_code"].isin(base_codes)].copy()
        newcomer_mean_5d_pct = float(newcomers["rank5d_pct"].mean()) if len(newcomers) else 1.0
        source_top5_bj_count = int((source_top5["board"] == "BJ").sum())
        use_source = newcomer_mean_5d_pct >= 0.4 and source_top5_bj_count == 0
        g["pred_prob"] = g["source_pred_prob"] if use_source else g["base_pred_prob"]
        g["used_source_gate"] = use_source
        output_parts.append(g)
        daily_flags.append(
            {
                "trade_date": trade_date,
                "used_source_gate": use_source,
                "newcomer_count": int(len(newcomers)),
                "newcomer_mean_5d_pct": newcomer_mean_5d_pct,
                "source_top5_bj_count": source_top5_bj_count,
            }
        )

    output = pd.concat(output_parts, ignore_index=True)
    daily_flag_df = pd.DataFrame(daily_flags)
    daily_flag_df.to_csv(OUT_DIR / "daily_gate_flags.csv", index=False, encoding="utf-8-sig")

    base_eval = evaluate(output.rename(columns={"base_pred_prob": "score"}), "score")
    source_eval = evaluate(output.rename(columns={"source_pred_prob": "score"}), "score")
    selected_eval = evaluate(output.rename(columns={"pred_prob": "score"}), "score")

    db_output = output[["trade_date", "stock_code", "pred_prob", "base_pred_prob", "source_pred_prob", "used_source_gate", LABEL_COL]]

    with sqlite3.connect(MODEL_DB) as conn:
        db_output.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_code on '{RESEARCH_TABLE}'(trade_date, stock_code)"
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
        "aux_5d_table": AUX_5D_TABLE,
        "used_source_days": int(daily_flag_df["used_source_gate"].sum()),
        "used_source_trade_dates": daily_flag_df.loc[daily_flag_df["used_source_gate"], "trade_date"].tolist(),
    }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_1d_gate092_new5d040_bj0_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "base_table": BASE_TABLE,
        "source_table": SOURCE_TABLE,
        "aux_5d_table": AUX_5D_TABLE,
        "rule": "if source top10 newcomers mean 5d rank pct >= 0.4 and source top5 contains no BJ stock, use gate092 source score; otherwise fallback to base 1D research score",
        "base_eval": base_eval,
        "source_eval": source_eval,
        "selected_eval": selected_eval,
        "delta_vs_source": {
            window: {
                metric: selected_eval[window][metric] - source_eval[window][metric]
                for metric in ["top1", "top3", "top5", "top10", "top20"]
            }
            for window in WINDOWS
        },
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
    (OUT_DIR / "db_summary.json").write_text(json.dumps(db_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "asset_role": "l4_research_prediction_asset",
                "approval_status": "research_only_not_for_l5",
                "promotion_requires_user_confirmation": True,
                "source_type": "sqlite_table",
                "label": LABEL_COL,
                "candidate_id": "model_agent_1d_gate092_new5d040_bj0_20260623",
                "db_path": "../MODEL_PREDICTIONS.db",
                "table": RESEARCH_TABLE,
                "market_db_path": "../../STOCK_DAILY_DATA.db",
                "generated_at": summary["generated_at"],
                "row_count": db_summary["row_count"],
                "trade_days": db_summary["trade_days"],
                "stock_count": db_summary["stock_count"],
                "min_trade_date": db_summary["min_trade_date"],
                "max_trade_date": db_summary["max_trade_date"],
                "duplicate_keys": db_summary["duplicate_key_groups"],
                "null_pred_prob": db_summary["null_pred_prob"],
                "base_source": f"MODEL_PREDICTIONS.db::{BASE_TABLE}",
                "aux_source": f"MODEL_PREDICTIONS.db::{SOURCE_TABLE}",
                "notes": "1D 候选在现有 newcomer 5D 门控基础上增加一层北交所约束：仅当 newcomer_mean_5d_pct >= 0.4 且 source top5 不含 BJ 股票时保留 source，否则回退到 base。该资产仅用于 research 验证，不进入 formal L4/L5。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# 1D newcomer + BJ 过滤研究候选",
        "",
        "## 当前结论",
        "",
        f"- 新候选 `1d_gate092_new5d040_bj0` 的 focus 分数为 `{summary['selected_focus_score']:.12f}`。",
        f"- 相比当前 1D research 候选，focus 由 `{summary['source_focus_score']:.12f}` 提升到 `{summary['selected_focus_score']:.12f}`。",
        "",
        "## 规则",
        "",
        "- 条件 1：`source top10 newcomers mean 5d rank pct >= 0.4`。",
        "- 条件 2：`source top5 contains no BJ stock`。",
        "- 仅当两个条件同时满足时保留 source；否则回退到 base。",
        "",
        "## 关键窗口",
        "",
        f"- recent63：`Top1={selected_eval['recent63']['top1']:.8f}` `Top3={selected_eval['recent63']['top3']:.8f}` `Top5={selected_eval['recent63']['top5']:.8f}`",
        f"- recent126：`Top1={selected_eval['recent126']['top1']:.8f}` `Top3={selected_eval['recent126']['top3']:.8f}` `Top5={selected_eval['recent126']['top5']:.8f}`",
        f"- full：`Top1={selected_eval['full']['top1']:.8f}` `Top3={selected_eval['full']['top3']:.8f}` `Top5={selected_eval['full']['top5']:.8f}`",
        "",
        "## 治理说明",
        "",
        "- 本轮未训练模型。",
        "- 本轮未发布 formal 资产。",
        "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'probe_summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'db_summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'daily_gate_flags.csv').as_posix()}`",
    ]
    (OUT_DIR / "probe_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
