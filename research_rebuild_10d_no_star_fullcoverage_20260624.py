from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_no_star_fullcoverage_rebuild_20260624"

BASE_10D = "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research"
AUX_3D = "stock_predict_data_model_agent_3d_guarded_lowvol_ext_20260624_executable_3d_open_return_research"
AUX_1D = "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_ext_20260624_executable_1d_open_return_research"
AUX_5D = "stock_predict_data_model_agent_5d_narrow_balanced_grid_20260624_executable_5d_open_return_research"
FORMAL_10D = "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal"

DATEGATE_TABLE = "stock_predict_data_model_agent_10d_dategate_3d_ext_20260624_executable_10d_open_return_research"
A015_TABLE = "stock_predict_data_model_agent_10d_dategate_3d_ext_plus_1dgate092_20260624_executable_10d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_10d_no_star_fullcoverage_rebuild_20260624_executable_10d_open_return_research"

LABEL_COL = "executable_10d_open_return"

DATEGATE_START = "20260401"
DATEGATE_THRESHOLD = 0.98
DATEGATE_ALPHA = 0.02
A015_START = "20260301"
A015_ZONE_PCT = 0.0015
A015_ALPHA = 0.015
NO_STAR_R3 = 0.991
NO_STAR_R5 = 0.98

WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}


def read_pred(conn: sqlite3.Connection, table: str, alias: str, include_label: bool = False) -> pd.DataFrame:
    cols = f"trade_date, stock_code, pred_prob as {alias}"
    if include_label:
        cols += f", [{LABEL_COL}] as {LABEL_COL}"
    frame = pd.read_sql_query(f"select {cols} from '{table}'", conn)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def evaluate(frame: pd.DataFrame, score_col: str) -> dict:
    data = frame[["trade_date", "stock_code", LABEL_COL, score_col]].dropna().copy()
    daily_rows = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 100:
            continue
        ordered = group.sort_values([score_col, "stock_code"], ascending=[False, True], kind="mergesort")
        daily_rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(group[score_col].corr(group[LABEL_COL], method="spearman")),
                "top1": float(ordered.head(1)[LABEL_COL].mean()),
                "top3": float(ordered.head(3)[LABEL_COL].mean()),
                "top5": float(ordered.head(5)[LABEL_COL].mean()),
                "top10": float(ordered.head(10)[LABEL_COL].mean()),
                "top20": float(ordered.head(20)[LABEL_COL].mean()),
                "top50": float(ordered.head(50)[LABEL_COL].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    out = {}
    for name, (start, end) in WINDOWS.items():
        win = daily[(daily["trade_date"] >= start) & (daily["trade_date"] <= end)].copy()
        out[name] = {
            "date_from": start,
            "date_to": end,
            "trade_days": int(win["trade_date"].nunique()),
            "rank_ic": float(win["rank_ic"].mean()),
            "top1": float(win["top1"].mean()),
            "top3": float(win["top3"].mean()),
            "top5": float(win["top5"].mean()),
            "top10": float(win["top10"].mean()),
            "top20": float(win["top20"].mean()),
            "top50": float(win["top50"].mean()),
        }
    return out


def table_summary(conn: sqlite3.Connection, table: str) -> dict:
    row = conn.execute(
        f"""
        select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
               count(distinct stock_code), sum(case when trade_date = '20260623' then 1 else 0 end),
               sum(case when pred_prob is null then 1 else 0 end)
        from '{table}'
        """
    ).fetchone()
    dup = conn.execute(
        f"""
        select count(*) from (
          select trade_date, stock_code, count(*) c
          from '{table}'
          group by trade_date, stock_code
          having c > 1
        )
        """
    ).fetchone()[0]
    return {
        "table": table,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "rows_20260623": int(row[5] or 0),
        "null_pred_prob": int(row[6] or 0),
        "duplicate_key_groups": int(dup),
    }


def board(stock_code: str) -> str:
    if stock_code.startswith(("8", "9")):
        return "BJ"
    if stock_code.startswith("688"):
        return "STAR"
    if stock_code.startswith("3"):
        return "GEM"
    return "MAIN"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with sqlite3.connect(MODEL_DB) as conn:
        base = read_pred(conn, BASE_10D, "base_10d", include_label=True)
        formal = read_pred(conn, FORMAL_10D, "formal_10d", include_label=True)
        aux3 = read_pred(conn, AUX_3D, "aux3")
        aux1 = read_pred(conn, AUX_1D, "aux1")
        aux5 = read_pred(conn, AUX_5D, "aux5")

    frame = base.merge(aux3, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame = frame.merge(aux1, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame = frame.merge(aux5, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame["base_rank"] = frame.groupby("trade_date")["base_10d"].rank(method="average", pct=True)
    frame["aux3_rank"] = frame.groupby("trade_date")["aux3"].rank(method="average", pct=True).fillna(frame["base_rank"])
    frame["aux1_rank"] = frame.groupby("trade_date")["aux1"].rank(method="average", pct=True)
    frame["aux5_rank"] = frame.groupby("trade_date")["aux5"].rank(method="average", pct=True)

    dategate_mask = frame["trade_date"].ge(DATEGATE_START) & frame["base_rank"].ge(DATEGATE_THRESHOLD)
    frame["dategate_score"] = frame["base_rank"]
    frame.loc[dategate_mask, "dategate_score"] = (
        frame.loc[dategate_mask, "base_rank"] + DATEGATE_ALPHA * frame.loc[dategate_mask, "aux3_rank"]
    )

    a015_threshold = 1.0 - A015_ZONE_PCT
    frame["dategate_rank"] = frame.groupby("trade_date")["dategate_score"].rank(method="average", pct=True)
    a015_mask = frame["trade_date"].ge(A015_START) & frame["dategate_rank"].ge(a015_threshold) & frame["aux1_rank"].notna()
    frame["a015_score"] = frame["dategate_rank"]
    frame.loc[a015_mask, "a015_score"] = (
        frame.loc[a015_mask, "dategate_rank"] + A015_ALPHA * frame.loc[a015_mask, "aux1_rank"]
    )

    output_parts = []
    flags = []
    frame["board"] = frame["stock_code"].map(board)
    for trade_date, group in frame.groupby("trade_date", sort=True):
        g = group.copy()
        source_top5 = g.sort_values(["a015_score", "stock_code"], ascending=[False, True], kind="mergesort").head(5)
        star_count = int((source_top5["board"] == "STAR").sum())
        mean_rank3 = float(source_top5["aux3_rank"].mean())
        mean_rank5 = float(source_top5["aux5_rank"].mean())
        keep_source = star_count == 0 and mean_rank3 >= NO_STAR_R3 and mean_rank5 >= NO_STAR_R5
        g["pred_prob"] = g["a015_score"] if keep_source else g["dategate_score"]
        g["keep_source"] = keep_source
        output_parts.append(g)
        flags.append(
            {
                "trade_date": trade_date,
                "keep_source": int(keep_source),
                "source_top5_star_count": star_count,
                "source_top5_mean_rank3": mean_rank3,
                "source_top5_mean_rank5": mean_rank5,
            }
        )

    final = pd.concat(output_parts, ignore_index=True)
    flags_df = pd.DataFrame(flags)
    flags_df.to_csv(OUT_DIR / "daily_gate_flags.csv", index=False, encoding="utf-8-sig")

    dategate_out = final[["trade_date", "stock_code", LABEL_COL, "dategate_score", "base_10d", "aux3", "base_rank", "aux3_rank"]].rename(
        columns={"dategate_score": "pred_prob"}
    )
    a015_out = final[["trade_date", "stock_code", LABEL_COL, "a015_score", "dategate_score", "aux1", "dategate_rank", "aux1_rank"]].rename(
        columns={"a015_score": "pred_prob"}
    )
    target_out = final[
        [
            "trade_date",
            "stock_code",
            LABEL_COL,
            "pred_prob",
            "dategate_score",
            "a015_score",
            "aux3_rank",
            "aux5_rank",
            "keep_source",
        ]
    ]

    with sqlite3.connect(MODEL_DB) as conn:
        for table, data in [(DATEGATE_TABLE, dategate_out), (A015_TABLE, a015_out), (TARGET_TABLE, target_out)]:
            conn.execute(f"drop table if exists '{table}'")
            data.to_sql(table, conn, if_exists="replace", index=False)
            conn.execute(f"create index if not exists idx_{table}_date_code on '{table}'(trade_date, stock_code)")
            conn.execute(f"create index if not exists idx_{table}_date_pred on '{table}'(trade_date, pred_prob desc)")
        summaries = {table: table_summary(conn, table) for table in [DATEGATE_TABLE, A015_TABLE, TARGET_TABLE]}

    formal_eval = evaluate(formal.rename(columns={"formal_10d": "score"}), "score")
    base_eval = evaluate(final.rename(columns={"base_rank": "score"}), "score")
    dategate_eval = evaluate(final.rename(columns={"dategate_score": "score"}), "score")
    a015_eval = evaluate(final.rename(columns={"a015_score": "score"}), "score")
    target_eval = evaluate(final.rename(columns={"pred_prob": "score"}), "score")

    metrics_rows = []
    for name, eval_data in [
        ("formal_10d", formal_eval),
        ("base_rank", base_eval),
        ("dategate", dategate_eval),
        ("a015", a015_eval),
        ("target_no_star", target_eval),
    ]:
        for window, values in eval_data.items():
            row = {"candidate": name, "window": window}
            row.update(values)
            metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "rebuild_metrics.csv", index=False, encoding="utf-8-sig")

    def delta(candidate: dict, baseline: dict) -> dict:
        out = {}
        for window in WINDOWS:
            out[window] = {
                metric: candidate[window][metric] - baseline[window][metric]
                for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]
            }
        return out

    report = {
        "generated_at": generated_at,
        "actor": "model-agent",
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "db_path": str(MODEL_DB),
        "target_table": TARGET_TABLE,
        "intermediate_tables": [DATEGATE_TABLE, A015_TABLE],
        "inputs": {
            "base_10d": BASE_10D,
            "aux_3d": AUX_3D,
            "aux_1d": AUX_1D,
            "aux_5d": AUX_5D,
            "formal_10d_baseline": FORMAL_10D,
        },
        "parameters": {
            "dategate_start": DATEGATE_START,
            "dategate_threshold": DATEGATE_THRESHOLD,
            "dategate_alpha": DATEGATE_ALPHA,
            "a015_start": A015_START,
            "a015_zone_pct": A015_ZONE_PCT,
            "a015_alpha": A015_ALPHA,
            "no_star_r3": NO_STAR_R3,
            "no_star_r5": NO_STAR_R5,
        },
        "summaries": summaries,
        "target_eval": target_eval,
        "delta_vs_formal": delta(target_eval, formal_eval),
        "delta_vs_dategate": delta(target_eval, dategate_eval),
        "gate_days": {
            "keep_source_days": int(flags_df["keep_source"].sum()),
            "keep_source_dates": flags_df.loc[flags_df["keep_source"] == 1, "trade_date"].tolist(),
        },
        "governance": {
            "no_training": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
            "promotion_requires_user_and_audit_confirmation": True,
        },
    }
    (OUT_DIR / "fullcoverage_rebuild_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "asset_role": "l4_research_prediction_asset",
                "approval_status": "research_only_not_approved_for_l4_or_l5",
                "source_type": "sqlite_table",
                "label": LABEL_COL,
                "candidate_id": "model_agent_10d_no_star_fullcoverage_rebuild_20260624",
                "db_path": "../MODEL_PREDICTIONS.db",
                "table": TARGET_TABLE,
                "market_db_path": "../../STOCK_DAILY_DATA.db",
                "generated_at": generated_at,
                "row_count": summaries[TARGET_TABLE]["row_count"],
                "trade_days": summaries[TARGET_TABLE]["trade_days"],
                "stock_count": summaries[TARGET_TABLE]["stock_count"],
                "min_trade_date": summaries[TARGET_TABLE]["min_trade_date"],
                "max_trade_date": summaries[TARGET_TABLE]["max_trade_date"],
                "duplicate_keys": summaries[TARGET_TABLE]["duplicate_key_groups"],
                "null_pred_prob": summaries[TARGET_TABLE]["null_pred_prob"],
                "notes": "10D no-star fullcoverage rebuild research asset; no training, no signal, no backtest, not formal.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# 10D no-star 全覆盖重建研究报告 20260624",
        "",
        "## 当前结论",
        "",
        f"- 新研究表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        f"- 覆盖：`{summaries[TARGET_TABLE]['min_trade_date']}` 到 `{summaries[TARGET_TABLE]['max_trade_date']}`，最新日行数 `{summaries[TARGET_TABLE]['rows_20260623']}`。",
        "- 本次使用全覆盖 3D/1D/5D 辅助研究资产重建门控链条，去掉上一版 20260623 formal fallback 覆盖风险。",
        "- 本次未训练模型、未修改生产 manifest、未生成交易信号、未跑回测。",
        "",
        "## 相对 formal 10D 的模型侧指标变化",
    ]
    for window in ["full", "recent126", "recent63"]:
        d = report["delta_vs_formal"][window]
        lines.extend(
            [
                f"### {window}",
                f"- ΔRankIC=`{d['rank_ic']:.6f}`",
                f"- ΔTop1=`{d['top1']:.6f}`",
                f"- ΔTop5=`{d['top5']:.6f}`",
                f"- ΔTop10=`{d['top10']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 风险说明",
            "",
            "- 该表仍是 research-only，不是 formal L4/L5 生产资产。",
            "- 辅助表口径从旧 `3d_guarded_lowvol` / 旧 5D topgate 切换到全覆盖研究资产，发布前需要单独审计。",
            "- 10D 成熟标签评价截止到 `20260528`，最新覆盖日 `20260623` 不进入效果评价。",
            "",
            "## 证据路径",
            "",
            f"- JSON：`{(OUT_DIR / 'fullcoverage_rebuild_report.json').as_posix()}`",
            f"- 指标：`{(OUT_DIR / 'rebuild_metrics.csv').as_posix()}`",
            f"- 日门控：`{(OUT_DIR / 'daily_gate_flags.csv').as_posix()}`",
        ]
    )
    (OUT_DIR / "fullcoverage_rebuild_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
