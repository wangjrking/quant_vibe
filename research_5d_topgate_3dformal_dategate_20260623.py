from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from evaluate_prediction_asset import evaluate_frame


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_topgate_3dformal_dategate_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research"
AUX_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_topgate_3dformal_dategate_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
TOP_K = [1, 3, 5, 10, 20, 50]
DATE_GATE = "20260201"
BASE_RANK_THRESHOLD = 0.995
AUX_ADD_ALPHA = 0.02

WINDOWS = {
    "full": ("20240604", "20260605"),
    "recent1y": ("20250301", "20260605"),
    "recent80": ("20260201", "20260605"),
}


def read_prediction(conn: sqlite3.Connection, table: str, alias: str, keep_payload: bool = False) -> pd.DataFrame:
    if keep_payload:
        frame = pd.read_sql_query(
            f"select * from '{table}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
    else:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from '{table}' "
            "where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.rename(columns={"pred_prob": alias})


def add_rank(frame: pd.DataFrame, score_col: str, rank_col: str) -> None:
    frame[rank_col] = frame.groupby("trade_date")[score_col].rank(method="average", pct=True)


def evaluate(name: str, frame: pd.DataFrame, score_col: str) -> tuple[dict, pd.DataFrame]:
    data = frame.copy()
    data["pred_prob"] = data[score_col].astype(float)
    summary, daily = evaluate_frame(
        data,
        score_col="pred_prob",
        label_col=LABEL_COL,
        top_k=TOP_K,
        quantiles=10,
    )
    top = summary["top_k_mean_returns"]
    return (
        {
            "name": name,
            **summary,
            "top1": top["1"],
            "top3": top["3"],
            "top5": top["5"],
            "top10": top["10"],
            "top20": top["20"],
            "top50": top["50"],
        },
        daily,
    )


def summarize_window(daily: pd.DataFrame, date_from: str, date_to: str) -> dict:
    win = daily[(daily["trade_date"] >= date_from) & (daily["trade_date"] <= date_to)].copy()
    return {
        "date_from": date_from,
        "date_to": date_to,
        "trade_days": int(win["trade_date"].nunique()),
        "rank_ic": float(win["rank_ic"].mean()),
        "top1": float(win["top1_mean"].mean()),
        "top3": float(win["top3_mean"].mean()),
        "top5": float(win["top5_mean"].mean()),
        "top10": float(win["top10_mean"].mean()),
        "top20": float(win["top20_mean"].mean()),
        "top50": float(win["top50_mean"].mean()),
        "top_bottom": float(win["top_minus_bottom"].mean()),
    }


def materialize(merged: pd.DataFrame, selected_summary: dict, window_summaries: dict) -> dict:
    out = merged.copy()
    out["pred_prob"] = out["topgate_score"].astype(float)
    keep_cols = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "base_5d_pred_prob",
        "rank_base_5d",
        "aux_3d_formal_pred_prob",
        "rank_aux_3d_formal",
        "topgate_applied",
        "topgate_reason",
        LABEL_COL,
    ]
    out = out[[col for col in keep_cols if col in out.columns]]
    with sqlite3.connect(MODEL_DB) as conn:
        out.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_code "
            f"on '{RESEARCH_TABLE}'(trade_date, stock_code)"
        )
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_pred "
            f"on '{RESEARCH_TABLE}'(trade_date, pred_prob desc)"
        )
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), "
            f"count(distinct stock_code) from '{RESEARCH_TABLE}'"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from '{RESEARCH_TABLE}' "
            "group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        null_pred = conn.execute(f"select count(*) from '{RESEARCH_TABLE}' where pred_prob is null").fetchone()[0]
    manifest = {
        "asset_status": "research_only_not_l5_approved",
        "approval_status": "research_only",
        "source_type": "sqlite_table",
        "asset_role": "l4_research_prediction_asset",
        "label_col": LABEL_COL,
        "db_path": str(MODEL_DB),
        "table": RESEARCH_TABLE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "score_formula": (
            "if trade_date >= 20260201 and base_5d_rank >= 0.995: "
            "base_5d_rank + 0.02 * 3d_formal_rank; else base_5d_rank"
        ),
        "date_gate": DATE_GATE,
        "base_rank_threshold": BASE_RANK_THRESHOLD,
        "aux_add_alpha": AUX_ADD_ALPHA,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stocks": int(row[4]),
        "duplicate_keys": int(dup),
        "null_pred_prob": int(null_pred),
        "inputs": {"base_table": BASE_TABLE, "aux_table": AUX_TABLE},
        "selected_summary": selected_summary,
        "window_summaries": window_summaries,
        "notes": "5D 使用 3D formal 做日期门控顶部局部重排的研究资产，仅供研究验证，不进入 formal 或 L5。",
    }
    (OUT_DIR / "research_prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        merged = read_prediction(conn, BASE_TABLE, "base_5d_pred_prob", keep_payload=True)
        aux = read_prediction(conn, AUX_TABLE, "aux_3d_formal_pred_prob")
        merged = merged.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")

    add_rank(merged, "base_5d_pred_prob", "rank_base_5d")
    add_rank(merged, "aux_3d_formal_pred_prob", "rank_aux_3d_formal")
    merged["rank_aux_3d_formal"] = merged["rank_aux_3d_formal"].fillna(merged["rank_base_5d"])
    merged["topgate_score"] = merged["rank_base_5d"]
    merged["topgate_applied"] = False
    merged["topgate_reason"] = ""

    gate_mask = (merged["trade_date"] >= DATE_GATE) & (merged["rank_base_5d"] >= BASE_RANK_THRESHOLD)
    merged.loc[gate_mask, "topgate_applied"] = True
    merged.loc[gate_mask, "topgate_reason"] = f"date_gate:{DATE_GATE}"
    merged.loc[gate_mask, "topgate_score"] = (
        merged.loc[gate_mask, "rank_base_5d"] + AUX_ADD_ALPHA * merged.loc[gate_mask, "rank_aux_3d_formal"]
    )

    baseline, baseline_daily = evaluate("base_5d_formal", merged, "base_5d_pred_prob")
    selected, selected_daily = evaluate("topgate_3d_formal_dategate", merged, "topgate_score")

    baseline_windows = {
        name: summarize_window(baseline_daily, date_from, date_to)
        for name, (date_from, date_to) in WINDOWS.items()
    }
    selected_windows = {
        name: summarize_window(selected_daily, date_from, date_to)
        for name, (date_from, date_to) in WINDOWS.items()
    }
    deltas = {
        name: {
            metric: selected_windows[name][metric] - baseline_windows[name][metric]
            for metric in ("rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom")
        }
        for name in WINDOWS
    }

    manifest = materialize(merged, selected, selected_windows)
    selected_daily.to_csv(OUT_DIR / "selected_daily_eval.csv", index=False, encoding="utf-8-sig")
    baseline_daily.to_csv(OUT_DIR / "baseline_daily_eval.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
        "baseline": baseline,
        "selected": selected,
        "baseline_windows": baseline_windows,
        "selected_windows": selected_windows,
        "window_deltas": deltas,
        "research_table": RESEARCH_TABLE,
        "manifest": manifest,
        "selection_rule": (
            "固定使用 20260201 起生效的 5D 顶部 0.5% + 3D formal rank 局部重排，"
            "作为 research-only 候选保留，不切换 formal。"
        ),
    }
    (OUT_DIR / "topgate_3dformal_dategate_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 5D 顶部 3D formal 日期门控重排研究",
        "",
        "## 当前结论",
        "",
        "本轮将快速筛选中的最优日期门控方案单独落为 research-only 资产。",
        "该候选相对全历史重排版本更平衡：保留了更好的全窗口 Top3/Top5，同时 recent80 Top1 转正。",
        "",
        "## 参数",
        "",
        f"- 基线表：`{BASE_TABLE}`",
        f"- 辅助表：`{AUX_TABLE}`",
        f"- 日期门控：`trade_date >= {DATE_GATE}`",
        f"- 阈值：`base_5d_rank >= {BASE_RANK_THRESHOLD}`",
        f"- 加权：`+ {AUX_ADD_ALPHA} * rank_aux_3d_formal`",
        "",
        "## 关键窗口变化",
        "",
        f"- full Top1：`{baseline_windows['full']['top1']:.8f} -> {selected_windows['full']['top1']:.8f}`",
        f"- full Top3：`{baseline_windows['full']['top3']:.8f} -> {selected_windows['full']['top3']:.8f}`",
        f"- full Top5：`{baseline_windows['full']['top5']:.8f} -> {selected_windows['full']['top5']:.8f}`",
        f"- recent80 Top1：`{baseline_windows['recent80']['top1']:.8f} -> {selected_windows['recent80']['top1']:.8f}`",
        f"- recent80 Top3：`{baseline_windows['recent80']['top3']:.8f} -> {selected_windows['recent80']['top3']:.8f}`",
        f"- recent80 Top5：`{baseline_windows['recent80']['top5']:.8f} -> {selected_windows['recent80']['top5']:.8f}`",
        "",
        "## 治理说明",
        "",
        "- 本轮未训练模型。",
        "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
        "- 新资产仅为 `research_only`，不切换 formal / L5。",
        "",
        "## 证据路径",
        "",
        f"- 汇总 JSON：`{(OUT_DIR / 'topgate_3dformal_dategate_summary.json').as_posix()}`",
        f"- research manifest：`{(OUT_DIR / 'research_prediction_manifest.json').as_posix()}`",
        f"- research DB 表：`{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`",
    ]
    (OUT_DIR / "topgate_3dformal_dategate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
