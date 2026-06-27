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
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_dense_aux_fusion_tune_20260622"

BASE_TABLE = (
    "stock_predict_data_model_agent_1d_aux_horizon_fusion_20260622_"
    "executable_1d_open_return_research"
)
AUX_TABLES = {
    "formal_1d": (
        "stock_predict_data_model_agent_toprank_latestfactor_20260618_"
        "executable_1d_open_return_score_20240604_20260618"
    ),
    "formal_3d": (
        "stock_predict_data_model_agent_toprank_latestfactor_20260618_"
        "executable_3d_open_return_score_20240604_20260618"
    ),
    "research_3d_aux": (
        "stock_predict_data_model_agent_3d_aux_horizon_fusion_20260622_"
        "executable_3d_open_return_research"
    ),
    "formal_5d": (
        "stock_predict_data_model_agent_5d_tune_20260620_"
        "executable_5d_open_return_d4_l4_fs160_gate1_score_20240604_20260618"
    ),
    "research_5d_aux": (
        "stock_predict_data_model_agent_5d_aux_horizon_fusion_20260622_"
        "executable_5d_open_return_research"
    ),
    "formal_10d": (
        "stock_predict_data_model_agent_10d_tune_20260620_"
        "executable_10d_open_return_d4_l4_fs160_gate2_score_20240604_20260618"
    ),
    "research_10d_aux": (
        "stock_predict_data_model_agent_10d_aux_horizon_fusion_20260622_"
        "executable_10d_open_return_research"
    ),
}
RESEARCH_TABLE = (
    "stock_predict_data_model_agent_1d_dense_aux_fusion_20260622_"
    "executable_1d_open_return_research"
)

LABEL_COL = "executable_1d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260618"
TOP_K = [1, 3, 5, 10, 20, 50]


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


def front_score(summary: dict) -> float:
    top = summary["top_k_mean_returns"]
    return (
        float(top["1"]) * 3.0
        + float(top["3"]) * 2.0
        + float(top["5"]) * 1.5
        + float(top["10"])
        + float(summary["daily_rank_ic_mean"]) * 0.10
        + float(summary["top_minus_bottom_mean"]) * 0.50
    )


def evaluate(name: str, base: pd.DataFrame, score: pd.Series) -> tuple[dict, pd.DataFrame]:
    frame = base.copy()
    frame["pred_prob"] = score.astype(float)
    summary, daily = evaluate_frame(
        frame,
        score_col="pred_prob",
        label_col=LABEL_COL,
        top_k=TOP_K,
        quantiles=10,
    )
    return {"name": name, **summary, "front_selection_score": front_score(summary)}, daily


def delta(candidate: dict, baseline: dict) -> dict:
    return {
        "daily_pearson_ic_mean": candidate["daily_pearson_ic_mean"] - baseline["daily_pearson_ic_mean"],
        "daily_rank_ic_mean": candidate["daily_rank_ic_mean"] - baseline["daily_rank_ic_mean"],
        "rank_ic_positive_ratio": candidate["rank_ic_positive_ratio"] - baseline["rank_ic_positive_ratio"],
        "top_minus_bottom_mean": candidate["top_minus_bottom_mean"] - baseline["top_minus_bottom_mean"],
        "top_minus_bottom_positive_ratio": (
            candidate["top_minus_bottom_positive_ratio"] - baseline["top_minus_bottom_positive_ratio"]
        ),
        "front_selection_score": candidate["front_selection_score"] - baseline["front_selection_score"],
        "top_k_mean_returns": {
            key: candidate["top_k_mean_returns"][key] - baseline["top_k_mean_returns"][key]
            for key in candidate["top_k_mean_returns"]
        },
    }


def score_name(prefix: str, weights: dict[str, float]) -> str:
    pieces = []
    for key, value in weights.items():
        pieces.append(f"{key}w{value:.2f}".replace(".", "p"))
    return prefix + "_" + "_".join(pieces)


def materialize(best_name: str, merged: pd.DataFrame, score: pd.Series) -> dict:
    out = merged.copy()
    out["pred_prob"] = score.astype(float)
    keep_cols = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "base_1d_aux_pred_prob",
        "rank_base_1d_aux",
        "formal_1d_pred_prob",
        "rank_formal_1d",
        "formal_3d_pred_prob",
        "rank_formal_3d",
        "research_3d_aux_pred_prob",
        "rank_research_3d_aux",
        "formal_5d_pred_prob",
        "rank_formal_5d",
        "research_5d_aux_pred_prob",
        "rank_research_5d_aux",
        "formal_10d_pred_prob",
        "rank_formal_10d",
        "research_10d_aux_pred_prob",
        "rank_research_10d_aux",
        "10d_yield_rate",
        "close",
        "pre_close",
        "industry",
        "industry_encode",
        "atr_qfq",
        "close_rate",
        "amount",
        "vol",
        "turnover_rate",
        "turnover_rate_f",
        "circ_mv",
        "total_mv",
        "volume_ratio",
        LABEL_COL,
        "source_fold",
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
        "score_formula": best_name,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stocks": int(row[4]),
        "duplicate_keys": int(dup),
        "null_pred_prob": int(null_pred),
        "inputs": {"base_table": BASE_TABLE, "aux_tables": AUX_TABLES},
        "notes": "1D 多周期 rank 密集融合研究资产；不作为 L5 正式入口。",
    }
    (OUT_DIR / "research_prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        merged = read_prediction(conn, BASE_TABLE, "base_1d_aux_pred_prob", keep_payload=True)
        for alias, table in AUX_TABLES.items():
            if f"{alias}_pred_prob" in merged.columns:
                continue
            aux = read_prediction(conn, table, f"{alias}_pred_prob")
            merged = merged.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")

    score_cols = ["base_1d_aux_pred_prob"] + [f"{alias}_pred_prob" for alias in AUX_TABLES]
    for col in score_cols:
        add_rank(merged, col, f"rank_{col.removesuffix('_pred_prob')}")

    rank_cols = ["rank_base_1d_aux"] + [f"rank_{alias}" for alias in AUX_TABLES]
    for col in rank_cols:
        merged[col] = merged[col].fillna(merged["rank_base_1d_aux"])

    candidates: dict[str, pd.Series] = {
        "base_1d_aux_raw": merged["base_1d_aux_pred_prob"],
        "base_1d_aux_rank": merged["rank_base_1d_aux"],
    }
    aux_aliases = list(AUX_TABLES)
    weights = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.40, 0.30]
    for alias in aux_aliases:
        rank_col = f"rank_{alias}"
        candidates[f"{alias}_rank"] = merged[rank_col]
        for base_weight in weights:
            rest = 1.0 - base_weight
            name = score_name("rank_blend_base", {"base": base_weight, alias: rest})
            candidates[name] = merged["rank_base_1d_aux"] * base_weight + merged[rank_col] * rest

    focused = ["research_3d_aux", "research_5d_aux", "research_10d_aux"]
    candidates["rank_avg_base_3d5d10d_research"] = merged[
        ["rank_base_1d_aux", *[f"rank_{alias}" for alias in focused]]
    ].mean(axis=1)
    candidates["rank_avg_3d5d10d_research"] = merged[[f"rank_{alias}" for alias in focused]].mean(axis=1)
    candidates["rank_consensus_max_base_3d5d10d_research"] = merged[
        ["rank_base_1d_aux", *[f"rank_{alias}" for alias in focused]]
    ].max(axis=1)
    candidates["rank_consensus_min_base_3d5d10d_research"] = merged[
        ["rank_base_1d_aux", *[f"rank_{alias}" for alias in focused]]
    ].min(axis=1)

    triplets = [
        {"base_1d_aux": 0.50, "research_5d_aux": 0.25, "research_10d_aux": 0.25},
        {"base_1d_aux": 0.40, "research_5d_aux": 0.30, "research_10d_aux": 0.30},
        {"base_1d_aux": 0.40, "research_3d_aux": 0.20, "research_10d_aux": 0.40},
        {"base_1d_aux": 0.35, "research_3d_aux": 0.20, "research_5d_aux": 0.20, "research_10d_aux": 0.25},
        {"base_1d_aux": 0.25, "research_3d_aux": 0.25, "research_5d_aux": 0.25, "research_10d_aux": 0.25},
        {"base_1d_aux": 0.20, "research_3d_aux": 0.20, "research_5d_aux": 0.30, "research_10d_aux": 0.30},
        {"base_1d_aux": 0.20, "research_5d_aux": 0.40, "research_10d_aux": 0.40},
    ]
    rank_lookup = {
        "base_1d_aux": "rank_base_1d_aux",
        **{alias: f"rank_{alias}" for alias in aux_aliases},
    }
    for weights_map in triplets:
        name = score_name("rank_weighted", weights_map)
        score = None
        for alias, weight in weights_map.items():
            part = merged[rank_lookup[alias]] * weight
            score = part if score is None else score + part
        candidates[name] = score

    summaries = []
    daily_by_name = {}
    for name, score in candidates.items():
        summary, daily = evaluate(name, merged, score)
        summaries.append(summary)
        daily_by_name[name] = daily

    by_name = {row["name"]: row for row in summaries}
    baseline = by_name["base_1d_aux_raw"]
    best = max(summaries, key=lambda row: row["front_selection_score"])
    best_delta = delta(best, baseline)

    def gate(row: dict) -> bool:
        row_delta = delta(row, baseline)
        return (
            row["name"] != "base_1d_aux_raw"
            and row_delta["front_selection_score"] > 0.001
            and row_delta["daily_rank_ic_mean"] >= -0.001
            and row_delta["top_k_mean_returns"]["1"] >= 0.001
            and row_delta["top_k_mean_returns"]["3"] >= 0.0
            and row_delta["top_k_mean_returns"]["5"] >= -0.0005
        )

    eligible = [row for row in summaries if gate(row)]
    selected = max(eligible, key=lambda row: row["front_selection_score"]) if eligible else None
    selected_delta = delta(selected, baseline) if selected else None
    manifest = None
    if selected is not None:
        manifest = materialize(selected["name"], merged, candidates[selected["name"]])
        daily_by_name[selected["name"]].to_csv(OUT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for item in summaries:
        top = item["top_k_mean_returns"]
        summary_rows.append(
            {
                "name": item["name"],
                "daily_pearson_ic_mean": item["daily_pearson_ic_mean"],
                "daily_rank_ic_mean": item["daily_rank_ic_mean"],
                "rank_ic_positive_ratio": item["rank_ic_positive_ratio"],
                "top_minus_bottom_mean": item["top_minus_bottom_mean"],
                "top_minus_bottom_positive_ratio": item["top_minus_bottom_positive_ratio"],
                "top1": top["1"],
                "top3": top["3"],
                "top5": top["5"],
                "top10": top["10"],
                "top20": top["20"],
                "top50": top["50"],
                "front_selection_score": item["front_selection_score"],
                "trade_days": item["trade_days"],
                "valid_rows": item["valid_rows"],
            }
        )
    pd.DataFrame(summary_rows).sort_values("front_selection_score", ascending=False).to_csv(
        OUT_DIR / "dense_fusion_eval_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "rows_base": int(len(merged)),
        "base_table": BASE_TABLE,
        "aux_tables": AUX_TABLES,
        "baseline": baseline,
        "best": best,
        "best_delta_vs_base_1d_aux": best_delta,
        "selected": selected,
        "selected_delta_vs_base_1d_aux": selected_delta,
        "materialized": selected is not None,
        "research_table": RESEARCH_TABLE if selected is not None else None,
        "manifest": manifest,
        "all_candidates": summaries,
        "selection_rule": (
            "以 front_selection_score 排序，同时要求相对当前 1D aux 的 RankIC 不明显回撤，"
            "Top1 明显提升、Top3 不回撤、Top5 不明显回撤；仅生成 research-only 资产。"
        ),
    }
    (OUT_DIR / "dense_fusion_eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_delta = selected_delta if selected_delta is not None else best_delta
    selected_name = selected["name"] if selected else None
    lines = [
        "# 1D 密集多周期融合研究报告",
        "",
        "## 结论",
        "",
    ]
    if selected is not None:
        lines.append(f"候选 `{selected_name}` 满足研究落库门槛，已生成 research-only L4 资产。")
    else:
        lines.append(f"最佳候选 `{best['name']}` 未满足研究落库门槛，未生成新资产。")
    lines.extend(
        [
            "",
            "## 相对当前 1D aux 的变化",
            "",
            f"- Top1 差值：`{report_delta['top_k_mean_returns']['1']:+.8f}`",
            f"- Top3 差值：`{report_delta['top_k_mean_returns']['3']:+.8f}`",
            f"- Top5 差值：`{report_delta['top_k_mean_returns']['5']:+.8f}`",
            f"- Top10 差值：`{report_delta['top_k_mean_returns']['10']:+.8f}`",
            f"- Rank IC 差值：`{report_delta['daily_rank_ic_mean']:+.8f}`",
            f"- Top-Bottom 差值：`{report_delta['top_minus_bottom_mean']:+.8f}`",
            "",
            "## 治理说明",
            "",
            "- 本轮未训练模型。",
            "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
            "- 若有落库，仅为 `research_only_not_l5_approved`，不改正式 manifest。",
            "",
            "## 证据路径",
            "",
            f"- 汇总 JSON：`{(OUT_DIR / 'dense_fusion_eval_summary.json').as_posix()}`",
            f"- 汇总 CSV：`{(OUT_DIR / 'dense_fusion_eval_summary.csv').as_posix()}`",
        ]
    )
    if manifest:
        lines.append(f"- research manifest：`{(OUT_DIR / 'research_prediction_manifest.json').as_posix()}`")
        lines.append(f"- research DB 表：`{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`")
    (OUT_DIR / "dense_fusion_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
