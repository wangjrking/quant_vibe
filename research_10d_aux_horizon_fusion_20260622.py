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
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_aux_fusion_tune_20260622"

BASE_TABLE = (
    "stock_predict_data_model_agent_10d_tune_20260620_"
    "executable_10d_open_return_d4_l4_fs160_gate2_score_20240604_20260618"
)
AUX_TABLES = {
    "formal_3d": (
        "stock_predict_data_model_agent_toprank_latestfactor_20260618_"
        "executable_3d_open_return_score_20240604_20260618"
    ),
    "formal_5d": (
        "stock_predict_data_model_agent_5d_tune_20260620_"
        "executable_5d_open_return_d4_l4_fs160_gate1_score_20240604_20260618"
    ),
    "blend_5d_research": (
        "stock_predict_data_model_agent_5d_blend50_gate6_formal_20260622_"
        "executable_5d_open_return_research"
    ),
    "research_1d_aux": (
        "stock_predict_data_model_agent_1d_aux_horizon_fusion_20260622_"
        "executable_1d_open_return_research"
    ),
}
RESEARCH_TABLE = (
    "stock_predict_data_model_agent_10d_aux_horizon_fusion_20260622_"
    "executable_10d_open_return_research"
)

LABEL_COL = "executable_10d_open_return"
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
    summary = {"name": name, **summary, "front_selection_score": front_score(summary)}
    return summary, daily


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


def materialize(best_name: str, merged: pd.DataFrame, score: pd.Series, best: dict) -> dict:
    out = merged.copy()
    out["pred_prob"] = score.astype(float)
    keep_cols = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "formal_10d_pred_prob",
        "rank_formal_10d",
        "formal_3d_pred_prob",
        "rank_formal_3d",
        "formal_5d_pred_prob",
        "rank_formal_5d",
        "blend_5d_research_pred_prob",
        "rank_blend_5d_research",
        "research_1d_aux_pred_prob",
        "rank_research_1d_aux",
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
    ]
    out = out[[col for col in keep_cols if col in out.columns]]
    with sqlite3.connect(MODEL_DB) as conn:
        out.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{RESEARCH_TABLE}_date_code on '{RESEARCH_TABLE}'(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{RESEARCH_TABLE}_date_pred on '{RESEARCH_TABLE}'(trade_date, pred_prob desc)")
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), count(distinct stock_code) "
            f"from '{RESEARCH_TABLE}'"
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
        "notes": "10D formal 与辅助周期预测分数rank融合研究资产；不作为L5正式入口。",
    }
    (OUT_DIR / "research_prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        merged = read_prediction(conn, BASE_TABLE, "formal_10d_pred_prob", keep_payload=True)
        for alias, table in AUX_TABLES.items():
            aux = read_prediction(conn, table, f"{alias}_pred_prob")
            merged = merged.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")

    score_cols = ["formal_10d_pred_prob"] + [f"{alias}_pred_prob" for alias in AUX_TABLES]
    for col in score_cols:
        add_rank(merged, col, f"rank_{col.removesuffix('_pred_prob')}")

    aux_rank_cols = [f"rank_{alias}" for alias in AUX_TABLES]
    for col in aux_rank_cols:
        merged[col] = merged[col].fillna(merged["rank_formal_10d"])

    candidates: dict[str, pd.Series] = {
        "formal_10d_raw": merged["formal_10d_pred_prob"],
        "formal_10d_rank": merged["rank_formal_10d"],
    }
    for alias in AUX_TABLES:
        candidates[f"{alias}_rank"] = merged[f"rank_{alias}"]
        for weight in [0.90, 0.85, 0.75, 0.65, 0.50]:
            rest = 1.0 - weight
            candidates[f"rank_blend_10dw{str(weight).replace('.', 'p')}_{alias}w{str(rest).replace('.', 'p')}"] = (
                merged["rank_formal_10d"] * weight + merged[f"rank_{alias}"] * rest
            )
    candidates["rank_avg_10d_5d_1daux"] = merged[["rank_formal_10d", "rank_formal_5d", "rank_research_1d_aux"]].mean(axis=1)
    candidates["rank_avg_10d_3d5d1daux"] = merged[["rank_formal_10d", "rank_formal_3d", "rank_formal_5d", "rank_research_1d_aux"]].mean(axis=1)
    candidates["rank_avg_10d_3d5dblend1daux"] = merged[["rank_formal_10d", "rank_formal_3d", "rank_blend_5d_research", "rank_research_1d_aux"]].mean(axis=1)
    candidates["rank_weighted_10d_0p70_5d_0p20_1daux_0p10"] = (
        merged["rank_formal_10d"] * 0.70 + merged["rank_formal_5d"] * 0.20 + merged["rank_research_1d_aux"] * 0.10
    )
    candidates["rank_weighted_10d_0p65_5d_0p20_1daux_0p15"] = (
        merged["rank_formal_10d"] * 0.65 + merged["rank_formal_5d"] * 0.20 + merged["rank_research_1d_aux"] * 0.15
    )
    candidates["rank_weighted_10d_0p70_5dblend_0p20_1daux_0p10"] = (
        merged["rank_formal_10d"] * 0.70 + merged["rank_blend_5d_research"] * 0.20 + merged["rank_research_1d_aux"] * 0.10
    )

    summaries = []
    daily_by_name = {}
    for name, score in candidates.items():
        summary, daily = evaluate(name, merged, score)
        summaries.append(summary)
        daily_by_name[name] = daily

    by_name = {row["name"]: row for row in summaries}
    baseline = by_name["formal_10d_raw"]
    best = max(summaries, key=lambda row: row["front_selection_score"])
    best_delta = delta(best, baseline)
    materialize_asset = (
        best["name"] != "formal_10d_raw"
        and best_delta["front_selection_score"] > 0.001
        and best_delta["daily_rank_ic_mean"] >= -0.001
        and best_delta["top_k_mean_returns"]["1"] >= -0.001
        and best_delta["top_k_mean_returns"]["3"] >= -0.001
    )

    manifest = None
    if materialize_asset:
        manifest = materialize(best["name"], merged, candidates[best["name"]], best)
        daily_by_name[best["name"]].to_csv(OUT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for item in summaries:
        summary_rows.append(
            {
                "name": item["name"],
                "daily_pearson_ic_mean": item["daily_pearson_ic_mean"],
                "daily_rank_ic_mean": item["daily_rank_ic_mean"],
                "rank_ic_positive_ratio": item["rank_ic_positive_ratio"],
                "top_minus_bottom_mean": item["top_minus_bottom_mean"],
                "top_minus_bottom_positive_ratio": item["top_minus_bottom_positive_ratio"],
                "top1": item["top_k_mean_returns"]["1"],
                "top3": item["top_k_mean_returns"]["3"],
                "top5": item["top_k_mean_returns"]["5"],
                "top10": item["top_k_mean_returns"]["10"],
                "top20": item["top_k_mean_returns"]["20"],
                "top50": item["top_k_mean_returns"]["50"],
                "front_selection_score": item["front_selection_score"],
                "trade_days": item["trade_days"],
                "valid_rows": item["valid_rows"],
            }
        )
    pd.DataFrame(summary_rows).sort_values("front_selection_score", ascending=False).to_csv(
        OUT_DIR / "aux_fusion_eval_summary.csv",
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
        "best_delta_vs_formal_10d": best_delta,
        "materialized": materialize_asset,
        "research_table": RESEARCH_TABLE if materialize_asset else None,
        "manifest": manifest,
        "all_candidates": summaries,
        "selection_rule": (
            "以 Top1/Top3/Top5/Top10 加权的 front_selection_score 为主，"
            "同时约束 RankIC、Top1、Top3 不出现明显回撤；辅助分数缺失日回退到10D基线rank。"
        ),
    }
    (OUT_DIR / "aux_fusion_eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 10D 多周期辅助分数融合研究报告",
        "",
        "## 结论",
        "",
    ]
    if materialize_asset:
        lines.append(f"最佳候选 `{best['name']}` 满足研究落库门槛，已生成 research-only L4 资产。")
    else:
        lines.append(f"最佳候选 `{best['name']}` 未满足研究落库门槛，未生成新研究资产。")
    lines.extend(
        [
            "",
            "## 相对正式 10D 的变化",
            "",
            f"- Top1 差值：`{best_delta['top_k_mean_returns']['1']:+.8f}`",
            f"- Top3 差值：`{best_delta['top_k_mean_returns']['3']:+.8f}`",
            f"- Top5 差值：`{best_delta['top_k_mean_returns']['5']:+.8f}`",
            f"- Top10 差值：`{best_delta['top_k_mean_returns']['10']:+.8f}`",
            f"- Rank IC 差值：`{best_delta['daily_rank_ic_mean']:+.8f}`",
            f"- Top-Bottom 差值：`{best_delta['top_minus_bottom_mean']:+.8f}`",
            "",
            "## 治理说明",
            "",
            "- 本轮未训练模型。",
            "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
            "- 若有落库，仅为 `research_only_not_l5_approved`，不改正式 manifest。",
            "- 辅助分数缺失日回退到 10D 基线 rank，避免缩短最新预测覆盖。",
            "",
            "## 证据路径",
            "",
            f"- 汇总 JSON：`{(OUT_DIR / 'aux_fusion_eval_summary.json').as_posix()}`",
            f"- 汇总 CSV：`{(OUT_DIR / 'aux_fusion_eval_summary.csv').as_posix()}`",
        ]
    )
    if manifest:
        lines.append(f"- research manifest：`{(OUT_DIR / 'research_prediction_manifest.json').as_posix()}`")
        lines.append(f"- research DB 表：`{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`")
    (OUT_DIR / "aux_fusion_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
