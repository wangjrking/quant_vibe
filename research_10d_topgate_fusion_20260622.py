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
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_topgate_fusion_tune_20260622"

BASE_TABLE = (
    "stock_predict_data_model_agent_10d_tune_20260620_"
    "executable_10d_open_return_d4_l4_fs160_gate2_score_20240604_20260618"
)
AUX_TABLE = (
    "stock_predict_data_model_agent_1d_dense_aux_fusion_20260622_"
    "executable_1d_open_return_research"
)
RESEARCH_TABLE = (
    "stock_predict_data_model_agent_10d_topgate_fusion_20260622_"
    "executable_10d_open_return_research"
)

LABEL_COL = "executable_10d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260618"
TOP_K = [1, 3, 5, 10, 20, 50]
FORMAL_RANK_THRESHOLD = 0.9975
AUX_ADD_ALPHA = 0.01


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


def materialize(merged: pd.DataFrame, selected: dict, selected_delta: dict) -> dict:
    out = merged.copy()
    out["pred_prob"] = out["topgate_score"].astype(float)
    keep_cols = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "formal_10d_pred_prob",
        "rank_formal_10d",
        "research_1d_dense_pred_prob",
        "rank_research_1d_dense",
        "topgate_applied",
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
            "formal_10d_rank + 0.01 * research_1d_dense_rank "
            "when formal_10d_rank >= 0.9975; otherwise formal_10d_rank"
        ),
        "formal_rank_threshold": FORMAL_RANK_THRESHOLD,
        "aux_add_alpha": AUX_ADD_ALPHA,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stocks": int(row[4]),
        "duplicate_keys": int(dup),
        "null_pred_prob": int(null_pred),
        "inputs": {"base_table": BASE_TABLE, "aux_table": AUX_TABLE},
        "selected_summary": selected,
        "selected_delta_vs_formal_10d": selected_delta,
        "notes": "10D formal 前排内部微重排研究资产；不作为 L5 正式入口。",
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
        aux = read_prediction(conn, AUX_TABLE, "research_1d_dense_pred_prob")
        merged = merged.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")

    add_rank(merged, "formal_10d_pred_prob", "rank_formal_10d")
    add_rank(merged, "research_1d_dense_pred_prob", "rank_research_1d_dense")
    merged["rank_research_1d_dense"] = merged["rank_research_1d_dense"].fillna(merged["rank_formal_10d"])
    merged["topgate_score"] = merged["rank_formal_10d"]
    merged["topgate_applied"] = merged["rank_formal_10d"] >= FORMAL_RANK_THRESHOLD
    merged.loc[merged["topgate_applied"], "topgate_score"] = (
        merged.loc[merged["topgate_applied"], "rank_formal_10d"]
        + AUX_ADD_ALPHA * merged.loc[merged["topgate_applied"], "rank_research_1d_dense"]
    )

    baseline, baseline_daily = evaluate("formal_10d_raw", merged, "formal_10d_pred_prob")
    selected, selected_daily = evaluate("topgate_formal_10d_plus_1d_dense", merged, "topgate_score")
    selected_delta = delta(selected, baseline)
    materialize_asset = (
        selected_delta["front_selection_score"] > 0.001
        and selected_delta["daily_rank_ic_mean"] >= -0.001
        and selected_delta["top_k_mean_returns"]["1"] > 0.005
        and selected_delta["top_k_mean_returns"]["3"] >= 0.0
        and selected_delta["top_k_mean_returns"]["5"] >= 0.0
        and selected_delta["top_k_mean_returns"]["10"] >= 0.0
    )

    manifest = None
    if materialize_asset:
        manifest = materialize(merged, selected, selected_delta)
        selected_daily.to_csv(OUT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
    baseline_daily.to_csv(OUT_DIR / "baseline_daily_eval.csv", index=False, encoding="utf-8-sig")

    rows = []
    for item in [baseline, selected]:
        top = item["top_k_mean_returns"]
        rows.append(
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
    pd.DataFrame(rows).to_csv(OUT_DIR / "topgate_fusion_eval_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "rows_base": int(len(merged)),
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
        "formal_rank_threshold": FORMAL_RANK_THRESHOLD,
        "aux_add_alpha": AUX_ADD_ALPHA,
        "baseline": baseline,
        "selected": selected,
        "selected_delta_vs_formal_10d": selected_delta,
        "materialized": materialize_asset,
        "research_table": RESEARCH_TABLE if materialize_asset else None,
        "manifest": manifest,
        "selection_rule": (
            "仅在 formal 10D 每日最前 0.25% 内用 1D dense rank 微重排；"
            "要求 Top1/Top3/Top5/Top10 均不低于 formal 10D，且 RankIC 不明显回撤。"
        ),
    }
    (OUT_DIR / "topgate_fusion_eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_delta = selected_delta
    lines = [
        "# 10D 前排微重排融合研究报告",
        "",
        "## 结论",
        "",
    ]
    if materialize_asset:
        lines.append("候选 `topgate_formal_10d_plus_1d_dense` 满足研究落库门槛，已生成 research-only L4 资产。")
    else:
        lines.append("候选 `topgate_formal_10d_plus_1d_dense` 未满足研究落库门槛，未生成新资产。")
    lines.extend(
        [
            "",
            "## 相对 formal 10D 的变化",
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
            f"- 汇总 JSON：`{(OUT_DIR / 'topgate_fusion_eval_summary.json').as_posix()}`",
            f"- 汇总 CSV：`{(OUT_DIR / 'topgate_fusion_eval_summary.csv').as_posix()}`",
        ]
    )
    if manifest:
        lines.append(f"- research manifest：`{(OUT_DIR / 'research_prediction_manifest.json').as_posix()}`")
        lines.append(f"- research DB 表：`{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`")
    (OUT_DIR / "topgate_fusion_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
