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
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_blend_tune_20260622"

GATE7_TABLE = (
    "stock_predict_data_model_agent_1d_gate7_research_20260621_"
    "executable_1d_open_return_score_20240604_20260618"
)
FORMAL_TABLE = (
    "stock_predict_data_model_agent_toprank_latestfactor_20260618_"
    "executable_1d_open_return_score_20240604_20260618"
)
RESEARCH_TABLE = (
    "stock_predict_data_model_agent_1d_gate7_invformal_blend_20260622_"
    "executable_1d_open_return_research"
)

LABEL_COL = "executable_1d_open_return"
TOP_K = [1, 3, 5, 10, 20, 50]
DATE_FROM = "20240604"
DATE_TO = "20260618"


def read_table(conn: sqlite3.Connection, table: str, score_alias: str) -> pd.DataFrame:
    sql = (
        f"select * from '{table}' "
        "where trade_date >= ? and trade_date <= ?"
    )
    frame = pd.read_sql_query(sql, conn, params=[DATE_FROM, DATE_TO])
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.rename(columns={"pred_prob": score_alias})
    return frame


def add_daily_rank(frame: pd.DataFrame, column: str, out_column: str) -> None:
    frame[out_column] = frame.groupby("trade_date")[column].rank(method="average", pct=True)


def score_summary(summary: dict) -> float:
    top = summary["top_k_mean_returns"]
    return (
        float(top["1"]) * 3.0
        + float(top["3"]) * 2.0
        + float(top["5"]) * 1.5
        + float(top["10"])
        + float(summary["daily_rank_ic_mean"]) * 0.10
        + float(summary["top_minus_bottom_mean"]) * 0.50
    )


def evaluate_candidate(name: str, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summary, daily = evaluate_frame(
        frame,
        score_col="pred_prob",
        label_col=LABEL_COL,
        top_k=TOP_K,
        quantiles=10,
    )
    summary = {
        "name": name,
        **summary,
        "front_selection_score": score_summary(summary),
    }
    return summary, daily


def metric_delta(candidate: dict, baseline: dict) -> dict:
    top_delta = {
        k: candidate["top_k_mean_returns"][k] - baseline["top_k_mean_returns"][k]
        for k in candidate["top_k_mean_returns"]
    }
    return {
        "daily_pearson_ic_mean": candidate["daily_pearson_ic_mean"] - baseline["daily_pearson_ic_mean"],
        "daily_rank_ic_mean": candidate["daily_rank_ic_mean"] - baseline["daily_rank_ic_mean"],
        "rank_ic_positive_ratio": candidate["rank_ic_positive_ratio"] - baseline["rank_ic_positive_ratio"],
        "top_minus_bottom_mean": candidate["top_minus_bottom_mean"] - baseline["top_minus_bottom_mean"],
        "top_minus_bottom_positive_ratio": (
            candidate["top_minus_bottom_positive_ratio"] - baseline["top_minus_bottom_positive_ratio"]
        ),
        "front_selection_score": candidate["front_selection_score"] - baseline["front_selection_score"],
        "top_k_mean_returns": top_delta,
    }


def write_research_asset(frame: pd.DataFrame, best: dict) -> dict:
    write_cols = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "gate7_pred_prob",
        "formal_pred_prob",
        "gate7_rank",
        "inv_formal_rank",
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
    out = frame[[col for col in write_cols if col in frame.columns]].copy()
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
    manifest = {
        "asset_status": "research_only_not_l5_approved",
        "approval_status": "research_only",
        "source_type": "sqlite_table",
        "asset_role": "l4_research_prediction_asset",
        "strategy_id": None,
        "label_col": LABEL_COL,
        "db_path": str(MODEL_DB),
        "table": RESEARCH_TABLE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stocks": int(row[4]),
        "duplicate_keys": int(dup),
        "score_formula": best["name"],
        "notes": "1D gate7 与正式1D反向日内rank融合研究资产；不作为L5正式入口。",
        "inputs": {
            "gate7_table": GATE7_TABLE,
            "formal_table": FORMAL_TABLE,
        },
    }
    (OUT_DIR / "research_prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        gate7 = read_table(conn, GATE7_TABLE, "gate7_pred_prob")
        formal = read_table(conn, FORMAL_TABLE, "formal_pred_prob")[
            ["trade_date", "stock_code", "formal_pred_prob"]
        ]

    merged = gate7.merge(formal, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    add_daily_rank(merged, "gate7_pred_prob", "gate7_rank")
    merged["inv_formal_pred_prob"] = -merged["formal_pred_prob"]
    add_daily_rank(merged, "inv_formal_pred_prob", "inv_formal_rank")
    add_daily_rank(merged, "formal_pred_prob", "formal_rank")

    candidates: list[tuple[str, pd.Series]] = [
        ("gate7_raw", merged["gate7_pred_prob"]),
        ("formal_raw", merged["formal_pred_prob"]),
        ("formal_inverted_raw", merged["inv_formal_pred_prob"]),
        ("gate7_rank", merged["gate7_rank"]),
        ("formal_inverted_rank", merged["inv_formal_rank"]),
    ]
    for weight in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]:
        inv_weight = 1.0 - weight
        safe = str(weight).replace(".", "p")
        candidates.append(
            (
                f"rank_blend_gate7w{safe}_invformalw{str(inv_weight).replace('.', 'p')}",
                merged["gate7_rank"] * weight + merged["inv_formal_rank"] * inv_weight,
            )
        )

    summaries = []
    daily_by_name = {}
    for name, score in candidates:
        frame = merged.copy()
        frame["pred_prob"] = score.astype(float)
        summary, daily = evaluate_candidate(name, frame)
        summaries.append(summary)
        daily_by_name[name] = daily

    summary_frame = pd.DataFrame(
        [
            {
                "name": s["name"],
                "daily_pearson_ic_mean": s["daily_pearson_ic_mean"],
                "daily_rank_ic_mean": s["daily_rank_ic_mean"],
                "rank_ic_positive_ratio": s["rank_ic_positive_ratio"],
                "top_minus_bottom_mean": s["top_minus_bottom_mean"],
                "top_minus_bottom_positive_ratio": s["top_minus_bottom_positive_ratio"],
                "top1": s["top_k_mean_returns"]["1"],
                "top3": s["top_k_mean_returns"]["3"],
                "top5": s["top_k_mean_returns"]["5"],
                "top10": s["top_k_mean_returns"]["10"],
                "top20": s["top_k_mean_returns"]["20"],
                "top50": s["top_k_mean_returns"]["50"],
                "front_selection_score": s["front_selection_score"],
                "trade_days": s["trade_days"],
                "valid_rows": s["valid_rows"],
            }
            for s in summaries
        ]
    ).sort_values("front_selection_score", ascending=False)
    summary_frame.to_csv(OUT_DIR / "blend_eval_summary.csv", index=False, encoding="utf-8-sig")

    by_name = {s["name"]: s for s in summaries}
    baseline = by_name["gate7_raw"]
    best = max(summaries, key=lambda item: item["front_selection_score"])
    best_delta = metric_delta(best, baseline)

    should_materialize = (
        best["name"] != "gate7_raw"
        and best_delta["front_selection_score"] > 0
        and best_delta["daily_rank_ic_mean"] >= -0.002
        and best_delta["top_k_mean_returns"]["1"] >= -0.0005
        and best_delta["top_k_mean_returns"]["3"] >= -0.0005
    )
    manifest = None
    if should_materialize:
        best_frame = merged.copy()
        best_frame["pred_prob"] = dict(candidates)[best["name"]].astype(float)
        manifest = write_research_asset(best_frame, best)
        daily_by_name[best["name"]].to_csv(OUT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")

    payload = {
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "rows_inner_join": int(len(merged)),
        "gate7_table": GATE7_TABLE,
        "formal_table": FORMAL_TABLE,
        "baseline": baseline,
        "best": best,
        "best_delta_vs_gate7": best_delta,
        "materialized": should_materialize,
        "research_table": RESEARCH_TABLE if should_materialize else None,
        "manifest": manifest,
        "all_candidates": summaries,
        "selection_rule": (
            "front_selection_score = 3*Top1 + 2*Top3 + 1.5*Top5 + Top10 "
            "+ 0.10*RankIC + 0.50*Top-Bottom; materialize only if front score "
            "improves and RankIC/Top1/Top3 do not materially degrade."
        ),
    }
    (OUT_DIR / "blend_eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 1D gate7 评分融合研究报告",
        "",
        "## 结论",
        "",
    ]
    if should_materialize:
        lines.append(
            f"本轮最佳候选为 `{best['name']}`，相对 gate7 的前排综合评分改善，已落地为 research-only L4 资产。"
        )
    else:
        lines.append(
            f"本轮最佳候选为 `{best['name']}`，但未满足稳健落库门槛；不生成新的研究资产。"
        )
    lines.extend(
        [
            "",
            "## 关键指标",
            "",
            f"- 评价区间：`{best['date_min']}` 至 `{best['date_max']}`，交易日 `{best['trade_days']}`。",
            f"- baseline gate7 Top1：`{baseline['top_k_mean_returns']['1']:.8f}`，最佳 Top1：`{best['top_k_mean_returns']['1']:.8f}`，差值：`{best_delta['top_k_mean_returns']['1']:+.8f}`。",
            f"- baseline gate7 Top3：`{baseline['top_k_mean_returns']['3']:.8f}`，最佳 Top3：`{best['top_k_mean_returns']['3']:.8f}`，差值：`{best_delta['top_k_mean_returns']['3']:+.8f}`。",
            f"- baseline gate7 Top5：`{baseline['top_k_mean_returns']['5']:.8f}`，最佳 Top5：`{best['top_k_mean_returns']['5']:.8f}`，差值：`{best_delta['top_k_mean_returns']['5']:+.8f}`。",
            f"- baseline gate7 Rank IC：`{baseline['daily_rank_ic_mean']:.8f}`，最佳 Rank IC：`{best['daily_rank_ic_mean']:.8f}`，差值：`{best_delta['daily_rank_ic_mean']:+.8f}`。",
            f"- baseline gate7 Top-Bottom：`{baseline['top_minus_bottom_mean']:.8f}`，最佳 Top-Bottom：`{best['top_minus_bottom_mean']:.8f}`，差值：`{best_delta['top_minus_bottom_mean']:+.8f}`。",
            "",
            "## 治理状态",
            "",
            "- 本轮未训练模型。",
            "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
            "- 如有落库，资产状态为 `research_only_not_l5_approved`，不改变正式 manifest。",
            "",
            "## 证据路径",
            "",
            f"- 汇总 JSON：`{(OUT_DIR / 'blend_eval_summary.json').as_posix()}`",
            f"- 汇总 CSV：`{(OUT_DIR / 'blend_eval_summary.csv').as_posix()}`",
        ]
    )
    if manifest:
        lines.append(f"- research manifest：`{(OUT_DIR / 'research_prediction_manifest.json').as_posix()}`")
        lines.append(f"- research DB 表：`{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`")
    (OUT_DIR / "blend_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
