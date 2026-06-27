from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_topzone_confirmation_scan_20260627"

LABEL = "executable_1d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_1d_daygate_best_20260627_executable_1d_open_return_research"
NEW_TABLE = "stock_predict_data_model_agent_1d_topzone_confirmation_best_20260627_executable_1d_open_return_research"

SOURCE_TABLES = {
    "base": BASE_TABLE,
    "rankblend": "stock_predict_data_model_agent_1d_rankblend_narrow_best_20260627_executable_1d_open_return_research",
    "risk_v7": "stock_predict_data_model_agent_1d_risk_balanced_v7_20260625_executable_1d_open_return_research",
    "lowstd_refined": "stock_predict_data_model_agent_1d_new5d_fixed4y_lowstd_gate_refined_20260626_executable_1d_open_return_research",
    "d3": "stock_predict_data_model_agent_3d_recent_top_daygate_best_20260627_executable_3d_open_return_research",
    "d5": "stock_predict_data_model_agent_5d_recent_top_condblend_latest_20260626_executable_5d_open_return_research",
    "d10": "stock_predict_data_model_agent_10d_regime_blend_best_20260627_executable_10d_open_return_research",
}

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_score_table(conn: sqlite3.Connection, table: str, name: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": f"{name}_score"})


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        merged: pd.DataFrame | None = None
        for name, table in SOURCE_TABLES.items():
            part = read_score_table(conn, table, name)
            if merged is None:
                merged = part
            else:
                merged = merged.merge(part, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    if merged is None:
        raise RuntimeError("no score tables loaded")
    for name in SOURCE_TABLES:
        merged[f"{name}_rank"] = merged.groupby("trade_date")[f"{name}_score"].rank(method="average", pct=True)
    return merged


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise RuntimeError("no labels loaded")
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def top_mean(group: pd.DataFrame, score_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), score_col)[LABEL].mean())


def daily_eval(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        score = group[score_col]
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(score.corr(label_rank)),
                "top1": top_mean(group, score_col, 1),
                "top3": top_mean(group, score_col, 3),
                "top5": top_mean(group, score_col, 5),
                "top10": top_mean(group, score_col, 10),
                "top20": top_mean(group, score_col, 20),
                "top50": top_mean(group, score_col, 50),
            }
        )
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> dict[str, float]:
    full = {m: float(daily[m].mean()) for m in METRICS}
    recent63 = {m: float(daily.tail(63)[m].mean()) for m in METRICS}
    recent20 = {m: float(daily.tail(20)[m].mean()) for m in METRICS}
    return {"full": full, "recent63": recent63, "recent20": recent20}


def objective(summary: dict[str, dict[str, float]]) -> float:
    return float(
        1.6 * summary["full"]["top1"]
        + 1.2 * summary["full"]["top5"]
        + 0.75 * summary["full"]["rank_ic"]
        + 1.8 * summary["recent63"]["top1"]
        + 1.2 * summary["recent63"]["top5"]
        + 0.55 * summary["recent63"]["rank_ic"]
        + 2.2 * summary["recent20"]["top1"]
        + 1.4 * summary["recent20"]["top5"]
        + 0.45 * summary["recent20"]["rank_ic"]
    )


def apply_candidate(frame: pd.DataFrame, aux: str, base_q: float, aux_hi: float, aux_lo: float, boost: float, penalty: float) -> pd.Series:
    score = frame["base_rank"].to_numpy(copy=True)
    base_rank = frame["base_rank"].to_numpy()
    aux_rank = frame[f"{aux}_rank"].to_numpy()
    top_zone = base_rank >= base_q
    confirm = top_zone & (aux_rank >= aux_hi)
    reject = top_zone & (aux_rank <= aux_lo)
    score[confirm] = score[confirm] + boost * (aux_rank[confirm] - aux_hi + 0.01)
    score[reject] = score[reject] - penalty * (aux_lo - aux_rank[reject] + 0.01)
    return pd.Series(score, index=frame.index)


def scan(eval_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    base_daily = daily_eval(eval_frame, "base_rank")
    base_summary = summarize(base_daily)
    base_objective = objective(base_summary)

    rows = []
    best_score = None
    best_payload = None
    for aux in ["rankblend", "risk_v7", "lowstd_refined"]:
        for base_q in [0.995, 0.999]:
            for aux_hi in [0.80, 0.95]:
                for aux_lo in [0.10, 0.30]:
                    for boost in [0.03, 0.10]:
                        for penalty in [0.03, 0.10]:
                            name = f"{aux}_bq{base_q}_hi{aux_hi}_lo{aux_lo}_b{boost}_p{penalty}"
                            eval_frame["_candidate_score"] = apply_candidate(eval_frame, aux, base_q, aux_hi, aux_lo, boost, penalty)
                            daily = daily_eval(eval_frame, "_candidate_score")
                            summary = summarize(daily)
                            obj = objective(summary)
                            payload = {
                                "name": name,
                                "aux": aux,
                                "base_q": base_q,
                                "aux_hi": aux_hi,
                                "aux_lo": aux_lo,
                                "boost": boost,
                                "penalty": penalty,
                                "objective": obj,
                                "delta_vs_base": obj - base_objective,
                                "summary": summary,
                            }
                            rows.append(
                                {
                                    "name": name,
                                    "aux": aux,
                                    "base_q": base_q,
                                    "aux_hi": aux_hi,
                                    "aux_lo": aux_lo,
                                    "boost": boost,
                                    "penalty": penalty,
                                    "objective": obj,
                                    "delta_vs_base": obj - base_objective,
                                    "full_rank_ic": summary["full"]["rank_ic"],
                                    "full_top1": summary["full"]["top1"],
                                    "full_top5": summary["full"]["top5"],
                                    "recent63_rank_ic": summary["recent63"]["rank_ic"],
                                    "recent63_top1": summary["recent63"]["top1"],
                                    "recent63_top5": summary["recent63"]["top5"],
                                    "recent20_rank_ic": summary["recent20"]["rank_ic"],
                                    "recent20_top1": summary["recent20"]["top1"],
                                    "recent20_top5": summary["recent20"]["top5"],
                                }
                            )
                            if best_payload is None or obj > best_payload["objective"]:
                                best_payload = payload
                                best_score = eval_frame["_candidate_score"].copy()

    result = pd.DataFrame(rows).sort_values(["objective", "recent63_top1", "full_top1"], ascending=False)
    best_daily = eval_frame.assign(_best_score=best_score)
    return result, {"objective": base_objective, "summary": base_summary}, best_payload | {"daily_eval": daily_eval(best_daily, "_best_score")}


def write_best_table(scores: pd.DataFrame, best: dict) -> dict:
    out = scores[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = apply_candidate(
        scores,
        best["aux"],
        best["base_q"],
        best["aux_hi"],
        best["aux_lo"],
        best["boost"],
        best["penalty"],
    )
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        out.to_sql(NEW_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{NEW_TABLE}_date_code on {quote(NEW_TABLE)} (trade_date, stock_code)")
        row_count, min_date, max_date, trade_days, nulls = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(NEW_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(NEW_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
    return {
        "table": NEW_TABLE,
        "row_count": int(row_count),
        "min_trade_date": str(min_date),
        "max_trade_date": str(max_date),
        "trade_days": int(trade_days),
        "null_pred_prob": int(nulls or 0),
        "duplicate_key_groups": int(dup),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    results, base, best = scan(eval_frame)
    results.to_csv(REPORT_DIR / "topzone_confirmation_scan_results.csv", index=False)
    best["daily_eval"].to_csv(REPORT_DIR / "best_daily_eval.csv", index=False)
    best.pop("daily_eval")

    db_summary = None
    if best["objective"] > base["objective"]:
        db_summary = write_best_table(scores, best)

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_topzone_confirmation_scan",
        "label": LABEL,
        "base_table": BASE_TABLE,
        "new_table": NEW_TABLE if db_summary else None,
        "sources": SOURCE_TABLES,
        "base": base,
        "best": best,
        "improves_over_base": bool(best["objective"] > base["objective"]),
        "db_summary": db_summary,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "topzone_confirmation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D Top 区域确认分扫描报告",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        f"- 基线 objective：`{base['objective']:.12f}`",
        f"- 最优候选 objective：`{best['objective']:.12f}`",
        f"- 相对提升：`{best['objective'] - base['objective']:.12f}`",
        f"- 是否生成 research-only 表：`{bool(db_summary)}`",
        "",
        "## 最优规则",
        "",
        f"- 辅助评分：`{best['aux']}`",
        f"- base top 区域阈值：`{best['base_q']}`",
        f"- 辅助确认阈值：`{best['aux_hi']}`",
        f"- 辅助否决阈值：`{best['aux_lo']}`",
        f"- 确认加权：`{best['boost']}`",
        f"- 否决惩罚：`{best['penalty']}`",
        "",
        "## 边界",
        "",
        "本实验仅属于 research-only。未训练模型，未修改 formal manifest，未修改 production manifest，未生成交易信号，未运行回测。",
    ]
    if db_summary:
        lines.extend(
            [
                "",
                "## 生成资产",
                "",
                f"- 表名：`{db_summary['table']}`",
                f"- 日期范围：`{db_summary['min_trade_date']}` 至 `{db_summary['max_trade_date']}`",
                f"- 行数：`{db_summary['row_count']}`",
                f"- 交易日：`{db_summary['trade_days']}`",
                f"- 空值：`{db_summary['null_pred_prob']}`",
                f"- 重复键组：`{db_summary['duplicate_key_groups']}`",
            ]
        )
    (REPORT_DIR / "topzone_confirmation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
