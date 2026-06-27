from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_consensus_rerank_scan_20260626"
LABEL = "executable_1d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_1d_fixed4y_refined_blend15_20260626_executable_1d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_1d_consensus_rerank_v1_20260626_executable_1d_open_return_research"
SOURCES = {
    "base": BASE_TABLE,
    "fixed4y_raw": "stock_predict_data_model_agent_1d_new5d_fixed4y_lowstd_gate_20260626_executable_1d_open_return_research",
    "proxy5d_topzone": "stock_predict_data_model_agent_1d_proxy5d_topzone_10dlite_20260623_executable_1d_open_return_research",
    "best_daygate": "stock_predict_data_model_agent_1d_best_with_daygate5dhi_20260623_executable_1d_open_return_research",
    "proxy10d_safe": "stock_predict_data_model_agent_1d_proxy10dnew_curgap_top1safe_20260623_executable_1d_open_return_research",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        out = None
        for alias, table in SOURCES.items():
            frame = pd.read_sql_query(
                f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
                conn,
            )
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame["stock_code"] = frame["stock_code"].astype(str)
            frame = frame.rename(columns={"pred_prob": f"{alias}_score"})
            out = frame if out is None else out.merge(frame, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    assert out is not None
    for alias in SOURCES:
        out[f"{alias}_rank"] = out.groupby("trade_date")[f"{alias}_score"].rank(method="average", pct=True)
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[LABEL])
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def daily_metrics(frame: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(group[rank_col].corr(label_rank)),
                "top1": top_mean(group, rank_col, 1),
                "top5": top_mean(group, rank_col, 5),
                "top10": top_mean(group, rank_col, 10),
                "top20": top_mean(group, rank_col, 20),
                "top_bottom": top_mean(group, rank_col, 50) - bottom_mean(group, rank_col, 50),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summary(daily: pd.DataFrame, window: int | None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {col: float(sub[col].mean()) for col in ["rank_ic", "top1", "top5", "top10", "top20", "top_bottom"]}


def score_candidate(base_full: dict[str, float], base_63: dict[str, float], base_20: dict[str, float], cand_full: dict[str, float], cand_63: dict[str, float], cand_20: dict[str, float]) -> float:
    return (
        2.8 * (cand_20["top1"] - base_20["top1"])
        + 2.2 * (cand_20["top5"] - base_20["top5"])
        + 1.2 * (cand_63["top1"] - base_63["top1"])
        + 0.9 * (cand_63["top5"] - base_63["top5"])
        + 0.8 * (cand_full["rank_ic"] - base_full["rank_ic"])
        + 0.3 * (cand_full["top5"] - base_full["top5"])
    )


def write_candidate(scores: pd.DataFrame, best: pd.Series) -> dict[str, object]:
    mode = str(best["mode"])
    out = scores[["trade_date", "stock_code", "base_rank"]].copy()
    if mode == "single_boost":
        alt = str(best["alt1"])
        thr = float(best["threshold"])
        alpha = float(best["alpha"])
        active = (scores["base_rank"] >= thr) & (scores[f"{alt}_rank"] >= thr)
        out["pred_prob"] = np.where(active, scores["base_rank"] + alpha * (scores[f"{alt}_rank"] - 0.5), scores["base_rank"])
        formula = f"if base_rank>={thr:.2f} and {alt}_rank>={thr:.2f} then base_rank + {alpha:.3f}*({alt}_rank-0.5) else base_rank"
    else:
        alt1 = str(best["alt1"])
        alt2 = str(best["alt2"])
        thr = float(best["threshold"])
        bonus = float(best["bonus"])
        w1 = float(best["w1"])
        w2 = float(best["w2"])
        active = (scores["base_rank"] >= thr) & (scores[f"{alt1}_rank"] >= thr) & (scores[f"{alt2}_rank"] >= thr)
        blended = 0.70 * scores["base_rank"] + w1 * scores[f"{alt1}_rank"] + w2 * scores[f"{alt2}_rank"] + bonus
        out["pred_prob"] = np.where(active, blended, scores["base_rank"])
        formula = f"if base_rank>={thr:.2f} and {alt1}_rank>={thr:.2f} and {alt2}_rank>={thr:.2f} then 0.70*base_rank + {w1:.2f}*{alt1}_rank + {w2:.2f}*{alt2}_rank + {bonus:.3f} else base_rank"
    out["score_formula"] = formula
    keep = ["trade_date", "stock_code", "pred_prob", "base_rank", "score_formula"]
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[keep].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_1d_consensus_rerank_v1_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_1d_consensus_rerank_v1_date_pred on {quote(TARGET_TABLE)}(trade_date, pred_prob desc)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
    return {
        "table": TARGET_TABLE,
        "formula": formula,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    base_daily = daily_metrics(eval_frame, "base_rank")
    base_full = summary(base_daily, None)
    base_63 = summary(base_daily, 63)
    base_20 = summary(base_daily, 20)

    results: list[dict[str, object]] = []
    alts = [alias for alias in SOURCES if alias != "base"]
    thresholds = [0.95, 0.96, 0.97]
    alphas = [0.02, 0.04, 0.06]
    single_alts = ["fixed4y_raw", "proxy5d_topzone", "best_daygate", "proxy10d_safe"]

    for alt in single_alts:
        for thr in thresholds:
            for alpha in alphas:
                rank_col = f"single_{alt}_{str(thr).replace('.', '')}_{int(alpha * 1000):03d}"
                active = (eval_frame["base_rank"] >= thr) & (eval_frame[f"{alt}_rank"] >= thr)
                eval_frame[rank_col] = np.where(
                    active,
                    eval_frame["base_rank"] + alpha * (eval_frame[f"{alt}_rank"] - 0.5),
                    eval_frame["base_rank"],
                )
                daily = daily_metrics(eval_frame, rank_col)
                cand_full = summary(daily, None)
                cand_63 = summary(daily, 63)
                cand_20 = summary(daily, 20)
                row = {
                    "mode": "single_boost",
                    "alt1": alt,
                    "alt2": "",
                    "threshold": thr,
                    "alpha": alpha,
                    "bonus": 0.0,
                    "w1": 0.0,
                    "w2": 0.0,
                    "full_rank_ic_delta": cand_full["rank_ic"] - base_full["rank_ic"],
                    "full_top1_delta": cand_full["top1"] - base_full["top1"],
                    "full_top5_delta": cand_full["top5"] - base_full["top5"],
                    "recent63_top1_delta": cand_63["top1"] - base_63["top1"],
                    "recent63_top5_delta": cand_63["top5"] - base_63["top5"],
                    "recent20_top1_delta": cand_20["top1"] - base_20["top1"],
                    "recent20_top5_delta": cand_20["top5"] - base_20["top5"],
                    "recent20_rank_ic_delta": cand_20["rank_ic"] - base_20["rank_ic"],
                    "score": score_candidate(base_full, base_63, base_20, cand_full, cand_63, cand_20),
                }
                row["pass_basic"] = bool(
                    row["recent20_top5_delta"] >= 0.0
                    and row["recent20_top1_delta"] >= 0.0
                    and row["recent63_top5_delta"] >= -0.00015
                    and row["full_rank_ic_delta"] >= -0.00030
                )
                results.append(row)

    pair_alts = ["fixed4y_raw", "proxy5d_topzone", "best_daygate"]
    pair_weights = [(0.20, 0.10), (0.15, 0.15)]
    bonuses = [0.00, 0.01]
    for alt1, alt2 in combinations(pair_alts, 2):
        for thr in [0.95, 0.96]:
            for w1, w2 in pair_weights:
                for bonus in bonuses:
                    rank_col = f"pair_{alt1}_{alt2}_{str(thr).replace('.', '')}_{int(w1*100):02d}_{int(w2*100):02d}_{int(bonus*100):02d}"
                    active = (
                        (eval_frame["base_rank"] >= thr)
                        & (eval_frame[f"{alt1}_rank"] >= thr)
                        & (eval_frame[f"{alt2}_rank"] >= thr)
                    )
                    eval_frame[rank_col] = np.where(
                        active,
                        0.70 * eval_frame["base_rank"] + w1 * eval_frame[f"{alt1}_rank"] + w2 * eval_frame[f"{alt2}_rank"] + bonus,
                        eval_frame["base_rank"],
                    )
                    daily = daily_metrics(eval_frame, rank_col)
                    cand_full = summary(daily, None)
                    cand_63 = summary(daily, 63)
                    cand_20 = summary(daily, 20)
                    row = {
                        "mode": "pair_consensus",
                        "alt1": alt1,
                        "alt2": alt2,
                        "threshold": thr,
                        "alpha": 0.0,
                        "bonus": bonus,
                        "w1": w1,
                        "w2": w2,
                        "full_rank_ic_delta": cand_full["rank_ic"] - base_full["rank_ic"],
                        "full_top1_delta": cand_full["top1"] - base_full["top1"],
                        "full_top5_delta": cand_full["top5"] - base_full["top5"],
                        "recent63_top1_delta": cand_63["top1"] - base_63["top1"],
                        "recent63_top5_delta": cand_63["top5"] - base_63["top5"],
                        "recent20_top1_delta": cand_20["top1"] - base_20["top1"],
                        "recent20_top5_delta": cand_20["top5"] - base_20["top5"],
                        "recent20_rank_ic_delta": cand_20["rank_ic"] - base_20["rank_ic"],
                        "score": score_candidate(base_full, base_63, base_20, cand_full, cand_63, cand_20),
                    }
                    row["pass_basic"] = bool(
                        row["recent20_top5_delta"] >= 0.00010
                        and row["recent20_top1_delta"] >= 0.0
                        and row["recent63_top5_delta"] >= -0.00010
                        and row["full_rank_ic_delta"] >= -0.00035
                    )
                    results.append(row)

    result = pd.DataFrame(results).sort_values(
        ["pass_basic", "score", "recent20_top1_delta", "recent20_top5_delta", "full_rank_ic_delta"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(REPORT_DIR / "consensus_rerank_results.csv", index=False, encoding="utf-8-sig")

    best = result.iloc[0].to_dict() if not result.empty else None
    asset_stats = None
    if not result.empty and bool(result.iloc[0]["pass_basic"]) and float(result.iloc[0]["score"]) > 0:
        asset_stats = write_candidate(scores, result.iloc[0])

    summary_obj = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_consensus_rerank_scan",
        "base_table": BASE_TABLE,
        "sources": SOURCES,
        "rows": int(len(result)),
        "pass_basic_count": int(result["pass_basic"].sum()) if not result.empty else 0,
        "best": best,
        "asset_stats": asset_stats,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "research_prediction_table_write_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "consensus_rerank_summary.json").write_text(
        json.dumps(summary_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
