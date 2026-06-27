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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_recent_top_condblend_extend_latest_20260626"
LABEL = "executable_5d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_5d_v6shrink_fixed4y_rankcorr_gate_refined_20260626_executable_5d_open_return_research"
CANDIDATE_TABLE = "stock_predict_data_model_agent_5d_daily_gate_d3d1_20260623_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_recent_top_condblend_latest_20260626_executable_5d_open_return_research"

THRESHOLD = 0.002730748225013513


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        base = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(BASE_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "base_score"})
        cand = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(CANDIDATE_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "cand_score"})
    base["trade_date"] = base["trade_date"].astype(str)
    base["stock_code"] = base["stock_code"].astype(str)
    cand["trade_date"] = cand["trade_date"].astype(str)
    cand["stock_code"] = cand["stock_code"].astype(str)
    scores = base.merge(cand, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    scores["base_rank"] = scores.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    scores["cand_rank"] = scores.groupby("trade_date")["cand_score"].rank(method="average", pct=True)
    return scores


def build_daily_candidate_gap(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.dropna(subset=["cand_rank"]).groupby("trade_date", sort=True):
        rows.append(
            {
                "trade_date": trade_date,
                "cand_top20_gap": float(group["cand_rank"].nlargest(20).mean() - group["cand_rank"].nlargest(50).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_extended_scores(scores: pd.DataFrame, daily_gap: pd.DataFrame) -> pd.DataFrame:
    active_dates = set(daily_gap.loc[daily_gap["cand_top20_gap"] <= THRESHOLD, "trade_date"])
    out = scores[["trade_date", "stock_code", "base_rank", "cand_rank"]].copy()
    use_candidate = out["trade_date"].isin(active_dates) & out["cand_rank"].notna()
    out["pred_prob"] = np.where(use_candidate, out["cand_rank"], out["base_rank"])
    out["score_formula"] = np.where(
        use_candidate,
        f"cand_top20_gap <= {THRESHOLD:.14g}: daily_gate_d3d1_rank",
        "base_rank_fallback",
    )
    return out[["trade_date", "stock_code", "pred_prob", "base_rank", "cand_rank", "score_formula"]]


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
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else 0.0


def eval_table(scores: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    ev = scores[["trade_date", "stock_code", "pred_prob"]].merge(labels, on=["trade_date", "stock_code"], how="inner")
    rows = []
    for trade_date, group in ev.groupby("trade_date", sort=True):
        rank = group["pred_prob"].rank(method="average", pct=True)
        label_rank = group[LABEL].rank(method="average", pct=True)
        group = group.assign(_rank=rank)
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(rank.corr(label_rank)),
                "top1": top_mean(group, "_rank", 1),
                "top5": top_mean(group, "_rank", 5),
                "top10": top_mean(group, "_rank", 10),
                "top20": top_mean(group, "_rank", 20),
            }
        )
    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    return {
        "days": int(len(daily)),
        "min_eval_date": str(daily["trade_date"].min()),
        "max_eval_date": str(daily["trade_date"].max()),
        "full": daily[["rank_ic", "top1", "top5", "top10", "top20"]].mean().to_dict(),
        "recent63": daily.tail(63)[["rank_ic", "top1", "top5", "top10", "top20"]].mean().to_dict(),
        "recent20": daily.tail(20)[["rank_ic", "top1", "top5", "top10", "top20"]].mean().to_dict(),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    daily_gap = build_daily_candidate_gap(scores)
    out = build_extended_scores(scores, daily_gap)

    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_5d_recent_top_condblend_latest_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 5"
        ).fetchall()

    labels = load_labels(str(out["trade_date"].min()), str(out["trade_date"].max()))
    base_scores = scores[["trade_date", "stock_code", "base_rank"]].rename(columns={"base_rank": "pred_prob"})
    new_eval = eval_table(out, labels)
    base_eval = eval_table(base_scores, labels)
    delta = {
        scope: {metric: new_eval[scope][metric] - base_eval[scope][metric] for metric in new_eval[scope]}
        for scope in ["full", "recent63", "recent20"]
    }

    candidate_coverage_max = str(daily_gap["trade_date"].max())
    active_days = int((daily_gap["cand_top20_gap"] <= THRESHOLD).sum())
    fallback_days = int(out.loc[out["cand_rank"].isna(), "trade_date"].nunique())
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_5d_recent_top_condblend_extend_latest",
        "target_table": TARGET_TABLE,
        "base_table": BASE_TABLE,
        "candidate_table": CANDIDATE_TABLE,
        "formula": f"candidate dates: if cand_top20_gap <= {THRESHOLD:.14g} then daily_gate_d3d1_rank else base_rank; candidate-missing dates: base_rank",
        "coverage": {
            "row_count": int(row[0]),
            "min_trade_date": str(row[1]),
            "max_trade_date": str(row[2]),
            "trade_days": int(row[3]),
            "null_pred_prob": int(row[4] or 0),
            "duplicate_key_groups": int(dup),
            "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
        },
        "candidate_coverage_max": candidate_coverage_max,
        "active_candidate_days": active_days,
        "base_fallback_days_due_candidate_missing": fallback_days,
        "eval": {
            "new": new_eval,
            "base": base_eval,
            "new_vs_base_delta": delta,
        },
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
    daily_gap.to_csv(REPORT_DIR / "candidate_daily_gap.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "extend_latest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
