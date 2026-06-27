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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_10d_regime_gate_scan_20260626"
LABEL = "executable_10d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_10d_risk_balanced_v5_shrink_20260625_executable_10d_open_return_research"
CANDIDATE_TABLE = "stock_predict_data_model_agent_10d_v5shrink_fixed4y_top20zero_gate_20260626_executable_10d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_10d_regime_gate_best_20260626_executable_10d_open_return_research"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        base = pd.read_sql_query(f"select trade_date, stock_code, pred_prob from {quote(BASE_TABLE)} order by trade_date, stock_code", conn)
        cand = pd.read_sql_query(f"select trade_date, stock_code, pred_prob from {quote(CANDIDATE_TABLE)} order by trade_date, stock_code", conn)
    base["trade_date"] = base["trade_date"].astype(str)
    base["stock_code"] = base["stock_code"].astype(str)
    cand["trade_date"] = cand["trade_date"].astype(str)
    cand["stock_code"] = cand["stock_code"].astype(str)
    scores = base.rename(columns={"pred_prob": "base_score"}).merge(
        cand.rename(columns={"pred_prob": "cand_score"}),
        on=["trade_date", "stock_code"],
        how="inner",
        validate="one_to_one",
    )
    scores["base_rank"] = scores.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    scores["cand_rank"] = scores.groupby("trade_date")["cand_score"].rank(method="average", pct=True)
    return scores


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
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summary(daily: pd.DataFrame, window: int | None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {col: float(sub[col].mean()) for col in ["rank_ic", "top1", "top5", "top10"]}


def objective(base_full: dict[str, float], base_63: dict[str, float], base_20: dict[str, float], cur_full: dict[str, float], cur_63: dict[str, float], cur_20: dict[str, float]) -> float:
    return (
        2.0 * (cur_20["top5"] - base_20["top5"])
        + 1.8 * (cur_20["top1"] - base_20["top1"])
        + 1.4 * (cur_63["top5"] - base_63["top5"])
        + 1.0 * (cur_63["rank_ic"] - base_63["rank_ic"])
        + 0.8 * (cur_full["rank_ic"] - base_full["rank_ic"])
        + 0.4 * (cur_full["top5"] - base_full["top5"])
    )


def write_candidate(scores: pd.DataFrame, best: pd.Series, daily_features: pd.DataFrame) -> dict[str, object]:
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    feature_map = daily_features.set_index("trade_date")[feature]
    active_dates = set(feature_map.index[feature_map <= threshold] if op == "<=" else feature_map.index[feature_map >= threshold])
    out = scores[["trade_date", "stock_code", "base_rank", "cand_rank"]].copy()
    out["pred_prob"] = np.where(out["trade_date"].isin(active_dates), out["cand_rank"], out["base_rank"])
    out["score_formula"] = f"if {feature} {op} {threshold:.12g} then cand_rank else base_rank"
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[["trade_date", "stock_code", "pred_prob", "base_rank", "cand_rank", "score_formula"]].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_10d_regime_gate_best_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
    return {
        "table": TARGET_TABLE,
        "formula": out["score_formula"].iloc[0],
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

    daily_features = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        daily_features.append(
            {
                "trade_date": trade_date,
                "base_score_std": float(group["base_score"].std()),
                "cand_score_std": float(group["cand_score"].std()),
                "base_top20_gap": float(group["base_rank"].nlargest(20).mean() - group["base_rank"].nlargest(50).mean()),
                "cand_top20_gap": float(group["cand_rank"].nlargest(20).mean() - group["cand_rank"].nlargest(50).mean()),
            }
        )
    daily_features = pd.DataFrame(daily_features)

    base_daily = daily_metrics(eval_frame, "base_rank")
    base_full = summary(base_daily, None)
    base_63 = summary(base_daily, 63)
    base_20 = summary(base_daily, 20)

    rows = []
    feature_names = ["base_score_std", "cand_score_std", "base_top20_gap", "cand_top20_gap"]
    quantiles = [0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90]
    for feature in feature_names:
        values = daily_features[feature]
        for threshold in sorted(set(float(values.quantile(q)) for q in quantiles)):
            for op in ["<=", ">="]:
                active_dates = set(daily_features.loc[daily_features[feature] <= threshold, "trade_date"]) if op == "<=" else set(daily_features.loc[daily_features[feature] >= threshold, "trade_date"])
                if len(active_dates) < 8 or len(active_dates) > 220:
                    continue
                rank_col = f"gate_{feature}_{op.replace('=','e').replace('<','l').replace('>','g')}_{abs(hash((feature, op, threshold))) % 100000}"
                eval_frame[rank_col] = np.where(eval_frame["trade_date"].isin(active_dates), eval_frame["cand_rank"], eval_frame["base_rank"])
                daily = daily_metrics(eval_frame, rank_col)
                cur_full = summary(daily, None)
                cur_63 = summary(daily, 63)
                cur_20 = summary(daily, 20)
                row = {
                    "feature": feature,
                    "op": op,
                    "threshold": threshold,
                    "active_days": len(active_dates),
                    "full_rank_ic_delta": cur_full["rank_ic"] - base_full["rank_ic"],
                    "full_top1_delta": cur_full["top1"] - base_full["top1"],
                    "full_top5_delta": cur_full["top5"] - base_full["top5"],
                    "recent63_rank_ic_delta": cur_63["rank_ic"] - base_63["rank_ic"],
                    "recent63_top1_delta": cur_63["top1"] - base_63["top1"],
                    "recent63_top5_delta": cur_63["top5"] - base_63["top5"],
                    "recent20_rank_ic_delta": cur_20["rank_ic"] - base_20["rank_ic"],
                    "recent20_top1_delta": cur_20["top1"] - base_20["top1"],
                    "recent20_top5_delta": cur_20["top5"] - base_20["top5"],
                }
                row["objective"] = objective(base_full, base_63, base_20, cur_full, cur_63, cur_20)
                row["pass_basic"] = bool(
                    row["full_rank_ic_delta"] >= 0.0010
                    and row["recent63_top5_delta"] >= 0.0010
                    and row["recent20_top5_delta"] >= -0.0030
                )
                rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["pass_basic", "objective", "full_rank_ic_delta", "recent63_top5_delta", "recent20_top5_delta"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(REPORT_DIR / "regime_gate_results.csv", index=False, encoding="utf-8-sig")

    best = result.iloc[0].to_dict() if not result.empty else None
    asset_stats = None
    if best and best["pass_basic"] and best["objective"] > 0:
        asset_stats = write_candidate(scores, result.iloc[0], daily_features)

    summary_obj = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_regime_gate_scan",
        "base_table": BASE_TABLE,
        "candidate_table": CANDIDATE_TABLE,
        "rows": int(len(result)),
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
    (REPORT_DIR / "regime_gate_summary.json").write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
