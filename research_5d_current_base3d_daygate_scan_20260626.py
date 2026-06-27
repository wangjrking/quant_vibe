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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_current_base3d_daygate_scan_20260626"
LABEL = "executable_5d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_5d_v6shrink_fixed4y_rankcorr_gate_refined_20260626_executable_5d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_rankic_recent_focus_20260625_executable_3d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_current_base3d_daygate_best_20260626_executable_5d_open_return_research"

START_DATES = ["20241001", "20250101", "20250301", "20250501", "20260101"]
QUANTILES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 0.95]
FEATURES = ["base_top5_mean_rank3", "base_top10_mean_rank3", "rank_corr", "mean_abs_rank_gap", "base_score_std", "aux3_score_std"]


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
        aux3 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(AUX3_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "aux3_score"})
    base["trade_date"] = base["trade_date"].astype(str)
    base["stock_code"] = base["stock_code"].astype(str)
    aux3["trade_date"] = aux3["trade_date"].astype(str)
    aux3["stock_code"] = aux3["stock_code"].astype(str)
    scores = base.merge(aux3, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    scores["base_rank"] = scores.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    scores["aux3_rank"] = scores.groupby("trade_date")["aux3_score"].rank(method="average", pct=True)
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


def build_daily_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        ordered_base = group.sort_values("base_score", ascending=False, kind="mergesort")
        rows.append(
            {
                "trade_date": trade_date,
                "base_top5_mean_rank3": float(ordered_base.head(5)["aux3_rank"].mean()),
                "base_top10_mean_rank3": float(ordered_base.head(10)["aux3_rank"].mean()),
                "rank_corr": float(group["base_rank"].corr(group["aux3_rank"])),
                "mean_abs_rank_gap": float((group["base_rank"] - group["aux3_rank"]).abs().mean()),
                "base_score_std": float(group["base_score"].std()),
                "aux3_score_std": float(group["aux3_score"].std()),
            }
        )
    return pd.DataFrame(rows)


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
                "top20": top_mean(group, rank_col, 20),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summary(daily: pd.DataFrame, window: int | None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {col: float(sub[col].mean()) for col in ["rank_ic", "top1", "top5", "top10", "top20"]}


def merge_daily(base_daily: pd.DataFrame, aux_daily: pd.DataFrame, active_dates: set[str]) -> pd.DataFrame:
    merged = base_daily.merge(aux_daily, on="trade_date", how="inner", suffixes=("_base", "_aux"), validate="one_to_one")
    use_aux = merged["trade_date"].isin(active_dates)
    out = pd.DataFrame({"trade_date": merged["trade_date"]})
    for col in ["rank_ic", "top1", "top5", "top10", "top20"]:
        out[col] = np.where(use_aux, merged[f"{col}_aux"], merged[f"{col}_base"])
    return out


def objective(base_full: dict[str, float], base_63: dict[str, float], base_20: dict[str, float], cur_full: dict[str, float], cur_63: dict[str, float], cur_20: dict[str, float]) -> float:
    return (
        2.8 * (cur_20["top1"] - base_20["top1"])
        + 2.2 * (cur_20["top5"] - base_20["top5"])
        + 1.2 * (cur_63["top1"] - base_63["top1"])
        + 1.0 * (cur_63["top5"] - base_63["top5"])
        + 1.0 * (cur_full["rank_ic"] - base_full["rank_ic"])
        + 0.5 * (cur_full["top5"] - base_full["top5"])
    )


def build_candidate(scores: pd.DataFrame, daily_features: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    start_date = str(best["start_date"])
    eligible = daily_features["trade_date"] >= start_date
    cond = daily_features[feature] <= threshold if op == "<=" else daily_features[feature] >= threshold
    active_dates = set(daily_features.loc[eligible & cond, "trade_date"])
    out = scores[["trade_date", "stock_code", "base_rank", "aux3_rank"]].copy()
    out["pred_prob"] = np.where(out["trade_date"].isin(active_dates), out["aux3_rank"], out["base_rank"])
    out["score_formula"] = np.where(
        out["trade_date"].isin(active_dates),
        f"if {feature} {op} {threshold:.12g} and trade_date >= {start_date}: aux3_rank",
        "base_rank",
    )
    return out


def write_candidate(scores: pd.DataFrame, daily_features: pd.DataFrame, best: pd.Series) -> dict[str, object]:
    out = build_candidate(scores, daily_features, best)
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[["trade_date", "stock_code", "pred_prob", "base_rank", "aux3_rank", "score_formula"]].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_5d_current_base3d_daygate_best_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
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
    return {
        "table": TARGET_TABLE,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    daily_features = build_daily_features(scores)
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    base_daily = daily_metrics(eval_frame, "base_rank")
    aux_daily = daily_metrics(eval_frame, "aux3_rank")
    base_full = summary(base_daily, None)
    base_63 = summary(base_daily, 63)
    base_20 = summary(base_daily, 20)

    rows = []
    for start_date in START_DATES:
        for feature in FEATURES:
            train_values = daily_features.loc[daily_features["trade_date"] >= start_date, feature].dropna()
            thresholds = sorted(set(float(train_values.quantile(q)) for q in QUANTILES))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    eligible = daily_features["trade_date"] >= start_date
                    cond = daily_features[feature] <= threshold if op == "<=" else daily_features[feature] >= threshold
                    active_dates = set(daily_features.loc[eligible & cond, "trade_date"])
                    if len(active_dates) < 5 or len(active_dates) > 260:
                        continue
                    daily = merge_daily(base_daily, aux_daily, active_dates)
                    cur_full = summary(daily, None)
                    cur_63 = summary(daily, 63)
                    cur_20 = summary(daily, 20)
                    row = {
                        "start_date": start_date,
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
                        row["recent20_top1_delta"] >= 0.0
                        and row["recent20_top5_delta"] >= 0.0
                        and row["recent63_top5_delta"] >= -0.001
                        and row["full_rank_ic_delta"] >= -0.01
                    )
                    rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["pass_basic", "objective", "recent20_top1_delta", "recent20_top5_delta", "full_rank_ic_delta"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(REPORT_DIR / "current_base3d_daygate_results.csv", index=False, encoding="utf-8-sig")
    daily_features.to_csv(REPORT_DIR / "daily_features.csv", index=False, encoding="utf-8-sig")

    best = result.iloc[0].to_dict() if len(result) else None
    asset_stats = None
    if best and best["pass_basic"] and best["objective"] > 0:
        asset_stats = write_candidate(scores, daily_features, result.iloc[0])

    summary_obj = {
        "generated_at": now_iso(),
        "scope": "research_only_5d_current_base3d_daygate_scan",
        "base_table": BASE_TABLE,
        "aux3_table": AUX3_TABLE,
        "base_metrics": {"full": base_full, "recent63": base_63, "recent20": base_20},
        "aux3_metrics": {"full": summary(aux_daily, None), "recent63": summary(aux_daily, 63), "recent20": summary(aux_daily, 20)},
        "rows": int(len(result)),
        "pass_basic_count": int(result["pass_basic"].sum()) if len(result) else 0,
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
    (REPORT_DIR / "current_base3d_daygate_summary.json").write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
