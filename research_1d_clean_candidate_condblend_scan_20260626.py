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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_clean_candidate_condblend_scan_20260626"
LABEL = "executable_1d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_1d_fixed4y_refined_blend15_20260626_executable_1d_open_return_research"
CANDIDATES = {
    "risk_v7": "stock_predict_data_model_agent_1d_risk_balanced_v7_20260625_executable_1d_open_return_research",
    "dense_guard_smallcap_vr_neutralfill": "stock_predict_data_model_agent_1d_dense_guard_smallcap_vr_neutralfill_20260623_executable_1d_open_return_research",
}
TARGET_TABLE = "stock_predict_data_model_agent_1d_clean_candidate_condblend_best_20260626_executable_1d_open_return_research"

FEATURE_NAMES = [
    "base_score_std",
    "cand_score_std",
    "base_top20_gap",
    "cand_top20_gap",
    "rank_corr",
    "top20_overlap",
    "mean_abs_rank_gap",
]
QUANTILES = [0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 0.95]
ALPHAS = [0.15, 0.25, 0.35, 0.50, 0.70]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        out = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(BASE_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "base_score"})
        for alias, table in CANDIDATES.items():
            frame = pd.read_sql_query(
                f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
                conn,
            ).rename(columns={"pred_prob": f"{alias}_score"})
            out = out.merge(frame, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    out["trade_date"] = out["trade_date"].astype(str)
    out["stock_code"] = out["stock_code"].astype(str)
    out["base_rank"] = out.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    for alias in CANDIDATES:
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
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else 0.0


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else 0.0


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


def build_daily_features(scores: pd.DataFrame, alias: str) -> pd.DataFrame:
    rows = []
    cand_score_col = f"{alias}_score"
    cand_rank_col = f"{alias}_rank"
    for trade_date, group in scores.groupby("trade_date", sort=True):
        top20_base = set(group.nlargest(min(20, len(group)), "base_rank")["stock_code"])
        top20_cand = set(group.nlargest(min(20, len(group)), cand_rank_col)["stock_code"])
        rows.append(
            {
                "trade_date": trade_date,
                "base_score_std": float(group["base_score"].std()),
                "cand_score_std": float(group[cand_score_col].std()),
                "base_top20_gap": float(group["base_rank"].nlargest(20).mean() - group["base_rank"].nlargest(50).mean()),
                "cand_top20_gap": float(group[cand_rank_col].nlargest(20).mean() - group[cand_rank_col].nlargest(50).mean()),
                "rank_corr": float(group["base_rank"].corr(group[cand_rank_col])),
                "top20_overlap": float(len(top20_base & top20_cand) / max(1, len(top20_base | top20_cand))),
                "mean_abs_rank_gap": float((group["base_rank"] - group[cand_rank_col]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def merge_daily_by_gate_and_alpha(base_daily: pd.DataFrame, cand_daily: pd.DataFrame, active_dates: set[str], alpha: float) -> pd.DataFrame:
    merged = base_daily.merge(cand_daily, on="trade_date", how="inner", suffixes=("_base", "_cand"), validate="one_to_one")
    use_cand = merged["trade_date"].isin(active_dates)
    out = pd.DataFrame({"trade_date": merged["trade_date"]})
    for col in ["rank_ic", "top1", "top5", "top10", "top20", "top_bottom"]:
        out[col] = np.where(
            use_cand,
            (1.0 - alpha) * merged[f"{col}_base"] + alpha * merged[f"{col}_cand"],
            merged[f"{col}_base"],
        )
    return out


def objective(
    base_full: dict[str, float],
    base_63: dict[str, float],
    base_20: dict[str, float],
    cur_full: dict[str, float],
    cur_63: dict[str, float],
    cur_20: dict[str, float],
) -> float:
    return (
        2.4 * (cur_20["top1"] - base_20["top1"])
        + 2.0 * (cur_20["top5"] - base_20["top5"])
        + 1.2 * (cur_63["top1"] - base_63["top1"])
        + 1.0 * (cur_63["top5"] - base_63["top5"])
        + 0.8 * (cur_full["rank_ic"] - base_full["rank_ic"])
        + 0.5 * (cur_full["top5"] - base_full["top5"])
        + 0.2 * (cur_20["rank_ic"] - base_20["rank_ic"])
    )


def write_candidate(scores: pd.DataFrame, best: pd.Series, daily_features: pd.DataFrame) -> dict[str, object]:
    alias = str(best["candidate"])
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    alpha = float(best["alpha"])
    feature_map = daily_features.set_index("trade_date")[feature]
    active_dates = set(feature_map.index[feature_map <= threshold] if op == "<=" else feature_map.index[feature_map >= threshold])
    out = scores[["trade_date", "stock_code", "base_rank", f"{alias}_rank"]].copy()
    out["pred_prob"] = np.where(
        out["trade_date"].isin(active_dates),
        (1.0 - alpha) * out["base_rank"] + alpha * out[f"{alias}_rank"],
        out["base_rank"],
    )
    out["score_formula"] = f"if {feature} {op} {threshold:.12g} then {(1.0-alpha):.2f}*base_rank + {alpha:.2f}*{alias}_rank else base_rank"

    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[["trade_date", "stock_code", "pred_prob", "base_rank", f"{alias}_rank", "score_formula"]].to_sql(
            TARGET_TABLE,
            conn,
            if_exists="replace",
            index=False,
        )
        conn.execute(f"create index if not exists idx_1d_clean_condblend_best_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
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

    base_daily = daily_metrics(eval_frame, "base_rank")
    base_full = summary(base_daily, None)
    base_63 = summary(base_daily, 63)
    base_20 = summary(base_daily, 20)

    rows = []
    feature_dumps: dict[str, pd.DataFrame] = {}
    candidate_daily_map: dict[str, pd.DataFrame] = {}
    for alias in CANDIDATES:
        daily_features = build_daily_features(scores, alias)
        feature_dumps[alias] = daily_features
        candidate_daily = daily_metrics(eval_frame, f"{alias}_rank")
        candidate_daily_map[alias] = candidate_daily
        for feature in FEATURE_NAMES:
            values = daily_features[feature].dropna()
            thresholds = sorted(set(float(values.quantile(q)) for q in QUANTILES))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active_dates = set(
                        daily_features.loc[daily_features[feature] <= threshold, "trade_date"]
                        if op == "<="
                        else daily_features.loc[daily_features[feature] >= threshold, "trade_date"]
                    )
                    if len(active_dates) < 6 or len(active_dates) > 180:
                        continue
                    for alpha in ALPHAS:
                        daily = merge_daily_by_gate_and_alpha(base_daily, candidate_daily, active_dates, alpha)
                        cur_full = summary(daily, None)
                        cur_63 = summary(daily, 63)
                        cur_20 = summary(daily, 20)
                        row = {
                            "candidate": alias,
                            "feature": feature,
                            "op": op,
                            "threshold": threshold,
                            "alpha": alpha,
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
                            and row["recent63_top5_delta"] >= 0.0
                            and row["full_rank_ic_delta"] >= -0.00025
                        )
                        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["pass_basic", "objective", "recent20_top1_delta", "recent20_top5_delta", "full_rank_ic_delta"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(REPORT_DIR / "clean_candidate_condblend_results.csv", index=False, encoding="utf-8-sig")
    for alias, dump in feature_dumps.items():
        dump.to_csv(REPORT_DIR / f"{alias}_daily_features.csv", index=False, encoding="utf-8-sig")

    best = result.iloc[0].to_dict() if not result.empty else None
    asset_stats = None
    if best and best["pass_basic"] and best["objective"] > 0:
        asset_stats = write_candidate(scores, result.iloc[0], feature_dumps[str(best["candidate"])])

    summary_obj = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_clean_candidate_condblend_scan",
        "base_table": BASE_TABLE,
        "candidate_tables": CANDIDATES,
        "base_metrics": {
            "full": base_full,
            "recent63": base_63,
            "recent20": base_20,
        },
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
    (REPORT_DIR / "clean_candidate_condblend_summary.json").write_text(
        json.dumps(summary_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
