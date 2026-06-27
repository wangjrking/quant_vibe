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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_recent_top_condblend_scan_20260627"
LABEL = "executable_3d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_3d_rankic_recent_focus_20260625_executable_3d_open_return_research"
CANDIDATES = {
    "proxy_bestmix": "stock_predict_data_model_agent_3d_proxy10d5d_bestmix_20260623_executable_3d_open_return_research",
    "proxy_refine": "stock_predict_data_model_agent_3d_proxy10d5d_bestmix_refine_20260623_executable_3d_open_return_research",
    "guarded": "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_research",
    "stability": "stock_predict_data_model_agent_3d_stability_anchor_fast_20260624_executable_3d_open_return_research",
}
TARGET_TABLE = "stock_predict_data_model_agent_3d_recent_top_condblend_best_20260627_executable_3d_open_return_research"

FEATURES = ["base_score_std", "cand_score_std", "base_top20_gap", "cand_top20_gap", "rank_corr", "top20_overlap", "mean_abs_rank_gap"]
QUANTILES = [0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 0.95]
ALPHAS = [0.25, 0.50, 0.70, 1.00]


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
            part = pd.read_sql_query(
                f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
                conn,
            ).rename(columns={"pred_prob": f"{alias}_score"})
            out = out.merge(part, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
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


def daily_metrics(eval_frame: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    rows = []
    valid = eval_frame.dropna(subset=[rank_col])
    for trade_date, group in valid.groupby("trade_date", sort=True):
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


def summarize(daily: pd.DataFrame, window: int | None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {col: float(sub[col].mean()) for col in ["rank_ic", "top1", "top5", "top10", "top20"]}


def daily_features(scores: pd.DataFrame, alias: str) -> pd.DataFrame:
    rows = []
    score_col = f"{alias}_score"
    rank_col = f"{alias}_rank"
    for trade_date, group in scores.dropna(subset=[rank_col]).groupby("trade_date", sort=True):
        top20_base = set(group.nlargest(min(20, len(group)), "base_rank")["stock_code"])
        top20_cand = set(group.nlargest(min(20, len(group)), rank_col)["stock_code"])
        rows.append(
            {
                "trade_date": trade_date,
                "base_score_std": float(group["base_score"].std()),
                "cand_score_std": float(group[score_col].std()),
                "base_top20_gap": float(group["base_rank"].nlargest(20).mean() - group["base_rank"].nlargest(50).mean()),
                "cand_top20_gap": float(group[rank_col].nlargest(20).mean() - group[rank_col].nlargest(50).mean()),
                "rank_corr": float(group["base_rank"].corr(group[rank_col])),
                "top20_overlap": float(len(top20_base & top20_cand) / max(1, len(top20_base | top20_cand))),
                "mean_abs_rank_gap": float((group["base_rank"] - group[rank_col]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def objective(base_full: dict[str, float], base_63: dict[str, float], base_20: dict[str, float], cur_full: dict[str, float], cur_63: dict[str, float], cur_20: dict[str, float]) -> float:
    return (
        2.6 * (cur_20["top1"] - base_20["top1"])
        + 2.2 * (cur_20["top5"] - base_20["top5"])
        + 1.2 * (cur_63["top1"] - base_63["top1"])
        + 1.0 * (cur_63["top5"] - base_63["top5"])
        + 1.0 * (cur_full["rank_ic"] - base_full["rank_ic"])
        + 0.4 * (cur_full["top5"] - base_full["top5"])
    )


def build_candidate(scores: pd.DataFrame, feature_frame: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    alias = str(best["candidate"])
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    alpha = float(best["alpha"])
    feature_map = feature_frame.set_index("trade_date")[feature]
    active_dates = set(feature_map.index[feature_map <= threshold] if op == "<=" else feature_map.index[feature_map >= threshold])
    out = scores[["trade_date", "stock_code", "base_rank", f"{alias}_rank"]].copy()
    use_cand = out["trade_date"].isin(active_dates) & out[f"{alias}_rank"].notna()
    out["pred_prob"] = np.where(use_cand, (1.0 - alpha) * out["base_rank"] + alpha * out[f"{alias}_rank"], out["base_rank"])
    out["score_formula"] = np.where(
        use_cand,
        f"if {feature} {op} {threshold:.12g}: {(1.0-alpha):.2f}*base_rank + {alpha:.2f}*{alias}_rank",
        "base_rank",
    )
    return out


def write_candidate(scores: pd.DataFrame, feature_frame: pd.DataFrame, best: pd.Series) -> dict[str, object]:
    out = build_candidate(scores, feature_frame, best)
    candidate_rank_col = f"{best['candidate']}_rank"
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[["trade_date", "stock_code", "pred_prob", "base_rank", candidate_rank_col, "score_formula"]].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_3d_recent_top_condblend_best_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
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
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    base_daily = daily_metrics(eval_frame, "base_rank")
    base_full = summarize(base_daily, None)
    base_63 = summarize(base_daily, 63)
    base_20 = summarize(base_daily, 20)
    rows = []
    feature_frames: dict[str, pd.DataFrame] = {}

    for alias in CANDIDATES:
        ff = daily_features(scores, alias)
        feature_frames[alias] = ff
        for feature in FEATURES:
            values = ff[feature].dropna()
            thresholds = sorted(set(float(values.quantile(q)) for q in QUANTILES))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active_dates = set(ff.loc[ff[feature] <= threshold, "trade_date"] if op == "<=" else ff.loc[ff[feature] >= threshold, "trade_date"])
                    if len(active_dates) < 5 or len(active_dates) > 220:
                        continue
                    for alpha in ALPHAS:
                        rank_col = f"tmp_{alias}_{feature}_{op}_{alpha}".replace("<", "l").replace(">", "g").replace("=", "e").replace(".", "_")
                        eval_frame[rank_col] = np.where(
                            eval_frame["trade_date"].isin(active_dates) & eval_frame[f"{alias}_rank"].notna(),
                            (1.0 - alpha) * eval_frame["base_rank"] + alpha * eval_frame[f"{alias}_rank"],
                            eval_frame["base_rank"],
                        )
                        daily = daily_metrics(eval_frame, rank_col)
                        cur_full = summarize(daily, None)
                        cur_63 = summarize(daily, 63)
                        cur_20 = summarize(daily, 20)
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
                            row["recent20_top1_delta"] >= 0
                            and row["recent20_top5_delta"] >= 0
                            and row["recent63_top5_delta"] >= -0.001
                            and row["full_rank_ic_delta"] >= -0.018
                        )
                        rows.append(row)
                        eval_frame.drop(columns=[rank_col], inplace=True)

    result = pd.DataFrame(rows).sort_values(
        ["pass_basic", "objective", "recent20_top1_delta", "recent20_top5_delta", "full_rank_ic_delta"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(REPORT_DIR / "recent_top_condblend_results.csv", index=False, encoding="utf-8-sig")
    for alias, ff in feature_frames.items():
        ff.to_csv(REPORT_DIR / f"{alias}_daily_features.csv", index=False, encoding="utf-8-sig")

    best = result.iloc[0].to_dict() if len(result) else None
    asset_stats = None
    if best and best["pass_basic"] and best["objective"] > 0:
        asset_stats = write_candidate(scores, feature_frames[str(best["candidate"])], result.iloc[0])

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_3d_recent_top_condblend_scan",
        "base_table": BASE_TABLE,
        "candidate_tables": CANDIDATES,
        "base_metrics": {"full": base_full, "recent63": base_63, "recent20": base_20},
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
    (REPORT_DIR / "recent_top_condblend_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
