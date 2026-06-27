from __future__ import annotations

import hashlib
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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_recent_focus_recent_scan_20260626"

LABEL = "executable_1d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_1d_new5d_fixed4y_lowstd_gate_refined_20260626_executable_1d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_1d_recent_focus_recent_scan_20260626_executable_1d_open_return_research"

SOURCES = {
    "base": BASE_TABLE,
    "cond_v6": "stock_predict_data_model_agent_1d_new5d_condition_gate_v6_20260625_executable_1d_open_return_research",
    "broad_guard": "stock_predict_data_model_agent_1d_broad_stability_guard_20260625_executable_1d_open_return_research",
    "dynamic_guard": "stock_predict_data_model_agent_1d_dynamic_guard_20260625_executable_1d_open_return_research",
    "cross_v2": "stock_predict_data_model_agent_1d_cross_horizon_guard_v2_20260625_executable_1d_open_return_research",
    "risk_v7": "stock_predict_data_model_agent_1d_risk_balanced_v7_20260625_executable_1d_open_return_research",
    "formal1": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
}

FEATURES = [
    "base_score_std",
    "alt_score_std",
    "rank_corr",
    "mean_abs_rank_gap",
    "top10_overlap",
    "top20_overlap",
    "base_top1_top5_gap",
    "base_top5_top10_gap",
]
METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_source(conn: sqlite3.Connection, alias: str, table: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": f"{alias}_score"})


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        out = read_source(conn, "base", SOURCES["base"])
        for alias, table in SOURCES.items():
            if alias == "base":
                continue
            out = out.merge(
                read_source(conn, alias, table),
                on=["trade_date", "stock_code"],
                how="left",
                validate="one_to_one",
            )
    for alias in SOURCES:
        score_col = f"{alias}_score"
        out[score_col] = out[score_col].fillna(out["base_score"])
        out[f"{alias}_rank"] = out.groupby("trade_date")[score_col].rank(method="average", pct=True)
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def build_feature_table(scores: pd.DataFrame, alt_alias: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        base_top10 = top_set(group, "base_rank", 10)
        alt_top10 = top_set(group, f"{alt_alias}_rank", 10)
        base_top20 = top_set(group, "base_rank", 20)
        alt_top20 = top_set(group, f"{alt_alias}_rank", 20)
        rows.append(
            {
                "trade_date": trade_date,
                "base_score_std": float(group["base_score"].std()),
                "alt_score_std": float(group[f"{alt_alias}_score"].std()),
                "rank_corr": float(group["base_rank"].corr(group[f"{alt_alias}_rank"])),
                "mean_abs_rank_gap": float((group["base_rank"] - group[f"{alt_alias}_rank"]).abs().mean()),
                "top10_overlap": float(len(base_top10 & alt_top10) / max(1, len(base_top10 | alt_top10))),
                "top20_overlap": float(len(base_top20 & alt_top20) / max(1, len(base_top20 | alt_top20))),
                "base_top1_top5_gap": float(
                    group.nlargest(1, "base_rank")["base_rank"].mean()
                    - group.nlargest(5, "base_rank")["base_rank"].min()
                ),
                "base_top5_top10_gap": float(
                    group.nlargest(5, "base_rank")["base_rank"].mean()
                    - group.nlargest(10, "base_rank")["base_rank"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("trade_date")


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def source_daily_metrics(eval_frame: pd.DataFrame, alias: str) -> pd.DataFrame:
    rank_col = f"{alias}_rank"
    rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        rows.append(
            {
                "trade_date": trade_date,
                "source": alias,
                "rank_ic": float(group[rank_col].corr(label_rank)),
                "top1": top_mean(group, rank_col, 1),
                "top3": top_mean(group, rank_col, 3),
                "top5": top_mean(group, rank_col, 5),
                "top10": top_mean(group, rank_col, 10),
                "top20": top_mean(group, rank_col, 20),
                "top50": top_mean(group, rank_col, 50),
                "top_bottom": top_mean(group, rank_col, 50) - bottom_mean(group, rank_col, 50),
            }
        )
    return pd.DataFrame(rows).set_index("trade_date")


def summarize(daily: pd.DataFrame, trade_dates: list[str], window: int | None) -> dict[str, float]:
    keep = trade_dates if window is None else trade_dates[-window:]
    sub = daily.loc[keep]
    return {metric: float(sub[metric].mean()) for metric in METRICS}


def period_stats(candidate_daily: pd.DataFrame, base_daily: pd.DataFrame) -> tuple[int, float, float]:
    periods = {
        "2024H2": ("20240604", "20241231"),
        "2025H1": ("20250101", "20250630"),
        "2025H2": ("20250701", "20251231"),
        "2026YTD": ("20260101", "99999999"),
    }
    merged = candidate_daily.join(base_daily, lsuffix="_cand", rsuffix="_base")
    top5_deltas = []
    rankic_deltas = []
    for lo, hi in periods.values():
        sub = merged[(merged.index >= lo) & (merged.index <= hi)]
        top5_deltas.append(float((sub["top5_cand"] - sub["top5_base"]).mean()))
        rankic_deltas.append(float((sub["rank_ic_cand"] - sub["rank_ic_base"]).mean()))
    return int(sum(value > 0 for value in top5_deltas)), float(min(top5_deltas)), float(min(rankic_deltas))


def objective(row: dict[str, float | int]) -> float:
    return (
        1.8 * row["recent63_top1_delta"]
        + 1.4 * row["recent63_top5_delta"]
        + 1.8 * row["recent20_top5_delta"]
        + 1.0 * row["recent20_rank_ic_delta"]
        + 0.5 * row["recent63_rank_ic_delta"]
        + 0.25 * row["full_rank_ic_delta"]
        + 0.5 * row["full_top5_delta"]
        + 0.006 * row["positive_top5_periods"]
        - 0.006 * row["active_ratio"]
    )


def scan(scores: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    source_daily = {alias: source_daily_metrics(eval_frame, alias) for alias in SOURCES}
    base_daily = source_daily["base"]
    trade_dates = sorted(base_daily.index.tolist())
    all_results = []
    feature_tables: dict[str, pd.DataFrame] = {}
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95]
    for alt in ["cond_v6", "broad_guard", "dynamic_guard", "cross_v2", "risk_v7", "formal1"]:
        feature_table = build_feature_table(scores, alt)
        feature_tables[alt] = feature_table
        alt_daily = source_daily[alt]
        for feature in FEATURES:
            values = feature_table[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if values.nunique() < 3:
                continue
            thresholds = sorted(set(float(values.quantile(q)) for q in quantiles))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active = feature_table[feature] <= threshold if op == "<=" else feature_table[feature] >= threshold
                    active_dates = set(active[active].index.tolist())
                    if len(active_dates) < 4 or len(active_dates) > int(0.22 * len(feature_table)):
                        continue
                    recent63_active = len(set(trade_dates[-63:]) & active_dates)
                    recent20_active = len(set(trade_dates[-20:]) & active_dates)
                    if recent63_active < 2 or recent20_active < 1:
                        continue
                    candidate = base_daily.copy()
                    overlap = [trade_date for trade_date in active_dates if trade_date in alt_daily.index]
                    if overlap:
                        candidate.loc[overlap, :] = alt_daily.loc[overlap, candidate.columns]
                    full_cand = summarize(candidate, trade_dates, None)
                    full_base = summarize(base_daily, trade_dates, None)
                    recent63_cand = summarize(candidate, trade_dates, 63)
                    recent63_base = summarize(base_daily, trade_dates, 63)
                    recent20_cand = summarize(candidate, trade_dates, 20)
                    recent20_base = summarize(base_daily, trade_dates, 20)
                    positive_top5_periods, min_period_top5_delta, min_period_rankic_delta = period_stats(candidate, base_daily)
                    row = {
                        "alt_source": alt,
                        "feature": feature,
                        "op": op,
                        "threshold": float(threshold),
                        "active_days": len(active_dates),
                        "active_ratio": len(active_dates) / len(feature_table),
                        "recent63_active_days": recent63_active,
                        "recent20_active_days": recent20_active,
                        "full_rank_ic_delta": full_cand["rank_ic"] - full_base["rank_ic"],
                        "full_top1_delta": full_cand["top1"] - full_base["top1"],
                        "full_top5_delta": full_cand["top5"] - full_base["top5"],
                        "recent63_rank_ic_delta": recent63_cand["rank_ic"] - recent63_base["rank_ic"],
                        "recent63_top1_delta": recent63_cand["top1"] - recent63_base["top1"],
                        "recent63_top5_delta": recent63_cand["top5"] - recent63_base["top5"],
                        "recent20_rank_ic_delta": recent20_cand["rank_ic"] - recent20_base["rank_ic"],
                        "recent20_top1_delta": recent20_cand["top1"] - recent20_base["top1"],
                        "recent20_top5_delta": recent20_cand["top5"] - recent20_base["top5"],
                        "positive_top5_periods": positive_top5_periods,
                        "min_period_top5_delta": min_period_top5_delta,
                        "min_period_rank_ic_delta": min_period_rankic_delta,
                    }
                    row["pass_basic"] = bool(
                        row["full_rank_ic_delta"] >= -0.00020
                        and row["recent63_top5_delta"] >= 0.0
                        and row["recent20_top5_delta"] >= 0.0
                        and row["recent63_top1_delta"] >= 0.0
                        and row["recent20_rank_ic_delta"] >= 0.0
                        and row["min_period_top5_delta"] >= 0.0
                        and row["positive_top5_periods"] >= 3
                    )
                    row["objective"] = objective(row)
                    all_results.append(row)
    result = pd.DataFrame(all_results)
    if not result.empty:
        result = result.sort_values(
            ["pass_basic", "objective", "recent20_rank_ic_delta", "recent63_top5_delta"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    source_daily_frame = pd.concat(
        [daily.assign(source=alias).reset_index() for alias, daily in source_daily.items()],
        ignore_index=True,
    )
    return result, feature_tables, source_daily_frame


def write_candidate(scores: pd.DataFrame, best: pd.Series, feature_tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    alt = str(best["alt_source"])
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    feature_table = feature_tables[alt]
    active = feature_table[feature] <= threshold if op == "<=" else feature_table[feature] >= threshold
    active_dates = set(active[active].index.tolist())
    out = scores[
        ["trade_date", "stock_code", "base_score", "base_rank", f"{alt}_score", f"{alt}_rank"]
    ].copy()
    out["guard_active"] = out["trade_date"].isin(active_dates)
    out["pred_prob"] = np.where(out["guard_active"], out[f"{alt}_rank"], out["base_rank"])
    out["score_formula"] = f"if {feature} {op} {threshold:.12g} then {alt}_rank else base_rank"
    keep = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "base_score",
        "base_rank",
        f"{alt}_score",
        f"{alt}_rank",
        "guard_active",
        "score_formula",
    ]
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[keep].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        short = hashlib.sha1(TARGET_TABLE.encode("utf-8")).hexdigest()[:12]
        conn.execute(f"create index if not exists idx_{short}_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{short}_date_pred on {quote(TARGET_TABLE)}(trade_date, pred_prob desc)")
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
        "formula": out["score_formula"].iloc[0],
        "active_days_full": int(out.groupby("trade_date")["guard_active"].first().sum()),
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
    result, feature_tables, source_daily = scan(scores, labels)
    result.to_csv(REPORT_DIR / "recent_focus_recent_scan_results.csv", index=False, encoding="utf-8-sig")
    source_daily.to_csv(REPORT_DIR / "recent_focus_recent_source_daily.csv", index=False, encoding="utf-8-sig")
    best = result.iloc[0].to_dict() if not result.empty else None
    asset_stats = None
    if not result.empty and bool(result.iloc[0]["pass_basic"]) and float(result.iloc[0]["objective"]) > 0:
        asset_stats = write_candidate(scores, result.iloc[0], feature_tables)
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_recent_focus_recent_scan",
        "base_table": BASE_TABLE,
        "sources": SOURCES,
        "best": best,
        "asset_stats": asset_stats,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "recent_focus_recent_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
