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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_3d_gate_search_promotion_safe_20260626"

LABEL = "executable_3d_open_return"
FORMAL_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
BASE_TABLE = "stock_predict_data_model_agent_3d_rankic_balanced_v5_20260625_executable_3d_open_return_research"
ALT_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_3d_scoreblend_fold08_20260626"
    / "blend40_60"
    / "fold_predictions"
    / "fold08.parquet"
)

FEATURES = ["base_std", "rank_corr", "mean_abs_rank_gap", "top10_overlap", "top20_overlap", "alt_std"]
METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_scores(conn: sqlite3.Connection, table: str, score_name: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": score_name})


def load_base_alt_formal() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        formal = read_scores(conn, FORMAL_TABLE, "formal_score")
        base = read_scores(conn, BASE_TABLE, "base_score")
    alt = pd.read_parquet(ALT_PATH, columns=["trade_date", "stock_code", "pred_prob"])
    alt["trade_date"] = alt["trade_date"].astype(str)
    alt["stock_code"] = alt["stock_code"].astype(str)
    alt = alt.rename(columns={"pred_prob": "alt_score"})

    frame = formal.merge(base, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(alt, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame["formal_rank"] = frame.groupby("trade_date")["formal_score"].rank(method="average", pct=True)
    frame["base_rank"] = frame.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    alt_mask = frame["alt_score"].notna()
    frame.loc[alt_mask, "alt_rank"] = frame.loc[alt_mask].groupby("trade_date")["alt_score"].rank(method="average", pct=True)
    return frame


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def build_daily_tables(scores: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    daily_base = []
    daily_formal = []
    daily_alt = []
    feature_rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        for name, col in [("base", "base_rank"), ("formal", "formal_rank"), ("alt", "alt_rank")]:
            if name == "alt" and group["alt_rank"].isna().all():
                row = {m: np.nan for m in METRICS}
            else:
                row = {
                    "rank_ic": float(group[col].corr(label_rank)),
                    "top1": top_mean(group, col, 1),
                    "top3": top_mean(group, col, 3),
                    "top5": top_mean(group, col, 5),
                    "top10": top_mean(group, col, 10),
                    "top20": top_mean(group, col, 20),
                    "top50": top_mean(group, col, 50),
                    "top_bottom": top_mean(group, col, 50) - bottom_mean(group, col, 50),
                }
            if name == "base":
                daily_base.append({"trade_date": trade_date, **row})
            elif name == "formal":
                daily_formal.append({"trade_date": trade_date, **row})
            else:
                daily_alt.append({"trade_date": trade_date, **row})
        if not group["alt_rank"].isna().all():
            top10_base = set(group.nlargest(min(10, len(group)), "base_rank")["stock_code"])
            top10_alt = set(group.nlargest(min(10, len(group)), "alt_rank")["stock_code"])
            top20_base = set(group.nlargest(min(20, len(group)), "base_rank")["stock_code"])
            top20_alt = set(group.nlargest(min(20, len(group)), "alt_rank")["stock_code"])
            feature_rows.append(
                {
                    "trade_date": trade_date,
                    "base_std": float(group["base_score"].std()),
                    "rank_corr": float(group["base_rank"].corr(group["alt_rank"])),
                    "mean_abs_rank_gap": float((group["base_rank"] - group["alt_rank"]).abs().mean()),
                    "top10_overlap": float(len(top10_base & top10_alt) / max(1, len(top10_base | top10_alt))),
                    "top20_overlap": float(len(top20_base & top20_alt) / max(1, len(top20_base | top20_alt))),
                    "alt_std": float(group["alt_score"].std()),
                }
            )
    return (
        pd.DataFrame(daily_base).set_index("trade_date"),
        pd.DataFrame(daily_formal).set_index("trade_date"),
        pd.DataFrame(daily_alt).set_index("trade_date"),
        pd.DataFrame(feature_rows).set_index("trade_date"),
    )


def summarize(daily: pd.DataFrame, trade_dates: list[str], window: int | None) -> dict[str, float]:
    keep = trade_dates if window is None else trade_dates[-window:]
    sub = daily.loc[keep]
    return {m: float(sub[m].mean()) for m in METRICS}


def objective(summary: dict[str, float]) -> float:
    return (
        1.8 * summary["top1"]
        + 1.2 * summary["top3"]
        + 1.0 * summary["top5"]
        + 0.4 * summary["top10"]
        + 0.45 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
    )


def period_top5_and_rankic(candidate_daily: pd.DataFrame, formal_daily: pd.DataFrame) -> tuple[int, float, float]:
    periods = {
        "2024H2": ("20240604", "20241231"),
        "2025H1": ("20250101", "20250630"),
        "2025H2": ("20250701", "20251231"),
        "2026YTD": ("20260101", "99999999"),
    }
    top5_deltas = []
    rankic_deltas = []
    merged = candidate_daily.join(formal_daily, lsuffix="_cand", rsuffix="_formal")
    for lo, hi in periods.values():
        sub = merged[(merged.index >= lo) & (merged.index <= hi)]
        top5_deltas.append(float((sub["top5_cand"] - sub["top5_formal"]).mean()))
        rankic_deltas.append(float((sub["rank_ic_cand"] - sub["rank_ic_formal"]).mean()))
    return int(sum(v > 0 for v in top5_deltas)), float(min(top5_deltas)), float(min(rankic_deltas))


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_base_alt_formal()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    base_daily, formal_daily, alt_daily, feature_daily = build_daily_tables(scores, labels)
    trade_dates = sorted(base_daily.index.tolist())
    overlap_days = feature_daily.index.tolist()

    rows = []
    for feature in FEATURES:
        values = sorted(feature_daily[feature].dropna().unique().tolist())
        for threshold in values:
            for direction in ["le", "ge"]:
                if direction == "le":
                    active_days = set(feature_daily.index[feature_daily[feature] <= threshold].tolist())
                else:
                    active_days = set(feature_daily.index[feature_daily[feature] >= threshold].tolist())

                mixed_rows = []
                for trade_date in trade_dates:
                    source = alt_daily.loc[trade_date] if trade_date in active_days else base_daily.loc[trade_date]
                    mixed_rows.append({"trade_date": trade_date, **{m: float(source[m]) for m in METRICS}})
                candidate_daily = pd.DataFrame(mixed_rows).set_index("trade_date")

                full_cand = summarize(candidate_daily, trade_dates, None)
                full_formal = summarize(formal_daily, trade_dates, None)
                recent63_cand = summarize(candidate_daily, trade_dates, 63)
                recent63_formal = summarize(formal_daily, trade_dates, 63)
                recent20_cand = summarize(candidate_daily, trade_dates, 20)
                recent20_formal = summarize(formal_daily, trade_dates, 20)
                positive_top5_periods, min_period_top5_delta, min_period_rankic_delta = period_top5_and_rankic(
                    candidate_daily, formal_daily
                )

                row = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": float(threshold),
                    "active_days": len(active_days),
                    "overlap_days": len(overlap_days),
                    "full_objective": objective(full_cand),
                    "full_rank_ic_delta": float(full_cand["rank_ic"] - full_formal["rank_ic"]),
                    "full_top5_delta": float(full_cand["top5"] - full_formal["top5"]),
                    "recent63_rank_ic_delta": float(recent63_cand["rank_ic"] - recent63_formal["rank_ic"]),
                    "recent63_top5_delta": float(recent63_cand["top5"] - recent63_formal["top5"]),
                    "recent20_rank_ic_delta": float(recent20_cand["rank_ic"] - recent20_formal["rank_ic"]),
                    "recent20_top5_delta": float(recent20_cand["top5"] - recent20_formal["top5"]),
                    "positive_top5_periods": positive_top5_periods,
                    "min_period_top5_delta": min_period_top5_delta,
                    "min_period_rank_ic_delta": min_period_rankic_delta,
                }
                row["promotion_safe"] = bool(
                    row["full_top5_delta"] >= 0.0
                    and row["recent63_top5_delta"] >= 0.0
                    and row["recent20_top5_delta"] >= 0.0
                    and row["positive_top5_periods"] >= 3
                    and row["full_rank_ic_delta"] >= -0.0015
                    and row["recent63_rank_ic_delta"] >= 0.0
                    and row["min_period_top5_delta"] >= 0.0
                )
                rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["promotion_safe", "full_objective", "recent63_top5_delta", "recent20_top5_delta"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    safe = result[result["promotion_safe"]].copy()
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_3d_promotion_safe_gate_search",
        "base_table": BASE_TABLE,
        "formal_table": FORMAL_TABLE,
        "alt_path": str(ALT_PATH),
        "safe_rule_count": int(len(safe)),
        "best_safe_rule": safe.iloc[0].to_dict() if len(safe) else None,
        "best_overall_rule": result.iloc[0].to_dict() if len(result) else None,
    }
    result.to_csv(REPORT_DIR / "promotion_safe_gate_search_results.csv", index=False)
    safe.to_csv(REPORT_DIR / "promotion_safe_gate_search_safe_only.csv", index=False)
    (REPORT_DIR / "promotion_safe_gate_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
