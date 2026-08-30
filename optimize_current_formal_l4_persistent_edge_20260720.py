from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
    / "current_formal_l4_rank_pool.duckdb"
)
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_persistent_edge_20260720"
)
FEATURE_DB = REPORT_DIR / "persistent_edge_features.duckdb"


BLENDS = {
    "w10_100": (0.0, 0.0, 1.0),
    "w5_20_10_80": (0.0, 0.2, 0.8),
    "w3_20_5_20_10_60": (0.2, 0.2, 0.6),
    "w3_10_5_30_10_60": (0.1, 0.3, 0.6),
    "w3_20_5_40_10_40": (0.2, 0.4, 0.4),
}

SCORES = {
    "plain": "blend",
    "agree": "blend - 0.20 * spread_3510",
    "persistent": "blend + 0.10 * rank10_lag1 + 0.05 * rank10_lag3",
    "improving": "blend + 0.20 * greatest(least(d10_1, 0.20), -0.20)",
    "stable_improving": (
        "blend - 0.15 * spread_3510 + 0.10 * rank10_lag1 "
        "+ 0.10 * greatest(least(d10_1, 0.20), -0.20)"
    ),
    "timed_1d": "blend + 0.10 * rank_1d",
    "liq_soft": "blend + 0.02 * amount_rank + 0.01 * mv_rank",
    "low_atr_soft": "blend + 0.02 * amount_rank + 0.01 * mv_rank - 0.02 * atr_rank",
}

FILTERS = {
    "broad": "signal_pct_chg_raw <= 3.0",
    "no_chase": "signal_pct_chg_raw <= 1.0",
    "pullback": "signal_pct_chg_raw BETWEEN -5.0 AND 0.5",
    "rank_persist": "rank10_lag1 >= 0.80 AND signal_pct_chg_raw <= 2.0",
    "rank_stable": "rank10_lag1 >= 0.85 AND abs(d10_1) <= 0.15 AND signal_pct_chg_raw <= 2.0",
    "multihorizon": "least(rank_3d, rank_5d, rank_10d) >= 0.70 AND signal_pct_chg_raw <= 2.0",
}


def period_stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    drawdown = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, drawdown


def materialize_features() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DB.unlink(missing_ok=True)
    con = duckdb.connect(str(FEATURE_DB))
    try:
        con.execute(f"ATTACH '{SOURCE_DB.as_posix()}' AS pooldb (READ_ONLY)")
        con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(
            """
            CREATE TABLE persistent_edge_features AS
            WITH score_history AS (
                SELECT p.*,
                       lag(rank_10d, 1) OVER w AS rank10_lag1,
                       lag(rank_10d, 3) OVER w AS rank10_lag3,
                       lag(rank_5d, 1) OVER w AS rank5_lag1,
                       abs(greatest(rank_3d, rank_5d, rank_10d)
                           - least(rank_3d, rank_5d, rank_10d)) AS spread_3510,
                       percent_rank() OVER (PARTITION BY trade_date ORDER BY atr_qfq) AS atr_rank
                FROM pooldb.current_formal_l4_rank_pool p
                WINDOW w AS (PARTITION BY stock_code ORDER BY trade_date)
            ), market_returns AS (
                SELECT trade_date, stock_code, open,
                       lead(open, 1) OVER w AS entry_open,
                       lead(open, 5) OVER w AS exit_open_5,
                       lead(open, 8) OVER w AS exit_open_8,
                       lead(open, 10) OVER w AS exit_open_10,
                       lead(open, 12) OVER w AS exit_open_12,
                       lead(open, 15) OVER w AS exit_open_15
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260717'
                  AND stock_code NOT LIKE '%.BJ'
                WINDOW w AS (PARTITION BY stock_code ORDER BY trade_date)
            )
            SELECT h.*,
                   h.rank_10d - h.rank10_lag1 AS d10_1,
                   h.rank_10d - h.rank10_lag3 AS d10_3,
                   h.rank_5d - h.rank5_lag1 AS d5_1,
                   m.exit_open_5 / nullif(m.entry_open, 0) - 1 AS ret_open_5,
                   m.exit_open_8 / nullif(m.entry_open, 0) - 1 AS ret_open_8,
                   m.exit_open_10 / nullif(m.entry_open, 0) - 1 AS ret_open_10,
                   m.exit_open_12 / nullif(m.entry_open, 0) - 1 AS ret_open_12,
                   m.exit_open_15 / nullif(m.entry_open, 0) - 1 AS ret_open_15
            FROM score_history h
            JOIN market_returns m USING (trade_date, stock_code)
            WHERE h.amount >= 90000
              AND h.total_mv >= 200000
              AND h.rank10_lag3 IS NOT NULL
            """
        )
        con.execute("CREATE INDEX idx_feature_date_code ON persistent_edge_features(trade_date, stock_code)")
    finally:
        con.close()


def select_daily_top5(
    con: duckdb.DuckDBPyConnection,
    blend_name: str,
    score_name: str,
    filter_name: str,
) -> pd.DataFrame:
    w3, w5, w10 = BLENDS[blend_name]
    blend = f"({w3} * rank_3d + {w5} * rank_5d + {w10} * rank_10d)"
    score = SCORES[score_name].replace("blend", blend)
    filt = FILTERS[filter_name]
    return con.execute(
        f"""
        WITH eligible AS (
            SELECT *, {blend} AS blend, {score} AS select_score
            FROM persistent_edge_features
            WHERE {filt}
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY trade_date ORDER BY select_score DESC, stock_code
            ) AS pick_rank
            FROM eligible
        )
        SELECT trade_date, stock_code, pick_rank, blend,
               ret_open_5, ret_open_8, ret_open_10, ret_open_12, ret_open_15
        FROM ranked WHERE pick_rank <= 5
        """
    ).fetchdf()


def evaluate_case(
    daily_top5: pd.DataFrame,
    all_dates: pd.Index,
    blend_name: str,
    score_name: str,
    filter_name: str,
    threshold: float,
    topn: int,
    hold: int,
) -> dict:
    ret_col = f"ret_open_{hold}"
    selected = daily_top5[
        (daily_top5["blend"] >= threshold)
        & (daily_top5["pick_rank"] <= topn)
        & daily_top5[ret_col].notna()
    ][["trade_date", "stock_code", "pick_rank", ret_col]].copy()
    selected = selected.rename(columns={ret_col: "gross_return"})
    name = f"{blend_name}_{score_name}_{filter_name}_m{int(threshold*100)}_top{topn}_h{hold}"
    if selected.empty:
        return {"name": name, "rows": 0, "days": 0, "robust_score": -999.0}
    # Each daily cohort receives 100% / hold. The 0.6% round-trip cost is charged once per cohort.
    selected["net_return"] = selected["gross_return"] - 0.006
    daily = selected.groupby("trade_date")["net_return"].mean() / hold
    daily = daily.reindex(all_dates, fill_value=0.0)
    early = daily[daily.index < "20250101"]
    late = daily[daily.index >= "20250101"]
    full_stats = period_stats(daily)
    early_stats = period_stats(early)
    late_stats = period_stats(late)
    dated = daily.copy()
    dated.index = pd.to_datetime(dated.index)
    year_stats = [period_stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 80]
    year_sharpes = [item[1] for item in year_stats]
    year_annuals = [item[0] for item in year_stats]
    gross = selected["gross_return"]
    return {
        "name": name,
        "blend": blend_name,
        "score": score_name,
        "filter": filter_name,
        "threshold": threshold,
        "topn": topn,
        "hold": hold,
        "rows": int(len(selected)),
        "days": int(selected["trade_date"].nunique()),
        "mean_gross": float(gross.mean()),
        "median_gross": float(gross.median()),
        "win_rate_gross": float((gross > 0).mean()),
        "full_annual_proxy": full_stats[0],
        "full_sharpe_proxy": full_stats[1],
        "full_mdd_proxy": full_stats[2],
        "early_annual_proxy": early_stats[0],
        "early_sharpe_proxy": early_stats[1],
        "late_annual_proxy": late_stats[0],
        "late_sharpe_proxy": late_stats[1],
        "worst_year_annual_proxy": min(year_annuals, default=-1.0),
        "worst_year_sharpe_proxy": min(year_sharpes, default=-9.0),
        "robust_score": min(early_stats[1], late_stats[1], min(year_sharpes, default=-9.0))
        + 0.20 * full_stats[1],
    }


def main() -> None:
    materialize_features()
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    rows: list[dict] = []
    try:
        all_dates = pd.Index(
            con.execute("SELECT DISTINCT trade_date FROM persistent_edge_features ORDER BY trade_date")
            .fetchnumpy()["trade_date"]
        )
        for blend_name in BLENDS:
            for score_name in SCORES:
                for filter_name in FILTERS:
                    daily_top5 = select_daily_top5(con, blend_name, score_name, filter_name)
                    for threshold in [0.80, 0.85, 0.90, 0.95]:
                        for topn in [1, 3, 5]:
                            for hold in [5, 8, 10, 12, 15]:
                                rows.append(
                                    evaluate_case(
                                        daily_top5,
                                        all_dates,
                                        blend_name,
                                        score_name,
                                        filter_name,
                                        threshold,
                                        topn,
                                        hold,
                                    )
                                )
    finally:
        con.close()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["robust_score", "full_sharpe_proxy"], ascending=False)
    frame.to_csv(REPORT_DIR / "local_screen.csv", index=False, encoding="utf-8-sig")
    eligible = frame[
        (frame["days"] >= 700)
        & (frame["early_annual_proxy"] > 0)
        & (frame["late_annual_proxy"] > 0)
        & (frame["worst_year_annual_proxy"] > -0.10)
    ].copy()
    eligible.head(100).to_csv(REPORT_DIR / "eligible_top100.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "research_only",
        "input": str(SOURCE_DB),
        "market": str(MARKET_DB),
        "cases": int(len(frame)),
        "eligible_cases": int(len(eligible)),
        "top": eligible.head(20).replace({np.nan: None}).to_dict(orient="records"),
        "selection_uses_future_data": False,
        "future_open_returns_used_only_for_evaluation": True,
        "round_trip_cost_proxy": 0.006,
    }
    (REPORT_DIR / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(frame), "eligible": len(eligible), "top": summary["top"][:3]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
