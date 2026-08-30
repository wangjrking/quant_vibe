from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
SIGNAL_DIR = REPORT_DIR / "signals"
SCORE_DIR = REPORT_DIR / "score_assets"
POOL_DB = REPORT_DIR / "current_formal_l4_rank_pool.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

L4 = {
    "1d": (
        ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate",
    ),
    "3d": (
        ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_3d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate",
    ),
    "5d": (
        ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate",
    ),
    "10d": (
        ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate",
    ),
}

BLENDS = {
    "b10_100": {"1d": 0.0, "3d": 0.0, "5d": 0.0, "10d": 1.0},
    "b5_20_10_80": {"1d": 0.0, "3d": 0.0, "5d": 0.20, "10d": 0.80},
    "b3_10_5_20_10_70": {"1d": 0.0, "3d": 0.10, "5d": 0.20, "10d": 0.70},
    "b1_05_3_10_5_15_10_70": {"1d": 0.05, "3d": 0.10, "5d": 0.15, "10d": 0.70},
}

CASES = [
    {
        "name": f"{blend}_top{topn}_{style}",
        "blend": blend,
        "topn": topn,
        "style": style,
        "amount_min": 90000.0 if style != "liq" else 150000.0,
        "mv_min": 200000.0 if style != "liq" else 500000.0,
        "cap": 0.90,
        "pos_gap_penalty": 0.05 if style == "smooth" else 0.0,
        "deep_drop_boost": 0.15 if style == "smooth" else 0.0,
        "weak_signal_penalty": 0.35 if style == "smooth" else 0.0,
        "floor": 0.20 if style == "smooth" else 1.0,
        "ceiling": 1.80 if style == "smooth" else 1.0,
        "liquidity_score_weight": 0.025 if style == "liq" else 0.0,
    }
    for blend in BLENDS
    for topn in [2, 3]
    for style in ["plain", "smooth", "liq"]
]


def symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def period_stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, mdd


def build_pool(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("PRAGMA threads=4")
    con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
    for label, (db, _table) in L4.items():
        con.execute(f"ATTACH '{db.as_posix()}' AS l4_{label} (READ_ONLY)")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE current_formal_l4_rank_pool AS
        WITH cal AS (
            SELECT
                trade_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                lead(trade_date, 2) OVER (ORDER BY trade_date) AS next_trade_date
            FROM (SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA)
        ),
        preds AS (
            SELECT
                p10.trade_date,
                p10.stock_code,
                p1.pred_prob AS pred_1d,
                p3.pred_prob AS pred_3d,
                p5.pred_prob AS pred_5d,
                p10.pred_prob AS pred_10d
            FROM l4_10d."{L4['10d'][1]}" p10
            JOIN l4_5d."{L4['5d'][1]}" p5 USING (trade_date, stock_code)
            JOIN l4_3d."{L4['3d'][1]}" p3 USING (trade_date, stock_code)
            JOIN l4_1d."{L4['1d'][1]}" p1 USING (trade_date, stock_code)
            WHERE p10.stock_code NOT LIKE '%.BJ'
        ),
        ranked AS (
            SELECT
                *,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS rank_1d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS rank_3d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS rank_5d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS rank_10d
            FROM preds
        )
        SELECT
            r.*,
            cal.buy_date,
            cal.next_trade_date,
            s.name,
            s.amount,
            s.turnover_rate,
            s.total_mv,
            s.atr_qfq,
            s.pct_chg AS signal_pct_chg_raw,
            b.open AS buy_open_raw,
            b.pre_close AS buy_pre_close_raw,
            b.open / NULLIF(b.pre_close, 0) * 100.0 - 100.0 AS buy_open_gap_raw_pct,
            n.open AS next_open_raw,
            n.open / NULLIF(b.open, 0) - 1.0 AS next_open_return_raw,
            percent_rank() OVER (PARTITION BY r.trade_date ORDER BY s.amount) AS amount_rank,
            percent_rank() OVER (PARTITION BY r.trade_date ORDER BY s.total_mv) AS mv_rank
        FROM ranked r
        JOIN cal ON cal.trade_date = r.trade_date
        JOIN marketdb.STOCK_DAILY_DATA s ON s.trade_date = r.trade_date AND s.stock_code = r.stock_code
        JOIN marketdb.STOCK_DAILY_DATA b ON b.trade_date = cal.buy_date AND b.stock_code = r.stock_code
        LEFT JOIN marketdb.STOCK_DAILY_DATA n ON n.trade_date = cal.next_trade_date AND n.stock_code = r.stock_code
        WHERE cal.buy_date IS NOT NULL
          AND s.amount IS NOT NULL
          AND s.total_mv IS NOT NULL
          AND s.atr_qfq IS NOT NULL
          AND b.open IS NOT NULL
          AND b.pre_close IS NOT NULL
          AND coalesce(cast(s.ST_TYPE AS VARCHAR), '') IN ('', '0', '0.0')
          AND coalesce(cast(b.ST_TYPE AS VARCHAR), '') IN ('', '0', '0.0')
          AND upper(coalesce(cast(s.ST_TYPE_name AS VARCHAR), '')) NOT LIKE '%ST%'
          AND upper(coalesce(cast(b.ST_TYPE_name AS VARCHAR), '')) NOT LIKE '%ST%'
          AND upper(coalesce(s.name, '')) NOT LIKE 'ST%'
          AND upper(coalesce(s.name, '')) NOT LIKE '*ST%'
          AND upper(coalesce(s.name, '')) NOT LIKE '退市%'
          AND upper(coalesce(b.name, '')) NOT LIKE 'ST%'
          AND upper(coalesce(b.name, '')) NOT LIKE '*ST%'
          AND upper(coalesce(b.name, '')) NOT LIKE '退市%'
          AND b.open < b.pre_close * (
              CASE WHEN r.stock_code LIKE '300%' OR r.stock_code LIKE '301%' OR r.stock_code LIKE '688%' THEN 1.20 ELSE 1.10 END
          ) * 0.995
        """
    )


def build_score_asset(con: duckdb.DuckDBPyConnection, blend: str, weights: dict[str, float]) -> Path:
    score_db = SCORE_DIR / f"{blend}.duckdb"
    score_db.unlink(missing_ok=True)
    expr = " + ".join(f"{weight} * rank_{label}" for label, weight in weights.items())
    con.execute(f"ATTACH '{score_db.as_posix()}' AS scoreout")
    con.execute(
        f"""
        CREATE TABLE scoreout.blended_rank_score AS
        SELECT trade_date, stock_code, cast({expr} AS DOUBLE) AS pred_prob
        FROM current_formal_l4_rank_pool
        """
    )
    con.execute("DETACH scoreout")
    return score_db


def build_case(con: duckdb.DuckDBPyConnection, case: dict, score_db: Path) -> dict:
    weights = BLENDS[case["blend"]]
    blend_expr = " + ".join(f"{weight} * rank_{label}" for label, weight in weights.items())
    frame = con.execute(
        f"""
        WITH eligible AS (
            SELECT
                *,
                ({blend_expr}) + {case['liquidity_score_weight']} * (amount_rank + mv_rank) / 2.0 AS select_score
            FROM current_formal_l4_rank_pool
            WHERE amount >= {case['amount_min']}
              AND total_mv >= {case['mv_min']}
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY trade_date ORDER BY select_score DESC, stock_code) AS pick_rank
            FROM eligible
        )
        SELECT *
        FROM ranked
        WHERE pick_rank <= {case['topn']}
        ORDER BY trade_date, pick_rank
        """
    ).fetchdf()
    positive_gap = np.maximum(frame["buy_open_gap_raw_pct"].to_numpy(float) - 0.5, 0.0)
    deep_drop = np.maximum(-4.0 - frame["signal_pct_chg_raw"].to_numpy(float), 0.0)
    weak_signal = np.maximum(frame["signal_pct_chg_raw"].to_numpy(float) + 2.0, 0.0)
    scale = np.clip(
        1.0
        - float(case["pos_gap_penalty"]) * positive_gap
        + float(case["deep_drop_boost"]) * deep_drop
        - float(case["weak_signal_penalty"]) * weak_signal,
        float(case["floor"]),
        float(case["ceiling"]),
    )
    frame["target_pct"] = float(case["cap"]) / int(case["topn"]) * scale
    daily_sum = frame.groupby("buy_date")["target_pct"].transform("sum")
    frame["target_pct"] *= np.minimum(1.0, float(case["cap"]) / daily_sum.replace(0.0, np.nan)).fillna(1.0)
    frame["symbol"] = frame["stock_code"].map(symbol)
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["rank"] = frame["pick_rank"].astype(int)
    frame["pred_prob"] = frame["select_score"].astype(float)
    frame["entry_score"] = frame["select_score"].astype(float)
    frame["holding_days"] = 1
    frame["max_holding_days"] = 3
    frame["score_exit_entry_ratio"] = 0.98
    frame["min_holding_days_before_score_exit"] = 1
    frame["score_continue_entry_ratio"] = 1.02
    frame["strategy_variant"] = case["name"]
    frame["filter_name"] = "current_formal_l4_rank_pool"
    frame["entry_weight_name"] = case["style"]
    frame["dynamic_hold_name"] = "h1_mh3_score_exit"
    frame["buy_day_market_available"] = True
    frame["buy_day_hard_gate_complete"] = True
    frame["buy_day_st_rejected"] = False
    frame["buy_day_open_limit_up_rejected"] = False
    frame["latest_market_date"] = str(frame["buy_date"].max())
    frame["buy_open_gap_pct"] = frame["buy_open_gap_raw_pct"]
    output_columns = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "rank_1d", "rank_3d", "rank_5d", "rank_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "signal_pct_chg_raw", "target_pct",
        "holding_days", "max_holding_days", "score_exit_entry_ratio", "min_holding_days_before_score_exit",
        "score_continue_entry_ratio", "strategy_variant", "filter_name", "entry_weight_name", "dynamic_hold_name",
        "buy_day_market_available", "buy_day_hard_gate_complete", "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected", "latest_market_date", "buy_open_gap_pct", "buy_open_gap_raw_pct",
        "next_open_return_raw",
    ]
    signal_file = SIGNAL_DIR / f"{case['name']}.csv"
    frame[output_columns].to_csv(signal_file, index=False, encoding="utf-8-sig")

    mature = frame.dropna(subset=["next_open_return_raw"]).copy()
    mature["weighted_return"] = mature["target_pct"] * (mature["next_open_return_raw"] - 0.006)
    daily = mature.groupby("buy_date")["weighted_return"].sum().sort_index()
    full_dates = pd.date_range(pd.to_datetime(daily.index.min()), pd.to_datetime(daily.index.max()), freq="B")
    daily = daily.reindex(full_dates.strftime("%Y%m%d"), fill_value=0.0)
    annual, sharpe, mdd = period_stats(daily)
    dated = daily.copy()
    dated.index = pd.to_datetime(dated.index)
    yearly = [period_stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 40]
    rolling = [period_stats(daily.iloc[start : start + 60]) for start in range(0, max(len(daily) - 59, 1), 20)]
    return {
        **case,
        "signal_file": str(signal_file),
        "score_db": str(score_db),
        "score_table": "blended_rank_score",
        "rows": int(len(frame)),
        "signal_days": int(frame["signal_date"].nunique()),
        "stock_count": int(frame["stock_code"].nunique()),
        "min_signal_date": str(frame["signal_date"].min()),
        "max_signal_date": str(frame["signal_date"].max()),
        "mean_daily_target": float(frame.groupby("buy_date")["target_pct"].sum().mean()),
        "local_annual": annual,
        "local_sharpe": sharpe,
        "local_mdd": mdd,
        "worst_year_sharpe": min((item[1] for item in yearly), default=0.0),
        "rolling60_min_sharpe": min((item[1] for item in rolling), default=0.0),
        "rolling60_median_sharpe": float(np.median([item[1] for item in rolling])) if rolling else 0.0,
        "bj_rows": int(frame["stock_code"].str.endswith(".BJ", na=False).sum()),
        "duplicate_keys": int(frame.duplicated(["signal_date", "stock_code"]).sum()),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(POOL_DB))
    try:
        # Research reruns must follow the active formal tables, even when an older
        # materialized pool is already present in the report directory.
        build_pool(con)
        score_assets = {blend: build_score_asset(con, blend, weights) for blend, weights in BLENDS.items()}
        rows = [build_case(con, case, score_assets[case["blend"]]) for case in CASES]
        pool_audit = con.execute(
            """
            select count(*) as row_count, count(distinct trade_date) trade_days, min(trade_date) min_date,
                   max(trade_date) max_date, count(*) filter (where stock_code like '%.BJ') bj_rows
            from current_formal_l4_rank_pool
            """
        ).fetchone()
    finally:
        con.close()
    frame = pd.DataFrame(rows).sort_values(["local_sharpe", "local_annual"], ascending=False)
    frame.to_csv(REPORT_DIR / "candidate_local_screen.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "research_only_current_active_formal_l4",
        "input_contract": {
            "l4": {label: {"db": str(db), "table": table} for label, (db, table) in L4.items()},
            "market_db": str(MARKET_DB),
            "factor_price_semantics": "qfq fields are explicit; atr_qfq only",
            "execution_price_semantics": "raw unadjusted open and pre_close",
            "next_open_return_usage": "evaluation only, never candidate selection",
        },
        "pool_audit": {
            "rows": int(pool_audit[0]), "trade_days": int(pool_audit[1]), "min_date": str(pool_audit[2]),
            "max_date": str(pool_audit[3]), "bj_rows": int(pool_audit[4]),
        },
        "cases": frame.to_dict("records"),
    }
    (REPORT_DIR / "candidate_local_screen.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.head(12).to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
