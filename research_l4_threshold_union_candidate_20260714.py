from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "l4_threshold_union_candidate"
LOG_DIR = REPORT_DIR / "logs" / "l4_threshold_union_candidate"
OUT_CSV = REPORT_DIR / "l4_threshold_union_candidate_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "l4_threshold_union_candidate_juejin_results_20260714.json"
POOL_PARQUET = REPORT_DIR / "l4_threshold_union_candidate_pool_20260714.parquet"

DUCK_DIR = ROOT / "quant" / "data_file" / "production_assets" / "duckdb"
L2_DB = DUCK_DIR / "l2_stock_daily_data.duckdb"
L4_1D_DB = DUCK_DIR / "l4_executable_1d_open_return_formal.duckdb"
L4_3D_DB = DUCK_DIR / "l4_executable_3d_open_return_formal.duckdb"
L4_5D_DB = DUCK_DIR / "l4_executable_5d_open_return_formal.duckdb"
L4_10D_DB = DUCK_DIR / "l4_executable_10d_open_return_formal.duckdb"

T1 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate"
T3 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate"
T5 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate"
T10 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"name": "tu10_98_p175_g05_amt12_t2", "w1": 0, "w3": 0, "w5": 0, "w10": 1, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 120000, "turn": 2, "atr": 12, "target": 0.50},
    {"name": "tu10_99_p175_g05_amt12_t2", "w1": 0, "w3": 0, "w5": 0, "w10": 1, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 120000, "turn": 2, "atr": 12, "target": 0.58, "min10": 0.99},
    {"name": "tu105_8020_p175_g05_amt12", "w1": 0, "w3": 0, "w5": 0.2, "w10": 0.8, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 120000, "turn": 2, "atr": 12, "target": 0.52},
    {"name": "tu105_7030_p175_g05_amt20", "w1": 0, "w3": 0, "w5": 0.3, "w10": 0.7, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 200000, "turn": 2, "atr": 12, "target": 0.54},
    {"name": "tu13510_0127_p175_g05_amt12", "w1": 0, "w3": 0.1, "w5": 0.2, "w10": 0.7, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 120000, "turn": 2, "atr": 12, "target": 0.54},
    {"name": "tu1510_guard_p175_g05_amt12", "w1": 0.08, "w3": 0, "w5": 0.17, "w10": 0.75, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 120000, "turn": 2, "atr": 12, "target": 0.52, "min1": 0.95},
    {"name": "tu1510_guard_p175_g05_amt20", "w1": 0.08, "w3": 0, "w5": 0.17, "w10": 0.75, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 200000, "turn": 2, "atr": 12, "target": 0.52, "min1": 0.95},
    {"name": "tu10_turn4_p175_g05", "w1": 0, "w3": 0, "w5": 0, "w10": 1, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 120000, "turn": 4, "atr": 8, "target": 0.65},
    {"name": "tu105_turn4_p175_g05", "w1": 0, "w3": 0, "w5": 0.2, "w10": 0.8, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5, "amount": 120000, "turn": 4, "atr": 8, "target": 0.65},
    {"name": "tu105_soft_p10_g05_amt20", "w1": 0, "w3": 0, "w5": 0.2, "w10": 0.8, "pct": -1.0, "gap_hi": 0.5, "gap_lo": -5, "amount": 200000, "turn": 2, "atr": 12, "target": 0.45},
    {"name": "tu105_widegap_p175_amt15", "w1": 0, "w3": 0, "w5": 0.2, "w10": 0.8, "pct": -1.75, "gap_hi": 1.0, "gap_lo": -6, "amount": 150000, "turn": 2, "atr": 12, "target": 0.48},
]


def build_pool() -> pd.DataFrame:
    if POOL_PARQUET.exists():
        return pd.read_parquet(POOL_PARQUET)
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"ATTACH '{L2_DB.as_posix()}' AS l2db (READ_ONLY)")
        con.execute(f"ATTACH '{L4_1D_DB.as_posix()}' AS db1 (READ_ONLY)")
        con.execute(f"ATTACH '{L4_3D_DB.as_posix()}' AS db3 (READ_ONLY)")
        con.execute(f"ATTACH '{L4_5D_DB.as_posix()}' AS db5 (READ_ONLY)")
        con.execute(f"ATTACH '{L4_10D_DB.as_posix()}' AS db10 (READ_ONLY)")
        sql = f"""
        WITH cal AS (
            SELECT trade_date, LEAD(trade_date) OVER (ORDER BY trade_date) AS buy_date
            FROM (SELECT DISTINCT trade_date FROM l2db.STOCK_DAILY_DATA)
        ),
        keys AS (
            SELECT trade_date, stock_code FROM db10.{T10} WHERE pred_prob >= 0.98
            UNION
            SELECT trade_date, stock_code FROM db5.{T5} WHERE pred_prob >= 0.98
            UNION
            SELECT trade_date, stock_code FROM db3.{T3} WHERE pred_prob >= 0.99
            UNION
            SELECT trade_date, stock_code FROM db1.{T1} WHERE pred_prob >= 0.99
        )
        SELECT
            k.trade_date AS signal_date,
            cal.buy_date,
            k.stock_code,
            m.name,
            s1.pred_prob AS pred_1d,
            s3.pred_prob AS pred_3d,
            s5.pred_prob AS pred_5d,
            s10.pred_prob AS pred_10d,
            m.pct_chg AS signal_pct_chg_raw,
            m.amount,
            m.turnover_rate,
            m.total_mv,
            m.atr_qfq,
            CASE WHEN b.pre_close IS NOT NULL AND b.pre_close != 0 AND b.open IS NOT NULL
                 THEN (b.open / b.pre_close - 1.0) * 100.0
                 ELSE NULL END AS buy_open_gap_pct
        FROM keys k
        JOIN db1.{T1} s1 USING (trade_date, stock_code)
        JOIN db3.{T3} s3 USING (trade_date, stock_code)
        JOIN db5.{T5} s5 USING (trade_date, stock_code)
        JOIN db10.{T10} s10 USING (trade_date, stock_code)
        JOIN l2db.STOCK_DAILY_DATA m ON m.trade_date = k.trade_date AND m.stock_code = k.stock_code
        JOIN cal ON cal.trade_date = k.trade_date AND cal.buy_date IS NOT NULL
        JOIN l2db.STOCK_DAILY_DATA b ON b.trade_date = cal.buy_date AND b.stock_code = k.stock_code
        WHERE k.stock_code NOT LIKE '%.BJ'
          AND (m.ST_TYPE IS NULL OR m.ST_TYPE = '' OR m.ST_TYPE = '0')
          AND (m.ST_TYPE_name IS NULL OR m.ST_TYPE_name = '' OR m.ST_TYPE_name = '正常')
          AND (m.name IS NULL OR (m.name NOT LIKE 'ST%' AND m.name NOT LIKE '*ST%' AND m.name NOT LIKE '%退%'))
          AND m.pct_chg <= -1.0
          AND m.amount >= 100000
          AND m.turnover_rate >= 1.5
          AND m.atr_qfq <= 15.0
          AND b.open IS NOT NULL
          AND b.pre_close IS NOT NULL
        """
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    for col, score_col in [("pred_1d", "s1"), ("pred_3d", "s3"), ("pred_5d", "s5"), ("pred_10d", "s10")]:
        df[score_col] = df.groupby("signal_date")[col].rank(pct=True, ascending=True)
    POOL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(POOL_PARQUET, index=False)
    return df


def symbol_of(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return ("SHSE." if suffix == "SH" else "SZSE.") + code


def write_case(pool: pd.DataFrame, case: dict) -> dict:
    df = pool.copy()
    df["blend"] = (
        df["s1"] * float(case["w1"]) + df["s3"] * float(case["w3"]) +
        df["s5"] * float(case["w5"]) + df["s10"] * float(case["w10"])
    )
    mask = (
        (df["signal_pct_chg_raw"] <= float(case["pct"])) &
        (df["buy_open_gap_pct"] <= float(case["gap_hi"])) &
        (df["buy_open_gap_pct"] >= float(case["gap_lo"])) &
        (df["amount"] >= float(case["amount"])) &
        (df["turnover_rate"] >= float(case["turn"])) &
        (df["atr_qfq"] <= float(case["atr"]))
    )
    if "min1" in case:
        mask &= df["pred_1d"] >= float(case["min1"])
    if "min10" in case:
        mask &= df["pred_10d"] >= float(case["min10"])
    cand = df[mask].copy()
    selected = (
        cand.sort_values(["signal_date", "blend", "pred_10d", "amount"], ascending=[True, False, False, False])
        .groupby("signal_date", as_index=False)
        .head(1)
        .copy()
    )
    selected["rank"] = 1
    selected["symbol"] = selected["stock_code"].map(symbol_of)
    selected["pred_prob"] = selected["pred_10d"]
    selected["entry_score"] = selected["blend"]
    selected["target_pct"] = float(case["target"])
    selected["holding_days"] = 1
    selected["max_holding_days"] = 3
    selected["score_exit_entry_ratio"] = "0.98000"
    selected["score_continue_entry_ratio"] = "1.02000"
    selected["min_holding_days_before_score_exit"] = 1
    selected["signal_stop_loss_pct"] = 0.05
    selected["signal_take_profit_pct"] = 0.08
    selected["strategy_variant"] = case["name"]
    selected["filter_name"] = case["name"]
    selected["daily_target_sum_after_cap"] = selected["target_pct"]
    selected["buy_open_gap_raw_pct"] = selected["buy_open_gap_pct"]
    keep_cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank",
        "pred_prob", "entry_score", "pred_1d", "pred_3d", "pred_5d", "pred_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "signal_pct_chg_raw",
        "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio",
        "signal_stop_loss_pct", "signal_take_profit_pct", "strategy_variant",
        "filter_name", "buy_open_gap_pct", "buy_open_gap_raw_pct",
        "daily_target_sum_after_cap", "s1", "s3", "s5", "s10", "blend",
    ]
    out = SIGNAL_DIR / f"{case['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    selected[keep_cols].to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["name"],
        "signal_file": str(out),
        "source_candidate_rows": int(len(cand)),
        "rows": int(len(selected)),
        "buy_days": int(selected["buy_date"].nunique()) if len(selected) else 0,
        "stock_count": int(selected["stock_code"].nunique()) if len(selected) else 0,
        "mean_daily_target_sum": float(selected.groupby("buy_date")["target_pct"].sum().mean()) if len(selected) else 0.0,
        "max_daily_target_sum": float(selected.groupby("buy_date")["target_pct"].sum().max()) if len(selected) else 0.0,
        **case,
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    results = []
    for case in CASES:
        row = write_case(pool, case)
        if row["rows"] == 0:
            results.append(row | {"returncode": -1, "indicator_error": "empty_signal"})
            continue
        result = base_mod.run_juejin(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
        ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "case": result.get("case"),
            "rows": result.get("rows"),
            "annual": result.get("pnl_ratio_annual"),
            "sharpe": result.get("sharp_ratio"),
            "mdd": result.get("max_drawdown"),
            "open_count": result.get("open_count"),
        }, ensure_ascii=False), flush=True)
    hits = [
        row for row in results
        if (row.get("pnl_ratio_annual") or 0) >= 5.0
        and (row.get("sharp_ratio") or 0) >= 4.0
        and (row.get("max_drawdown") or 1) <= 0.40
    ]
    summary = {
        "pool_path": str(POOL_PARQUET),
        "pool_rows": int(len(pool)),
        "pool_days": int(pool["signal_date"].nunique()),
        "result_csv": str(OUT_CSV),
        "result_json": str(OUT_JSON),
        "cases": len(results),
        "target_hits": len(hits),
        "target_hits_table": sorted(hits, key=lambda r: (r.get("sharp_ratio") or -999, r.get("pnl_ratio_annual") or -999), reverse=True),
        "best_by_sharpe": sorted(results, key=lambda r: (r.get("sharp_ratio") or -999, r.get("pnl_ratio_annual") or -999), reverse=True)[:5],
        "best_by_annual": sorted(results, key=lambda r: (r.get("pnl_ratio_annual") or -999, r.get("sharp_ratio") or -999), reverse=True)[:5],
    }
    (REPORT_DIR / "l4_threshold_union_candidate_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
