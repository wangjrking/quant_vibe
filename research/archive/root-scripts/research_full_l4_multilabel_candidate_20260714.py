from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "full_l4_multilabel_candidate"
LOG_DIR = REPORT_DIR / "logs" / "full_l4_multilabel_candidate"
OUT_CSV = REPORT_DIR / "full_l4_multilabel_candidate_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "full_l4_multilabel_candidate_juejin_results_20260714.json"
BASE_PARQUET = REPORT_DIR / "full_l4_multilabel_candidate_pool_20260714.parquet"

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
    {"name": "ml10_top1_deep_amt12", "w1": 0.0, "w3": 0.0, "w5": 0.0, "w10": 1.0, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 120000, "turn": 2.0, "atr": 12.0, "target": 0.46},
    {"name": "ml10_top1_deep_amt20", "w1": 0.0, "w3": 0.0, "w5": 0.0, "w10": 1.0, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 200000, "turn": 2.0, "atr": 12.0, "target": 0.50},
    {"name": "ml10_5_8020_amt12", "w1": 0.0, "w3": 0.0, "w5": 0.2, "w10": 0.8, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 120000, "turn": 2.0, "atr": 12.0, "target": 0.50},
    {"name": "ml10_5_7030_amt20", "w1": 0.0, "w3": 0.0, "w5": 0.3, "w10": 0.7, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 200000, "turn": 2.0, "atr": 12.0, "target": 0.52},
    {"name": "ml10_3_5_721_amt12", "w1": 0.0, "w3": 0.1, "w5": 0.2, "w10": 0.7, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 120000, "turn": 2.0, "atr": 12.0, "target": 0.52},
    {"name": "ml10_3_5_631_deep", "w1": 0.0, "w3": 0.1, "w5": 0.3, "w10": 0.6, "pct": -2.5, "gap_hi": 0.5, "gap_lo": -6.0, "amount": 120000, "turn": 2.0, "atr": 12.0, "target": 0.55},
    {"name": "ml10_5_1_guard", "w1": 0.08, "w3": 0.0, "w5": 0.17, "w10": 0.75, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 120000, "turn": 2.0, "atr": 12.0, "target": 0.50, "min_r1": 0.45},
    {"name": "ml10_5_1_strong", "w1": 0.15, "w3": 0.0, "w5": 0.20, "w10": 0.65, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 120000, "turn": 2.0, "atr": 12.0, "target": 0.50, "min_r1": 0.55},
    {"name": "ml10_top1_widegap", "w1": 0.0, "w3": 0.0, "w5": 0.0, "w10": 1.0, "pct": -1.75, "gap_hi": 1.0, "gap_lo": -6.0, "amount": 150000, "turn": 2.0, "atr": 12.0, "target": 0.45},
    {"name": "ml10_top1_turn4", "w1": 0.0, "w3": 0.0, "w5": 0.0, "w10": 1.0, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 120000, "turn": 4.0, "atr": 8.0, "target": 0.60},
    {"name": "ml10_5_8020_turn4", "w1": 0.0, "w3": 0.0, "w5": 0.2, "w10": 0.8, "pct": -1.75, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 120000, "turn": 4.0, "atr": 8.0, "target": 0.60},
    {"name": "ml10_5_softpct", "w1": 0.0, "w3": 0.0, "w5": 0.2, "w10": 0.8, "pct": -1.0, "gap_hi": 0.5, "gap_lo": -5.0, "amount": 200000, "turn": 2.0, "atr": 12.0, "target": 0.42},
]


def build_pool() -> pd.DataFrame:
    if BASE_PARQUET.exists():
        return pd.read_parquet(BASE_PARQUET)

    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"ATTACH '{L2_DB.as_posix()}' AS l2db (READ_ONLY)")
        con.execute(f"ATTACH '{L4_1D_DB.as_posix()}' AS db1 (READ_ONLY)")
        con.execute(f"ATTACH '{L4_3D_DB.as_posix()}' AS db3 (READ_ONLY)")
        con.execute(f"ATTACH '{L4_5D_DB.as_posix()}' AS db5 (READ_ONLY)")
        con.execute(f"ATTACH '{L4_10D_DB.as_posix()}' AS db10 (READ_ONLY)")
        query = f"""
        WITH cal AS (
            SELECT trade_date, LEAD(trade_date) OVER (ORDER BY trade_date) AS buy_date
            FROM (SELECT DISTINCT trade_date FROM l2db.STOCK_DAILY_DATA)
        ),
        joined AS (
            SELECT
                s10.trade_date AS signal_date,
                cal.buy_date,
                s10.stock_code,
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
                b.open AS buy_open,
                b.pre_close AS buy_pre_close,
                CASE WHEN b.pre_close IS NOT NULL AND b.pre_close != 0 AND b.open IS NOT NULL
                     THEN (b.open / b.pre_close - 1.0) * 100.0
                     ELSE NULL END AS buy_open_gap_pct,
                m.ST_TYPE,
                m.ST_TYPE_name
            FROM db10.{T10} s10
            JOIN db5.{T5} s5 USING (trade_date, stock_code)
            JOIN db3.{T3} s3 USING (trade_date, stock_code)
            JOIN db1.{T1} s1 USING (trade_date, stock_code)
            JOIN l2db.STOCK_DAILY_DATA m ON m.trade_date = s10.trade_date AND m.stock_code = s10.stock_code
            JOIN cal ON cal.trade_date = s10.trade_date AND cal.buy_date IS NOT NULL
            JOIN l2db.STOCK_DAILY_DATA b ON b.trade_date = cal.buy_date AND b.stock_code = s10.stock_code
            WHERE s10.stock_code NOT LIKE '%.BJ'
              AND (m.ST_TYPE IS NULL OR m.ST_TYPE = '' OR m.ST_TYPE = '0')
              AND (m.ST_TYPE_name IS NULL OR m.ST_TYPE_name = '' OR m.ST_TYPE_name = '正常')
              AND (m.name IS NULL OR (m.name NOT LIKE 'ST%' AND m.name NOT LIKE '*ST%' AND m.name NOT LIKE '%退%'))
              AND m.amount IS NOT NULL
              AND m.turnover_rate IS NOT NULL
              AND m.total_mv IS NOT NULL
              AND m.atr_qfq IS NOT NULL
              AND m.pct_chg IS NOT NULL
              AND b.open IS NOT NULL
              AND b.pre_close IS NOT NULL
        ),
        ranked AS (
            SELECT *,
                (COUNT(*) OVER (PARTITION BY signal_date) - RANK() OVER (PARTITION BY signal_date ORDER BY pred_1d DESC))::DOUBLE
                  / NULLIF((COUNT(*) OVER (PARTITION BY signal_date) - 1), 0) AS r1,
                (COUNT(*) OVER (PARTITION BY signal_date) - RANK() OVER (PARTITION BY signal_date ORDER BY pred_3d DESC))::DOUBLE
                  / NULLIF((COUNT(*) OVER (PARTITION BY signal_date) - 1), 0) AS r3,
                (COUNT(*) OVER (PARTITION BY signal_date) - RANK() OVER (PARTITION BY signal_date ORDER BY pred_5d DESC))::DOUBLE
                  / NULLIF((COUNT(*) OVER (PARTITION BY signal_date) - 1), 0) AS r5,
                (COUNT(*) OVER (PARTITION BY signal_date) - RANK() OVER (PARTITION BY signal_date ORDER BY pred_10d DESC))::DOUBLE
                  / NULLIF((COUNT(*) OVER (PARTITION BY signal_date) - 1), 0) AS r10
            FROM joined
        )
        SELECT * FROM ranked
        WHERE signal_pct_chg_raw <= -1.0
          AND buy_open_gap_pct BETWEEN -7.5 AND 1.0
          AND amount >= 100000
          AND turnover_rate >= 1.5
          AND atr_qfq <= 15.0
          AND r10 >= 0.60
        """
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    BASE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BASE_PARQUET, index=False)
    return df


def symbol_of(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return ("SHSE." if suffix == "SH" else "SZSE.") + code


def write_case(pool: pd.DataFrame, case: dict) -> dict:
    df = pool.copy()
    df["blend"] = (
        df["r1"] * float(case["w1"])
        + df["r3"] * float(case["w3"])
        + df["r5"] * float(case["w5"])
        + df["r10"] * float(case["w10"])
    )
    mask = (
        (df["signal_pct_chg_raw"] <= float(case["pct"]))
        & (df["buy_open_gap_pct"] <= float(case["gap_hi"]))
        & (df["buy_open_gap_pct"] >= float(case["gap_lo"]))
        & (df["amount"] >= float(case["amount"]))
        & (df["turnover_rate"] >= float(case["turn"]))
        & (df["atr_qfq"] <= float(case["atr"]))
    )
    if "min_r1" in case:
        mask &= df["r1"] >= float(case["min_r1"])
    cand = df[mask].copy()
    if cand.empty:
        selected = cand
    else:
        selected = (
            cand.sort_values(
                ["signal_date", "blend", "r10", "pred_10d", "amount"],
                ascending=[True, False, False, False, False],
            )
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
    keep_cols = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "entry_score",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "target_pct",
        "holding_days",
        "max_holding_days",
        "score_exit_entry_ratio",
        "min_holding_days_before_score_exit",
        "score_continue_entry_ratio",
        "signal_stop_loss_pct",
        "signal_take_profit_pct",
        "strategy_variant",
        "filter_name",
        "buy_open_gap_pct",
        "daily_target_sum_after_cap",
        "r1",
        "r3",
        "r5",
        "r10",
        "blend",
    ]
    out = SIGNAL_DIR / f"{case['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    selected[keep_cols].to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["name"],
        "signal_file": str(out),
        "rows": int(len(selected)),
        "buy_days": int(selected["buy_date"].nunique()) if len(selected) else 0,
        "stock_count": int(selected["stock_code"].nunique()) if len(selected) else 0,
        "source_candidate_rows": int(len(cand)),
        "mean_daily_target_sum": float(selected.groupby("buy_date")["target_pct"].sum().mean()) if len(selected) else 0.0,
        "max_daily_target_sum": float(selected.groupby("buy_date")["target_pct"].sum().max()) if len(selected) else 0.0,
        "w1": case["w1"],
        "w3": case["w3"],
        "w5": case["w5"],
        "w10": case["w10"],
        "pct": case["pct"],
        "gap_hi": case["gap_hi"],
        "gap_lo": case["gap_lo"],
        "amount": case["amount"],
        "turn": case["turn"],
        "atr": case["atr"],
        "target": case["target"],
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
        result = base_mod.run_juejin(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
        ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "rows": result.get("rows"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    hits = [
        row
        for row in results
        if (row.get("pnl_ratio_annual") or 0) >= 5.0
        and (row.get("sharp_ratio") or 0) >= 4.0
        and (row.get("max_drawdown") or 1) <= 0.40
    ]
    summary = {
        "pool_path": str(BASE_PARQUET),
        "result_csv": str(OUT_CSV),
        "result_json": str(OUT_JSON),
        "cases": len(results),
        "target_hits": len(hits),
        "target_hits_table": sorted(
            hits,
            key=lambda row: (row.get("sharp_ratio") or -999, row.get("pnl_ratio_annual") or -999),
            reverse=True,
        ),
        "best_by_sharpe": sorted(
            results,
            key=lambda row: (row.get("sharp_ratio") or -999, row.get("pnl_ratio_annual") or -999),
            reverse=True,
        )[:5],
        "best_by_annual": sorted(
            results,
            key=lambda row: (row.get("pnl_ratio_annual") or -999, row.get("sharp_ratio") or -999),
            reverse=True,
        )[:5],
    }
    (REPORT_DIR / "full_l4_multilabel_candidate_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
