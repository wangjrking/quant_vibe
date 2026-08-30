from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "active_l4_wide_buy_quality"
LOG_DIR = REPORT_DIR / "logs" / "active_l4_wide_buy_quality"
ASSET_DIR = ROOT / "quant" / "data_file" / "production_assets" / "duckdb"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


DBS = {
    "p1": ASSET_DIR / "l4_executable_1d_open_return_formal.duckdb",
    "p3": ASSET_DIR / "l4_executable_3d_open_return_formal.duckdb",
    "p5": ASSET_DIR / "l4_executable_5d_open_return_formal.duckdb",
    "p10": ASSET_DIR / "l4_executable_10d_open_return_formal.duckdb",
    "m": ASSET_DIR / "l2_stock_daily_data.duckdb",
}
TABLES = {
    "p1": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate",
    "p3": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate",
    "p5": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate",
    "p10": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate",
    "m": "STOCK_DAILY_DATA",
}

CASES = [
    {"case": "wbq_a_pctl_m175_gap15_amt12_mv20_t3", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_b_pctl_m125_gap15_amt12_mv20_t3", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.25, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_c_pctl_m175_gap05_amt12_mv20_t3", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.75, "gap_max": 0.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_d_pctl_m175_gap25_amt12_mv20_t3", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.75, "gap_max": 2.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_e_pctl_m175_gap15_amt20_mv20_t3", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 200000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_f_pctl_m175_gap15_amt12_mv50_t3", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 120000, "mv_min": 500000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_g_10d80_5d20_m175_gap15_t3", "w": (0.00, 0.00, 0.20, 0.80), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_h_10d60_5d30_3d10_m175_gap15_t3", "w": (0.00, 0.10, 0.30, 0.60), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_i_10d50_5d30_3d10_1d10_m175_t3", "w": (0.10, 0.10, 0.30, 0.50), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
    {"case": "wbq_j_pctl_m175_gap15_amt12_mv20_t2", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 2, "cap": 0.96},
    {"case": "wbq_k_pctl_m175_gap15_amt12_mv20_t4", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -1.75, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 4, "cap": 0.96},
    {"case": "wbq_l_pctl_m250_gap15_amt12_mv20_t3", "w": (0.05, 0.10, 0.20, 0.65), "pct_max": -2.50, "gap_max": 1.5, "amount_min": 120000, "mv_min": 200000, "top_n": 3, "cap": 0.96},
]


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


def build_base_pool() -> pd.DataFrame:
    con = duckdb.connect()
    for alias, path in DBS.items():
        con.execute(f"ATTACH '{path.as_posix()}' AS {alias} (READ_ONLY)")
    query = f"""
    WITH calendar AS (
        SELECT
            trade_date AS signal_date,
            LEAD(trade_date) OVER (ORDER BY trade_date) AS buy_date
        FROM (SELECT DISTINCT trade_date FROM m.{TABLES['m']})
    ),
    joined AS (
        SELECT
            p10.trade_date AS signal_date,
            cal.buy_date,
            p10.stock_code,
            sm.name,
            p1.pred_prob AS pred_1d,
            p3.pred_prob AS pred_3d,
            p5.pred_prob AS pred_5d,
            p10.pred_prob AS pred_10d,
            sm.amount,
            sm.turnover_rate,
            sm.total_mv,
            sm.pct_chg AS signal_pct_chg_raw,
            sm.close AS signal_close_raw,
            sm.close_qfq AS signal_close_qfq,
            sm.ST_TYPE AS signal_st_type,
            sm.ST_TYPE_name AS signal_st_name,
            bm.open AS buy_open_raw,
            bm.pre_close AS buy_pre_close_raw,
            bm.ST_TYPE AS buy_st_type,
            bm.ST_TYPE_name AS buy_st_name,
            bm.amount AS buy_amount,
            bm.turnover_rate AS buy_turnover_rate,
            bm.total_mv AS buy_total_mv
        FROM p10.{TABLES['p10']} p10
        JOIN p5.{TABLES['p5']} p5 USING (trade_date, stock_code)
        JOIN p3.{TABLES['p3']} p3 USING (trade_date, stock_code)
        JOIN p1.{TABLES['p1']} p1 USING (trade_date, stock_code)
        JOIN calendar cal ON cal.signal_date = p10.trade_date
        JOIN m.{TABLES['m']} sm ON sm.trade_date = p10.trade_date AND sm.stock_code = p10.stock_code
        JOIN m.{TABLES['m']} bm ON bm.trade_date = cal.buy_date AND bm.stock_code = p10.stock_code
        WHERE cal.buy_date IS NOT NULL
          AND p10.stock_code NOT LIKE '%%.BJ'
          AND sm.market <> '北交所'
          AND sm.close IS NOT NULL
          AND bm.open IS NOT NULL
          AND bm.pre_close IS NOT NULL
          AND sm.amount IS NOT NULL
          AND sm.total_mv IS NOT NULL
          AND sm.pct_chg IS NOT NULL
          AND COALESCE(sm.ST_TYPE, '') IN ('', '0')
          AND COALESCE(bm.ST_TYPE, '') IN ('', '0')
          AND COALESCE(sm.ST_TYPE_name, '') NOT LIKE '%%ST%%'
          AND COALESCE(bm.ST_TYPE_name, '') NOT LIKE '%%ST%%'
          AND sm.name NOT LIKE '%%退%%'
    )
    SELECT * FROM joined
    WHERE signal_date >= '20220606'
      AND signal_date <= '20260713'
    """
    df = con.execute(query).fetchdf()
    con.close()
    df["buy_open_gap_raw_pct"] = (df["buy_open_raw"] / df["signal_close_raw"] - 1.0) * 100.0
    df["buy_open_limit_up_rejected"] = df.apply(
        lambda r: bool(r["buy_open_raw"] >= r["buy_pre_close_raw"] * (1.198 if str(r["stock_code"]).startswith(("300", "301", "688")) else 1.098)),
        axis=1,
    )
    return df


def add_scores(df: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.DataFrame:
    out = df.copy()
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        out[f"{col}_rank"] = out.groupby("signal_date")[col].rank(method="average", pct=True)
    w1, w3, w5, w10 = weights
    out["blend_score"] = (
        w1 * out["pred_1d_rank"]
        + w3 * out["pred_3d_rank"]
        + w5 * out["pred_5d_rank"]
        + w10 * out["pred_10d_rank"]
    )
    return out


def target_for(row: pd.Series, rank: int, top_n: int, cap: float) -> float:
    base = cap / max(top_n, 1)
    if rank == 1:
        base *= 1.10
    elif rank >= 3:
        base *= 0.92
    if row["buy_open_gap_raw_pct"] > 0.5:
        base *= 0.85
    if row["signal_pct_chg_raw"] <= -5.0:
        base *= 1.08
    return max(0.0, base)


def write_case(pool: pd.DataFrame, case: dict) -> dict:
    df = add_scores(pool, case["w"])
    filtered = df[
        (df["signal_pct_chg_raw"] <= case["pct_max"])
        & (df["buy_open_gap_raw_pct"] <= case["gap_max"])
        & (df["buy_open_gap_raw_pct"] >= -8.0)
        & (df["amount"] >= case["amount_min"])
        & (df["total_mv"] >= case["mv_min"])
        & (~df["buy_open_limit_up_rejected"])
        & (df["pred_10d"] >= 0.70)
    ].copy()
    filtered = filtered.sort_values(
        ["buy_date", "blend_score", "pred_10d", "amount"],
        ascending=[True, False, False, False],
    )
    selected = filtered.groupby("buy_date", group_keys=False).head(case["top_n"]).copy()
    selected["rank"] = selected.groupby("buy_date").cumcount() + 1
    selected["target_pct"] = selected.apply(lambda r: target_for(r, int(r["rank"]), case["top_n"], case["cap"]), axis=1)
    sums = selected.groupby("buy_date")["target_pct"].transform("sum")
    selected["target_pct"] *= (case["cap"] / sums).clip(upper=1.0)
    selected["symbol"] = selected["stock_code"].map(to_symbol)
    selected["pred_prob"] = selected["blend_score"]
    selected["entry_score"] = selected["blend_score"]
    selected["atr_qfq"] = None
    selected["holding_days"] = 1
    selected["max_holding_days"] = 3
    selected["score_exit_entry_ratio"] = "0.98000"
    selected["min_holding_days_before_score_exit"] = 1
    selected["score_continue_entry_ratio"] = "1.02000"
    selected["signal_stop_loss_pct"] = 0.05
    selected["signal_take_profit_pct"] = 0.08
    selected["strategy_variant"] = case["case"]
    selected["source_strategy_variant"] = "active_l4_wide_buy_quality_research"
    selected["filter_name"] = case["case"]
    selected["entry_weight_name"] = "rank_blend_active_l4"
    selected["dynamic_hold_name"] = "h1_mh3_score_continue"
    selected["buy_day_market_available"] = True
    selected["buy_day_hard_gate_complete"] = True
    selected["buy_day_st_rejected"] = False
    selected["buy_day_open_limit_up_rejected"] = False
    selected["latest_market_date"] = "20260713"
    selected["buy_open_gap_pct"] = selected["buy_open_gap_raw_pct"]
    selected["hybrid_source"] = "active_l4_formal_duckdb_wide_pool"
    selected["feature_weight_scale"] = 1.0
    selected["daily_target_sum_after_cap"] = selected.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
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
        "source_strategy_variant",
        "filter_name",
        "entry_weight_name",
        "dynamic_hold_name",
        "buy_day_market_available",
        "buy_day_hard_gate_complete",
        "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected",
        "latest_market_date",
        "buy_open_gap_pct",
        "hybrid_source",
        "buy_open_gap_raw_pct",
        "feature_weight_scale",
        "daily_target_sum_after_cap",
    ]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    selected[cols].to_csv(out, index=False, encoding="utf-8-sig")
    shortage = selected.groupby("buy_date").size()
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(selected)),
        "buy_days": int(selected["buy_date"].nunique()) if len(selected) else 0,
        "stock_count": int(selected["stock_code"].nunique()) if len(selected) else 0,
        "mean_target": float(selected["target_pct"].mean()) if len(selected) else 0.0,
        "mean_daily_target_sum": float(selected.groupby("buy_date")["target_pct"].sum().mean()) if len(selected) else 0.0,
        "days_below_topn": int((shortage < case["top_n"]).sum()) if len(selected) else 0,
        **{k: v for k, v in case.items() if k != "w"},
        "w1": case["w"][0],
        "w3": case["w"][1],
        "w5": case["w"][2],
        "w10": case["w"][3],
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_base_pool()
    pool_path = REPORT_DIR / "active_l4_wide_buy_quality_base_pool_20260714.parquet"
    pool.to_parquet(pool_path, index=False)
    manifest = [write_case(pool, case) for case in CASES]
    results = []
    for row in manifest:
        result = base_mod.run_juejin(row)
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "rows": result.get("rows"),
                    "buy_days": result.get("buy_days"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    csv_path = REPORT_DIR / "active_l4_wide_buy_quality_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "active_l4_wide_buy_quality_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "pool": str(pool_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
