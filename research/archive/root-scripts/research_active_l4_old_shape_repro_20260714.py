from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "active_l4_old_shape_repro"
LOG_DIR = REPORT_DIR / "logs" / "active_l4_old_shape_repro"
POOL_PATH = REPORT_DIR / "active_l4_wide_buy_quality_base_pool_20260714.parquet"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"case": "osr_a_p10_098_p5_097_gapm08_top1", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_b_p10_098_p5_097_gapm20_top1", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -2.0, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_c_p10_098_p5_097_p1_090_gapm08_top1", "p10": 0.98, "p5": 0.97, "p3": None, "p1": 0.90, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_d_p10_098_p5_097_p3_090_gapm08_top1", "p10": 0.98, "p5": 0.97, "p3": 0.90, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_e_p10_099_p5_098_gapm08_top1", "p10": 0.99, "p5": 0.98, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_f_p10_098_p5_097_gapm08_top2", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 2, "high": 0.58, "mid": 0.32, "low": 0.18},
    {"case": "osr_g_p10_098_p5_097_gapm08_amt20_top1", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 200000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_h_p10_098_p5_097_gapm08_turn25_top1", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 2.5, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_i_p10_098_p5_097_gapm08_pct300_top1", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -3.0, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24},
    {"case": "osr_j_p10_098_p5_097_gapm08_mh1", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24, "mh": 1},
    {"case": "osr_k_p10_098_p5_097_gapm08_mh2", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.96, "mid": 0.42, "low": 0.24, "mh": 2},
    {"case": "osr_l_p10_098_p5_097_gapm08_scale80", "p10": 0.98, "p5": 0.97, "p3": None, "p1": None, "pct": -1.75, "gap": -0.8, "amount": 120000, "turn": 1.4, "top_n": 1, "high": 0.80, "mid": 0.36, "low": 0.20},
]


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


def prepare_pool() -> pd.DataFrame:
    df = pd.read_parquet(POOL_PATH)
    df["buy_open_gap_raw_pct"] = (df["buy_open_raw"] / df["buy_pre_close_raw"] - 1.0) * 100.0
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        df[f"{col}_rank"] = df.groupby("signal_date")[col].rank(method="average", pct=True)
    df["blend_score"] = 0.10 * df["pred_1d_rank"] + 0.10 * df["pred_3d_rank"] + 0.30 * df["pred_5d_rank"] + 0.50 * df["pred_10d_rank"]
    return df


def target_for(row: pd.Series, case: dict) -> float:
    if row["pred_10d"] >= 0.995 and row["pred_5d"] >= 0.99 and row["buy_open_gap_raw_pct"] <= -2.0:
        target = case["high"]
    elif row["pred_10d"] >= 0.985 and row["pred_5d"] >= 0.97:
        target = case["mid"]
    else:
        target = case["low"]
    if row["signal_pct_chg_raw"] <= -5.0:
        target *= 1.08
    if row["buy_open_gap_raw_pct"] > -0.5:
        target *= 0.85
    return min(0.96, target)


def write_case(pool: pd.DataFrame, case: dict) -> dict:
    cond = (
        (pool["pred_10d"] >= case["p10"])
        & (pool["pred_5d"] >= case["p5"])
        & (pool["signal_pct_chg_raw"] <= case["pct"])
        & (pool["buy_open_gap_raw_pct"] <= case["gap"])
        & (pool["buy_open_gap_raw_pct"] >= -8.0)
        & (pool["amount"] >= case["amount"])
        & (pool["turnover_rate"] >= case["turn"])
        & (~pool["buy_open_limit_up_rejected"])
    )
    if case.get("p1") is not None:
        cond &= pool["pred_1d"] >= case["p1"]
    if case.get("p3") is not None:
        cond &= pool["pred_3d"] >= case["p3"]
    df = pool[cond].copy()
    df = df.sort_values(["buy_date", "blend_score", "pred_10d", "amount"], ascending=[True, False, False, False])
    selected = df.groupby("buy_date", group_keys=False).head(case["top_n"]).copy()
    selected["rank"] = selected.groupby("buy_date").cumcount() + 1
    selected["target_pct"] = selected.apply(lambda r: target_for(r, case), axis=1)
    selected["symbol"] = selected["stock_code"].map(to_symbol)
    selected["pred_prob"] = selected["blend_score"]
    selected["entry_score"] = selected["blend_score"]
    selected["atr_qfq"] = None
    selected["holding_days"] = 1
    selected["max_holding_days"] = int(case.get("mh", 3))
    selected["score_exit_entry_ratio"] = "0.98000"
    selected["min_holding_days_before_score_exit"] = 1
    selected["score_continue_entry_ratio"] = "1.02000"
    selected["signal_stop_loss_pct"] = 0.05
    selected["signal_take_profit_pct"] = 0.08
    selected["strategy_variant"] = case["case"]
    selected["source_strategy_variant"] = "active_l4_old_shape_repro_research"
    selected["filter_name"] = case["case"]
    selected["entry_weight_name"] = "old_shape_active_l4"
    selected["dynamic_hold_name"] = "h1_score_continue"
    selected["buy_day_market_available"] = True
    selected["buy_day_hard_gate_complete"] = True
    selected["buy_day_st_rejected"] = False
    selected["buy_day_open_limit_up_rejected"] = False
    selected["latest_market_date"] = "20260713"
    selected["buy_open_gap_pct"] = selected["buy_open_gap_raw_pct"]
    selected["hybrid_source"] = "active_l4_formal_duckdb_old_shape"
    selected["feature_weight_scale"] = 1.0
    selected["daily_target_sum_after_cap"] = selected.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "amount", "turnover_rate", "total_mv", "atr_qfq",
        "signal_pct_chg_raw", "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio", "signal_stop_loss_pct",
        "signal_take_profit_pct", "strategy_variant", "source_strategy_variant", "filter_name",
        "entry_weight_name", "dynamic_hold_name", "buy_day_market_available", "buy_day_hard_gate_complete",
        "buy_day_st_rejected", "buy_day_open_limit_up_rejected", "latest_market_date", "buy_open_gap_pct",
        "hybrid_source", "buy_open_gap_raw_pct", "feature_weight_scale", "daily_target_sum_after_cap",
    ]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    selected[cols].to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(selected)),
        "buy_days": int(selected["buy_date"].nunique()) if len(selected) else 0,
        "stock_count": int(selected["stock_code"].nunique()) if len(selected) else 0,
        "mean_target": float(selected["target_pct"].mean()) if len(selected) else 0.0,
        "mean_daily_target_sum": float(selected.groupby("buy_date")["target_pct"].sum().mean()) if len(selected) else 0.0,
        **case,
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pool = prepare_pool()
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
    csv_path = REPORT_DIR / "active_l4_old_shape_repro_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "active_l4_old_shape_repro_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
