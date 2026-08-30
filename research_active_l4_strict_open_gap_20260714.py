from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "active_l4_strict_open_gap"
LOG_DIR = REPORT_DIR / "logs" / "active_l4_strict_open_gap"
POOL_PATH = REPORT_DIR / "active_l4_wide_buy_quality_base_pool_20260714.parquet"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"case": "sog_a_p10_095_p5_090_gap05_t3", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_b_p10_095_p5_095_gap05_t3", "p10": 0.95, "p5": 0.95, "p3": None, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_c_p10_098_p5_095_gap05_t3", "p10": 0.98, "p5": 0.95, "p3": None, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_d_p10_095_p5_090_gapm05_t3", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -1.75, "gap": -0.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_e_p10_095_p5_090_gapm15_t3", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -1.75, "gap": -1.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_f_p10_095_p5_090_gap05_pct250_t3", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -2.50, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_g_p10_095_p5_090_gap05_p1_075_t3", "p10": 0.95, "p5": 0.90, "p3": None, "p1": 0.75, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_h_p10_095_p5_090_gap05_p3_080_t3", "p10": 0.95, "p5": 0.90, "p3": 0.80, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_i_p10_095_p5_090_gap05_t2", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 2, "cap": 0.96},
    {"case": "sog_j_p10_095_p5_090_gap05_t4", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 1.4, "top_n": 4, "cap": 0.96},
    {"case": "sog_k_p10_095_p5_090_gap05_amt20_t3", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 200000, "turn": 1.4, "top_n": 3, "cap": 0.96},
    {"case": "sog_l_p10_095_p5_090_gap05_turn25_t3", "p10": 0.95, "p5": 0.90, "p3": None, "p1": None, "pct": -1.75, "gap": 0.5, "amount": 110000, "turn": 2.5, "top_n": 3, "cap": 0.96},
]


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


def add_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["buy_open_gap_raw_pct"] = (out["buy_open_raw"] / out["buy_pre_close_raw"] - 1.0) * 100.0
    out["blend_score"] = (
        0.10 * out.groupby("signal_date")["pred_1d"].rank(method="average", pct=True)
        + 0.10 * out.groupby("signal_date")["pred_3d"].rank(method="average", pct=True)
        + 0.30 * out.groupby("signal_date")["pred_5d"].rank(method="average", pct=True)
        + 0.50 * out.groupby("signal_date")["pred_10d"].rank(method="average", pct=True)
    )
    return out


def target_for(row: pd.Series, rank: int, top_n: int, cap: float) -> float:
    target = cap / top_n
    if row["buy_open_gap_raw_pct"] <= -2.0:
        target *= 1.08
    elif row["buy_open_gap_raw_pct"] > 0:
        target *= 0.88
    if rank == 1:
        target *= 1.08
    elif rank >= 3:
        target *= 0.92
    return target


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
    if case["p3"] is not None:
        cond &= pool["pred_3d"] >= case["p3"]
    if case["p1"] is not None:
        cond &= pool["pred_1d"] >= case["p1"]
    df = pool[cond].copy()
    df = df.sort_values(["buy_date", "blend_score", "pred_10d", "amount"], ascending=[True, False, False, False])
    selected = df.groupby("buy_date", group_keys=False).head(case["top_n"]).copy()
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
    selected["source_strategy_variant"] = "active_l4_strict_open_gap_research"
    selected["filter_name"] = case["case"]
    selected["entry_weight_name"] = "strict_10d5d_open_gap"
    selected["dynamic_hold_name"] = "h1_mh3_score_continue"
    selected["buy_day_market_available"] = True
    selected["buy_day_hard_gate_complete"] = True
    selected["buy_day_st_rejected"] = False
    selected["buy_day_open_limit_up_rejected"] = False
    selected["latest_market_date"] = "20260713"
    selected["buy_open_gap_pct"] = selected["buy_open_gap_raw_pct"]
    selected["hybrid_source"] = "active_l4_formal_duckdb_strict_open_gap"
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
        **case,
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pool = add_score(pd.read_parquet(POOL_PATH))
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
    csv_path = REPORT_DIR / "active_l4_strict_open_gap_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "active_l4_strict_open_gap_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
