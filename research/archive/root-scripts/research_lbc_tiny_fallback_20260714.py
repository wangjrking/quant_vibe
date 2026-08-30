from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
MAIN_SIGNAL = REPORT_DIR / "signals" / "lowbucket_compensation" / "lbc_q0b150_nondeep.csv"
POOL = REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"
SIGNAL_DIR = REPORT_DIR / "signals" / "lbc_tiny_fallback"
LOG_DIR = REPORT_DIR / "logs" / "lbc_tiny_fallback"
OUT_CSV = REPORT_DIR / "lbc_tiny_fallback_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "lbc_tiny_fallback_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"case": "ltf_t01_strict", "fallback_target": 0.01, "p10": 0.98, "amount": 300000, "gap_hi": 0.5, "gap_lo": -3.5, "atr": 8.0, "turn": 2.0, "pct_lo": -4.5, "pct_hi": 2.0},
    {"case": "ltf_t02_strict", "fallback_target": 0.02, "p10": 0.98, "amount": 300000, "gap_hi": 0.5, "gap_lo": -3.5, "atr": 8.0, "turn": 2.0, "pct_lo": -4.5, "pct_hi": 2.0},
    {"case": "ltf_t03_strict", "fallback_target": 0.03, "p10": 0.98, "amount": 300000, "gap_hi": 0.5, "gap_lo": -3.5, "atr": 8.0, "turn": 2.0, "pct_lo": -4.5, "pct_hi": 2.0},
    {"case": "ltf_t01_mid", "fallback_target": 0.01, "p10": 0.97, "amount": 200000, "gap_hi": 1.0, "gap_lo": -4.0, "atr": 8.0, "turn": 2.0, "pct_lo": -4.5, "pct_hi": 2.0},
    {"case": "ltf_t02_mid", "fallback_target": 0.02, "p10": 0.97, "amount": 200000, "gap_hi": 1.0, "gap_lo": -4.0, "atr": 8.0, "turn": 2.0, "pct_lo": -4.5, "pct_hi": 2.0},
    {"case": "ltf_t03_mid", "fallback_target": 0.03, "p10": 0.97, "amount": 200000, "gap_hi": 1.0, "gap_lo": -4.0, "atr": 8.0, "turn": 2.0, "pct_lo": -4.5, "pct_hi": 2.0},
]


def _symbol(code: str) -> str:
    if str(code).endswith(".SH"):
        return "SHSE." + str(code).split(".")[0]
    if str(code).endswith(".SZ"):
        return "SZSE." + str(code).split(".")[0]
    return str(code)


def build_fallback(pool: pd.DataFrame, main_buy_dates: set[str], case: dict) -> pd.DataFrame:
    df = pool.copy()
    for col in ["pred_10d", "pred_5d", "pred_1d", "amount", "buy_open_gap_raw_pct", "atr_qfq", "turnover_rate", "signal_pct_chg_raw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (
        (~df["buy_date"].astype(str).isin(main_buy_dates))
        & (df["pred_10d"] >= float(case["p10"]))
        & (df["amount"] >= float(case["amount"]))
        & (df["buy_open_gap_raw_pct"].between(float(case["gap_lo"]), float(case["gap_hi"])))
        & (df["atr_qfq"] <= float(case["atr"]))
        & (df["turnover_rate"] >= float(case["turn"]))
        & (df["signal_pct_chg_raw"].between(float(case["pct_lo"]), float(case["pct_hi"])))
        & (~df["stock_code"].astype(str).str.endswith(".BJ"))
        & (df["ST_TYPE"].fillna("").astype(str).isin(["", "0", "0.0"]))
        & (df["ST_TYPE_name"].fillna("").astype(str).isin(["", "0", "0.0"]))
    )
    df = df[mask].copy()
    if df.empty:
        return df
    df["sort_score"] = df["pred_10d"] * 10 + df["pred_5d"].fillna(0) + df["pred_1d"].fillna(0) * 0.2
    df = df.sort_values(["buy_date", "sort_score", "amount"], ascending=[True, False, False])
    df = df.groupby("buy_date", as_index=False).head(1).copy()
    df["symbol"] = df["stock_code"].map(_symbol)
    df["rank"] = 99
    df["pred_prob"] = df["pred_10d"]
    df["entry_score"] = df["pred_10d"]
    df["target_pct"] = float(case["fallback_target"])
    df["holding_days"] = 1
    df["max_holding_days"] = 2
    df["score_exit_entry_ratio"] = "0.98000"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = "1.02000"
    df["signal_stop_loss_pct"] = 0.05
    df["signal_take_profit_pct"] = 0.08
    df["source_strategy_variant"] = "active_l4_tiny_fallback"
    df["entry_weight_name"] = "tiny_fallback"
    df["dynamic_hold_name"] = "fallback_h1m2"
    df["buy_day_market_available"] = True
    df["buy_day_hard_gate_complete"] = True
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["latest_market_date"] = df["signal_date"]
    df["buy_open_gap_pct"] = df["buy_open_gap_raw_pct"]
    df["hybrid_source"] = "active_l4_tiny_fallback"
    df["feature_weight_scale"] = 1.0
    df["daily_target_sum_after_cap"] = df["target_pct"]
    df["exec_open_gap_pct"] = df["buy_open_gap_raw_pct"]
    df["quality_bucket"] = -1
    return df


def write_variant(main_signal: pd.DataFrame, pool: pd.DataFrame, case: dict) -> dict:
    fallback = build_fallback(pool, set(main_signal["buy_date"].astype(str)), case)
    columns = list(main_signal.columns)
    for col in columns:
        if col not in fallback.columns:
            fallback[col] = pd.NA
    df = pd.concat([main_signal, fallback[columns]], ignore_index=True)
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df = df.sort_values(["buy_date", "rank", "pred_prob"], ascending=[True, True, False])
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        **case,
        "signal_file": str(out),
        "rows": int(len(df)),
        "main_rows": int(len(main_signal)),
        "fallback_rows": int(len(fallback)),
        "buy_days": int(df["buy_date"].nunique()),
        "fallback_buy_days": int(fallback["buy_date"].nunique()) if len(fallback) else 0,
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(pd.to_numeric(df["target_pct"], errors="coerce").groupby(df["buy_date"]).sum().mean()),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    main_signal = pd.read_csv(MAIN_SIGNAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    pool = pd.read_parquet(POOL)
    pool["signal_date"] = pool["signal_date"].astype(str)
    pool["buy_date"] = pool["buy_date"].astype(str)
    pool["stock_code"] = pool["stock_code"].astype(str)
    manifest = [write_variant(main_signal, pool, case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        results.append(result)
        print(json.dumps({"case": result.get("case"), "fallback_rows": result.get("fallback_rows"), "pnl_ratio_annual": result.get("pnl_ratio_annual"), "sharp_ratio": result.get("sharp_ratio"), "max_drawdown": result.get("max_drawdown"), "open_count": result.get("open_count")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = frame[(pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0) & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0) & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)]
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results), "target_hits": int(len(hits))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
