from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
POOL_SCRIPT = MAIN / "research_active_l4_multilabel_pool_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "pool_pullback_gap_rule"
LOG_DIR = REPORT_DIR / "logs" / "pool_pullback_gap_rule"
OUT_CSV = REPORT_DIR / "pool_pullback_gap_rule_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "pool_pullback_gap_rule_juejin_results_20260714.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


base_mod = load_module(BASE_SCRIPT, "base_three_tier")
pool_mod = load_module(POOL_SCRIPT, "active_l4_pool")


CASES = [
    {
        "case": "pgr_w10p1_t1_p990_pct8_15_gap5_0_pos20",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.20,
        "strong_target": 0.34,
        "cap": 0.80,
    },
    {
        "case": "pgr_w10p1_t1_p990_pct8_15_gap5_0_pos30",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.30,
        "strong_target": 0.48,
        "cap": 0.90,
    },
    {
        "case": "pgr_w10p1_t1_p990_pct8_3_gap5_0_pos35",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -3.0,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.35,
        "strong_target": 0.55,
        "cap": 0.90,
    },
    {
        "case": "pgr_w10p1_t2_p990_pct8_15_gap5_0_pos18",
        "top_n": 2,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.18,
        "strong_target": 0.30,
        "cap": 0.80,
    },
    {
        "case": "pgr_w10p1_t3_p990_pct8_15_gap5_0_pos15",
        "top_n": 3,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.15,
        "strong_target": 0.24,
        "cap": 0.75,
    },
    {
        "case": "pgr_w10_t2_p990_pct8_15_gap5_0_pos20",
        "top_n": 2,
        "weights": {"p10": 1.00, "p1": 0.00, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.20,
        "strong_target": 0.32,
        "cap": 0.80,
    },
    {
        "case": "pgr_w10p5_t2_p990_pct8_15_gap5_0_pos20",
        "top_n": 2,
        "weights": {"p10": 0.85, "p1": 0.00, "p5": 0.15, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.20,
        "strong_target": 0.32,
        "cap": 0.80,
    },
    {
        "case": "pgr_w10p1_t1_p990_pct8_3_gap3_05_pos35",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -3.0,
        "gap_min": -3.0,
        "gap_max": 0.5,
        "base_target": 0.35,
        "strong_target": 0.55,
        "cap": 0.90,
    },
    {
        "case": "pgr_w10p1_t3_p990_pct8_15_gap3_05_pos16",
        "top_n": 3,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.0,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -3.0,
        "gap_max": 0.5,
        "base_target": 0.16,
        "strong_target": 0.25,
        "cap": 0.80,
    },
    {
        "case": "pgr_w10_t1_p990_p1p9_pct15_3_gap5_0_pos50",
        "top_n": 1,
        "weights": {"p10": 1.00, "p1": 0.00, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -15.0,
        "pct_max": -3.0,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.50,
        "strong_target": 0.75,
        "cap": 1.0,
    },
    {
        "case": "pgr_w10_t1_p990_p1p9_pct15_3_gap5_0_pos70",
        "top_n": 1,
        "weights": {"p10": 1.00, "p1": 0.00, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -15.0,
        "pct_max": -3.0,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.70,
        "strong_target": 0.95,
        "cap": 1.0,
    },
    {
        "case": "pgr_w10_t1_p990_p1p9_pct10_3_gap10_1_pos60",
        "top_n": 1,
        "weights": {"p10": 1.00, "p1": 0.00, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -10.0,
        "pct_max": -3.0,
        "gap_min": -10.0,
        "gap_max": 1.0,
        "base_target": 0.60,
        "strong_target": 0.90,
        "cap": 1.0,
    },
    {
        "case": "pgr_w10_t2_p990_p1p9_pct8_3_gap5_0_pos38",
        "top_n": 2,
        "weights": {"p10": 1.00, "p1": 0.00, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -8.0,
        "pct_max": -3.0,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.38,
        "strong_target": 0.55,
        "cap": 1.0,
    },
    {
        "case": "pgr_w10p1_t1_p990_p1p9_pct8_15_gap5_0_pos50",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -8.0,
        "pct_max": -1.5,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.50,
        "strong_target": 0.80,
        "cap": 1.0,
    },
    {
        "case": "pgr_w10p5_t2_p990_p1p9_pct8_3_gap5_0_pos35",
        "top_n": 2,
        "weights": {"p10": 0.85, "p1": 0.00, "p5": 0.15, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -8.0,
        "pct_max": -3.0,
        "gap_min": -5.0,
        "gap_max": 0.0,
        "base_target": 0.35,
        "strong_target": 0.52,
        "cap": 1.0,
    },
    {
        "case": "pgr_stab_w10p1_t1_p990_p1p9_pct10_15_gap3_05_pos68",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -10.0,
        "pct_max": -1.5,
        "gap_min": -3.0,
        "gap_max": 0.5,
        "base_target": 0.68,
        "strong_target": 0.68,
        "cap": 1.0,
    },
    {
        "case": "pgr_stab_w10p1_t1_p990_p1p9_pct15_15_gap3_05_pos68",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -15.0,
        "pct_max": -1.5,
        "gap_min": -3.0,
        "gap_max": 0.5,
        "base_target": 0.68,
        "strong_target": 0.68,
        "cap": 1.0,
    },
    {
        "case": "pgr_stab_w10_t1_p990_p1p9_pct10_3_gap8_1_pos68",
        "top_n": 1,
        "weights": {"p10": 1.00, "p1": 0.00, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -10.0,
        "pct_max": -3.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "base_target": 0.68,
        "strong_target": 0.68,
        "cap": 1.0,
    },
    {
        "case": "pgr_stab_w10_t1_p990_p1p9_pct15_3_gap8_1_pos68",
        "top_n": 1,
        "weights": {"p10": 1.00, "p1": 0.00, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -15.0,
        "pct_max": -3.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "base_target": 0.68,
        "strong_target": 0.68,
        "cap": 1.0,
    },
    {
        "case": "pgr_stab_w10p1_t2_p990_p1p9_pct10_15_gap3_05_pos34",
        "top_n": 2,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.990,
        "pred1_min": 0.90,
        "pct_min": -10.0,
        "pct_max": -1.5,
        "gap_min": -3.0,
        "gap_max": 0.5,
        "base_target": 0.34,
        "strong_target": 0.34,
        "cap": 0.80,
    },
    {
        "case": "pgr_stab_w10p1_t1_p980_p1p9_pct10_15_gap3_05_pos60",
        "top_n": 1,
        "weights": {"p10": 0.80, "p1": 0.20, "p5": 0.00, "p3": 0.00},
        "pred10_min": 0.980,
        "pred1_min": 0.90,
        "pct_min": -10.0,
        "pct_max": -1.5,
        "gap_min": -3.0,
        "gap_max": 0.5,
        "base_target": 0.60,
        "strong_target": 0.60,
        "cap": 1.0,
    },
]


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return (
        float(weights["p10"]) * pd.to_numeric(df["pred_10d"], errors="coerce").fillna(0.0)
        + float(weights["p1"]) * pd.to_numeric(df["pred_1d"], errors="coerce").fillna(0.0)
        + float(weights["p5"]) * pd.to_numeric(df["pred_5d"], errors="coerce").fillna(0.0)
        + float(weights["p3"]) * pd.to_numeric(df["pred_3d"], errors="coerce").fillna(0.0)
    )


def write_signal(candidates: pd.DataFrame, case: dict) -> dict:
    df = candidates.copy()
    df["entry_score"] = weighted_score(df, case["weights"])
    pct = pd.to_numeric(df["signal_pct_chg_raw"], errors="coerce")
    raw_gap = pd.to_numeric(df["buy_open_gap_raw_pct"], errors="coerce")
    mask = (
        (pd.to_numeric(df["pred_10d"], errors="coerce") >= float(case["pred10_min"]))
        & (pd.to_numeric(df["pred_1d"], errors="coerce") >= float(case["pred1_min"]))
        & (pct >= float(case["pct_min"]))
        & (pct <= float(case["pct_max"]))
        & (raw_gap >= float(case["gap_min"]))
        & (raw_gap <= float(case["gap_max"]))
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise RuntimeError(f"empty candidate for {case['case']}")
    df["sort_score"] = df["entry_score"] + 0.02 * pd.to_numeric(df["amount"], errors="coerce").rank(pct=True)
    df = (
        df.sort_values(["buy_date", "sort_score", "pred_10d"], ascending=[True, False, False])
        .groupby("buy_date", group_keys=False)
        .head(int(case["top_n"]))
        .copy()
    )
    strong = (df["pred_10d"] >= 0.995) & (df["signal_pct_chg_raw"] <= -5.0) & (df["buy_open_gap_raw_pct"] <= 0.0)
    df["target_pct"] = float(case["base_target"])
    df.loc[strong, "target_pct"] = float(case["strong_target"])
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    df["target_pct"] = df["target_pct"] * (float(case["cap"]) / daily_sum).clip(upper=1.0)
    df["rank"] = df.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
    df["pred_prob"] = df["entry_score"]
    df["holding_days"] = 1
    df["max_holding_days"] = 3
    df["score_exit_entry_ratio"] = "0.98000"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = "1.02000"
    df["signal_stop_loss_pct"] = 0.05
    df["signal_take_profit_pct"] = 0.08
    df["strategy_variant"] = case["case"]
    df["source_strategy_variant"] = "active_formal_l4_pullback_gap_pool"
    df["filter_name"] = case["case"]
    df["entry_weight_name"] = json.dumps(case["weights"], ensure_ascii=False)
    df["dynamic_hold_name"] = "h1m3_exit098_cont102"
    df["buy_day_market_available"] = df["buy_open_raw"].notna()
    df["buy_day_hard_gate_complete"] = df["buy_open_raw"].notna()
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["hybrid_source"] = "active_formal_l4_multilabel_pullback_gap"
    df["feature_weight_scale"] = 1.0
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    out_cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "amount", "turnover_rate", "total_mv", "atr_qfq",
        "signal_pct_chg_raw", "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio", "signal_stop_loss_pct",
        "signal_take_profit_pct", "strategy_variant", "source_strategy_variant", "filter_name",
        "entry_weight_name", "dynamic_hold_name", "buy_day_market_available", "buy_day_hard_gate_complete",
        "buy_day_st_rejected", "buy_day_open_limit_up_rejected", "buy_open_gap_pct", "hybrid_source",
        "buy_open_gap_raw_pct", "feature_weight_scale", "daily_target_sum_after_cap", "sort_score",
    ]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        "top_n": int(case["top_n"]),
        "pred10_min": float(case["pred10_min"]),
        "pct_min": float(case["pct_min"]),
        "pct_max": float(case["pct_max"]),
        "gap_min": float(case["gap_min"]),
        "gap_max": float(case["gap_max"]),
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
    candidates = pool_mod.build_candidates()
    manifest = [write_signal(candidates, case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
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
                    "buy_days": result.get("buy_days"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    summary = {
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best_by_sharpe": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(5).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"),
    }
    (REPORT_DIR / "pool_pullback_gap_rule_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
