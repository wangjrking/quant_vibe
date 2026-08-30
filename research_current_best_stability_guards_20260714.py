from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE = REPORT_DIR / "signals" / "open_gap_deep_rebalance" / "ogd_gap1x50_deep8up110.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "current_best_stability_guards"
LOG_DIR = REPORT_DIR / "logs" / "current_best_stability_guards"
OUT_CSV = REPORT_DIR / "current_best_stability_guards_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "current_best_stability_guards_juejin_results_20260714.json"
SUMMARY_JSON = REPORT_DIR / "current_best_stability_guards_summary_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"case": "guard_target_cap070", "target_cap": 0.70},
    {"case": "guard_target_cap068", "target_cap": 0.68},
    {"case": "guard_target_cap066", "target_cap": 0.66},
    {"case": "guard_target_cap064", "target_cap": 0.64},
    {"case": "guard_target_cap062", "target_cap": 0.62},
    {"case": "guard_target_cap060", "target_cap": 0.60},
    {"case": "guard_target_cap055", "target_cap": 0.55},
    {"case": "guard_target_cap050", "target_cap": 0.50},
    {"case": "guard_target_cap045", "target_cap": 0.45},
    {"case": "guard_bucket2_x085", "bucket2_scale": 0.85},
    {"case": "guard_bucket2_x075", "bucket2_scale": 0.75},
    {"case": "guard_mv_min30w", "total_mv_min": 300000},
    {"case": "guard_mv_min50w", "total_mv_min": 500000},
    {"case": "guard_amount_min15w", "amount_min": 150000},
    {"case": "guard_amount_min20w", "amount_min": 200000},
    {"case": "guard_turnover_max10", "turnover_max": 10.0},
    {"case": "guard_turnover_max8", "turnover_max": 8.0},
    {"case": "guard_atr_max5", "atr_max": 5.0},
    {"case": "guard_target_cap055_bucket2_x085", "target_cap": 0.55, "bucket2_scale": 0.85},
    {"case": "guard_target_cap055_turnover_max10", "target_cap": 0.55, "turnover_max": 10.0},
    {"case": "guard_target_cap055_amount_min15w", "target_cap": 0.55, "amount_min": 150000},
]


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in ["target_pct", "quality_bucket", "total_mv", "amount", "turnover_rate", "atr_qfq"]:
        if col in df.columns:
            df[col] = num(df, col)
    return df


def write_variant(source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    before_rows = len(df)
    if "total_mv_min" in case:
        df = df[df["total_mv"] >= float(case["total_mv_min"])].copy()
    if "amount_min" in case:
        df = df[df["amount"] >= float(case["amount_min"])].copy()
    if "turnover_max" in case:
        df = df[df["turnover_rate"] <= float(case["turnover_max"])].copy()
    if "atr_max" in case:
        df = df[df["atr_qfq"] <= float(case["atr_max"])].copy()
    if "bucket2_scale" in case:
        mask = df["quality_bucket"] == 2
        df.loc[mask, "target_pct"] = df.loc[mask, "target_pct"] * float(case["bucket2_scale"])
    if "target_cap" in case:
        df["target_pct"] = df["target_pct"].clip(upper=float(case["target_cap"]))
    if df.empty:
        raise RuntimeError(f"empty candidate for {case['case']}")
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    df["target_pct"] = df["target_pct"] * (1.0 / daily_sum).clip(upper=1.0)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "removed_rows": int(before_rows - len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        **{k: v for k, v in case.items() if k != "case"},
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
    source = load_source()
    manifest = [write_variant(source, case) for case in CASES]
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
                    "removed": result.get("removed_rows"),
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
        "best_by_sharpe": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(8).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
