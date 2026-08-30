from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE_FILE = (
    REPORT_DIR
    / "signals"
    / "top3_sell_frequency_refine"
    / "sell_ms_t3_sc110_cap91_h3_exit098_cont102.csv"
)
SIGNAL_DIR = REPORT_DIR / "signals" / "highsharpe_lift_refine"
LOG_DIR = REPORT_DIR / "logs" / "highsharpe_lift_refine"
OUT_CSV = REPORT_DIR / "highsharpe_lift_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "highsharpe_lift_refine_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"name": "hsl_global_x104", "base": 1.04},
    {"name": "hsl_global_x106", "base": 1.06},
    {"name": "hsl_global_x108", "base": 1.08},
    {"name": "hsl_global_x110", "base": 1.10},
    {"name": "hsl_pred99_deep5_x122_else100", "base": 1.00, "pred99_deep5": 1.22},
    {"name": "hsl_pred99_deep5_x130_else098", "base": 0.98, "pred99_deep5": 1.30},
    {"name": "hsl_q2_pred99_x120_else098", "base": 0.98, "q2_pred99": 1.20},
    {"name": "hsl_liq_pred99_x125_else098", "base": 0.98, "liq_pred99": 1.25},
    {"name": "hsl_gap_neg_pred99_x125_else098", "base": 0.98, "gap_neg_pred99": 1.25},
    {"name": "hsl_combo_strong_x135_else096", "base": 0.96, "combo_strong": 1.35},
    {"name": "hsl_combo_strong_x145_else094", "base": 0.94, "combo_strong": 1.45},
    {"name": "hsl_trim_bad_boost_good", "base": 1.03, "trim_bad": 0.82, "combo_strong": 1.35},
]


def n(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def write_variant(case: dict) -> dict:
    df = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    target = n(df, "target_pct") * float(case.get("base", 1.0))
    pred10 = n(df, "pred_10d")
    pred1 = n(df, "pred_1d")
    pct = n(df, "signal_pct_chg_raw")
    amount = n(df, "amount")
    atr = n(df, "atr_qfq", 99.0)
    qbucket = n(df, "quality_bucket").astype(int)
    gap = n(df, "buy_open_gap_raw_pct", 999.0)
    if "buy_open_gap_pct" in df.columns:
        gap = gap.where(gap != 999.0, n(df, "buy_open_gap_pct", 999.0))

    scale = pd.Series(1.0, index=df.index, dtype="float64")
    pred99_deep5 = (pred10 >= 0.99) & (pct <= -5.0)
    q2_pred99 = (qbucket == 2) & (pred10 >= 0.99)
    liq_pred99 = (pred10 >= 0.99) & (amount >= 500000)
    gap_neg_pred99 = (pred10 >= 0.99) & (gap <= 0.5) & (gap >= -5.0)
    combo_strong = (pred10 >= 0.985) & (pred1 >= 0.90) & (pct <= -5.0) & (amount >= 300000) & (atr <= 8.0)
    bad = (pct > -2.5) | (gap > 0.75) | (atr >= 10.0)

    if "pred99_deep5" in case:
        scale.loc[pred99_deep5] *= float(case["pred99_deep5"])
    if "q2_pred99" in case:
        scale.loc[q2_pred99] *= float(case["q2_pred99"])
    if "liq_pred99" in case:
        scale.loc[liq_pred99] *= float(case["liq_pred99"])
    if "gap_neg_pred99" in case:
        scale.loc[gap_neg_pred99] *= float(case["gap_neg_pred99"])
    if "combo_strong" in case:
        scale.loc[combo_strong] *= float(case["combo_strong"])
    if "trim_bad" in case:
        scale.loc[bad] *= float(case["trim_bad"])

    target = target * scale
    daily_sum = target.groupby(df["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = target * cap_scale
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
    df["lift_combo_strong"] = combo_strong.astype(int)
    df["lift_bad"] = bad.astype(int)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    out = SIGNAL_DIR / f"{case['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["name"],
        "source_file": str(SOURCE_FILE),
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "combo_strong_rows": int(combo_strong.sum()),
        "bad_rows": int(bad.sum()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
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

    rows = [write_variant(case) for case in CASES]
    results = []
    for row in rows:
        result = run_or_parse(row)
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
