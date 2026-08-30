from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE_FILE = (
    REPORT_DIR
    / "signals"
    / "top3_sell_scale_refine"
    / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv"
)
SIGNAL_DIR = REPORT_DIR / "signals" / "score_bucket_weight_refine"
LOG_DIR = REPORT_DIR / "logs" / "score_bucket_weight_refine"
OUT_CSV = REPORT_DIR / "score_bucket_weight_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "score_bucket_weight_refine_juejin_results_20260714.json"

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"name": "sb_entry99_x118_else096", "base": 0.96, "entry99": 1.18},
    {"name": "sb_entry985_x115_else098", "base": 0.98, "entry985": 1.15},
    {"name": "sb_pred10_99_x115_else098", "base": 0.98, "pred10_99": 1.15},
    {"name": "sb_entry99_pred99_x125_else095", "base": 0.95, "entry99": 1.15, "pred10_99": 1.10},
    {"name": "sb_deep5_entry97_x112_else098", "base": 0.98, "deep5": 1.08, "entry97": 1.04},
    {"name": "sb_deep8_entry98_x118_else096", "base": 0.96, "deep8": 1.10, "entry98": 1.08},
    {"name": "sb_midbucket_entry98_x118_else095", "base": 0.95, "midbucket": 1.10, "entry98": 1.08},
    {"name": "sb_lowmid_boost_high_trim", "base": 1.00, "lowbucket": 1.20, "midbucket": 1.12, "highbucket": 0.96},
    {"name": "sb_lowmid_score_boost", "base": 0.98, "lowbucket": 1.15, "midbucket": 1.12, "entry98": 1.05, "highbucket": 0.98},
    {"name": "sb_balanced_edge", "base": 1.00, "entry985": 1.08, "pred10_99": 1.06, "pos_gap": 0.75, "deep_gap": 1.06},
    {"name": "sb_balanced_edge_x102", "base": 1.02, "entry985": 1.08, "pred10_99": 1.06, "pos_gap": 0.75, "deep_gap": 1.06},
    {"name": "sb_balanced_edge_x104", "base": 1.04, "entry985": 1.08, "pred10_99": 1.06, "pos_gap": 0.75, "deep_gap": 1.06},
]


def write_variant(case: dict) -> dict:
    df = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig")
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    entry = pd.to_numeric(df["entry_score"], errors="coerce")
    pred10 = pd.to_numeric(df["pred_10d"], errors="coerce")
    pct = pd.to_numeric(df["signal_pct_chg_raw"], errors="coerce")
    bucket = pd.to_numeric(df["quality_bucket"], errors="coerce").fillna(0).astype(int)
    gap = pd.to_numeric(df.get("buy_open_gap_pct"), errors="coerce")

    scale = pd.Series(float(case.get("base", 1.0)), index=df.index)
    if "entry99" in case:
        scale.loc[entry >= 0.99] *= float(case["entry99"])
    if "entry985" in case:
        scale.loc[entry >= 0.985] *= float(case["entry985"])
    if "entry98" in case:
        scale.loc[entry >= 0.98] *= float(case["entry98"])
    if "entry97" in case:
        scale.loc[entry >= 0.97] *= float(case["entry97"])
    if "pred10_99" in case:
        scale.loc[pred10 >= 0.99] *= float(case["pred10_99"])
    if "deep5" in case:
        scale.loc[pct <= -5.0] *= float(case["deep5"])
    if "deep8" in case:
        scale.loc[pct <= -8.0] *= float(case["deep8"])
    if "lowbucket" in case:
        scale.loc[bucket == 0] *= float(case["lowbucket"])
    if "midbucket" in case:
        scale.loc[bucket == 1] *= float(case["midbucket"])
    if "highbucket" in case:
        scale.loc[bucket == 2] *= float(case["highbucket"])
    if "pos_gap" in case:
        scale.loc[gap > 0.5] *= float(case["pos_gap"])
    if "deep_gap" in case:
        scale.loc[gap <= -3.0] *= float(case["deep_gap"])

    scaled = target * scale
    daily_sum = scaled.groupby(df["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = scaled * cap_scale
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
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
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        row = write_variant(case)
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
    (REPORT_DIR / "score_bucket_weight_refine_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
