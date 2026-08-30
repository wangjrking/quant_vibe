from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "target_hit_inverse_refine"
LOG_DIR = REPORT_DIR / "logs" / "target_hit_inverse_refine"
OUT_CSV = REPORT_DIR / "target_hit_inverse_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "target_hit_inverse_refine_juejin_results_20260714.json"

SOURCES = {
    "mf092": REPORT_DIR
    / "signals"
    / "top3_sell_scale_refine"
    / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv",
    "hbb090": REPORT_DIR
    / "signals"
    / "top3_sell_scale_refine"
    / "ss_hbb_t3_h78_m33_l18_h3e98c102_sc090.csv",
    "hbb086": REPORT_DIR
    / "signals"
    / "top3_sell_scale_refine"
    / "ss_hbb_t3_h78_m33_l18_h3e98c102_sc086.csv",
}

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"source": "mf092", "name": "inv_mf092_l115_m108_h100", "bucket_scale": {0: 1.15, 1: 1.08, 2: 1.00}},
    {"source": "mf092", "name": "inv_mf092_l120_m110_h098", "bucket_scale": {0: 1.20, 1: 1.10, 2: 0.98}},
    {"source": "mf092", "name": "inv_mf092_l125_m112_h096", "bucket_scale": {0: 1.25, 1: 1.12, 2: 0.96}},
    {"source": "mf092", "name": "inv_mf092_l130_m115_h095", "bucket_scale": {0: 1.30, 1: 1.15, 2: 0.95}},
    {"source": "mf092", "name": "inv_mf092_l120_m115_h100", "bucket_scale": {0: 1.20, 1: 1.15, 2: 1.00}},
    {"source": "mf092", "name": "inv_mf092_l110_m115_h102", "bucket_scale": {0: 1.10, 1: 1.15, 2: 1.02}},
    {"source": "hbb090", "name": "hbb090_global_x102", "bucket_scale": {0: 1.02, 1: 1.02, 2: 1.02}},
    {"source": "hbb090", "name": "hbb090_global_x104", "bucket_scale": {0: 1.04, 1: 1.04, 2: 1.04}},
    {"source": "hbb090", "name": "hbb090_global_x106", "bucket_scale": {0: 1.06, 1: 1.06, 2: 1.06}},
    {"source": "hbb086", "name": "hbb086_global_x110", "bucket_scale": {0: 1.10, 1: 1.10, 2: 1.10}},
    {"source": "hbb086", "name": "hbb086_global_x115", "bucket_scale": {0: 1.15, 1: 1.15, 2: 1.15}},
    {"source": "hbb086", "name": "hbb086_global_x120", "bucket_scale": {0: 1.20, 1: 1.20, 2: 1.20}},
]


def write_variant(case: dict) -> dict:
    source = SOURCES[case["source"]]
    df = pd.read_csv(source, encoding="utf-8-sig")
    bucket = pd.to_numeric(df["quality_bucket"], errors="coerce").fillna(0).astype(int)
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    scale = bucket.map(case["bucket_scale"]).fillna(1.0).astype(float)
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
        "source": case["source"],
        "source_file": str(source),
        "signal_file": str(out),
        "bucket_scale": case["bucket_scale"],
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
    (REPORT_DIR / "target_hit_inverse_refine_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
