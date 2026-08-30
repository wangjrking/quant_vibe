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
SIGNAL_DIR = REPORT_DIR / "signals" / "target_hit_refine"
LOG_DIR = REPORT_DIR / "logs" / "target_hit_refine"
OUT_CSV = REPORT_DIR / "target_hit_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "target_hit_refine_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    # Current close neighbor: raise annual return while preserving the sc092 sell rule.
    {"name": "thr_global_x1035_e98_c102", "bucket_scale": {0: 1.035, 1: 1.035, 2: 1.035}, "exit": 0.98, "cont": 1.02},
    {"name": "thr_global_x1040_e98_c102", "bucket_scale": {0: 1.04, 1: 1.04, 2: 1.04}, "exit": 0.98, "cont": 1.02},
    {"name": "thr_global_x1045_e98_c102", "bucket_scale": {0: 1.045, 1: 1.045, 2: 1.045}, "exit": 0.98, "cont": 1.02},
    # Quality tilt: reduce lower bucket exposure and move risk to historically cleaner high bucket rows.
    {"name": "thr_tilt_h106_m104_l098_e98_c102", "bucket_scale": {0: 0.98, 1: 1.04, 2: 1.06}, "exit": 0.98, "cont": 1.02},
    {"name": "thr_tilt_h108_m102_l096_e98_c102", "bucket_scale": {0: 0.96, 1: 1.02, 2: 1.08}, "exit": 0.98, "cont": 1.02},
    {"name": "thr_tilt_h110_m100_l092_e98_c102", "bucket_scale": {0: 0.92, 1: 1.00, 2: 1.10}, "exit": 0.98, "cont": 1.02},
    {"name": "thr_tilt_h112_m098_l090_e98_c102", "bucket_scale": {0: 0.90, 1: 0.98, 2: 1.12}, "exit": 0.98, "cont": 1.02},
    # Exit threshold neighborhood on the most direct annual-return candidate.
    {"name": "thr_global_x1040_e975_c102", "bucket_scale": {0: 1.04, 1: 1.04, 2: 1.04}, "exit": 0.975, "cont": 1.02},
    {"name": "thr_global_x1040_e985_c102", "bucket_scale": {0: 1.04, 1: 1.04, 2: 1.04}, "exit": 0.985, "cont": 1.02},
    {"name": "thr_global_x1040_e98_c1015", "bucket_scale": {0: 1.04, 1: 1.04, 2: 1.04}, "exit": 0.98, "cont": 1.015},
    {"name": "thr_global_x1040_e98_c1025", "bucket_scale": {0: 1.04, 1: 1.04, 2: 1.04}, "exit": 0.98, "cont": 1.025},
    {"name": "thr_tilt_h108_m102_l096_e975_c1015", "bucket_scale": {0: 0.96, 1: 1.02, 2: 1.08}, "exit": 0.975, "cont": 1.015},
    {"name": "thr_tilt_h108_m102_l096_e985_c1025", "bucket_scale": {0: 0.96, 1: 1.02, 2: 1.08}, "exit": 0.985, "cont": 1.025},
]


def write_variant(case: dict) -> dict:
    df = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig")
    bucket = pd.to_numeric(df["quality_bucket"], errors="coerce").fillna(0).astype(int)
    base_target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    scale = bucket.map(case["bucket_scale"]).fillna(1.0).astype(float)
    target = base_target * scale
    daily_sum = target.groupby(df["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = target * cap_scale
    df["score_exit_entry_ratio"] = f"{float(case['exit']):.5f}"
    df["score_continue_entry_ratio"] = f"{float(case['cont']):.5f}"
    df["min_holding_days_before_score_exit"] = 1
    df["holding_days"] = 1
    df["max_holding_days"] = 3
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
        "bucket_scale": case["bucket_scale"],
        "score_exit_entry_ratio": float(case["exit"]),
        "score_continue_entry_ratio": float(case["cont"]),
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
    manifest = [write_variant(case) for case in CASES]

    results = []
    for row in manifest:
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
        "source_file": str(SOURCE_FILE),
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
    (REPORT_DIR / "target_hit_refine_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
