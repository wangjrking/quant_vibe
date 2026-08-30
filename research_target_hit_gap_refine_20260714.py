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
SIGNAL_DIR = REPORT_DIR / "signals" / "target_hit_gap_refine"
LOG_DIR = REPORT_DIR / "logs" / "target_hit_gap_refine"
OUT_CSV = REPORT_DIR / "target_hit_gap_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "target_hit_gap_refine_juejin_results_20260714.json"

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"name": "gap_cut_pos05_x108", "global_scale": 1.08, "pos05_scale": 0.0, "pos10_scale": 0.0, "deep_scale": 1.0, "core_scale": 1.0},
    {"name": "gap_half_pos05_x108", "global_scale": 1.08, "pos05_scale": 0.5, "pos10_scale": 0.5, "deep_scale": 1.0, "core_scale": 1.0},
    {"name": "gap_cut_pos10_x107", "global_scale": 1.07, "pos05_scale": 1.0, "pos10_scale": 0.0, "deep_scale": 1.0, "core_scale": 1.0},
    {"name": "gap_half_pos10_x108", "global_scale": 1.08, "pos05_scale": 1.0, "pos10_scale": 0.5, "deep_scale": 1.0, "core_scale": 1.0},
    {"name": "gap_core_boost_x106", "global_scale": 1.06, "pos05_scale": 0.5, "pos10_scale": 0.0, "deep_scale": 0.9, "core_scale": 1.10},
    {"name": "gap_core_boost_x108", "global_scale": 1.08, "pos05_scale": 0.5, "pos10_scale": 0.0, "deep_scale": 0.9, "core_scale": 1.10},
    {"name": "gap_soft_core_x106", "global_scale": 1.06, "pos05_scale": 0.8, "pos10_scale": 0.4, "deep_scale": 0.95, "core_scale": 1.05},
    {"name": "gap_soft_core_x108", "global_scale": 1.08, "pos05_scale": 0.8, "pos10_scale": 0.4, "deep_scale": 0.95, "core_scale": 1.05},
    {"name": "gap_neg1_boost_x106", "global_scale": 1.06, "pos05_scale": 0.8, "pos10_scale": 0.4, "deep_scale": 0.95, "core_scale": 1.0, "neg1_scale": 1.10},
    {"name": "gap_neg1_boost_x108", "global_scale": 1.08, "pos05_scale": 0.8, "pos10_scale": 0.4, "deep_scale": 0.95, "core_scale": 1.0, "neg1_scale": 1.10},
]


def write_variant(case: dict) -> dict:
    df = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig")
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    gap = pd.to_numeric(df.get("buy_open_gap_pct"), errors="coerce")
    scale = pd.Series(float(case["global_scale"]), index=df.index)

    deep = gap.notna() & (gap <= -5.0)
    core = gap.notna() & (gap > -3.0) & (gap <= 0.5)
    neg1 = gap.notna() & (gap > -1.0) & (gap <= 0.0)
    pos05 = gap.notna() & (gap > 0.5) & (gap <= 1.0)
    pos10 = gap.notna() & (gap > 1.0)

    scale.loc[deep] *= float(case.get("deep_scale", 1.0))
    scale.loc[core] *= float(case.get("core_scale", 1.0))
    scale.loc[neg1] *= float(case.get("neg1_scale", 1.0))
    scale.loc[pos05] *= float(case.get("pos05_scale", 1.0))
    scale.loc[pos10] *= float(case.get("pos10_scale", 1.0))

    scaled = target * scale
    df = df[scaled > 0].copy()
    scaled = scaled[scaled > 0]
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
    (REPORT_DIR / "target_hit_gap_refine_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
