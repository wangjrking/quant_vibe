from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_sell_frequency_refine"
LOG_DIR = REPORT_DIR / "logs" / "top3_sell_frequency_refine"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


BASE_CASES = [
    {
        "name": "mf_t2_sc1200_cap100",
        "path": REPORT_DIR / "signals" / "top3_multi_open_quality_sizing" / "mf_t2_sc1200_cap100.csv",
    },
    {
        "name": "ms_t3_sc110_cap91",
        "path": REPORT_DIR / "signals" / "top3_multi_open_quality_sizing" / "ms_t3_sc110_cap91.csv",
    },
    {
        "name": "hbb_t3_h78_m33_l18",
        "path": REPORT_DIR / "signals" / "top3_high_bucket_boost" / "hbb_t3_h78_m33_l18.csv",
    },
]

SELL_CASES = [
    {
        "suffix": "h1_base",
        "holding_days": 1,
        "max_holding_days": 1,
        "score_exit_entry_ratio": 0.96,
        "score_continue_entry_ratio": 9.99,
        "min_holding_days_before_score_exit": 1,
    },
    {
        "suffix": "h2_exit098_cont100",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit_entry_ratio": 0.98,
        "score_continue_entry_ratio": 1.00,
        "min_holding_days_before_score_exit": 1,
    },
    {
        "suffix": "h2_exit096_cont098",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit_entry_ratio": 0.96,
        "score_continue_entry_ratio": 0.98,
        "min_holding_days_before_score_exit": 1,
    },
    {
        "suffix": "h2_exit094_cont098",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit_entry_ratio": 0.94,
        "score_continue_entry_ratio": 0.98,
        "min_holding_days_before_score_exit": 1,
    },
    {
        "suffix": "h3_exit096_cont098",
        "holding_days": 1,
        "max_holding_days": 3,
        "score_exit_entry_ratio": 0.96,
        "score_continue_entry_ratio": 0.98,
        "min_holding_days_before_score_exit": 1,
    },
    {
        "suffix": "h3_exit098_cont102",
        "holding_days": 1,
        "max_holding_days": 3,
        "score_exit_entry_ratio": 0.98,
        "score_continue_entry_ratio": 1.02,
        "min_holding_days_before_score_exit": 1,
    },
]


def write_variant(base_name: str, source: Path, case: dict) -> dict:
    df = pd.read_csv(source, encoding="utf-8-sig")
    variant = f"sell_{base_name}_{case['suffix']}"
    df["holding_days"] = case["holding_days"]
    df["max_holding_days"] = case["max_holding_days"]
    df["score_exit_entry_ratio"] = f"{case['score_exit_entry_ratio']:.5f}"
    df["score_continue_entry_ratio"] = f"{case['score_continue_entry_ratio']:.5f}"
    df["min_holding_days_before_score_exit"] = case["min_holding_days_before_score_exit"]
    df["strategy_variant"] = variant
    df["filter_name"] = variant
    out = SIGNAL_DIR / f"{variant}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": variant,
        "base_case": base_name,
        "sell_suffix": case["suffix"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "mean_target": float(pd.to_numeric(df["target_pct"], errors="coerce").mean()) if len(df) else 0.0,
        "mean_daily_target_sum": float(pd.to_numeric(df["target_pct"], errors="coerce").groupby(df["buy_date"]).sum().mean())
        if len(df)
        else 0.0,
        **case,
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for base in BASE_CASES:
        if not base["path"].exists():
            raise FileNotFoundError(base["path"])
        for case in SELL_CASES:
            manifest.append(write_variant(base["name"], base["path"], case))

    results = []
    for row in manifest:
        result = base_mod.run_juejin(row)
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

    csv_path = REPORT_DIR / "top3_sell_frequency_refine_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_sell_frequency_refine_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
