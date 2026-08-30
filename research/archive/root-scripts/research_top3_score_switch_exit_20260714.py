from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SWITCH_STRATEGY_DIR = REPORT_DIR / "code_snapshots" / "score_switch_exit"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_score_switch_exit"
LOG_DIR = REPORT_DIR / "logs" / "top3_score_switch_exit"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = [
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

CASES = [
    {"max_hold": 2, "ratio": 0.98, "reference": "min"},
    {"max_hold": 2, "ratio": 1.00, "reference": "min"},
    {"max_hold": 2, "ratio": 1.02, "reference": "min"},
    {"max_hold": 3, "ratio": 0.98, "reference": "min"},
    {"max_hold": 3, "ratio": 1.00, "reference": "min"},
    {"max_hold": 3, "ratio": 1.02, "reference": "min"},
    {"max_hold": 4, "ratio": 1.00, "reference": "min"},
    {"max_hold": 3, "ratio": 0.98, "reference": "mean"},
    {"max_hold": 3, "ratio": 1.00, "reference": "mean"},
]


def write_signal(source: dict, case: dict) -> dict:
    df = pd.read_csv(source["path"], encoding="utf-8-sig")
    variant = f"sw_{source['name']}_mh{case['max_hold']}_r{int(case['ratio'] * 100):03d}_{case['reference']}"
    df["holding_days"] = 1
    df["max_holding_days"] = int(case["max_hold"])
    df["score_exit_entry_ratio"] = "9.99000"
    df["score_continue_entry_ratio"] = "9.99000"
    df["min_holding_days_before_score_exit"] = 1
    df["strategy_variant"] = variant
    df["filter_name"] = variant
    out = SIGNAL_DIR / f"{variant}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": variant,
        "base_case": source["name"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "mean_target": float(pd.to_numeric(df["target_pct"], errors="coerce").mean()) if len(df) else 0.0,
        "mean_daily_target_sum": float(pd.to_numeric(df["target_pct"], errors="coerce").groupby(df["buy_date"]).sum().mean())
        if len(df)
        else 0.0,
        "switch_max_hold": int(case["max_hold"]),
        "switch_ratio": float(case["ratio"]),
        "switch_reference": case["reference"],
    }


def run_with_switch(row: dict) -> dict:
    os.environ["GM_SCORE_SWITCH_EXIT_MODE"] = "1"
    os.environ["GM_SCORE_SWITCH_SIGNAL_FIELD"] = "pred_10d"
    os.environ["GM_SCORE_SWITCH_REFERENCE"] = str(row["switch_reference"])
    os.environ["GM_SCORE_SWITCH_RATIO"] = str(row["switch_ratio"])
    os.environ["GM_SCORE_SWITCH_MIN_HOLDING_DAYS"] = "1"
    os.environ["GM_SCORE_SWITCH_MAX_HOLDING_DAYS"] = str(row["switch_max_hold"])
    os.environ["GM_SCORE_SWITCH_KEEP_IF_IN_TODAY_SIGNALS"] = "1"
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.STRATEGY_DIR = SWITCH_STRATEGY_DIR
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for source in SOURCES:
        if not source["path"].exists():
            raise FileNotFoundError(source["path"])
        for case in CASES:
            manifest.append(write_signal(source, case))

    results = []
    for row in manifest:
        result = run_with_switch(row)
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

    csv_path = REPORT_DIR / "top3_score_switch_exit_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_score_switch_exit_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
