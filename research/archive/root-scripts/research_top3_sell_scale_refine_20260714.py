from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_sell_scale_refine"
LOG_DIR = REPORT_DIR / "logs" / "top3_sell_scale_refine"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = [
    {
        "name": "ms_t3_sc110_cap91",
        "path": REPORT_DIR / "signals" / "top3_multi_open_quality_sizing" / "ms_t3_sc110_cap91.csv",
        "max_holding_days": 3,
        "score_exit_entry_ratio": 0.98,
        "score_continue_entry_ratio": 1.02,
        "scales": [1.08, 1.12, 1.16, 1.20, 1.24, 1.28],
    },
    {
        "name": "mf_t2_sc1200_cap100",
        "path": REPORT_DIR / "signals" / "top3_multi_open_quality_sizing" / "mf_t2_sc1200_cap100.csv",
        "max_holding_days": 3,
        "score_exit_entry_ratio": 0.98,
        "score_continue_entry_ratio": 1.02,
        "scales": [0.84, 0.88, 0.92, 0.96],
    },
    {
        "name": "hbb_t3_h78_m33_l18",
        "path": REPORT_DIR / "signals" / "top3_high_bucket_boost" / "hbb_t3_h78_m33_l18.csv",
        "max_holding_days": 3,
        "score_exit_entry_ratio": 0.98,
        "score_continue_entry_ratio": 1.02,
        "scales": [0.78, 0.82, 0.86, 0.90],
    },
]


def write_variant(source: dict, scale: float) -> dict:
    df = pd.read_csv(source["path"], encoding="utf-8-sig")
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0) * scale
    daily_sum = target.groupby(df["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = target * cap_scale
    df["holding_days"] = 1
    df["max_holding_days"] = int(source["max_holding_days"])
    df["score_exit_entry_ratio"] = f"{float(source['score_exit_entry_ratio']):.5f}"
    df["score_continue_entry_ratio"] = f"{float(source['score_continue_entry_ratio']):.5f}"
    df["min_holding_days_before_score_exit"] = 1
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case = f"ss_{source['name']}_h3e98c102_sc{int(scale * 100):03d}"
    df["strategy_variant"] = case
    df["filter_name"] = case
    out = SIGNAL_DIR / f"{case}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case,
        "base_case": source["name"],
        "target_scale": scale,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "mean_target": float(df["target_pct"].mean()) if len(df) else 0.0,
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()) if len(df) else 0.0,
        "holding_days": 1,
        "max_holding_days": int(source["max_holding_days"]),
        "score_exit_entry_ratio": float(source["score_exit_entry_ratio"]),
        "score_continue_entry_ratio": float(source["score_continue_entry_ratio"]),
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for source in SOURCES:
        if not source["path"].exists():
            raise FileNotFoundError(source["path"])
        for scale in source["scales"]:
            manifest.append(write_variant(source, scale))

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

    csv_path = REPORT_DIR / "top3_sell_scale_refine_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_sell_scale_refine_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
