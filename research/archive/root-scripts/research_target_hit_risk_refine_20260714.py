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
    / "highsharpe_position_scale"
    / "hps_sc092_x1040_cap100.csv"
)
SIGNAL_DIR = REPORT_DIR / "signals" / "target_hit_risk_refine"
LOG_DIR = REPORT_DIR / "logs" / "target_hit_risk_refine"
OUT_CSV = REPORT_DIR / "target_hit_risk_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "target_hit_risk_refine_juejin_results_20260714.json"

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = []
for stop_loss in [0.04, 0.045, 0.05, 0.055, 0.06]:
    for take_profit in [0.07, 0.08, 0.09, 0.10, 0.12]:
        if (stop_loss, take_profit) in {(0.05, 0.08)}:
            continue
        CASES.append(
            {
                "name": f"risk_sl{int(stop_loss * 1000):03d}_tp{int(take_profit * 1000):03d}",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        )


def write_variant(case: dict) -> dict:
    df = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig")
    df["signal_stop_loss_pct"] = f"{case['stop_loss']:.5f}"
    df["signal_take_profit_pct"] = f"{case['take_profit']:.5f}"
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
    out = SIGNAL_DIR / f"{case['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["name"],
        "source_file": str(SOURCE_FILE),
        "signal_file": str(out),
        "stop_loss": float(case["stop_loss"]),
        "take_profit": float(case["take_profit"]),
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
    (REPORT_DIR / "target_hit_risk_refine_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
