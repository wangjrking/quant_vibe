from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE_SIGNAL = REPORT_DIR / "signals" / "top3_sell_scale_refine" / "ss_mf_t2_sc1200_cap100_h3e98c102_sc096.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "boundary_quality_reweight"
LOG_DIR = REPORT_DIR / "logs" / "boundary_quality_reweight"
OUT_CSV = REPORT_DIR / "boundary_quality_reweight_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "boundary_quality_reweight_juejin_results_20260714.json"

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"low": 0.90, "mid": 0.95, "high": 1.00, "cap": 0.96},
    {"low": 0.80, "mid": 0.90, "high": 1.00, "cap": 0.96},
    {"low": 0.70, "mid": 0.85, "high": 1.00, "cap": 0.96},
    {"low": 0.80, "mid": 0.90, "high": 1.05, "cap": 0.96},
    {"low": 0.70, "mid": 0.85, "high": 1.08, "cap": 0.96},
    {"low": 0.70, "mid": 0.80, "high": 1.10, "cap": 0.96},
    {"low": 0.90, "mid": 0.95, "high": 1.05, "cap": 1.00},
    {"low": 0.80, "mid": 0.90, "high": 1.08, "cap": 1.00},
    {"low": 0.70, "mid": 0.85, "high": 1.12, "cap": 1.00},
    {"low": 0.60, "mid": 0.80, "high": 1.15, "cap": 1.00},
]


def write_variant(params: dict) -> dict:
    df = pd.read_csv(SOURCE_SIGNAL, encoding="utf-8-sig")
    bucket = pd.to_numeric(df.get("quality_bucket"), errors="coerce").fillna(0).astype(int)
    mult = bucket.map({0: params["low"], 1: params["mid"], 2: params["high"]}).fillna(params["low"])
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0) * mult
    daily_sum = target.groupby(df["buy_date"]).transform("sum")
    cap_scale = (float(params["cap"]) / daily_sum).clip(upper=1.0)
    df["target_pct"] = target * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case = (
        f"bqr_l{int(params['low']*100):03d}_m{int(params['mid']*100):03d}_"
        f"h{int(params['high']*100):03d}_cap{int(params['cap']*100):03d}"
    )
    df["strategy_variant"] = case
    df["filter_name"] = case
    out = SIGNAL_DIR / f"{case}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case,
        "signal_file": str(out),
        "low_quality_scale": params["low"],
        "mid_quality_scale": params["mid"],
        "high_quality_scale": params["high"],
        "daily_cap": params["cap"],
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_target": float(df["target_pct"].mean()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
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
    results = []
    for params in CASES:
        row = write_variant(params)
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
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    hits = frame[(frame["pnl_ratio_annual"] >= 5.0) & (frame["sharp_ratio"] >= 4.0) & (frame["max_drawdown"] <= 0.4)]
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results), "target_hits": len(hits)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
