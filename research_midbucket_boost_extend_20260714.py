from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE = REPORT_DIR / "signals" / "highsharpe_position_scale" / "hps_sc092_x1040_cap100.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "midbucket_boost_extend"
LOG_DIR = REPORT_DIR / "logs" / "midbucket_boost_extend"
OUT_CSV = REPORT_DIR / "midbucket_boost_extend_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "midbucket_boost_extend_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = []
for global_scale in [1.070, 1.075]:
    for q1_boost in [1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20]:
        CASES.append(
            {
                "case": f"mbe_x86_g{int(global_scale*1000):04d}_q1b{int(q1_boost*1000):04d}",
                "deep_scale": 0.86,
                "global_scale": global_scale,
                "q1_boost": q1_boost,
            }
        )


def write_variant(source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    pct = pd.to_numeric(df["signal_pct_chg_raw"], errors="coerce")
    bucket = pd.to_numeric(df["quality_bucket"], errors="coerce")
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    deep_mask = pct < -4.5
    q1_mask = (~deep_mask) & (bucket == 1)
    scale = pd.Series(float(case["global_scale"]), index=df.index)
    scale = scale.mask(deep_mask, float(case["global_scale"]) * float(case["deep_scale"]))
    scale = scale.mask(q1_mask, scale * float(case["q1_boost"]))

    df["target_pct"] = (target * scale).clip(upper=1.0)
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = df["target_pct"] * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["deepdrop_gate_scale"] = float(case["deep_scale"])
    df["midbucket_q1_boost"] = float(case["q1_boost"])

    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        **case,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "deep_scaled_rows": int(deep_mask.sum()),
        "q1_boost_rows": int(q1_mask.sum()),
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
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    manifest = [write_variant(source, case) for case in CASES]
    results = []
    for row in manifest:
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
    frame = pd.DataFrame(results)
    frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    print(
        json.dumps(
            {"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results), "target_hits": int(len(hits))},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
