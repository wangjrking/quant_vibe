from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "highsharpe_position_scale"
LOG_DIR = REPORT_DIR / "logs" / "highsharpe_position_scale"
OUT_CSV = REPORT_DIR / "highsharpe_position_scale_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "highsharpe_position_scale_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "sc092": REPORT_DIR / "signals" / "top3_sell_scale_refine" / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv",
    "sc088": REPORT_DIR / "signals" / "top3_sell_scale_refine" / "ss_mf_t2_sc1200_cap100_h3e98c102_sc088.csv",
    "ms_h3": REPORT_DIR / "signals" / "top3_sell_frequency_refine" / "sell_ms_t3_sc110_cap91_h3_exit098_cont102.csv",
    "ms_base": REPORT_DIR / "signals" / "top3_multi_open_quality_sizing" / "ms_t3_sc110_cap91.csv",
}


CASES = []
for source, scales in {
    "sc092": [1.04, 1.06, 1.08, 1.10, 1.12, 1.15],
    "sc088": [1.10, 1.15, 1.20, 1.25],
    "ms_h3": [1.12, 1.18, 1.24, 1.30, 1.36],
    "ms_base": [1.15, 1.25, 1.35, 1.45],
}.items():
    for scale in scales:
        CASES.append({"source": source, "scale": scale, "cap": 1.0})


def write_variant(case: dict) -> dict:
    src = SOURCES[case["source"]]
    df = pd.read_csv(src, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0) * float(case["scale"])
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (float(case["cap"]) / daily_sum).clip(upper=1.0)
    df["target_pct"] = df["target_pct"] * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    name = f"hps_{case['source']}_x{int(case['scale'] * 1000):04d}_cap100"
    df["strategy_variant"] = name
    df["filter_name"] = name
    out = SIGNAL_DIR / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": name,
        "source": case["source"],
        "source_file": str(src),
        "scale": float(case["scale"]),
        "cap": float(case["cap"]),
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
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
    results = []
    for case in CASES:
        row = write_variant(case)
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
                    "mean_daily_target_sum": result.get("mean_daily_target_sum"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    hits = frame[
        (frame["pnl_ratio_annual"] >= 5.0)
        & (frame["sharp_ratio"] >= 4.0)
        & (frame["max_drawdown"] <= 0.4)
    ]
    summary = {
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best_by_sharpe": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(10).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"),
    }
    (REPORT_DIR / "highsharpe_position_scale_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
