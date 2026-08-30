from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE_SIGNAL = REPORT_DIR / "signals" / "boundary_lowbase_highboost" / "blhb_o0925_h1040.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "boundary_gap_shallow_patch"
LOG_DIR = REPORT_DIR / "logs" / "boundary_gap_shallow_patch"
OUT_CSV = REPORT_DIR / "boundary_gap_shallow_patch_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "boundary_gap_shallow_patch_juejin_results_20260714.json"

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"pos_gap_cut": 0.0, "pos_gap_scale": 0.95, "shallow_cut": -2.5, "shallow_scale": 0.95, "cap": 1.0},
    {"pos_gap_cut": 0.0, "pos_gap_scale": 0.90, "shallow_cut": -2.5, "shallow_scale": 0.95, "cap": 1.0},
    {"pos_gap_cut": 0.0, "pos_gap_scale": 0.85, "shallow_cut": -2.5, "shallow_scale": 0.95, "cap": 1.0},
    {"pos_gap_cut": 0.5, "pos_gap_scale": 0.90, "shallow_cut": -2.5, "shallow_scale": 0.95, "cap": 1.0},
    {"pos_gap_cut": 0.5, "pos_gap_scale": 0.85, "shallow_cut": -2.5, "shallow_scale": 0.95, "cap": 1.0},
    {"pos_gap_cut": 1.0, "pos_gap_scale": 0.80, "shallow_cut": -2.5, "shallow_scale": 0.95, "cap": 1.0},
    {"pos_gap_cut": 0.0, "pos_gap_scale": 0.90, "shallow_cut": -3.0, "shallow_scale": 0.90, "cap": 1.0},
    {"pos_gap_cut": 0.5, "pos_gap_scale": 0.90, "shallow_cut": -3.0, "shallow_scale": 0.90, "cap": 1.0},
    {"pos_gap_cut": 0.0, "pos_gap_scale": 0.85, "shallow_cut": -3.0, "shallow_scale": 0.90, "cap": 1.0},
    {"pos_gap_cut": 0.0, "pos_gap_scale": 0.90, "shallow_cut": -2.5, "shallow_scale": 0.90, "cap": 1.0},
]


def write_variant(params: dict) -> dict:
    df = pd.read_csv(SOURCE_SIGNAL, encoding="utf-8-sig")
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    gap = pd.to_numeric(df.get("exec_open_gap_pct"), errors="coerce")
    pct = pd.to_numeric(df.get("signal_pct_chg_raw"), errors="coerce")
    mult = pd.Series(1.0, index=df.index)
    mult = mult.mask(gap >= params["pos_gap_cut"], params["pos_gap_scale"])
    mult = mult.mask(pct >= params["shallow_cut"], mult * params["shallow_scale"])
    adjusted = target * mult
    daily_sum = adjusted.groupby(df["buy_date"]).transform("sum")
    cap_scale = (float(params["cap"]) / daily_sum).clip(upper=1.0)
    df["target_pct"] = adjusted * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case = (
        f"bgsp_pg{int(params['pos_gap_cut']*10):+03d}s{int(params['pos_gap_scale']*100):03d}_"
        f"sh{int(abs(params['shallow_cut'])*10):03d}s{int(params['shallow_scale']*100):03d}"
    ).replace("+", "p").replace("-", "m")
    df["strategy_variant"] = case
    df["filter_name"] = case
    out = SIGNAL_DIR / f"{case}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case,
        "signal_file": str(out),
        **params,
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
    if not SOURCE_SIGNAL.exists():
        raise FileNotFoundError(SOURCE_SIGNAL)
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
