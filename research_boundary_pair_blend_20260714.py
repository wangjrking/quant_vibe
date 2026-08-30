from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
HIGH_ANNUAL = REPORT_DIR / "signals" / "highsharpe_position_scale" / "hps_sc092_x1040_cap100.csv"
HIGH_SHARPE = REPORT_DIR / "signals" / "target_hit_inverse_refine" / "inv_mf092_l120_m110_h098.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "boundary_pair_blend"
LOG_DIR = REPORT_DIR / "logs" / "boundary_pair_blend"
OUT_CSV = REPORT_DIR / "boundary_pair_blend_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "boundary_pair_blend_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = []
for alpha in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
    for scale in [1.00, 1.015, 1.03]:
        CASES.append({"alpha": alpha, "scale": scale, "cap": 1.0})


KEYS = ["signal_date", "buy_date", "stock_code"]


def load_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(HIGH_ANNUAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    s = pd.read_csv(HIGH_SHARPE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    return a, s


def write_variant(annual: pd.DataFrame, sharpe: pd.DataFrame, case: dict) -> dict:
    alpha = float(case["alpha"])
    scale = float(case["scale"])
    name = f"bpb_a{int(alpha * 100):02d}_sc{int(scale * 1000):04d}"
    right = sharpe[KEYS + ["target_pct"]].rename(columns={"target_pct": "target_sharpe"})
    df = annual.merge(right, on=KEYS, how="left")
    if df["target_sharpe"].isna().any():
        missing = int(df["target_sharpe"].isna().sum())
        raise RuntimeError(f"{name} missing paired target rows: {missing}")
    target_annual = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    target_sharpe = pd.to_numeric(df["target_sharpe"], errors="coerce").fillna(0.0)
    blended = (alpha * target_annual + (1.0 - alpha) * target_sharpe) * scale
    daily_sum = blended.groupby(df["buy_date"]).transform("sum")
    cap_scale = (float(case["cap"]) / daily_sum).clip(upper=1.0)
    df["target_pct"] = blended * cap_scale
    df["strategy_variant"] = name
    df["filter_name"] = name
    df["blend_alpha_high_annual"] = alpha
    df["blend_scale"] = scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df = df.drop(columns=["target_sharpe"])
    out = SIGNAL_DIR / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": name,
        "alpha_high_annual": alpha,
        "scale": scale,
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
    annual, sharpe = load_pair()
    rows = [write_variant(annual, sharpe, case) for case in CASES]
    results = []
    for row in rows:
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
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
