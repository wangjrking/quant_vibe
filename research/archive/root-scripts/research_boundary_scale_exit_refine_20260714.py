from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE_SIGNAL = REPORT_DIR / "signals" / "top3_multi_open_quality_sizing" / "mf_t2_sc1200_cap100.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "boundary_scale_exit_refine"
LOG_DIR = REPORT_DIR / "logs" / "boundary_scale_exit_refine"
OUT_CSV = REPORT_DIR / "boundary_scale_exit_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "boundary_scale_exit_refine_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SCALES = [0.925, 0.930, 0.935, 0.940, 0.945, 0.950, 0.955, 0.960]
EXIT_RATIOS = [0.975, 0.980, 0.985]
CONTINUE_RATIOS = [1.015, 1.020, 1.025, 1.030]


def write_variant(scale: float, exit_ratio: float, continue_ratio: float) -> dict:
    df = pd.read_csv(SOURCE_SIGNAL, encoding="utf-8-sig")
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0) * scale
    daily_sum = target.groupby(df["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = target * cap_scale
    df["holding_days"] = 1
    df["max_holding_days"] = 3
    df["score_exit_entry_ratio"] = f"{exit_ratio:.5f}"
    df["score_continue_entry_ratio"] = f"{continue_ratio:.5f}"
    df["min_holding_days_before_score_exit"] = 1
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case = f"bser_mf_sc{int(scale * 1000):04d}_e{int(exit_ratio * 1000):03d}_c{int(continue_ratio * 1000):04d}"
    df["strategy_variant"] = case
    df["filter_name"] = case
    out = SIGNAL_DIR / f"{case}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case,
        "base_case": "mf_t2_sc1200_cap100",
        "target_scale": scale,
        "score_exit_entry_ratio": exit_ratio,
        "score_continue_entry_ratio": continue_ratio,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "mean_target": float(df["target_pct"].mean()) if len(df) else 0.0,
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()) if len(df) else 0.0,
        "holding_days": 1,
        "max_holding_days": 3,
    }


def main() -> None:
    if not SOURCE_SIGNAL.exists():
        raise FileNotFoundError(SOURCE_SIGNAL)
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        write_variant(scale=scale, exit_ratio=exit_ratio, continue_ratio=continue_ratio)
        for scale in SCALES
        for exit_ratio in EXIT_RATIOS
        for continue_ratio in CONTINUE_RATIOS
    ]

    results = []
    for row in rows:
        log_path = LOG_DIR / f"{row['case']}.log"
        if log_path.exists() and log_path.stat().st_size > 0:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            ind = base_mod.extract_indicator(text)
            if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
                result = dict(row)
                result["returncode"] = 0
                result["log_file"] = str(log_path)
                result.update(ind)
            else:
                result = base_mod.run_juejin(row)
        else:
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
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "returncode": result.get("returncode"),
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
