from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
MAIN = ROOT / "quant/main"
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
SIGNAL_DIR = OUT_DIR / "scale124_risk_bucket_downweight_signals"
RUN_DIR = OUT_DIR / "scale124_risk_bucket_downweight"
SUMMARY = OUT_DIR / "scale124_risk_bucket_downweight_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, sig5_8_mult, sig_gt8_mult, mid_mv_mult, deep_gap_possig_mult, global_scale, cap
    ("base", 1.0, 1.0, 1.0, 1.0, 1.00, 0.82),
    ("sig5_8_half", 0.50, 1.0, 1.0, 1.0, 1.00, 0.82),
    ("sig_gt5_half", 0.50, 0.50, 1.0, 1.0, 1.00, 0.82),
    ("mid_mv_half", 1.0, 1.0, 0.50, 1.0, 1.00, 0.82),
    ("deep_possig_half", 1.0, 1.0, 1.0, 0.50, 1.00, 0.82),
    ("sig_mid_combo", 0.55, 0.70, 0.65, 1.0, 1.00, 0.82),
    ("sig_mid_combo_scale110", 0.55, 0.70, 0.65, 1.0, 1.10, 0.82),
    ("allrisk_combo", 0.55, 0.70, 0.65, 0.65, 1.00, 0.82),
    ("allrisk_combo_scale115", 0.55, 0.70, 0.65, 0.65, 1.15, 0.82),
]


def parse_indicator(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if not isinstance(indicator, str):
        return indicator or {}
    text = re.sub(r"datetime\.datetime\(.*?\)\)", "'datetime'", indicator)
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_signal(case: tuple) -> tuple[Path, dict[str, object]]:
    name, sig5_8_mult, sig_gt8_mult, mid_mv_mult, deep_gap_possig_mult, global_scale, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["buy_total_mv", "signal_pct_chg", "buy_open_gap_raw_pct", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mult = pd.Series(1.0, index=df.index)
    mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= float(sig5_8_mult)
    mult.loc[df["signal_pct_chg"] > 8.0] *= float(sig_gt8_mult)
    mult.loc[(df["buy_total_mv"] > 300000) & (df["buy_total_mv"] <= 500000)] *= float(mid_mv_mult)
    mult.loc[(df["buy_open_gap_raw_pct"] <= -2.5) & (df["signal_pct_chg"] > 0.0)] *= float(deep_gap_possig_mult)
    df["target_pct"] = (df["target_pct"] * mult * float(global_scale)).clip(lower=0.05, upper=float(cap))
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    touched = int((mult != 1.0).sum())
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "downweighted_rows": touched,
        "avg_target": float(df["target_pct"].mean()),
        "min_target": float(df["target_pct"].min()),
        "max_target": float(df["target_pct"].max()),
    }


def run_case(case: tuple) -> dict[str, object]:
    name, sig5_8_mult, sig_gt8_mult, mid_mv_mult, deep_gap_possig_mult, global_scale, cap = case
    signal_file, meta = build_signal(case)
    log_file = RUN_DIR / f"{name}.log"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--holding-days",
        "2",
        "--target-position-pct",
        str(cap),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        "0.900",
        "--min-holding-days-before-score-exit",
        "2",
        "--score-continue-entry-ratio",
        "1.000",
        "--max-holding-days",
        "4",
        "--light-stop-loss-pct",
        "0.080",
        "--min-holding-days-before-light-stop",
        "1",
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    indicator = parse_indicator(proc.stdout)
    return {
        "name": name,
        "sig5_8_mult": sig5_8_mult,
        "sig_gt8_mult": sig_gt8_mult,
        "mid_mv_mult": mid_mv_mult,
        "deep_gap_possig_mult": deep_gap_possig_mult,
        "global_scale": global_scale,
        "cap": cap,
        **meta,
        "returncode": proc.returncode,
        "log_file": str(log_file),
        "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
        "sharp_ratio": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "pnl_ratio": indicator.get("pnl_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "win_ratio": indicator.get("win_ratio"),
        "calmar_ratio": indicator.get("calmar_ratio"),
    }


def main() -> int:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        print(f"RUN {case[0]}", flush=True)
        rows.append(run_case(case))
    rows = sorted(rows, key=lambda r: (float(r.get("sharp_ratio") or -999), float(r.get("pnl_ratio_annual") or -999)), reverse=True)
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
