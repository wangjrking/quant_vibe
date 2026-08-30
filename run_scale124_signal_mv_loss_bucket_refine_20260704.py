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
SIGNAL_DIR = OUT_DIR / "scale124_signal_mv_loss_bucket_refine_signals"
RUN_DIR = OUT_DIR / "scale124_signal_mv_loss_bucket_refine_grid"
SUMMARY = OUT_DIR / "scale124_signal_mv_loss_bucket_refine_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, sig5_8, sig_gt8, mid_mv, missing_mv, small_mv_boost, low_mv_boost, global_scale
    ("base_signal_mv045_s120", 0.45, 0.55, 0.45, 1.00, 1.00, 1.00, 1.20),
    ("missing_half_s125", 0.45, 0.55, 0.45, 0.50, 1.00, 1.00, 1.25),
    ("missing_zero_s130", 0.45, 0.55, 0.45, 0.00, 1.00, 1.00, 1.30),
    ("mid025_missing050_s135", 0.45, 0.55, 0.25, 0.50, 1.00, 1.00, 1.35),
    ("sig025_mid025_missing050_s145", 0.25, 0.45, 0.25, 0.50, 1.00, 1.00, 1.45),
    ("sig035_mid025_missing030_s145", 0.35, 0.50, 0.25, 0.30, 1.00, 1.00, 1.45),
    ("boost_small_mid025_missing050_s130", 0.45, 0.55, 0.25, 0.50, 1.10, 1.05, 1.30),
    ("boost_small_sig035_mid025_s140", 0.35, 0.50, 0.25, 0.50, 1.15, 1.08, 1.40),
    ("no_mid_missing_s150", 0.45, 0.55, 0.00, 0.00, 1.15, 1.08, 1.50),
    ("soft_mid035_missing030_s135", 0.40, 0.50, 0.35, 0.30, 1.10, 1.05, 1.35),
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


def build_signal(case: tuple[str, float, float, float, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, sig5_8, sig_gt8, mid_mv, missing_mv, small_mv_boost, low_mv_boost, global_scale = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["signal_total_mv", "signal_pct_chg", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mult = pd.Series(1.0, index=df.index)
    mv = df["signal_total_mv"]
    mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= sig5_8
    mult.loc[df["signal_pct_chg"] > 8.0] *= sig_gt8
    mult.loc[(mv > 300000) & (mv <= 500000)] *= mid_mv
    mult.loc[mv.isna()] *= missing_mv
    mult.loc[mv <= 200000] *= small_mv_boost
    mult.loc[(mv > 200000) & (mv <= 300000)] *= low_mv_boost
    df["target_pct"] = (df["target_pct"] * mult * global_scale).clip(lower=0.03, upper=0.82)
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(df["target_pct"].mean()),
        "downweighted_rows": int((mult < 1.0).sum()),
        "boosted_rows": int((mult > 1.0).sum()),
        "zero_rows": int((df["target_pct"] <= 0.031).sum()),
    }


def run_case(case: tuple[str, float, float, float, float, float, float, float]) -> dict[str, object]:
    name, sig5_8, sig_gt8, mid_mv, missing_mv, small_mv_boost, low_mv_boost, global_scale = case
    signal_file, meta = build_signal(case)
    log_file = RUN_DIR / f"{name}.log"
    runner_json = RUN_DIR / f"{name}.runner.json"
    if runner_json.exists() and log_file.exists():
        stdout = runner_json.read_text(encoding="utf-8", errors="ignore")
        returncode = 0
    else:
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
            "0.82",
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
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")
    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "sig5_8": sig5_8,
        "sig_gt8": sig_gt8,
        "mid_mv": mid_mv,
        "missing_mv": missing_mv,
        "small_mv_boost": small_mv_boost,
        "low_mv_boost": low_mv_boost,
        "global_scale": global_scale,
        **meta,
        "returncode": returncode,
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
    rows.sort(key=lambda r: (float(r.get("sharp_ratio") or -999), float(r.get("pnl_ratio_annual") or -999)), reverse=True)
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
