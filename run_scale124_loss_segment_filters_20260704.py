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
SIGNAL_DIR = OUT_DIR / "scale124_loss_segment_filter_signals"
RUN_DIR = OUT_DIR / "scale124_loss_segment_filters"
SUMMARY = OUT_DIR / "scale124_loss_segment_filter_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, filter_mid_mv, filter_sigchg_5_8, filter_missing_mv, target_scale, target_cap
    ("base", False, False, False, 1.00, 0.82),
    ("no_mv30_50", True, False, True, 1.00, 0.82),
    ("no_sigchg5_8", False, True, False, 1.00, 0.82),
    ("no_mv30_50_no_sig5_8", True, True, True, 1.00, 0.82),
    ("no_mv30_50_scale110", True, False, True, 1.10, 0.82),
    ("no_sig5_8_scale110", False, True, False, 1.10, 0.82),
    ("no_both_scale110", True, True, True, 1.10, 0.82),
    ("no_both_scale120", True, True, True, 1.20, 0.82),
    ("no_both_scale090", True, True, True, 0.90, 0.75),
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
    name, filter_mid_mv, filter_sigchg_5_8, filter_missing_mv, scale, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["buy_total_mv", "signal_pct_chg", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if filter_missing_mv:
        mask &= df["buy_total_mv"].notna()
    if filter_mid_mv:
        mask &= ~((df["buy_total_mv"] > 300000) & (df["buy_total_mv"] <= 500000))
    if filter_sigchg_5_8:
        mask &= ~((df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0))
    outdf = df.loc[mask].copy()
    outdf["target_pct"] = (outdf["target_pct"] * float(scale)).clip(lower=0.05, upper=float(cap))
    signal_file = SIGNAL_DIR / f"{name}.csv"
    outdf.to_csv(signal_file, index=False, encoding="utf-8-sig")
    meta = {
        "signal_file": str(signal_file),
        "rows": int(len(outdf)),
        "buy_days": int(outdf["buy_date"].nunique()) if len(outdf) else 0,
        "removed_rows": int(len(df) - len(outdf)),
        "avg_target": float(outdf["target_pct"].mean()) if len(outdf) else None,
    }
    return signal_file, meta


def run_case(case: tuple) -> dict[str, object]:
    name, filter_mid_mv, filter_sigchg_5_8, filter_missing_mv, scale, cap = case
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
        "filter_mid_mv": filter_mid_mv,
        "filter_sigchg_5_8": filter_sigchg_5_8,
        "filter_missing_mv": filter_missing_mv,
        "target_scale": scale,
        "target_cap": cap,
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
    rows = sorted(
        rows,
        key=lambda r: (
            float(r.get("sharp_ratio") or -999),
            float(r.get("pnl_ratio_annual") or -999),
        ),
        reverse=True,
    )
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
