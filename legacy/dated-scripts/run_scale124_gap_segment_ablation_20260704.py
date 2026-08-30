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
SIGNAL_DIR = OUT_DIR / "scale124_gap_segment_signals"
RUN_DIR = OUT_DIR / "scale124_gap_segment_ablation"
SUMMARY = OUT_DIR / "scale124_gap_segment_ablation_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    ("all_base", -3.0, 0.0, 1.00, 0.82),
    ("deep_gap_only", -3.0, -2.5, 1.00, 0.82),
    ("mild_gap_only", -1.5, 0.0, 1.00, 0.82),
    ("mild_gap_m15_m05", -1.5, -0.5, 1.00, 0.82),
    ("mild_gap_m10_0", -1.0, 0.0, 1.00, 0.82),
    ("deep_scaled90", -3.0, -2.5, 0.90, 0.75),
    ("mild_scaled90", -1.5, 0.0, 0.90, 0.75),
    ("mild_scaled110", -1.5, 0.0, 1.10, 0.82),
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
    name, low, high, scale, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    gap = pd.to_numeric(df["buy_open_gap_raw_pct"], errors="coerce")
    df = df[(gap >= float(low)) & (gap <= float(high))].copy()
    df["target_pct"] = (df["target_pct"].astype(float) * float(scale)).clip(lower=0.05, upper=float(cap))
    out = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out, {
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "avg_target": float(df["target_pct"].mean()) if len(df) else None,
        "avg_gap": float(pd.to_numeric(df["buy_open_gap_raw_pct"], errors="coerce").mean()) if len(df) else None,
    }


def run_case(case: tuple) -> dict[str, object]:
    name, low, high, scale, cap = case
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
        "gap_low": low,
        "gap_high": high,
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
    rows = sorted(rows, key=lambda r: (float(r.get("sharp_ratio") or -999), float(r.get("pnl_ratio_annual") or -999)), reverse=True)
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
