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
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_SIGNAL = (
    ROOT
    / "quant/main/strategy_library/production/prod_repro500_p435_s96_p10d70_v20260702/signals/full_history_repro500_p435_s96_p10d70.csv"
)
OUT_DIR = REPORT_DIR / "repro500_exit_frequency_frontier"
SIGNAL_DIR = OUT_DIR / "signals"
LOG_DIR = OUT_DIR / "juejin_runs"
SUMMARY_FILE = OUT_DIR / "exit_frequency_frontier_summary.csv"

STRATEGY_DIR = (
    ROOT
    / "quant/data_file/reports/strategy_agent_latest_l4_weight_candidates_20260702/postrank_open_filter_candidates/code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant/main/run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    {"name": "base_sl04_tp07_exit096", "target": 0.435, "sl": 0.04, "tp": 0.07, "exit": 0.96},
    {"name": "base_sl05_tp08_exit096", "target": 0.435, "sl": 0.05, "tp": 0.08, "exit": 0.96},
    {"name": "base_sl06_tp09_exit096", "target": 0.435, "sl": 0.06, "tp": 0.09, "exit": 0.96},
    {"name": "base_sl08_tp12_exit096", "target": 0.435, "sl": 0.08, "tp": 0.12, "exit": 0.96},
    {"name": "base_nostop_exit096", "target": 0.435, "sl": None, "tp": None, "exit": 0.96},
    {"name": "high_sl04_tp07_exit096", "target": 0.445, "sl": 0.04, "tp": 0.07, "exit": 0.96},
    {"name": "high_sl05_tp08_exit096", "target": 0.445, "sl": 0.05, "tp": 0.08, "exit": 0.96},
    {"name": "high_sl06_tp09_exit096", "target": 0.445, "sl": 0.06, "tp": 0.09, "exit": 0.96},
    {"name": "high_sl05_tp08_exit094", "target": 0.445, "sl": 0.05, "tp": 0.08, "exit": 0.94},
    {"name": "high_sl05_tp08_exit098", "target": 0.445, "sl": 0.05, "tp": 0.08, "exit": 0.98},
]


def write_signal(case: dict) -> Path:
    df = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = f"{float(case['target']):.5f}"
    df["holding_days"] = "1"
    df["max_holding_days"] = "1"
    df["score_exit_entry_ratio"] = f"{float(case['exit']):.5f}"
    if case["sl"] is not None:
        df["signal_stop_loss_pct"] = f"{float(case['sl']):.5f}"
    if case["tp"] is not None:
        df["signal_take_profit_pct"] = f"{float(case['tp']):.5f}"
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
    case_dir = SIGNAL_DIR / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "signals.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    return out


def parse_indicator(stdout: str) -> dict:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        text = re.sub(r"datetime\.datetime\(.*?\)\)", "'datetime'", indicator)
        try:
            return ast.literal_eval(text)
        except Exception:
            return {}
    return indicator or {}


def run_case(signal_file: Path, case: dict) -> tuple[int, dict, Path]:
    log_file = LOG_DIR / f"{case['name']}.log"
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
        "3",
        "--holding-days",
        "1",
        "--max-holding-days",
        "1",
        "--score-exit-entry-ratio",
        str(case["exit"]),
        "--min-holding-days-before-score-exit",
        "1",
        "--score-continue-entry-ratio",
        "9.99",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    if case["sl"] is not None:
        cmd.extend(["--light-stop-loss-pct", str(case["sl"]), "--min-holding-days-before-light-stop", "1"])
    if case["tp"] is not None:
        cmd.extend(["--take-profit-pct", str(case["tp"])])
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (LOG_DIR / f"{case['name']}.runner.json").write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, parse_indicator(proc.stdout), log_file


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        signal_file = write_signal(case)
        print("RUN", case["name"], flush=True)
        returncode, indicator, log_file = run_case(signal_file, case)
        rows.append(
            {
                **case,
                "returncode": returncode,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                "sharp_ratio": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
            }
        )
    rows = sorted(rows, key=lambda x: (x.get("sharp_ratio") or -999, x.get("pnl_ratio_annual") or -999), reverse=True)
    with SUMMARY_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
