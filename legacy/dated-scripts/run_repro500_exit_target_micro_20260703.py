from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_SIGNAL = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_repro500_p435_s96_p10d70_v20260702"
    / "signals"
    / "full_history_repro500_p435_s96_p10d70.csv"
)
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant" / "main" / "run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

OUT_DIR = REPORT_DIR / "repro500_exit_target_micro"
LOG_DIR = OUT_DIR / "juejin_runs"


CASES = [
    # Current production baseline, included for exact same-run comparison.
    {"name": "base_p435_s960_sl05_tp08", "target": 0.435, "exit": 0.960, "stop": 0.050, "take": 0.080},
    # Target micro-neighborhood around the annual-return boundary.
    {"name": "p432_s960_sl05_tp08", "target": 0.432, "exit": 0.960, "stop": 0.050, "take": 0.080},
    {"name": "p438_s960_sl05_tp08", "target": 0.438, "exit": 0.960, "stop": 0.050, "take": 0.080},
    {"name": "p440_s960_sl05_tp08", "target": 0.440, "exit": 0.960, "stop": 0.050, "take": 0.080},
    # Score-exit sensitivity.
    {"name": "p435_s955_sl05_tp08", "target": 0.435, "exit": 0.955, "stop": 0.050, "take": 0.080},
    {"name": "p435_s965_sl05_tp08", "target": 0.435, "exit": 0.965, "stop": 0.050, "take": 0.080},
    {"name": "p435_s970_sl05_tp08", "target": 0.435, "exit": 0.970, "stop": 0.050, "take": 0.080},
    # Light stop / take-profit sensitivity.
    {"name": "p435_s960_sl04_tp08", "target": 0.435, "exit": 0.960, "stop": 0.040, "take": 0.080},
    {"name": "p435_s960_sl06_tp08", "target": 0.435, "exit": 0.960, "stop": 0.060, "take": 0.080},
    {"name": "p435_s960_sl05_tp07", "target": 0.435, "exit": 0.960, "stop": 0.050, "take": 0.070},
    {"name": "p435_s960_sl05_tp09", "target": 0.435, "exit": 0.960, "stop": 0.050, "take": 0.090},
    # Combined small nudges.
    {"name": "p432_s965_sl04_tp08", "target": 0.432, "exit": 0.965, "stop": 0.040, "take": 0.080},
    {"name": "p438_s955_sl06_tp09", "target": 0.438, "exit": 0.955, "stop": 0.060, "take": 0.090},
]


def _copy_signal(case: dict) -> Path:
    case_dir = OUT_DIR / "signals" / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "signals.csv"
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as src, out.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if "target_pct" in row:
                row["target_pct"] = f"{float(case['target']):.5f}"
            if "score_exit_entry_ratio" in row:
                row["score_exit_entry_ratio"] = f"{float(case['exit']):.5f}"
            if "signal_stop_loss_pct" in row:
                row["signal_stop_loss_pct"] = f"{float(case['stop']):.5f}"
            if "signal_take_profit_pct" in row:
                row["signal_take_profit_pct"] = f"{float(case['take']):.5f}"
            if "strategy_variant" in row:
                row["strategy_variant"] = case["name"]
            if "filter_name" in row:
                row["filter_name"] = case["name"]
            writer.writerow(row)
    return out


def _indicator_from_runner_stdout(text: str) -> dict:
    payload = json.loads(text)
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        try:
            indicator = ast.literal_eval(indicator)
        except Exception:
            indicator = {}
    return indicator or {}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        signal_file = _copy_signal(case)
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
            "--target-position-pct",
            str(case["target"]),
            "--score-exit-entry-ratio",
            str(case["exit"]),
            "--min-holding-days-before-score-exit",
            "1",
            "--score-continue-entry-ratio",
            "9.99",
            "--light-stop-loss-pct",
            str(case["stop"]),
            "--min-holding-days-before-light-stop",
            "1",
            "--take-profit-pct",
            str(case["take"]),
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
        print("RUN", case["name"], flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (LOG_DIR / f"{case['name']}.runner.json").write_text(proc.stdout, encoding="utf-8")
        indicator = _indicator_from_runner_stdout(proc.stdout) if proc.returncode == 0 else {}
        rows.append(
            {
                **case,
                "returncode": proc.returncode,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                "sharp_ratio": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
                "calmar_ratio": indicator.get("calmar_ratio"),
            }
        )
    out_csv = OUT_DIR / "repro500_exit_target_micro_summary.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_csv)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
