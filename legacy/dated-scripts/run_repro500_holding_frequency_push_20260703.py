from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_SIGNAL = (
    ROOT
    / "quant/main/strategy_library/production/prod_repro500_p435_s96_p10d70_v20260702/signals/full_history_repro500_p435_s96_p10d70.csv"
)
OUT_DIR = REPORT_DIR / "repro500_holding_frequency_push"
SIGNAL_DIR = OUT_DIR / "signals"
LOG_DIR = OUT_DIR / "juejin_runs"
SUMMARY_FILE = OUT_DIR / "holding_frequency_summary.csv"

STRATEGY_DIR = (
    ROOT
    / "quant/data_file/reports/strategy_agent_latest_l4_weight_candidates_20260702/postrank_open_filter_candidates/code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant/main/run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = []
for target in [0.432, 0.435, 0.438]:
    for hold in [1, 2, 3]:
        for exit_ratio in [0.94, 0.96, 9.99]:
            CASES.append(
                {
                    "name": f"p{str(target).replace('.', 'p')}_h{hold}_exit{str(exit_ratio).replace('.', 'p')}",
                    "target": target,
                    "hold": hold,
                    "exit_ratio": exit_ratio,
                }
            )


def write_signal(case: dict) -> Path:
    case_dir = SIGNAL_DIR / case["name"]
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
            row["target_pct"] = f"{case['target']:.5f}"
            row["holding_days"] = str(case["hold"])
            row["max_holding_days"] = str(case["hold"])
            row["score_exit_entry_ratio"] = f"{case['exit_ratio']:.5f}"
            row["min_holding_days_before_score_exit"] = "1"
            row["strategy_variant"] = case["name"]
            row["filter_name"] = case["name"]
            writer.writerow(row)
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        signal_file = write_signal(case)
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
            str(case["hold"]),
            "--max-holding-days",
            str(case["hold"]),
            "--target-position-pct",
            str(case["target"]),
            "--score-exit-entry-ratio",
            str(case["exit_ratio"]),
            "--min-holding-days-before-score-exit",
            "1",
            "--score-continue-entry-ratio",
            "9.99",
            "--light-stop-loss-pct",
            "0.05",
            "--min-holding-days-before-light-stop",
            "1",
            "--take-profit-pct",
            "0.08",
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
        indicator = parse_indicator(proc.stdout) if proc.returncode == 0 else {}
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
            }
        )
    with SUMMARY_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    rows = sorted(rows, key=lambda x: (x.get("sharp_ratio") or -999, x.get("pnl_ratio_annual") or -999), reverse=True)
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
