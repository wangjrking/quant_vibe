from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("D:/work/quant/quant_mcp")
MAIN = ROOT / "quant/main"
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
RUN_DIR = OUT_DIR / "scale124_exit_frequency_grid"
SUMMARY = OUT_DIR / "scale124_exit_frequency_grid_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SIGNAL_FILE = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, score_exit, min_score_hold, max_hold, light_stop, min_light_hold, take_profit
    ("base_h2m4_e090_ls08", 0.90, 2, 4, 0.08, 1, None),
    ("fast_h2m3_e095_ls06_tp08", 0.95, 1, 3, 0.06, 1, 0.08),
    ("fast_h2m2_e095_ls05_tp08", 0.95, 1, 2, 0.05, 1, 0.08),
    ("fast_h2m3_e092_ls05_tp07", 0.92, 1, 3, 0.05, 1, 0.07),
    ("fast_h2m2_e092_ls05_tp07", 0.92, 1, 2, 0.05, 1, 0.07),
    ("tight_h2m3_e090_ls04_tp06", 0.90, 1, 3, 0.04, 1, 0.06),
    ("tight_h2m2_e090_ls04_tp06", 0.90, 1, 2, 0.04, 1, 0.06),
    ("hold_h2m5_e090_ls08_tp10", 0.90, 2, 5, 0.08, 1, 0.10),
    ("hold_h2m6_e088_ls08_tp12", 0.88, 2, 6, 0.08, 1, 0.12),
    ("loose_h2m4_e085_ls08_tp12", 0.85, 2, 4, 0.08, 1, 0.12),
    ("day1_h1m2_e095_ls05_tp08", 0.95, 1, 2, 0.05, 1, 0.08),
    ("day1_h1m1_e999_ls05_tp08", 9.99, 1, 1, 0.05, 1, 0.08),
    ("stop_only_h2m4_e090_ls04", 0.90, 2, 4, 0.04, 1, None),
    ("profit_only_h2m4_e090_tp08", 0.90, 2, 4, 0.08, 1, 0.08),
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


def run_case(case: tuple) -> dict[str, object]:
    name, exit_ratio, min_score_hold, max_hold, light_stop, min_light_hold, take_profit = case
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
            str(SIGNAL_FILE),
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
            str(exit_ratio),
            "--min-holding-days-before-score-exit",
            str(min_score_hold),
            "--score-continue-entry-ratio",
            "1.000",
            "--max-holding-days",
            str(max_hold),
            "--light-stop-loss-pct",
            str(light_stop),
            "--min-holding-days-before-light-stop",
            str(min_light_hold),
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        if take_profit is not None:
            cmd.extend(["--take-profit-pct", str(take_profit)])
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")
    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "score_exit_entry_ratio": exit_ratio,
        "min_score_hold": min_score_hold,
        "max_hold": max_hold,
        "light_stop": light_stop,
        "min_light_hold": min_light_hold,
        "take_profit": take_profit,
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
