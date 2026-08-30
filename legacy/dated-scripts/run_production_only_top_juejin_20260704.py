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
SUMMARY_IN = REPORT_DIR / "local_grid_summary.csv"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
CODE_SNAPSHOT = OUT_DIR / "code_snapshot_sell_available_safe"
LOG_DIR = OUT_DIR / "top20_runs"
SUMMARY_OUT = OUT_DIR / "top20_juejin_summary.csv"
RUNNER = ROOT / "quant/main/run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


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


def already_done(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    return "GM_BACKTEST_INDICATOR:" in text


def run_case(row: dict[str, str]) -> dict[str, object]:
    name = row["name"]
    signal_file = Path(row["signal_file"])
    log_file = LOG_DIR / f"{name}.log"
    runner_json = LOG_DIR / f"{name}.runner.json"
    if already_done(log_file) and runner_json.exists():
        stdout = runner_json.read_text(encoding="utf-8", errors="ignore")
        returncode = 0
    else:
        cmd = [
            sys.executable,
            str(RUNNER),
            "--strategy-dir",
            str(CODE_SNAPSHOT),
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
            "9.99",
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
        "returncode": returncode,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "local_annual": row.get("local_annual"),
        "local_sharpe": row.get("local_sharpe"),
        "local_max_drawdown": row.get("local_max_drawdown"),
        "rows": row.get("rows"),
        "signal_days": row.get("signal_days"),
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_IN)
    df = df.sort_values(["local_annual", "local_sharpe"], ascending=[False, False]).head(20)
    rows: list[dict[str, object]] = []
    for row in df.to_dict("records"):
        print(f"RUN {row['name']}", flush=True)
        rows.append(run_case({k: str(v) for k, v in row.items()}))
    rows = sorted(
        rows,
        key=lambda x: (
            float(x.get("pnl_ratio_annual") or -999),
            float(x.get("sharp_ratio") or -999),
        ),
        reverse=True,
    )
    with SUMMARY_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
