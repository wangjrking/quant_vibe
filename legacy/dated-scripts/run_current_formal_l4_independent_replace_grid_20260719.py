from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719"
STRATEGY_DIR = REPORT_DIR / "research_code_snapshot"
SIGNAL_FILE = REPORT_DIR / "signals" / "b510_support_ma10_m90_top5_h5.csv"
LOG_DIR = REPORT_DIR / "independent_replace_logs"
OUT_CSV = REPORT_DIR / "independent_replace_juejin.csv"
OUT_JSON = REPORT_DIR / "independent_replace_juejin.json"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
SCORE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
    / "score_assets"
    / "b5_20_10_80.duckdb"
)
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
MAX_POSITIONS = 5

CASES = [
    {"min_hold": 1, "max_hold": 2, "ratio": 1.00, "edge": 0.000},
    {"min_hold": 1, "max_hold": 2, "ratio": 1.00, "edge": 0.005},
    {"min_hold": 1, "max_hold": 2, "ratio": 1.02, "edge": 0.005},
    {"min_hold": 1, "max_hold": 3, "ratio": 1.00, "edge": 0.000},
    {"min_hold": 1, "max_hold": 3, "ratio": 1.00, "edge": 0.010},
    {"min_hold": 1, "max_hold": 3, "ratio": 1.02, "edge": 0.005},
    {"min_hold": 1, "max_hold": 5, "ratio": 1.00, "edge": 0.000},
    {"min_hold": 1, "max_hold": 5, "ratio": 1.02, "edge": 0.005},
]


def parse_indicator(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result = {}
            for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"]:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            for key in ["open_count", "close_count"]:
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    result[key] = int(match.group(1))
            return result
    return {}


def run_case(case: dict) -> dict:
    name = "replace_min{min_hold}_max{max_hold}_r{ratio:.3f}_e{edge:.3f}".format(**case)
    log_file = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(case["min_hold"]),
            "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(case["max_hold"]),
            "GM_INDEPENDENT_REPLACE_RATIO": str(case["ratio"]),
            "GM_INDEPENDENT_REPLACE_EDGE": str(case["edge"]),
        }
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(SIGNAL_FILE),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(MAX_POSITIONS),
        "--holding-days",
        "5",
        "--max-holding-days",
        "5",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "blended_rank_score",
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.003",
        "--backtest-start",
        "2022-06-07 09:00:00",
        "--backtest-end",
        "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log_file)
    return {
        "name": name,
        **case,
        "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "log_file": str(log_file),
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        frame = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
        frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(
            json.dumps({"status": "research_only_in_progress", "results": frame.to_dict("records")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
    OUT_JSON.write_text(
        json.dumps({"status": "research_only_complete", "results": frame.to_dict("records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
