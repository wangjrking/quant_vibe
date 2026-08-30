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
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_robust_rules_20260719"
    / "research_code_snapshot"
)
SOURCE_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_multihorizon_refine_juejin_20260720"
)
SIGNAL_FILE = SOURCE_DIR / "signals" / "w10_100_plain_mh65_pct3_m98_top1_h12_full100.csv"
SCORE_DB = SOURCE_DIR / "score_assets" / "w10_100.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_multihorizon_independent_sell_20260720"
)
LOG_DIR = REPORT_DIR / "logs"


CASES = [
    {"min_hold": 8, "max_hold": 12, "ratio": 1.00, "edge": 0.00},
    {"min_hold": 8, "max_hold": 15, "ratio": 1.02, "edge": 0.01},
    {"min_hold": 10, "max_hold": 12, "ratio": 1.00, "edge": 0.00},
    {"min_hold": 10, "max_hold": 15, "ratio": 1.00, "edge": 0.00},
    {"min_hold": 10, "max_hold": 15, "ratio": 1.02, "edge": 0.01},
    {"min_hold": 10, "max_hold": 18, "ratio": 1.02, "edge": 0.01},
    {"min_hold": 12, "max_hold": 15, "ratio": 1.00, "edge": 0.00},
    {"min_hold": 12, "max_hold": 18, "ratio": 1.02, "edge": 0.01},
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


def run(case: dict) -> dict:
    name = (
        f"ind_min{case['min_hold']}_max{case['max_hold']}"
        f"_r{str(case['ratio']).replace('.', 'p')}_e{str(case['edge']).replace('.', 'p')}"
    )
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
        "12",
        "--holding-days",
        "12",
        "--max-holding-days",
        str(case["max_hold"]),
        "--target-position-pct",
        str(1.0 / 12.0),
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
        "signal_file": str(SIGNAL_FILE),
        "score_db": str(SCORE_DB),
        "log_file": str(log_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    rows = []
    for case in CASES:
        result = run(case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last").to_csv(
            REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    (REPORT_DIR / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
