from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_robust_rules_20260719"
    / "research_code_snapshot"
)
SOURCE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_5d10d_top3_independent_sell_refine_20260720"
)
SIGNAL = SOURCE / "signals" / "r1ge88_top3_min12_max13_ratio100_edge0.csv"
SCORE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_5d10d_gate_juejin_20260720"
    / "score_assets"
    / "w10_100.duckdb"
)
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_5d10d_top3_independent_sell_validation_20260720"
)
LOGS = OUT / "logs"
WINDOWS = [
    ("full_repeat", "2022-06-07 09:00:00", "2026-07-17 15:30:00"),
    ("2022h2", "2022-06-07 09:00:00", "2022-12-30 15:30:00"),
    ("2023", "2023-01-03 09:00:00", "2023-12-29 15:30:00"),
    ("2024", "2024-01-02 09:00:00", "2024-12-31 15:30:00"),
    ("2025", "2025-01-02 09:00:00", "2025-12-31 15:30:00"),
    ("2026ytd", "2026-01-05 09:00:00", "2026-07-17 15:30:00"),
    ("recent60", "2026-04-22 09:00:00", "2026-07-17 15:30:00"),
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
            for key in ("pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"):
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            for key in ("open_count", "close_count"):
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    result[key] = int(match.group(1))
            return result
    return {}


def run_window(name: str, start: str, end: str) -> dict:
    log_path = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12",
            "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
            "GM_INDEPENDENT_REPLACE_RATIO": "1.0",
            "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
        }
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY),
        "--signal-file",
        str(SIGNAL),
        "--log-file",
        str(log_path),
        "--max-positions",
        "3",
        "--holding-days",
        "12",
        "--max-holding-days",
        "13",
        "--target-position-pct",
        str(1 / 3),
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
        start,
        "--backtest-end",
        end,
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log_path)
    return {
        "name": name,
        "start": start,
        "end": end,
        "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "log_file": str(log_path),
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in WINDOWS:
        result = run_window(*window)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    payload = {
        "status": "research_only",
        "candidate": "r1ge88_top3_min12_max13_ratio100_edge0",
        "signal_sha256": hashlib.sha256(SIGNAL.read_bytes()).hexdigest(),
        "signal_file": str(SIGNAL),
        "score_db": str(SCORE_DB),
        "market_db": str(MARKET_DB),
        "backtest_adjust": "none",
        "one_side_slippage_ratio": 0.003,
        "results": rows,
    }
    (OUT / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
