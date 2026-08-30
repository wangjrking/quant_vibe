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
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_open_gap_sell_20260720" / "research_code_snapshot"
SIGNAL = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_top3_independent_sell_refine_20260720" / "signals" / "r1ge88_top3_min12_max13_ratio100_edge0.csv"
SCORE_DB = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_gate_juejin_20260720" / "score_assets" / "w10_100.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_open_gap_sell_20260720"
LOGS = OUT / "logs"

CASES = [
    {"name": "disabled", "gap_pct": None, "early_min_hold": 1},
    *[
        {"name": f"gap_m{abs(int(gap))}_h{hold}", "gap_pct": gap, "early_min_hold": hold}
        for gap in (-1.0, -2.0, -3.0, -4.0, -5.0)
        for hold in (1, 2)
    ],
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
            for key in ("pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio", "open_count", "close_count"):
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            return result
    return {}


def run_case(case: dict) -> dict:
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_EARLY_OPEN_GAP_MIN_HOLD": str(case["early_min_hold"]),
        "GM_INDEPENDENT_EARLY_OPEN_GAP_PCT": "" if case["gap_pct"] is None else str(case["gap_pct"]),
        "GM_RESIZE_HELD_ON_SIGNAL": "0",
    })
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY), "--signal-file", str(SIGNAL),
        "--log-file", str(log), "--max-positions", "3", "--holding-days", "12",
        "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(SCORE_DB), "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log)
    return {
        **case, "returncode": process.returncode, "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "log_file": str(log),
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last").to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
