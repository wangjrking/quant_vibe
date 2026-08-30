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
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720"
LOGS = OUT / "logs"
STRATEGY = OUT / "research_code_snapshot"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SIGNAL_A = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"
SIGNAL_B = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_lexicographic_entry_20260720" / "signals" / "current_postgate.csv"
SCORE_A = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "score_assets" / "w10_100.duckdb"
SCORE_B = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_persistent_edge_juejin_20260720" / "score_assets" / "w10_100.duckdb"
CASES = {
    "signal_a_score_a": (SIGNAL_A, SCORE_A),
    "signal_a_score_b": (SIGNAL_A, SCORE_B),
    "signal_b_score_a": (SIGNAL_B, SCORE_A),
    "signal_b_score_b": (SIGNAL_B, SCORE_B),
}


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


def run_case(name: str, signal: Path, score: Path) -> dict:
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
    })
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY), "--signal-file", str(signal),
        "--log-file", str(log), "--max-positions", "3", "--holding-days", "12",
        "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(score), "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log)
    return {
        "name": name, "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "score_db": str(score), "log_file": str(log),
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (signal, score) in CASES.items():
        result = run_case(name, signal, score)
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / "juejin_results.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    comparable = [
        (
            round(float(row["annual_return"]), 12),
            round(float(row["sharpe"]), 12),
            round(float(row["max_drawdown"]), 12),
            int(row["open_count"]),
            int(row["close_count"]),
        )
        for row in rows if row["returncode"] == 0
    ]
    exact = len(comparable) == len(CASES) and len(set(comparable)) == 1
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "cross_input_exact": exact, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
