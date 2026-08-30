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
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719" / "research_code_snapshot"
SOURCE = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_gate_juejin_20260720"
BASE_SIGNAL = SOURCE / "signals" / "w10_100_plain_f50_pct3_m98_top1_h12_full100.csv"
SCORE_DB = SOURCE / "score_assets" / "w10_100.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_position_count_neighborhood_20260720"
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"


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
            for key in ("pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"):
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            return result
    return {}


def run(max_positions: int) -> dict:
    name = f"r1ge88_pos{max_positions}_min12_max13"
    frame = pd.read_csv(BASE_SIGNAL)
    frame = frame[frame["rank_1d"] >= 0.88].copy()
    frame["target_pct"] = 1.0 / max_positions
    frame["holding_days"] = 12
    frame["max_holding_days"] = 13
    signal = SIGNALS / f"{name}.csv"
    frame.to_csv(signal, index=False, encoding="utf-8-sig")
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_RESIZE_HELD_ON_SIGNAL": "0",
    })
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY), "--signal-file", str(signal),
        "--log-file", str(log), "--max-positions", str(max_positions), "--holding-days", "12",
        "--max-holding-days", "13", "--target-position-pct", str(1.0 / max_positions),
        "--score-db", str(SCORE_DB), "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log)
    return {
        "name": name, "max_positions": max_positions, "target_pct": 1.0 / max_positions,
        "returncode": process.returncode, "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for positions in (3, 4, 5, 6):
        result = run(positions)
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / "juejin_results.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
