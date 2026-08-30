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
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_sell_frequency_grid_20260720"
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"
CASES = [
    (max_positions, rank1, min_hold)
    for max_positions in (2, 3)
    for rank1 in (0.88, 0.90)
    for min_hold in (1, 3, 5, 7)
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


def run_case(max_positions: int, rank1: float, min_hold: int) -> dict:
    name = f"top{max_positions}_r1ge{int(rank1 * 100)}_min{min_hold}_max13"
    frame = pd.read_csv(BASE_SIGNAL)
    frame = frame[frame["rank_1d"] >= rank1].copy()
    frame["target_pct"] = 1 / max_positions
    frame["holding_days"] = min_hold
    frame["max_holding_days"] = 13
    signal_path = SIGNALS / f"{name}.csv"
    frame.to_csv(signal_path, index=False, encoding="utf-8-sig")
    log_path = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(min_hold),
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
        str(signal_path),
        "--log-file",
        str(log_path),
        "--max-positions",
        str(max_positions),
        "--holding-days",
        str(min_hold),
        "--max-holding-days",
        "13",
        "--target-position-pct",
        str(1 / max_positions),
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
    indicator = parse_indicator(log_path)
    return {
        "name": name,
        "max_positions": max_positions,
        "rank1_min": rank1,
        "min_hold": min_hold,
        "max_hold": 13,
        "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "signal_file": str(signal_path),
        "log_file": str(log_path),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(*case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False, na_position="last").to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
