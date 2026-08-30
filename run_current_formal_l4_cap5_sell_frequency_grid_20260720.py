from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
SOURCE = MAIN / "run_current_formal_l4_open_gap_sell_neighborhood_20260720.py"
SPEC = importlib.util.spec_from_file_location("common", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_sell_frequency_grid_20260720"
SIGNALS, LOGS = OUT / "signals", OUT / "logs"
BASE = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719" / "research_code_snapshot"
CASES = [
    {"min_hold": min_hold, "max_hold": max_hold}
    for min_hold in (2, 4, 6, 8, 10, 12)
    for max_hold in (10, 13, 16)
    if max_hold > min_hold
]


def run_case(base: pd.DataFrame, case: dict) -> dict:
    name = f"min{case['min_hold']}_max{case['max_hold']}_r100_e0"
    frame = base.copy()
    frame["holding_days"] = case["min_hold"]
    frame["max_holding_days"] = case["max_hold"]
    signal = SIGNALS / f"{name}.csv"
    frame.to_csv(signal, index=False, encoding="utf-8-sig")
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(case["min_hold"]),
        "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(case["max_hold"]),
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_RESIZE_HELD_ON_SIGNAL": "0",
    })
    command = [
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", str(case["min_hold"]), "--max-holding-days", str(case["max_hold"]),
        "--target-position-pct", str(1.0 / 3.0), "--score-db", str(MODULE.SCORE_DB),
        "--score-table", "blended_rank_score", "--market-db", str(MODULE.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    return {
        "name": name, **case, "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)
    result_path = OUT / "juejin_results.csv"
    if result_path.exists():
        existing = pd.read_csv(result_path).to_dict("records")
    else:
        existing = []
    rows = existing
    completed = {str(row["name"]) for row in existing if int(row.get("returncode", -1)) == 0}
    for case in CASES:
        name = f"min{case['min_hold']}_max{case['max_hold']}_r100_e0"
        if name in completed:
            continue
        result = run_case(base, case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).to_csv(
            result_path, index=False, encoding="utf-8-sig"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
