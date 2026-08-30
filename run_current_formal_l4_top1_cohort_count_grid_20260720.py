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
SOURCE_PATH = MAIN / "run_current_formal_l4_open_gap_sell_neighborhood_20260720.py"
SPEC = importlib.util.spec_from_file_location("top1_common", SOURCE_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

SOURCE_SIGNAL = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_top1_open_gap_risk_20260720/signals/high_gt4_s00.csv"
STRATEGY = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_score_quantized_20260720/research_code_snapshot"
OUT = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_top1_cohort_count_grid_20260720"
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"

CASES = [
    {"name": f"positions_{positions}", "max_positions": positions, "target_pct": 1.0 / positions}
    for positions in (1, 2, 3, 4, 5)
]


def build_signal(case: dict) -> Path:
    frame = pd.read_csv(SOURCE_SIGNAL)
    frame["target_pct"] = float(case["target_pct"])
    frame["holding_days"] = 10
    frame["max_holding_days"] = 13
    frame["strategy_variant"] = case["name"]
    frame["entry_weight_name"] = "equal_weight_independent_cohort"
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(case: dict) -> dict:
    signal = build_signal(case)
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "10", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log),
        "--max-positions", str(case["max_positions"]), "--holding-days", "10",
        "--max-holding-days", "13", "--target-position-pct", str(case["target_pct"]),
        "--score-db", str(COMMON.SCORE_DB), "--score-table", "blended_rank_score",
        "--market-db", str(COMMON.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    return {
        **case, "returncode": int(process.returncode), "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in (
            "name", "annual_return", "sharpe", "max_drawdown", "open_count"
        )}, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only_not_admitted", "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
