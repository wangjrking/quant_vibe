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
COMMON_PATH = MAIN / "run_current_formal_l4_currentcoverage_entry_exit_grid_20260720.py"
SPEC = importlib.util.spec_from_file_location("currentcoverage_common", COMMON_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_entry_exit_grid_20260720"
    / "signals"
    / "buygap_gt5_scale50.csv"
)
SCORE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
    / "score_assets"
    / "w10_100.duckdb"
)
STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_score_state_exit_20260720"
    / "research_code_snapshot"
)
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_score_state_exit_20260720"
)
LOGS = OUT / "logs"

CASES = [
    {"min_hold": min_hold, "max_hold": 16, "current_max": current_max}
    for min_hold in (8, 10, 12)
    for current_max in (0.70, 0.80, 0.85, 0.90, 0.95)
]


def run_case(case: dict) -> dict:
    name = f"min{case['min_hold']}_max16_cur{int(case['current_max'] * 100)}"
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(case["min_hold"]),
            "GM_INDEPENDENT_REPLACE_MAX_HOLD": "16",
            "GM_INDEPENDENT_REPLACE_RATIO": "1.0",
            "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10",
            "GM_INDEPENDENT_REPLACE_CURRENT_SCORE_MAX": str(case["current_max"]),
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_EQUITY_DD_RISK_MODE": "0",
        }
    )
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(SIGNAL), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", str(case["min_hold"]), "--max-holding-days", "16",
        "--target-position-pct", str(1.0 / 3.0), "--score-db", str(SCORE_DB),
        "--score-table", "blended_rank_score", "--market-db", str(COMMON.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    return {
        "name": name, **case, "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "log_file": str(log),
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in ("name", "annual_return", "sharpe", "max_drawdown")}, ensure_ascii=False), flush=True)
    payload = {
        "status": "research_only_not_admitted",
        "signal_file": str(SIGNAL), "signal_sha256": COMMON.sha256(SIGNAL),
        "score_db": str(SCORE_DB), "score_db_sha256": COMMON.sha256(SCORE_DB),
        "strategy_snapshot": str(STRATEGY),
        "results": rows,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
