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

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_gap4_sell_advantage_20260720"
LOGS = OUT / "logs"
SIGNAL = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_open_gap_risk_20260720" / "signals" / "high_gt4_s00.csv"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720" / "research_code_snapshot"
CASES = (
    [{"name": f"ratio_{str(ratio).replace('.', 'p')}_edge0", "ratio": ratio, "edge": 0.0} for ratio in (1.0, 1.001, 1.002, 1.005, 1.01, 1.02)]
    + [{"name": f"ratio1_edge_{str(edge).replace('.', 'p')}", "ratio": 1.0, "edge": edge} for edge in (0.0005, 0.001, 0.002, 0.005, 0.01)]
)


def run_case(case: dict) -> dict:
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "10", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": str(case["ratio"]), "GM_INDEPENDENT_REPLACE_EDGE": str(case["edge"]),
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(SIGNAL), "--log-file", str(log), "--max-positions", "1",
        "--holding-days", "10", "--max-holding-days", "13", "--target-position-pct", "1.0",
        "--score-db", str(MODULE.SCORE_DB), "--score-table", "blended_rank_score",
        "--market-db", str(MODULE.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    return {
        **case, "returncode": process.returncode, "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(SIGNAL), "log_file": str(log),
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
