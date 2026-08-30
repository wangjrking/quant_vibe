from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_contribution_audit_20260720"
)
LOG = OUT / "verbose_full.log"
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
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_score_quantized_20260720"
    / "research_code_snapshot"
)
RUNNER = MAIN / "run_juejin_signal_backtest.py"


def run_verbose_full() -> dict:
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
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10",
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_EQUITY_DD_RISK_MODE": "0",
            "GM_VERBOSE_TRADES": "1",
        }
    )
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(SIGNAL), "--log-file", str(LOG), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13",
        "--target-position-pct", str(1.0 / 3.0), "--score-db", str(SCORE_DB),
        "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    return {"returncode": int(process.returncode), "stdout_tail": process.stdout[-2000:], "stderr_tail": process.stderr[-2000:]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verbose_result = run_verbose_full()
    if verbose_result["returncode"] != 0:
        raise RuntimeError(json.dumps(verbose_result, ensure_ascii=False))

    audit = importlib.import_module("build_current_formal_l4_contribution_pressure_20260720")
    audit.REPORT = OUT
    audit.LOG = LOG
    audit.BASE_SIGNAL = SIGNAL
    audit.PRESSURE = OUT / "pressure_signals"
    audit.main()

    pressure = importlib.import_module("run_current_formal_l4_contribution_pressure_juejin_20260720")
    pressure.OUT = OUT
    pressure.SIGNALS = OUT / "pressure_signals"
    pressure.LOGS = OUT / "pressure_logs"
    pressure.STRATEGY_DIR = STRATEGY
    pressure.MODULE.SCORE_DB = SCORE_DB
    pressure.MODULE.MARKET_DB = MARKET_DB
    pressure.main()

    (OUT / "execution_summary.json").write_text(
        json.dumps(
            {
                "status": "research_only_not_admitted",
                "verbose_backtest": verbose_result,
                "signal_file": str(SIGNAL),
                "score_db": str(SCORE_DB),
                "market_db": str(MARKET_DB),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
