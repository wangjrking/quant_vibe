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

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_drawdown_overlay_20260720"
LOGS = OUT / "logs"
SIGNAL = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_top1_independent_sell_20260720" / "signals" / "r1ge90_top1_min10_max13.csv"
SCORE_DB = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_gate_juejin_20260720" / "score_assets" / "w10_100.duckdb"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720" / "research_code_snapshot"
CASES = [
    {"name": "off", "enabled": False},
    {"name": "dd_06_10_s65_45", "enabled": True, "soft": 0.06, "hard": 0.10, "recover": 0.03, "soft_scale": 0.65, "hard_scale": 0.45},
    {"name": "dd_08_12_s75_50", "enabled": True, "soft": 0.08, "hard": 0.12, "recover": 0.04, "soft_scale": 0.75, "hard_scale": 0.50},
    {"name": "dd_10_15_s80_60", "enabled": True, "soft": 0.10, "hard": 0.15, "recover": 0.05, "soft_scale": 0.80, "hard_scale": 0.60},
    {"name": "dd_12_18_s85_65", "enabled": True, "soft": 0.12, "hard": 0.18, "recover": 0.06, "soft_scale": 0.85, "hard_scale": 0.65},
    {"name": "dd_15_25_s85_65", "enabled": True, "soft": 0.15, "hard": 0.25, "recover": 0.07, "soft_scale": 0.85, "hard_scale": 0.65},
]


def run_case(case: dict) -> dict:
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "10", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "1" if case["enabled"] else "0",
        "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    })
    if case["enabled"]:
        env.update({
            "GM_EQUITY_DD_SOFT_TRIGGER": str(case["soft"]), "GM_EQUITY_DD_HARD_TRIGGER": str(case["hard"]),
            "GM_EQUITY_DD_RECOVER_TRIGGER": str(case["recover"]), "GM_EQUITY_DD_SOFT_SCALE": str(case["soft_scale"]),
            "GM_EQUITY_DD_HARD_SCALE": str(case["hard_scale"]),
        })
    command = [
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY), "--signal-file", str(SIGNAL),
        "--log-file", str(log), "--max-positions", "1", "--holding-days", "10",
        "--max-holding-days", "13", "--target-position-pct", "1.0",
        "--score-db", str(SCORE_DB), "--score-table", "blended_rank_score", "--market-db", str(MODULE.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    return {
        **case, "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "log_file": str(log),
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / "juejin_results.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
