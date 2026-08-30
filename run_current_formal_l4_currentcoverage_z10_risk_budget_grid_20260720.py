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

SOURCE_SIGNAL = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_currentcoverage_z10_confidence_grid_20260720/signals/soft_2p1_2p7_s50.csv"
SCORE_DB = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720/score_assets/w10_100.duckdb"
OUT = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_currentcoverage_z10_risk_budget_grid_20260720"
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"

CASES = [
    {"name": f"base{int(base * 100)}_mp{max_positions}_max{max_hold}", "base": base,
     "max_positions": max_positions, "max_hold": max_hold}
    for base, max_positions in ((1.0 / 3.0, 3), (0.40, 3), (0.45, 3), (0.50, 2))
    for max_hold in (13, 15)
]


def build_signal(case: dict) -> Path:
    frame = pd.read_csv(SOURCE_SIGNAL)
    frame["target_pct"] = float(case["base"])
    frame.loc[~frame["confidence_in_range"].astype(bool), "target_pct"] *= 0.50
    frame.loc[pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce") > 5.0, "target_pct"] *= 0.50
    frame["holding_days"] = 12
    frame["max_holding_days"] = int(case["max_hold"])
    frame["strategy_variant"] = case["name"]
    frame["entry_weight_name"] = "z10_2p1_2p7_outside_half_then_buy_gap_gt5_half"
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
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12",
        "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(case["max_hold"]),
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(COMMON.STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log),
        "--max-positions", str(case["max_positions"]), "--holding-days", "12",
        "--max-holding-days", str(case["max_hold"]), "--target-position-pct", str(case["base"]),
        "--score-db", str(SCORE_DB), "--score-table", "blended_rank_score", "--market-db", str(COMMON.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    return {
        **case, "returncode": int(process.returncode), "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(signal),
        "signal_sha256": COMMON.sha256(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in (
            "name", "annual_return", "sharpe", "max_drawdown"
        )}, ensure_ascii=False), flush=True)
    payload = {
        "status": "research_only_not_admitted", "source_signal": str(SOURCE_SIGNAL),
        "source_signal_sha256": COMMON.sha256(SOURCE_SIGNAL), "score_db": str(SCORE_DB),
        "score_db_sha256": COMMON.sha256(SCORE_DB), "results": rows,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
