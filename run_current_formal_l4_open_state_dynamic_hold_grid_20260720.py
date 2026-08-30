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

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_open_state_dynamic_hold_20260720"
SIGNALS, LOGS = OUT / "signals", OUT / "logs"
BASE_SIGNAL = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"
STRATEGY = OUT / "research_code_snapshot"
CASES = (
    [{"name": "fixed_12_13", "default": (12, 13)}]
    + [{"name": "fixed_13_14", "default": (13, 14)}, {"name": "fixed_14_15", "default": (14, 15)}]
    + [
        {"name": f"weak_le_m{abs(int(threshold * 10)):02d}_to_{hold}_{hold + 1}", "default": (12, 13), "weak": (threshold, hold, hold + 1)}
        for threshold in (-3.0, -2.0, -1.0, 0.0)
        for hold in (13, 14)
    ]
    + [
        {"name": f"strong_ge_{str(threshold).replace('.', 'p')}_to_11_12", "default": (12, 13), "strong": (threshold, 11, 12)}
        for threshold in (1.5, 3.0, 5.0)
    ]
    + [
        {
            "name": f"weak_m2_to_{hold}_{hold + 1}_strong3_to_11_12",
            "default": (12, 13), "weak": (-2.0, hold, hold + 1), "strong": (3.0, 11, 12),
        }
        for hold in (13, 14)
    ]
)


def build_signal(case: dict) -> Path:
    frame = pd.read_csv(BASE_SIGNAL)
    default_hold, default_max = case["default"]
    frame["holding_days"], frame["max_holding_days"] = default_hold, default_max
    if "weak" in case:
        threshold, hold, max_hold = case["weak"]
        mask = frame["buy_open_gap_raw_pct"] <= threshold
        frame.loc[mask, ["holding_days", "max_holding_days"]] = [hold, max_hold]
    if "strong" in case:
        threshold, hold, max_hold = case["strong"]
        mask = frame["buy_open_gap_raw_pct"] >= threshold
        frame.loc[mask, ["holding_days", "max_holding_days"]] = [hold, max_hold]
    frame["strategy_variant"] = case["name"]
    frame["dynamic_hold_name"] = "buy_open_gap_dynamic_independent_replace"
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(case: dict, signal: Path) -> dict:
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_USE_SIGNAL_HOLD": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
    })
    command = [
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(MODULE.SCORE_DB), "--score-table", "blended_rank_score",
        "--market-db", str(MODULE.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    frame = pd.read_csv(signal)
    return {
        "name": case["name"], "rows": int(len(frame)),
        "dynamic_rows": int(((frame["holding_days"] != 12) | (frame["max_holding_days"] != 13)).sum()),
        "returncode": process.returncode, "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "juejin_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {str(row["name"]) for row in rows if int(row.get("returncode", -1)) == 0}
    for case in CASES:
        if case["name"] in completed:
            continue
        result = run_case(case, build_signal(case))
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
