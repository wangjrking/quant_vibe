from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719" / "research_code_snapshot"
SOURCE = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_top3_independent_sell_refine_20260720"
BASE_SIGNAL = SOURCE / "signals" / "r1ge88_top3_min12_max13_ratio100_edge0.csv"
SCORE_DB = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_gate_juejin_20260720" / "score_assets" / "w10_100.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_open_gap_entry_neighborhood_20260720"
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"

CASES = [
    {"name": "all", "floor": None, "cap": None, "scale": "none"},
    *[{"name": f"cap{cap}", "floor": None, "cap": cap, "scale": "none"} for cap in (0.0, 1.0, 2.0, 3.0, 5.0)],
    {"name": "range_m5_p3", "floor": -5.0, "cap": 3.0, "scale": "none"},
    {"name": "range_m3_p3", "floor": -3.0, "cap": 3.0, "scale": "none"},
    {"name": "pos_scale_05", "floor": None, "cap": None, "scale": "positive_005"},
    {"name": "pos_scale_10", "floor": None, "cap": None, "scale": "positive_010"},
    {"name": "abs_scale_05", "floor": None, "cap": None, "scale": "absolute_005"},
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
            for key in ("pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"):
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            return result
    return {}


def build_signal(case: dict) -> tuple[Path, dict]:
    frame = pd.read_csv(BASE_SIGNAL)
    original_rows = len(frame)
    gap = pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce")
    if case["floor"] is not None:
        frame = frame[gap >= float(case["floor"])].copy()
    if case["cap"] is not None:
        frame = frame[pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce") <= float(case["cap"])].copy()
    gap = pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce").fillna(0.0).to_numpy(float)
    if case["scale"] == "positive_005":
        scale = np.clip(1.0 - 0.05 * np.maximum(gap, 0.0), 0.50, 1.0)
        frame["target_pct"] = (1.0 / 3.0) * scale
    elif case["scale"] == "positive_010":
        scale = np.clip(1.0 - 0.10 * np.maximum(gap, 0.0), 0.40, 1.0)
        frame["target_pct"] = (1.0 / 3.0) * scale
    elif case["scale"] == "absolute_005":
        scale = np.clip(1.0 - 0.05 * np.abs(gap), 0.50, 1.0)
        frame["target_pct"] = (1.0 / 3.0) * scale
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    audit = {
        "original_rows": original_rows, "rows": len(frame), "removed_rows": original_rows - len(frame),
        "signal_days": int(frame["signal_date"].nunique()), "mean_target_pct": float(frame["target_pct"].mean()),
    }
    return path, audit


def run_case(case: dict) -> dict:
    signal, audit = build_signal(case)
    log = LOGS / f"{case['name']}.log"
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
        "--log-file", str(log), "--max-positions", "3", "--holding-days", "12",
        "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(SCORE_DB), "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log)
    return {
        **case, **audit, "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last").to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in ("name", "rows", "returncode", "annual_return", "sharpe", "max_drawdown")}, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
