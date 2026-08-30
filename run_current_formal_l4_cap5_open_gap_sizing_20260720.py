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

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_open_gap_sizing_20260720"
SIGNALS, LOGS = OUT / "signals", OUT / "logs"
BASE = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720" / "research_code_snapshot"
CASES = [
    {"name": "base", "boost": 1.0, "boost_low": -3.0, "boost_high": -1.0, "high_gap": 99.0, "high_scale": 1.0},
    {"name": "reject_gap_gt5", "boost": 1.0, "boost_low": -3.0, "boost_high": -1.0, "high_gap": 5.0, "high_scale": 0.0},
    {"name": "boost125_m3_m1", "boost": 1.25, "boost_low": -3.0, "boost_high": -1.0, "high_gap": 99.0, "high_scale": 1.0},
    {"name": "boost150_m3_m1", "boost": 1.50, "boost_low": -3.0, "boost_high": -1.0, "high_gap": 99.0, "high_scale": 1.0},
    {"name": "boost125_m3_0", "boost": 1.25, "boost_low": -3.0, "boost_high": 0.0, "high_gap": 99.0, "high_scale": 1.0},
    {"name": "boost125_m3_m1_reduce_gt5", "boost": 1.25, "boost_low": -3.0, "boost_high": -1.0, "high_gap": 5.0, "high_scale": 0.5},
    *[
        {
            "name": f"reduce_gt{int(high_gap)}_scale{int(high_scale * 100)}",
            "boost": 1.0,
            "boost_low": -3.0,
            "boost_high": -1.0,
            "high_gap": high_gap,
            "high_scale": high_scale,
        }
        for high_gap in (3.0, 4.0, 5.0, 6.0)
        for high_scale in (0.25, 0.50, 0.75)
    ],
]


def build_signal(base: pd.DataFrame, case: dict) -> Path:
    frame = base.copy()
    gap = pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce")
    boost_mask = gap.between(case["boost_low"], case["boost_high"], inclusive="both")
    high_mask = gap > case["high_gap"]
    frame.loc[boost_mask, "target_pct"] = (1.0 / 3.0) * case["boost"]
    frame.loc[high_mask, "target_pct"] = (1.0 / 3.0) * case["high_scale"]
    frame = frame[pd.to_numeric(frame["target_pct"], errors="coerce") > 0].copy()
    frame["strategy_variant"] = case["name"]
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(case: dict, signal: Path) -> dict:
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
    })
    command = [
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(MODULE.SCORE_DB), "--score-table", "blended_rank_score", "--market-db", str(MODULE.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    frame = pd.read_csv(signal)
    return {
        "name": case["name"], "rows": int(len(frame)), "returncode": process.returncode,
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
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {str(row["name"]) for row in rows if int(row.get("returncode", -1)) == 0}
    for case in CASES:
        if case["name"] in completed:
            continue
        signal = build_signal(base, case)
        result = run_case(case, signal)
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
