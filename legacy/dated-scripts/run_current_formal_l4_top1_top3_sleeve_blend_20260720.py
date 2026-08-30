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

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_top3_sleeve_blend_20260720"
SIGNALS, LOGS = OUT / "signals", OUT / "logs"
TOP1 = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_open_gap_risk_20260720" / "signals" / "high_gt4_s00.csv"
TOP3 = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_open_state_dynamic_hold_20260720" / "research_code_snapshot"
CASES = [
    {"name": f"top1_{int(weight * 100):02d}_top3_{int((1 - weight) * 100):02d}", "top1_weight": weight}
    for weight in (0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0)
]


def build_signal(case: dict) -> Path:
    w1 = float(case["top1_weight"])
    frames = []
    if w1 > 0:
        top1 = pd.read_csv(TOP1)
        top1["target_pct"] = w1
        top1["sleeve"] = "top1"
        frames.append(top1)
    if w1 < 1:
        top3 = pd.read_csv(TOP3)
        top3["target_pct"] = (1.0 - w1) / 3.0
        top3["sleeve"] = "top3"
        frames.append(top3)
    frame = pd.concat(frames, ignore_index=True)
    frame["sleeve_priority"] = frame["sleeve"].map({"top1": 0, "top3": 1})
    frame = frame.sort_values(["buy_date", "stock_code", "sleeve_priority"])
    grouped = []
    for _, group in frame.groupby(["buy_date", "stock_code"], sort=False):
        row = group.iloc[0].copy()
        row["target_pct"] = min(0.98, float(group["target_pct"].sum()))
        row["holding_days"] = int(group["holding_days"].min())
        row["max_holding_days"] = int(group["max_holding_days"].max())
        row["sleeve"] = "+".join(sorted(set(group["sleeve"])))
        grouped.append(row)
    result = pd.DataFrame(grouped).drop(columns=["sleeve_priority"], errors="ignore")
    result["strategy_variant"] = case["name"]
    result["filter_name"] = "top1_top3_sleeve_blend"
    result["dynamic_hold_name"] = "signal_hold_independent_replace"
    path = SIGNALS / f"{case['name']}.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(case: dict, signal: Path) -> dict:
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_USE_SIGNAL_HOLD": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "10", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "4",
        "--holding-days", "10", "--max-holding-days", "13", "--target-position-pct", "0.25",
        "--score-db", str(MODULE.SCORE_DB), "--score-table", "blended_rank_score",
        "--market-db", str(MODULE.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    return {
        **case, "rows": len(pd.read_csv(signal)), "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        result = run_case(case, build_signal(case))
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
