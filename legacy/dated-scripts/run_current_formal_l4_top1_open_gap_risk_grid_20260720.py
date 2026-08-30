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

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_open_gap_risk_20260720"
SIGNALS, LOGS = OUT / "signals", OUT / "logs"
BASE_SIGNAL = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_top1_independent_sell_20260720" / "signals" / "r1ge90_top1_min10_max13.csv"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720" / "research_code_snapshot"
CASES = (
    [{"name": "base", "mode": "none", "threshold": None, "scale": 1.0}]
    + [
        {"name": f"high_gt{threshold}_s{int(scale * 100):02d}", "mode": "high", "threshold": float(threshold), "scale": scale}
        for threshold in (3, 4, 5, 6)
        for scale in (0.0, 0.25, 0.50)
    ]
    + [
        {"name": f"low_lt_m{threshold}_s{int(scale * 100):02d}", "mode": "low", "threshold": -float(threshold), "scale": scale}
        for threshold in (3, 4, 5)
        for scale in (0.0, 0.50)
    ]
    + [
        {"name": f"abs_gt{threshold}_s{int(scale * 100):02d}", "mode": "abs", "threshold": float(threshold), "scale": scale}
        for threshold in (4, 5)
        for scale in (0.25, 0.50)
    ]
)


def build_signal(case: dict) -> tuple[Path, int]:
    frame = pd.read_csv(BASE_SIGNAL)
    if case["mode"] == "none":
        mask = pd.Series(False, index=frame.index)
    elif case["mode"] == "high":
        mask = frame["buy_open_gap_raw_pct"] > case["threshold"]
    elif case["mode"] == "low":
        mask = frame["buy_open_gap_raw_pct"] < case["threshold"]
    else:
        mask = frame["buy_open_gap_raw_pct"].abs() > case["threshold"]
    affected = int(mask.sum())
    if case["scale"] == 0.0:
        frame = frame.loc[~mask].copy()
    else:
        frame.loc[mask, "target_pct"] = case["scale"]
    frame["strategy_variant"] = case["name"]
    frame["filter_name"] = "top1_next_open_gap_risk"
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path, affected


def run_case(case: dict, signal: Path, affected: int) -> dict:
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
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "1",
        "--holding-days", "10", "--max-holding-days", "13", "--target-position-pct", "1.0",
        "--score-db", str(MODULE.SCORE_DB), "--score-table", "blended_rank_score",
        "--market-db", str(MODULE.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    return {
        **case, "affected_signal_rows": affected, "rows": len(pd.read_csv(signal)),
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
        signal, affected = build_signal(case)
        result = run_case(case, signal, affected)
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
