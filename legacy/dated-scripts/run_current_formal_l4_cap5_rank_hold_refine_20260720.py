from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
SOURCE = MAIN / "run_current_formal_l4_persistent_edge_juejin_20260720.py"
SPEC = importlib.util.spec_from_file_location("persistent_edge", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
SIGNALS, SCORES, LOGS = OUT / "signals", OUT / "score_assets", OUT / "logs"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719" / "research_code_snapshot"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
MODULE.FILTERS["cap5"] = "least(rank_5d, rank_10d) >= 0.50 AND signal_pct_chg_raw <= 5.0"
CASES = [
    {"rank1": rank1, "min_hold": min_hold, "max_hold": max_hold}
    for rank1 in (0.84, 0.86, 0.88, 0.90, 0.92)
    for min_hold, max_hold in ((11, 13), (12, 13), (13, 14))
]


def run_case(base: pd.DataFrame, case: dict, score_db: Path) -> dict:
    name = f"r1_{int(case['rank1']*100)}_min{case['min_hold']}_max{case['max_hold']}"
    frame = base[base["rank_1d"] >= case["rank1"]].copy()
    frame["target_pct"] = 1.0 / 3.0
    frame["holding_days"] = case["min_hold"]
    frame["max_holding_days"] = case["max_hold"]
    signal = SIGNALS / f"{name}.csv"
    frame.to_csv(signal, index=False, encoding="utf-8-sig")
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(case["min_hold"]),
        "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(case["max_hold"]),
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_RESIZE_HELD_ON_SIGNAL": "0",
    })
    command = [
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY), "--signal-file", str(signal),
        "--log-file", str(log), "--max-positions", "3", "--holding-days", str(case["min_hold"]),
        "--max-holding-days", str(case["max_hold"]), "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(score_db), "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log)
    return {
        "name": name, **case, "rows": int(len(frame)), "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, SCORES, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    MODULE.REPORT_DIR, MODULE.SIGNAL_DIR, MODULE.SCORE_DIR, MODULE.LOG_DIR = OUT, SIGNALS, SCORES, LOGS
    con = duckdb.connect(str(MODULE.FEATURE_DB), read_only=True)
    rows = []
    try:
        score_db = MODULE.build_score(con, "w10_100")
        case = {"blend": "w10_100", "score": "plain", "filter": "cap5", "threshold": 0.98, "topn": 1, "hold": 12}
        _, source_signal, _, _ = MODULE.build_signal(con, case)
        base = pd.read_csv(source_signal)
        for params in CASES:
            result = run_case(base, params, score_db)
            rows.append(result)
            pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False, na_position="last").to_csv(
                OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (OUT / "result.json").write_text(json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
