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
SOURCE = MAIN / "run_current_formal_l4_open_gap_sell_neighborhood_20260720.py"
SPEC = importlib.util.spec_from_file_location("common", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_score_persistence_grid_20260720"
SIGNALS, LOGS = OUT / "signals", OUT / "logs"
BASE = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"
FEATURE_DB = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_persistent_edge_20260720" / "persistent_edge_features.duckdb"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720" / "research_code_snapshot"
CASES = {
    "base": "TRUE",
    "lag1_ge50": "rank10_lag1 >= 0.50",
    "lag1_ge70": "rank10_lag1 >= 0.70",
    "lag1_ge80": "rank10_lag1 >= 0.80",
    "lag1_ge90": "rank10_lag1 >= 0.90",
    "lag3_ge50": "rank10_lag3 >= 0.50",
    "lag3_ge70": "rank10_lag3 >= 0.70",
    "lag3_ge80": "rank10_lag3 >= 0.80",
    "lag1_70_lag3_50": "rank10_lag1 >= 0.70 AND rank10_lag3 >= 0.50",
    "lag1_80_lag3_70": "rank10_lag1 >= 0.80 AND rank10_lag3 >= 0.70",
}


def build_signal(base: pd.DataFrame, feature: pd.DataFrame, name: str, expression: str) -> Path:
    left = base.copy()
    right = feature.copy()
    left["signal_date"] = left["signal_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    right["trade_date"] = right["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    frame = left.merge(right, left_on=["signal_date", "stock_code"], right_on=["trade_date", "stock_code"], how="left")
    if expression != "TRUE":
        con = duckdb.connect()
        try:
            con.register("frame", frame)
            keep = con.execute(f"SELECT signal_date, stock_code FROM frame WHERE {expression}").fetchdf()
        finally:
            con.close()
        frame = frame.merge(keep, on=["signal_date", "stock_code"], how="inner")
    frame = frame[base.columns].copy()
    frame["strategy_variant"] = name
    path = SIGNALS / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(name: str, signal: Path) -> dict:
    log = LOGS / f"{name}.log"
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
        "name": name, "rows": int(len(frame)), "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    try:
        feature = con.execute("SELECT trade_date, stock_code, rank10_lag1, rank10_lag3 FROM persistent_edge_features").fetchdf()
    finally:
        con.close()
    rows = []
    for name, expression in CASES.items():
        signal = build_signal(base, feature, name, expression)
        result = run_case(name, signal)
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
