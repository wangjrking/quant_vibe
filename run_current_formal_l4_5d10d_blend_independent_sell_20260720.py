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
MODULE_PATH = MAIN / "run_current_formal_l4_persistent_edge_juejin_20260720.py"
SPEC = importlib.util.spec_from_file_location("persistent_edge_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_blend_independent_sell_20260720"
SIGNALS = OUT / "signals"
SCORES = OUT / "score_assets"
LOGS = OUT / "logs"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719" / "research_code_snapshot"

MODULE.REPORT_DIR = OUT
MODULE.SIGNAL_DIR = SIGNALS
MODULE.SCORE_DIR = SCORES
MODULE.LOG_DIR = LOGS
MODULE.BLENDS.update(
    {
        "w5_40_10_60": (0.0, 0.4, 0.6),
        "w5_60_10_40": (0.0, 0.6, 0.4),
    }
)
for rank1 in (88, 90):
    MODULE.FILTERS[f"f50_pct3_r1{rank1}"] = (
        f"least(rank_5d, rank_10d) >= 0.50 AND signal_pct_chg_raw <= 3.0 "
        f"AND rank_1d >= {rank1 / 100:.2f}"
    )

CASES = [
    {
        "blend": blend,
        "score": "plain",
        "filter": f"f50_pct3_r1{rank1}",
        "threshold": threshold,
        "topn": 1,
        "hold": 12,
    }
    for blend in ("w10_100", "w5_20_10_80", "w5_40_10_60", "w5_60_10_40")
    for rank1 in (88, 90)
    for threshold in (0.95, 0.98)
]


def run_case(case: dict, signal_file: Path, score_db: Path) -> dict:
    name = signal_file.stem + "_top2_independent_min12_max13"
    frame = pd.read_csv(signal_file)
    frame["target_pct"] = 0.5
    frame["holding_days"] = 12
    frame["max_holding_days"] = 13
    adjusted_signal = SIGNALS / f"{name}.csv"
    frame.to_csv(adjusted_signal, index=False, encoding="utf-8-sig")
    log_file = LOGS / f"{name}.log"
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
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
        }
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY),
        "--signal-file",
        str(adjusted_signal),
        "--log-file",
        str(log_file),
        "--max-positions",
        "2",
        "--holding-days",
        "12",
        "--max-holding-days",
        "13",
        "--target-position-pct",
        "0.5",
        "--score-db",
        str(score_db),
        "--score-table",
        "blended_rank_score",
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.003",
        "--backtest-start",
        "2022-06-07 09:00:00",
        "--backtest-end",
        "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.parse_indicator(log_file)
    return {
        "name": name,
        **case,
        "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "signal_file": str(adjusted_signal),
        "score_db": str(score_db),
        "log_file": str(log_file),
    }


def main() -> None:
    for path in (OUT, SIGNALS, SCORES, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(MODULE.FEATURE_DB), read_only=True)
    rows = []
    try:
        score_assets = {blend: MODULE.build_score(con, blend) for blend in {case["blend"] for case in CASES}}
        for case in CASES:
            _, signal_file, _, _ = MODULE.build_signal(con, case)
            result = run_case(case, signal_file, score_assets[case["blend"]])
            rows.append(result)
            pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False, na_position="last").to_csv(
                OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
