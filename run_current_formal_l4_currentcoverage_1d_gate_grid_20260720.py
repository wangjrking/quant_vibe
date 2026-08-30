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
COMMON_PATH = MAIN / "run_current_formal_l4_currentcoverage_entry_exit_grid_20260720.py"
SPEC = importlib.util.spec_from_file_location("currentcoverage_common", COMMON_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

FEATURE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_persistent_edge_20260720"
    / "persistent_edge_features.duckdb"
)
SCORE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
    / "score_assets"
    / "w10_100.duckdb"
)
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_1d_gate_grid_20260720"
)
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"

CASES = (
    [{"name": "no_1d_gate", "operator": "all", "threshold": None}]
    + [
        {"name": f"rank1_ge_{int(threshold * 100)}", "operator": "ge", "threshold": threshold}
        for threshold in (0.20, 0.40, 0.60, 0.70, 0.80, 0.84, 0.86, 0.88, 0.90)
    ]
    + [
        {"name": f"rank1_le_{int(threshold * 100)}", "operator": "le", "threshold": threshold}
        for threshold in (0.30, 0.50, 0.70)
    ]
)


def symbol(stock_code: str) -> str:
    return ("SHSE." if stock_code.endswith(".SH") else "SZSE.") + stock_code[:6]


def source_frame() -> pd.DataFrame:
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    try:
        return con.execute(
            """
            WITH eligible AS (
                SELECT *, rank_10d AS blend, rank_10d AS select_score
                FROM persistent_edge_features
                WHERE rank_10d >= 0.98
                  AND least(rank_5d, rank_10d) >= 0.50
                  AND signal_pct_chg_raw <= 5.0
            ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY trade_date ORDER BY select_score DESC, stock_code
                ) AS pick_rank
                FROM eligible
            )
            SELECT * FROM ranked WHERE pick_rank = 1 ORDER BY trade_date
            """
        ).fetchdf()
    finally:
        con.close()


def build_signal(source: pd.DataFrame, case: dict) -> Path:
    frame = source.copy()
    if case["operator"] == "ge":
        frame = frame[frame["rank_1d"] >= float(case["threshold"])].copy()
    elif case["operator"] == "le":
        frame = frame[frame["rank_1d"] <= float(case["threshold"])].copy()
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["symbol"] = frame["stock_code"].map(symbol)
    frame["rank"] = 1
    frame["pred_prob"] = frame["blend"].astype(float)
    frame["entry_score"] = frame["blend"].astype(float)
    frame["target_pct"] = 1.0 / 3.0
    high_gap = pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce") > 5.0
    frame.loc[high_gap, "target_pct"] *= 0.50
    frame["holding_days"] = 12
    frame["max_holding_days"] = 13
    frame["score_exit_entry_ratio"] = 9.99
    frame["min_holding_days_before_score_exit"] = 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = case["name"]
    frame["filter_name"] = "current_formal_l4_1d_gate_grid"
    frame["entry_weight_name"] = "buy_gap_gt5_half"
    frame["dynamic_hold_name"] = "independent_score_replace_min12_max13"
    frame["buy_day_market_available"] = True
    frame["buy_day_hard_gate_complete"] = True
    frame["buy_day_st_rejected"] = False
    frame["buy_day_open_limit_up_rejected"] = False
    frame["latest_market_date"] = str(frame["buy_date"].max())
    frame["buy_open_gap_pct"] = frame["buy_open_gap_raw_pct"]
    columns = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "rank_1d", "rank_3d", "rank_5d", "rank_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "signal_pct_chg_raw", "target_pct",
        "holding_days", "max_holding_days", "score_exit_entry_ratio", "min_holding_days_before_score_exit",
        "score_continue_entry_ratio", "strategy_variant", "filter_name", "entry_weight_name", "dynamic_hold_name",
        "buy_day_market_available", "buy_day_hard_gate_complete", "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected", "latest_market_date", "buy_open_gap_pct", "buy_open_gap_raw_pct",
    ]
    path = SIGNALS / f"{case['name']}.csv"
    frame[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(source: pd.DataFrame, case: dict) -> dict:
    signal = build_signal(source, case)
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
            "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_EQUITY_DD_RISK_MODE": "0",
        }
    )
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(COMMON.STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(SCORE_DB), "--score-table", "blended_rank_score",
        "--market-db", str(COMMON.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    signal_frame = pd.read_csv(signal)
    return {
        **case, "signal_rows": int(len(signal_frame)),
        "signal_days": int(signal_frame["signal_date"].nunique()),
        "returncode": int(process.returncode), "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(signal),
        "signal_sha256": COMMON.sha256(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    source = source_frame()
    rows: list[dict] = []
    for case in CASES:
        result = run_case(source, case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in ("name", "signal_days", "annual_return", "sharpe", "max_drawdown")}, ensure_ascii=False), flush=True)
    payload = {
        "status": "research_only_not_admitted", "feature_db": str(FEATURE_DB),
        "feature_db_sha256": COMMON.sha256(FEATURE_DB), "score_db": str(SCORE_DB),
        "score_db_sha256": COMMON.sha256(SCORE_DB), "results": rows,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
