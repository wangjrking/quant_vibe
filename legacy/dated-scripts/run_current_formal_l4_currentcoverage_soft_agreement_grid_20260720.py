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

FEATURE_DB = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_persistent_edge_20260720/persistent_edge_features.duckdb"
OUT = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_currentcoverage_soft_agreement_grid_20260720"
SIGNALS = OUT / "signals"
SCORES = OUT / "score_assets"
LOGS = OUT / "logs"

CASES = [
    {"name": "base_r1_88", "rank1_min": 0.88, "rank5_min": 0.50, "w1": 0.00, "w5": 0.00},
    {"name": "r1_80_w1_01_w5_01", "rank1_min": 0.80, "rank5_min": 0.50, "w1": 0.01, "w5": 0.01},
    {"name": "r1_80_w1_02_w5_01", "rank1_min": 0.80, "rank5_min": 0.50, "w1": 0.02, "w5": 0.01},
    {"name": "r1_80_w1_05_w5_02", "rank1_min": 0.80, "rank5_min": 0.50, "w1": 0.05, "w5": 0.02},
    {"name": "r1_84_w1_01_w5_01", "rank1_min": 0.84, "rank5_min": 0.50, "w1": 0.01, "w5": 0.01},
    {"name": "r1_84_w1_02_w5_01", "rank1_min": 0.84, "rank5_min": 0.50, "w1": 0.02, "w5": 0.01},
    {"name": "r1_86_w1_01_w5_01", "rank1_min": 0.86, "rank5_min": 0.50, "w1": 0.01, "w5": 0.01},
    {"name": "r1_86_w1_02_w5_01", "rank1_min": 0.86, "rank5_min": 0.50, "w1": 0.02, "w5": 0.01},
    {"name": "r1_88_w1_01_w5_01", "rank1_min": 0.88, "rank5_min": 0.50, "w1": 0.01, "w5": 0.01},
    {"name": "r1_88_w1_02_w5_01", "rank1_min": 0.88, "rank5_min": 0.50, "w1": 0.02, "w5": 0.01},
    {"name": "r1_84_r5_40_w1_02_w5_01", "rank1_min": 0.84, "rank5_min": 0.40, "w1": 0.02, "w5": 0.01},
]


def symbol(stock_code: str) -> str:
    return ("SHSE." if stock_code.endswith(".SH") else "SZSE.") + stock_code[:6]


def score_expression(case: dict) -> str:
    return f"rank_10d + {float(case['w1'])} * rank_1d + {float(case['w5'])} * rank_5d"


def build_assets(con: duckdb.DuckDBPyConnection, case: dict) -> tuple[Path, Path]:
    score_expr = score_expression(case)
    frame = con.execute(
        f"""
        WITH eligible AS (
            SELECT *, {score_expr} AS blend, {score_expr} AS select_score
            FROM persistent_edge_features
            WHERE rank_10d >= 0.98
              AND rank_5d >= {float(case['rank5_min'])}
              AND signal_pct_chg_raw <= 5.0
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY trade_date ORDER BY select_score DESC, stock_code
            ) AS pick_rank
            FROM eligible
        )
        SELECT * FROM ranked
        WHERE pick_rank = 1 AND rank_1d >= {float(case['rank1_min'])}
        ORDER BY trade_date
        """
    ).fetchdf()
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["symbol"] = frame["stock_code"].map(symbol)
    frame["rank"] = 1
    frame["pred_prob"] = frame["blend"].astype(float)
    frame["entry_score"] = frame["blend"].astype(float)
    frame["target_pct"] = 1.0 / 3.0
    frame.loc[pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce") > 5.0, "target_pct"] *= 0.50
    frame["holding_days"] = 12
    frame["max_holding_days"] = 13
    frame["score_exit_entry_ratio"] = 9.99
    frame["min_holding_days_before_score_exit"] = 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = case["name"]
    frame["filter_name"] = "current_formal_l4_soft_agreement"
    frame["entry_weight_name"] = "buy_gap_gt5_half"
    frame["dynamic_hold_name"] = "same_soft_score_independent_replace_min12_max13"
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
    signal_path = SIGNALS / f"{case['name']}.csv"
    frame[columns].to_csv(signal_path, index=False, encoding="utf-8-sig")

    score_path = SCORES / f"{case['name']}.duckdb"
    if score_path.exists():
        score_path.unlink()
    score_frame = con.execute(
        f"SELECT trade_date, stock_code, {score_expr} AS pred_prob FROM persistent_edge_features"
    ).fetchdf()
    out = duckdb.connect(str(score_path))
    try:
        out.register("score_frame", score_frame)
        out.execute("CREATE TABLE blended_rank_score AS SELECT * FROM score_frame")
    finally:
        out.close()
    return signal_path, score_path


def run_case(con: duckdb.DuckDBPyConnection, case: dict) -> dict:
    signal, score_db = build_assets(con, case)
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(COMMON.STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(score_db), "--score-table", "blended_rank_score", "--market-db", str(COMMON.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    signal_frame = pd.read_csv(signal)
    return {
        **case, "score_expression": score_expression(case),
        "signal_days": int(signal_frame["signal_date"].nunique()), "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "signal_sha256": COMMON.sha256(signal),
        "score_db": str(score_db), "score_db_sha256": COMMON.sha256(score_db), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, SCORES, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    rows: list[dict] = []
    try:
        for case in CASES:
            result = run_case(con, case)
            rows.append(result)
            pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
                OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
            )
            print(json.dumps({key: result.get(key) for key in (
                "name", "signal_days", "annual_return", "sharpe", "max_drawdown"
            )}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    payload = {
        "status": "research_only_not_admitted", "feature_db": str(FEATURE_DB),
        "feature_db_sha256": COMMON.sha256(FEATURE_DB), "results": rows,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
