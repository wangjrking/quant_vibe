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
SPEC = importlib.util.spec_from_file_location("common", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_lexicographic_entry_20260720"
SIGNALS, SCORES, LOGS = OUT / "signals", OUT / "score_assets", OUT / "logs"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719" / "research_code_snapshot"
BASE_FILTER = "rank_10d >= 0.98 AND least(rank_5d, rank_10d) >= 0.50 AND signal_pct_chg_raw <= 5.0"
CASES = [
    {"name": "current_postgate", "shortlist": 1, "selector": "rank_10d", "post_rank1": True},
    {"name": "full_refill_r1", "shortlist": 9999, "selector": "rank_10d", "pre_rank1": True},
    {"name": "top3_choose_r1", "shortlist": 3, "selector": "rank_1d"},
    {"name": "top5_choose_r1", "shortlist": 5, "selector": "rank_1d"},
    {"name": "top10_choose_r1", "shortlist": 10, "selector": "rank_1d"},
    {"name": "top3_choose_r5", "shortlist": 3, "selector": "rank_5d"},
    {"name": "top5_choose_r5", "shortlist": 5, "selector": "rank_5d"},
    {"name": "top5_choose_min", "shortlist": 5, "selector": "least(rank_1d, rank_5d, rank_10d)"},
]


def build_signal(con: duckdb.DuckDBPyConnection, case: dict) -> Path:
    pre_rank1 = "AND rank_1d >= 0.88" if case.get("pre_rank1") else ""
    frame = con.execute(
        f"""
        WITH eligible AS (
            SELECT *, row_number() OVER (
                PARTITION BY trade_date ORDER BY rank_10d DESC, stock_code
            ) AS rank10_short
            FROM persistent_edge_features
            WHERE {BASE_FILTER} {pre_rank1}
        ), shortlisted AS (
            SELECT * FROM eligible WHERE rank10_short <= {int(case['shortlist'])}
        ), selected AS (
            SELECT *, row_number() OVER (
                PARTITION BY trade_date ORDER BY {case['selector']} DESC, rank_10d DESC, stock_code
            ) AS pick_rank
            FROM shortlisted
        )
        SELECT * FROM selected WHERE pick_rank = 1 ORDER BY trade_date
        """
    ).fetchdf()
    if case.get("post_rank1"):
        frame = frame[frame["rank_1d"] >= 0.88].copy()
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["symbol"] = frame["stock_code"].map(MODULE.symbol)
    frame["rank"] = 1
    frame["pred_prob"] = frame["rank_10d"]
    frame["entry_score"] = frame["rank_10d"]
    frame["target_pct"] = 1.0 / 3.0
    frame["holding_days"], frame["max_holding_days"] = 12, 13
    frame["score_exit_entry_ratio"], frame["min_holding_days_before_score_exit"] = 9.99, 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = case["name"]
    frame["filter_name"] = "cap5_lexicographic_entry"
    frame["entry_weight_name"] = "top10_lexicographic"
    frame["dynamic_hold_name"] = "independent_current_score_12_13"
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


def run_case(case: dict, signal: Path, score_db: Path) -> dict:
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
        sys.executable, str(MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(score_db), "--score-table", "blended_rank_score", "--market-db", str(MODULE.MARKET_DB),
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
    for path in (OUT, SIGNALS, SCORES, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "juejin_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {str(row["name"]) for row in rows if int(row.get("returncode", -1)) == 0}
    con = duckdb.connect(str(MODULE.FEATURE_DB), read_only=True)
    try:
        score_db = MODULE.build_score(con, "w10_100")
        for case in CASES:
            if case["name"] in completed:
                continue
            signal = build_signal(con, case)
            result = run_case(case, signal, score_db)
            rows.append(result)
            pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).to_csv(
                result_path, index=False, encoding="utf-8-sig"
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
