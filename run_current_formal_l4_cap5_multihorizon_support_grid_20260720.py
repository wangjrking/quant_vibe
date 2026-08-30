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
SOURCE = MAIN / "run_current_formal_l4_cap5_lexicographic_entry_20260720.py"
SPEC = importlib.util.spec_from_file_location("lex", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_multihorizon_support_20260720"
SIGNALS, LOGS = OUT / "signals", OUT / "logs"
STRATEGY = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720" / "research_code_snapshot"
CASES = (
    [{"name": f"post_r3_{int(floor * 100)}", "mode": "post", "floor": floor} for floor in (0.50, 0.70, 0.80, 0.90, 0.95, 0.98)]
    + [{"name": f"joint_{int(floor * 100)}", "mode": "joint", "floor": floor} for floor in (0.50, 0.70, 0.80, 0.90)]
    + [{"name": f"fallback_top{topk}_joint50", "mode": "fallback", "floor": 0.50, "topk": topk} for topk in (3, 5)]
)


def prepare_signal(frame: pd.DataFrame, name: str) -> Path:
    frame = frame.copy()
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["symbol"] = frame["stock_code"].map(MODULE.MODULE.symbol)
    frame["rank"] = 1
    frame["pred_prob"] = frame["rank_10d"]
    frame["entry_score"] = frame["rank_10d"]
    frame["target_pct"] = 1.0 / 3.0
    frame["holding_days"], frame["max_holding_days"] = 12, 13
    frame["score_exit_entry_ratio"], frame["min_holding_days_before_score_exit"] = 9.99, 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = name
    frame["filter_name"] = "cap5_multihorizon_support"
    frame["entry_weight_name"] = "10d_primary_support_gate"
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
    path = SIGNALS / f"{name}.csv"
    frame[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_signal(con: duckdb.DuckDBPyConnection, case: dict) -> Path:
    floor = float(case["floor"])
    base = "rank_10d >= 0.98 AND rank_5d >= 0.50 AND signal_pct_chg_raw <= 5.0"
    if case["mode"] == "post":
        where, post = base, f"rank_1d >= 0.88 AND rank_3d >= {floor}"
        shortlist = 1
    elif case["mode"] == "joint":
        where, post = base, f"least(rank_1d, rank_3d, rank_5d) >= {floor}"
        shortlist = 1
    else:
        where, post = base, f"least(rank_1d, rank_3d, rank_5d) >= {floor}"
        shortlist = int(case["topk"])
    frame = con.execute(
        f"""
        WITH ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY trade_date ORDER BY rank_10d DESC, stock_code
            ) AS rank10_short
            FROM persistent_edge_features
            WHERE {where}
        )
        SELECT * FROM ranked
        WHERE rank10_short <= {shortlist} AND {post}
        QUALIFY row_number() OVER (PARTITION BY trade_date ORDER BY rank_10d DESC, stock_code) = 1
        ORDER BY trade_date
        """
    ).fetchdf()
    return prepare_signal(frame, case["name"])


def run_case(case: dict, signal: Path, score_db: Path) -> dict:
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
        sys.executable, str(MODULE.MODULE.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(score_db), "--score-table", "blended_rank_score",
        "--market-db", str(MODULE.MODULE.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = MODULE.MODULE.parse_indicator(log)
    return {
        "name": case["name"], "mode": case["mode"], "floor": case["floor"],
        "topk": case.get("topk", 1), "rows": len(pd.read_csv(signal)), "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(MODULE.MODULE.FEATURE_DB), read_only=True)
    rows = []
    try:
        score_db = MODULE.MODULE.build_score(con, "w10_100")
        for case in CASES:
            result = run_case(case, build_signal(con, case), score_db)
            rows.append(result)
            pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).to_csv(
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
