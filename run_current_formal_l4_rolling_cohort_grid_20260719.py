from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
SOURCE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_rolling_cohort_20260719"
POOL_DB = SOURCE_DIR / "current_formal_l4_rank_pool.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_high_return_frs_scale090_cap090_v20260716" / "code_snapshot"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
SIGNAL_DIR = REPORT_DIR / "signals"
LOG_DIR = REPORT_DIR / "logs"

BLENDS = {
    "b10": ({"rank_5d": 0.0, "rank_10d": 1.0}, SOURCE_DIR / "score_assets" / "b10_100.duckdb"),
    "b510": ({"rank_5d": 0.2, "rank_10d": 0.8}, SOURCE_DIR / "score_assets" / "b5_20_10_80.duckdb"),
}

CASES = [
    {"blend": blend, "topn": topn, "hold": hold, "model_min": model_min}
    for blend in BLENDS
    for topn in [3, 5]
    for hold in [5, 10]
    for model_min in [0.90, 0.95]
]


def symbol(stock_code: str) -> str:
    return ("SHSE." if stock_code.endswith(".SH") else "SZSE.") + stock_code[:6]


def parse_indicator(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result = {}
            for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"]:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            for key in ["open_count", "close_count"]:
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    result[key] = int(match.group(1))
            return result
    return {}


def build_signal(con: duckdb.DuckDBPyConnection, case: dict) -> tuple[str, Path, int, float]:
    weights, _score_db = BLENDS[case["blend"]]
    score_expr = " + ".join(f"{weight} * {column}" for column, weight in weights.items())
    topn = int(case["topn"])
    hold = int(case["hold"])
    model_min = float(case["model_min"])
    frame = con.execute(
        f"""
        WITH eligible AS (
            SELECT *, ({score_expr}) AS model_score,
                   ({score_expr}) + 0.005 * (amount_rank + mv_rank) / 2.0 AS select_score
            FROM current_formal_l4_rank_pool
            WHERE amount >= 90000
              AND total_mv >= 200000
              AND signal_pct_chg_raw <= 2.0
              AND ({score_expr}) >= {model_min}
        ), ranked AS (
            SELECT *, row_number() OVER (PARTITION BY trade_date ORDER BY select_score DESC, stock_code) AS pick_rank
            FROM eligible
        )
        SELECT * FROM ranked WHERE pick_rank <= {topn} ORDER BY trade_date, pick_rank
        """
    ).fetchdf()
    max_positions = topn * hold
    target_pct = min(0.12, 1.15 / max_positions)
    name = f"{case['blend']}_m{int(model_min * 100)}_daily{topn}_hold{hold}_cohort"
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["symbol"] = frame["stock_code"].map(symbol)
    frame["rank"] = frame["pick_rank"].astype(int)
    frame["pred_prob"] = frame["model_score"]
    frame["entry_score"] = frame["model_score"]
    frame["target_pct"] = target_pct
    frame["holding_days"] = hold
    frame["max_holding_days"] = hold
    frame["score_exit_entry_ratio"] = 9.99
    frame["min_holding_days_before_score_exit"] = 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = name
    frame["filter_name"] = "current_formal_l4_rolling_cohort"
    frame["entry_weight_name"] = "rolling_cohort"
    frame["dynamic_hold_name"] = f"fixed_h{hold}"
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
    path = SIGNAL_DIR / f"{name}.csv"
    frame[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return name, path, max_positions, target_pct


def run_case(case: dict, name: str, signal_file: Path, max_positions: int, target_pct: float) -> dict:
    _weights, score_db = BLENDS[case["blend"]]
    log_file = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env.update({"GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0", "GM_OPEN_DAILY_SCORE_EXIT": "0"})
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY_DIR), "--signal-file", str(signal_file),
        "--log-file", str(log_file), "--max-positions", str(max_positions), "--holding-days", str(case["hold"]),
        "--max-holding-days", str(case["hold"]), "--target-position-pct", str(target_pct),
        "--score-db", str(score_db), "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log_file)
    return {
        "name": name, **case, "max_positions": max_positions, "target_pct": target_pct,
        "returncode": int(process.returncode), "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(signal_file), "log_file": str(log_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(POOL_DB), read_only=True)
    rows = []
    try:
        for case in CASES:
            name, signal_file, max_positions, target_pct = build_signal(con, case)
            result = run_case(case, name, signal_file, max_positions, target_pct)
            rows.append(result)
            frame = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
            frame.to_csv(REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig")
            print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (REPORT_DIR / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
