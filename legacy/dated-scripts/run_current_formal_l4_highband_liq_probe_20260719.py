from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
POOL_DB = REPORT_DIR / "current_formal_l4_rank_pool.duckdb"
SIGNAL_DIR = REPORT_DIR / "highband_liq_signals"
LOG_DIR = REPORT_DIR / "highband_liq_logs"
OUT_CSV = REPORT_DIR / "highband_liq_juejin.csv"
OUT_JSON = REPORT_DIR / "highband_liq_juejin.json"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_high_return_frs_scale090_cap090_v20260716" / "code_snapshot"

BLENDS = {
    "b5_20_10_80": {"rank_1d": 0.0, "rank_3d": 0.0, "rank_5d": 0.20, "rank_10d": 0.80},
    "b3_10_5_20_10_70": {"rank_1d": 0.0, "rank_3d": 0.10, "rank_5d": 0.20, "rank_10d": 0.70},
}


def symbol(code: str) -> str:
    return ("SHSE." if code.endswith(".SH") else "SZSE.") + code[:6]


def parse_indicator(log_file: Path) -> dict:
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
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


def build_signal(con: duckdb.DuckDBPyConnection, blend: str, model_min: float, topn: int) -> Path:
    weights = BLENDS[blend]
    model_score = " + ".join(f"{weight} * {column}" for column, weight in weights.items())
    frame = con.execute(
        f"""
        WITH scored AS (
            SELECT
                *,
                ({model_score}) AS model_score,
                0.65 * amount_rank + 0.30 * mv_rank + 0.05 * ({model_score}) AS select_score
            FROM current_formal_l4_rank_pool
            WHERE amount >= 150000
              AND total_mv >= 500000
              AND ({model_score}) >= {model_min}
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY trade_date ORDER BY select_score DESC, stock_code) pick_rank
            FROM scored
        )
        SELECT * FROM ranked WHERE pick_rank <= {topn} ORDER BY trade_date, pick_rank
        """
    ).fetchdf()
    name = f"{blend}_m{int(model_min*100)}_liq_top{topn}_h5"
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["symbol"] = frame["stock_code"].map(symbol)
    frame["rank"] = frame["pick_rank"].astype(int)
    frame["pred_prob"] = frame["model_score"]
    frame["entry_score"] = frame["model_score"]
    frame["target_pct"] = 0.90 / topn
    frame["holding_days"] = 5
    frame["max_holding_days"] = 5
    frame["score_exit_entry_ratio"] = 9.99
    frame["min_holding_days_before_score_exit"] = 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = name
    frame["filter_name"] = "current_formal_l4_highband_liquidity"
    frame["entry_weight_name"] = "highband_liquidity"
    frame["dynamic_hold_name"] = "fixed_h5"
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
    return path


def run_case(blend: str, model_min: float, topn: int, signal_file: Path) -> dict:
    name = signal_file.stem
    log_file = LOG_DIR / f"{name}.log"
    score_db = REPORT_DIR / "score_assets" / f"{blend}.duckdb"
    existing_indicator = parse_indicator(log_file)
    if existing_indicator:
        return {
            "case": name, "blend": blend, "model_min": model_min, "topn": topn,
            "returncode": 0, "signal_file": str(signal_file), "score_db": str(score_db),
            "log_file": str(log_file), "annual_return": existing_indicator.get("pnl_ratio_annual"),
            "sharpe": existing_indicator.get("sharp_ratio"),
            "max_drawdown": existing_indicator.get("max_drawdown"),
            "win_ratio": existing_indicator.get("win_ratio"),
            "open_count": existing_indicator.get("open_count"),
            "close_count": existing_indicator.get("close_count"),
        }
    command = [
        sys.executable, str(RUNNER),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", str(topn),
        "--holding-days", "5",
        "--max-holding-days", "5",
        "--score-exit-entry-ratio", "9.99",
        "--score-continue-entry-ratio", "9.99",
        "--min-holding-days-before-score-exit", "99",
        "--score-db", str(score_db),
        "--score-table", "blended_rank_score",
        "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0", "GM_STOP_LOSS_PCT": "", "GM_TAKE_PROFIT_PCT": "",
            "GM_LIGHT_STOP_LOSS_PCT": "", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0", "GM_OPEN_DAILY_SCORE_EXIT": "0",
        }
    )
    process = None
    indicator = {}
    for attempt in range(1, 4):
        log_file.unlink(missing_ok=True)
        process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
        indicator = parse_indicator(log_file)
        text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
        if process.returncode == 0 and indicator:
            break
        if "1026" not in text or attempt == 3:
            break
        time.sleep(12)
    assert process is not None
    time.sleep(4)
    return {
        "case": name, "blend": blend, "model_min": model_min, "topn": topn,
        "returncode": int(process.returncode), "signal_file": str(signal_file), "score_db": str(score_db),
        "log_file": str(log_file), "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
    }


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(POOL_DB), read_only=True)
    rows = []
    try:
        for blend in BLENDS:
            for model_min in [0.90, 0.95]:
                for topn in [3, 5, 10]:
                    signal_file = build_signal(con, blend, model_min, topn)
                    result = run_case(blend, model_min, topn, signal_file)
                    rows.append(result)
                    checkpoint = pd.DataFrame(rows).sort_values(
                        ["sharpe", "annual_return"], ascending=False, na_position="last"
                    )
                    checkpoint.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
                    OUT_JSON.write_text(
                        json.dumps(
                            {"status": "research_only_juejin_in_progress", "results": checkpoint.to_dict("records")},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(json.dumps({key: result.get(key) for key in ["case", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    frame = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps({"status": "research_only_juejin", "results": frame.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
