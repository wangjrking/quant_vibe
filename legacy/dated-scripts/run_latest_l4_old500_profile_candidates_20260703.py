from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
WIDE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
OUT_DIR = REPORT_DIR / "latest_l4_old500_profile_candidates"
LOG_DIR = OUT_DIR / "juejin_runs"
RUNNER = ROOT / "quant" / "main" / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "code_snapshot_sell_available_safe_20260702"
)
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "prof_rank16_pct175_gap15_lowgap",
        "rank_max": 16,
        "pct_max": -1.75,
        "gap_low": -8.0,
        "gap_high": 1.5,
        "amount_min": 90000,
        "order": "buy_open_gap_raw_pct ASC, score_desc_rank ASC, stock_code",
        "topn": 3,
        "pos": [0.32, 0.32, 0.32],
    },
    {
        "name": "prof_rank16_pct25_gap10_lowgap",
        "rank_max": 16,
        "pct_max": -2.5,
        "gap_low": -8.0,
        "gap_high": 1.0,
        "amount_min": 90000,
        "order": "buy_open_gap_raw_pct ASC, score_desc_rank ASC, stock_code",
        "topn": 3,
        "pos": [0.32, 0.32, 0.32],
    },
    {
        "name": "prof_rank32_pct175_gap15_score",
        "rank_max": 32,
        "pct_max": -1.75,
        "gap_low": -8.0,
        "gap_high": 1.5,
        "amount_min": 90000,
        "order": "score_desc_rank ASC, buy_open_gap_raw_pct ASC, stock_code",
        "topn": 3,
        "pos": [0.32, 0.32, 0.32],
    },
    {
        "name": "prof_rank32_pct25_gap10_score",
        "rank_max": 32,
        "pct_max": -2.5,
        "gap_low": -8.0,
        "gap_high": 1.0,
        "amount_min": 90000,
        "order": "score_desc_rank ASC, buy_open_gap_raw_pct ASC, stock_code",
        "topn": 3,
        "pos": [0.32, 0.32, 0.32],
    },
    {
        "name": "prof_rank16_pct175_gap15_rankpos",
        "rank_max": 16,
        "pct_max": -1.75,
        "gap_low": -8.0,
        "gap_high": 1.5,
        "amount_min": 90000,
        "order": "buy_open_gap_raw_pct ASC, score_desc_rank ASC, stock_code",
        "topn": 3,
        "pos": [0.45, 0.30, 0.20],
    },
    {
        "name": "prof_rank16_pct175_gap15_top2",
        "rank_max": 16,
        "pct_max": -1.75,
        "gap_low": -8.0,
        "gap_high": 1.5,
        "amount_min": 90000,
        "order": "buy_open_gap_raw_pct ASC, score_desc_rank ASC, stock_code",
        "topn": 2,
        "pos": [0.48, 0.48],
    },
    {
        "name": "prof_rank16_pct175_gap15_top1",
        "rank_max": 16,
        "pct_max": -1.75,
        "gap_low": -8.0,
        "gap_high": 1.5,
        "amount_min": 90000,
        "order": "buy_open_gap_raw_pct ASC, score_desc_rank ASC, stock_code",
        "topn": 1,
        "pos": [0.70],
    },
    {
        "name": "prof_rank24_pct20_gap075_amt20",
        "rank_max": 24,
        "pct_max": -2.0,
        "gap_low": -8.0,
        "gap_high": 0.75,
        "amount_min": 200000,
        "order": "buy_open_gap_raw_pct ASC, score_desc_rank ASC, stock_code",
        "topn": 3,
        "pos": [0.34, 0.31, 0.27],
    },
]


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _extract(stdout: str) -> dict:
    payload = json.loads(stdout)
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        try:
            return ast.literal_eval(indicator)
        except Exception:
            return {}
    return indicator or {}


def _export_case(con: duckdb.DuckDBPyConnection, case: dict) -> Path:
    query = f"""
    WITH filtered AS (
      SELECT *
      FROM active_l4_wide
      WHERE score_desc_rank <= {int(case['rank_max'])}
        AND signal_pct_chg <= {float(case['pct_max'])}
        AND buy_open_gap_raw_pct >= {float(case['gap_low'])}
        AND buy_open_gap_raw_pct <= {float(case['gap_high'])}
        AND signal_amount >= {float(case['amount_min'])}
        AND ret_h1 IS NOT NULL
    ),
    picked AS (
      SELECT
        *,
        row_number() OVER (
          PARTITION BY signal_date
          ORDER BY {case['order']}
        ) AS pick_rank
      FROM filtered
    )
    SELECT *
    FROM picked
    WHERE pick_rank <= {int(case['topn'])}
    ORDER BY signal_date, pick_rank
    """
    df = con.execute(query).fetchdf()
    df.insert(2, "symbol", df["stock_code"].map(_symbol))
    df["rank"] = df["pick_rank"].astype(int)
    pos = list(case["pos"])
    df["target_pct"] = df["rank"].map(lambda r: f"{pos[int(r)-1]:.5f}")
    df["pred_prob"] = df["pred_prob"].astype(float)
    df["entry_score"] = df["pred_prob"]
    df["holding_days"] = 1
    df["max_holding_days"] = 1
    df["score_exit_entry_ratio"] = "9.99000"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = "9.99000"
    df["signal_stop_loss_pct"] = "0.05000"
    df["signal_take_profit_pct"] = "0.08000"
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
    df["entry_weight_name"] = "latest_l4_old500_profile"
    df["dynamic_hold_name"] = "h1m1_profile"
    df["buy_day_market_available"] = True
    df["buy_day_hard_gate_complete"] = True
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["latest_market_date"] = "20260702"
    out_cols = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "entry_score",
        "score_pct_rank",
        "score_desc_rank",
        "signal_pct_chg",
        "buy_open_gap_raw_pct",
        "signal_amount",
        "signal_turnover_rate",
        "signal_total_mv",
        "signal_atr_qfq",
        "target_pct",
        "holding_days",
        "max_holding_days",
        "score_exit_entry_ratio",
        "min_holding_days_before_score_exit",
        "score_continue_entry_ratio",
        "signal_stop_loss_pct",
        "signal_take_profit_pct",
        "strategy_variant",
        "filter_name",
        "entry_weight_name",
        "dynamic_hold_name",
        "buy_day_market_available",
        "buy_day_hard_gate_complete",
        "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected",
        "latest_market_date",
    ]
    case_dir = OUT_DIR / "signals" / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "signals.csv"
    df[out_cols].to_csv(out, index=False, encoding="utf-8")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    con = duckdb.connect(str(WIDE_DB), read_only=True)
    try:
        for case in CASES:
            signal_file = _export_case(con, case)
            sig = pd.read_csv(signal_file, dtype={"signal_date": str})
            target_by_day = sig.assign(tp=sig["target_pct"].astype(float)).groupby("signal_date")["tp"].sum()
            log_file = LOG_DIR / f"{case['name']}.log"
            cmd = [
                sys.executable,
                str(RUNNER),
                "--strategy-dir",
                str(STRATEGY_DIR),
                "--signal-file",
                str(signal_file),
                "--log-file",
                str(log_file),
                "--max-positions",
                str(int(case["topn"])),
                "--holding-days",
                "1",
                "--max-holding-days",
                "1",
                "--score-exit-entry-ratio",
                "9.99",
                "--min-holding-days-before-score-exit",
                "1",
                "--score-continue-entry-ratio",
                "9.99",
                "--light-stop-loss-pct",
                "0.05",
                "--min-holding-days-before-light-stop",
                "1",
                "--take-profit-pct",
                "0.08",
                "--score-db",
                str(SCORE_DB),
                "--score-table",
                "score",
                "--market-db",
                str(MARKET_DB),
                "--backtest-adjust",
                "none",
                "--backtest-slippage-ratio",
                "0.0015",
            ]
            print("RUN", case["name"], flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (LOG_DIR / f"{case['name']}.runner.json").write_text(proc.stdout, encoding="utf-8")
            indicator = _extract(proc.stdout) if proc.returncode == 0 else {}
            rows.append(
                {
                    "name": case["name"],
                    "returncode": proc.returncode,
                    "rows": int(len(sig)),
                    "signal_days": int(sig["signal_date"].nunique()),
                    "avg_target_sum": float(target_by_day.mean()) if len(target_by_day) else 0.0,
                    "max_target_sum": float(target_by_day.max()) if len(target_by_day) else 0.0,
                    "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                    "sharp_ratio": indicator.get("sharp_ratio"),
                    "max_drawdown": indicator.get("max_drawdown"),
                    "open_count": indicator.get("open_count"),
                    "close_count": indicator.get("close_count"),
                    "win_ratio": indicator.get("win_ratio"),
                    "signal_file": str(signal_file),
                    "log_file": str(log_file),
                }
            )
    finally:
        con.close()
    df = pd.DataFrame(rows).sort_values(["pnl_ratio_annual", "sharp_ratio"], ascending=False)
    out = OUT_DIR / "old500_profile_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(out)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
