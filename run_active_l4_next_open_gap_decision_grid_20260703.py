from __future__ import annotations

import re
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "code_snapshot_sell_available_safe_20260702"
)
OUT_DIR = REPORT_DIR / "variants_next_open_gap_decision"
RUN_DIR = REPORT_DIR / "juejin_runs_next_open_gap_decision"
MANIFEST_CSV = REPORT_DIR / "variants_next_open_gap_decision_manifest.csv"
RAW_CSV = REPORT_DIR / "juejin_next_open_gap_decision_raw.csv"
SUMMARY_CSV = REPORT_DIR / "juejin_next_open_gap_decision_summary.csv"


SORTS = {
    "score": "pred_prob desc, buy_amount desc, stock_code",
    "lowgap": "buy_open_gap_raw_pct asc, pred_prob desc, stock_code",
    "near_flat": "abs(buy_open_gap_raw_pct - 0.0) asc, pred_prob desc, stock_code",
    "near_m1": "abs(buy_open_gap_raw_pct + 1.0) asc, pred_prob desc, stock_code",
}

CASES = [
    ("gap_m300_p150_lowgap_h2", -3.0, 1.5, "lowgap", 2, 0.50),
    ("gap_m300_p150_score_h2", -3.0, 1.5, "score", 2, 0.50),
    ("gap_m300_p150_nearflat_h2", -3.0, 1.5, "near_flat", 2, 0.50),
    ("gap_m300_p150_nearm1_h2", -3.0, 1.5, "near_m1", 2, 0.50),
    ("gap_m200_p100_score_h2", -2.0, 1.0, "score", 2, 0.50),
    ("gap_m200_p100_nearflat_h2", -2.0, 1.0, "near_flat", 2, 0.50),
    ("gap_m200_p100_nearm1_h2", -2.0, 1.0, "near_m1", 2, 0.50),
    ("gap_m150_p100_score_h2", -1.5, 1.0, "score", 2, 0.50),
    ("gap_m150_p100_nearflat_h2", -1.5, 1.0, "near_flat", 2, 0.50),
    ("gap_m100_p100_score_h2", -1.0, 1.0, "score", 2, 0.50),
    ("gap_m100_p100_nearflat_h2", -1.0, 1.0, "near_flat", 2, 0.50),
    ("gap_m300_000_score_h2", -3.0, 0.0, "score", 2, 0.50),
    ("gap_m300_000_nearm1_h2", -3.0, 0.0, "near_m1", 2, 0.50),
    ("gap_m050_p150_score_h2", -0.5, 1.5, "score", 2, 0.50),
    ("gap_000_p150_score_h2", 0.0, 1.5, "score", 2, 0.50),
    ("gap_m300_p150_score_h1", -3.0, 1.5, "score", 1, 0.50),
    ("gap_m200_p100_score_h1", -2.0, 1.0, "score", 1, 0.50),
    ("gap_m100_p100_score_h1", -1.0, 1.0, "score", 1, 0.50),
    ("gap_m300_000_score_h1", -3.0, 0.0, "score", 1, 0.50),
    ("gap_m050_p150_score_h1", -0.5, 1.5, "score", 1, 0.50),
]


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _indicator_from_log(text: str) -> dict:
    match = re.search(r"GM_BACKTEST_INDICATOR:\s*(\{.*?\})(?:\r?\n|$)", text, re.S)
    if not match:
        return {}
    raw = match.group(1)
    out: dict[str, object] = {}
    for key in [
        "pnl_ratio",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "risk_ratio",
        "open_count",
        "close_count",
        "win_count",
        "lose_count",
        "win_ratio",
        "calmar_ratio",
    ]:
        value_match = re.search(rf"'{key}':\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", raw)
        if value_match:
            value = float(value_match.group(1))
            out[key] = int(value) if key.endswith("_count") else value
    return out


def export_case(con: duckdb.DuckDBPyConnection, case: tuple) -> dict:
    name, gap_min, gap_max, sort_name, hold, target = case
    ret_col = f"ret_h{hold}"
    order_expr = SORTS[sort_name]
    sig = con.execute(
        f"""
        WITH picked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY buy_date
                    ORDER BY {order_expr}
                ) AS pick_rank
            FROM active_l4_wide
            WHERE score_pct_rank >= 0.95
              AND signal_amount >= 50000
              AND buy_amount >= 50000
              AND {ret_col} IS NOT NULL
              AND buy_open_gap_raw_pct >= {gap_min}
              AND buy_open_gap_raw_pct <= {gap_max}
        )
        SELECT
            signal_date,
            buy_date,
            stock_code,
            name,
            market,
            pick_rank AS rank,
            pred_prob,
            score_pct_rank,
            signal_pct_chg,
            signal_open_gap_raw_pct,
            buy_open_gap_raw_pct,
            signal_amount,
            buy_amount,
            signal_turnover_rate,
            buy_turnover_rate,
            signal_total_mv,
            buy_total_mv,
            signal_atr_qfq,
            buy_atr_qfq
        FROM picked
        WHERE pick_rank <= 1
        ORDER BY buy_date, pick_rank
        """
    ).fetchdf()
    sig["symbol"] = np.where(
        sig["stock_code"].str.endswith(".SH"),
        "SHSE." + sig["stock_code"].str.split(".").str[0],
        "SZSE." + sig["stock_code"].str.split(".").str[0],
    )
    sig["target_pct"] = target
    sig["holding_days"] = hold
    sig["max_holding_days"] = hold
    sig["score_exit_entry_ratio"] = 9.99
    sig["min_holding_days_before_score_exit"] = hold
    sig["score_continue_entry_ratio"] = 9.99
    sig["buy_day_market_available"] = True
    sig["buy_day_hard_gate_complete"] = True
    sig["buy_day_st_rejected"] = False
    sig["buy_day_open_limit_up_rejected"] = False
    sig["latest_market_date"] = "20260702"
    path = OUT_DIR / f"nextopen_{name}.csv"
    sig.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "name": name,
        "signal_file": str(path),
        "gap_min": gap_min,
        "gap_max": gap_max,
        "sort_name": sort_name,
        "holding_days": hold,
        "target": target,
        "signal_rows": int(len(sig)),
        "signal_buy_days": int(sig["buy_date"].nunique()),
        "avg_buy_open_gap_raw_pct": float(sig["buy_open_gap_raw_pct"].mean()) if len(sig) else None,
    }


def run_juejin(row: dict) -> dict:
    log_file = RUN_DIR / f"{row['name']}.log"
    cmd = [
        str(PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        row["signal_file"],
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--holding-days",
        str(int(row["holding_days"])),
        "--target-position-pct",
        f"{float(row['target']):.2f}",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        "9.99",
        "--min-holding-days-before-score-exit",
        str(int(row["holding_days"])),
        "--score-continue-entry-ratio",
        "9.99",
        "--max-holding-days",
        str(int(row["holding_days"])),
        "--light-stop-loss-pct",
        "0.08",
        "--min-holding-days-before-light-stop",
        "1",
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    print(f"RUN {row['name']}", flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=180)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else completed.stdout + completed.stderr
    out = dict(row)
    out.update(
        {
            "log_file": str(log_file),
            "returncode": completed.returncode,
            "has_indicator": False,
        }
    )
    indicator = _indicator_from_log(text)
    out["has_indicator"] = bool(indicator)
    out.update(indicator)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    try:
        manifest = [export_case(con, case) for case in CASES]
    finally:
        con.close()
    pd.DataFrame(manifest).to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")
    rows = [run_juejin(row) for row in manifest]
    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    summary = raw.sort_values(
        ["pnl_ratio_annual", "sharp_ratio", "max_drawdown"],
        ascending=[False, False, True],
    )
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(summary[[
        "name",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "open_count",
        "close_count",
        "win_ratio",
        "signal_rows",
        "signal_buy_days",
        "avg_buy_open_gap_raw_pct",
    ]].to_string(index=False))
    print(SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
