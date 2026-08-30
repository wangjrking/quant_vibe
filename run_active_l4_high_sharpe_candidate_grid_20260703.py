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
OUT_DIR = REPORT_DIR / "variants_high_sharpe_candidate_grid"
RUN_DIR = REPORT_DIR / "juejin_runs_high_sharpe_candidate_grid"
SUMMARY_CSV = REPORT_DIR / "juejin_high_sharpe_candidate_grid_summary.csv"
RAW_CSV = REPORT_DIR / "juejin_high_sharpe_candidate_grid_raw.csv"


# name, rank_min, gap_min, gap_max, filter_sql, scale, cap
CASES = [
    ("hs_q97_gm25_gp08_sigle2_s050", 0.97, -2.5, 0.8, "AND signal_pct_chg <= 2.0", 0.50, 0.70),
    ("hs_q97_gm25_gp08_sigle2_s070", 0.97, -2.5, 0.8, "AND signal_pct_chg <= 2.0", 0.70, 0.70),
    ("hs_q97_gm25_gp08_sigle2_s100", 0.97, -2.5, 0.8, "AND signal_pct_chg <= 2.0", 1.00, 0.70),
    ("hs_q96_gm30_gp00_sig02_s050", 0.96, -3.0, 0.0, "AND signal_pct_chg > 0.0 AND signal_pct_chg <= 2.0", 0.50, 0.70),
    ("hs_q96_gm30_gp00_sig02_s070", 0.96, -3.0, 0.0, "AND signal_pct_chg > 0.0 AND signal_pct_chg <= 2.0", 0.70, 0.70),
    ("hs_q96_gm30_gp08_sig02_s050", 0.96, -3.0, 0.8, "AND signal_pct_chg > 0.0 AND signal_pct_chg <= 2.0", 0.50, 0.70),
    ("hs_q95_gm30_gp08_nosiggt2_midatr_s050", 0.95, -3.0, 0.8, "AND signal_pct_chg <= 2.0 AND NOT (buy_open_gap_raw_pct > -2.5 AND buy_open_gap_raw_pct <= -1.5) AND NOT (buy_atr_qfq >= 2.0 AND buy_atr_qfq < 4.0)", 0.50, 0.70),
    ("hs_q95_gm30_gp08_nosiggt2_midatr_s070", 0.95, -3.0, 0.8, "AND signal_pct_chg <= 2.0 AND NOT (buy_open_gap_raw_pct > -2.5 AND buy_open_gap_raw_pct <= -1.5) AND NOT (buy_atr_qfq >= 2.0 AND buy_atr_qfq < 4.0)", 0.70, 0.70),
    ("hs_q98_gm30_gp00_sig02_s050", 0.98, -3.0, 0.0, "AND signal_pct_chg > 0.0 AND signal_pct_chg <= 2.0", 0.50, 0.70),
    ("hs_q98_gm25_gp08_sigle2_s050", 0.98, -2.5, 0.8, "AND signal_pct_chg <= 2.0", 0.50, 0.70),
]


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


def base_target(row: pd.Series) -> float:
    gap = float(row["buy_open_gap_raw_pct"])
    sigchg = float(row["signal_pct_chg"])
    atr = float(row["buy_atr_qfq"])
    if gap <= -2.5:
        target = 0.57
    elif gap <= -1.5:
        target = 0.32
    elif gap <= 0.0:
        target = 0.62
    else:
        target = 0.40
    if 0.0 < sigchg <= 2.0:
        target += 0.34
    elif sigchg > 2.0:
        target += 0.04
    elif sigchg <= -2.0:
        target -= 0.03
    else:
        target += 0.03
    if 2.0 <= atr < 4.0:
        target -= 0.22
    return float(max(0.05, min(0.70, target)))


def export_case(con: duckdb.DuckDBPyConnection, case: tuple) -> dict:
    name, rank_min, gap_min, gap_max, filter_sql, scale, cap = case
    df = con.execute(
        f"""
        WITH picked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY buy_date
                    ORDER BY buy_open_gap_raw_pct ASC, pred_prob DESC, stock_code
                ) AS pick_rank
            FROM active_l4_wide
            WHERE score_pct_rank >= {rank_min}
              AND signal_amount >= 50000
              AND buy_amount >= 50000
              AND ret_h2 IS NOT NULL
              AND buy_open_gap_raw_pct >= {gap_min}
              AND buy_open_gap_raw_pct <= {gap_max}
              {filter_sql}
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
    df["symbol"] = np.where(
        df["stock_code"].str.endswith(".SH"),
        "SHSE." + df["stock_code"].str.split(".").str[0],
        "SZSE." + df["stock_code"].str.split(".").str[0],
    )
    df["target_pct"] = [max(0.05, min(cap, base_target(row) * scale)) for _, row in df.iterrows()]
    df["holding_days"] = 2
    df["max_holding_days"] = 4
    df["score_exit_entry_ratio"] = 0.90
    df["min_holding_days_before_score_exit"] = 2
    df["score_continue_entry_ratio"] = 1.00
    df["buy_day_market_available"] = True
    df["buy_day_hard_gate_complete"] = True
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["latest_market_date"] = "20260702"
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "name": name,
        "signal_file": str(path),
        "signal_rows": int(len(df)),
        "signal_buy_days": int(df["buy_date"].nunique()),
        "shortage_days": int(985 - df["buy_date"].nunique()),
        "avg_target": float(df["target_pct"].mean()),
        "max_target": float(df["target_pct"].max()),
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
        "2",
        "--target-position-pct",
        "0.70",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        "0.900",
        "--min-holding-days-before-score-exit",
        "2",
        "--score-continue-entry-ratio",
        "1.000",
        "--max-holding-days",
        "4",
        "--light-stop-loss-pct",
        "0.080",
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
    indicator = _indicator_from_log(text)
    out = dict(row)
    out.update({"log_file": str(log_file), "returncode": completed.returncode, "has_indicator": bool(indicator)})
    out.update(indicator)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    try:
        rows = [export_case(con, case) for case in CASES]
    finally:
        con.close()
    results = [run_juejin(row) for row in rows]
    raw = pd.DataFrame(results)
    raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    summary = raw.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False])
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(summary[[
        "name",
        "signal_buy_days",
        "shortage_days",
        "avg_target",
        "max_target",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "open_count",
        "close_count",
        "win_ratio",
    ]].to_string(index=False))
    print(SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
