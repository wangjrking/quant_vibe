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
OUT_DIR = REPORT_DIR / "variants_lowgap_multitop_exit"
RUN_DIR = REPORT_DIR / "juejin_runs_lowgap_multitop_exit"
MANIFEST_CSV = REPORT_DIR / "variants_lowgap_multitop_exit_manifest.csv"
RAW_CSV = REPORT_DIR / "juejin_lowgap_multitop_exit_raw.csv"
SUMMARY_CSV = REPORT_DIR / "juejin_lowgap_multitop_exit_summary.csv"


CASES = [
    # name, topn, target_each, max_positions
    ("top1_t542", 1, 0.542, 1),
    ("top2_t25", 2, 0.25, 2),
    ("top2_t30", 2, 0.30, 2),
    ("top2_t35", 2, 0.35, 2),
    ("top2_t40", 2, 0.40, 2),
    ("top3_t18", 3, 0.18, 3),
    ("top3_t22", 3, 0.22, 3),
    ("top3_t25", 3, 0.25, 3),
    ("top4_t15", 4, 0.15, 4),
    ("top4_t18", 4, 0.18, 4),
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


def export_case(con: duckdb.DuckDBPyConnection, name: str, topn: int, target_each: float, max_positions: int) -> dict:
    sig = con.execute(
        f"""
        WITH picked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY buy_date
                    ORDER BY buy_open_gap_raw_pct ASC, pred_prob DESC, stock_code
                ) AS pick_rank
            FROM active_l4_wide
            WHERE score_pct_rank >= 0.95
              AND signal_amount >= 50000
              AND buy_amount >= 50000
              AND ret_h2 IS NOT NULL
              AND buy_open_gap_raw_pct >= -3.0
              AND buy_open_gap_raw_pct <= 1.5
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
        WHERE pick_rank <= {topn}
        ORDER BY buy_date, pick_rank
        """
    ).fetchdf()
    sig["symbol"] = np.where(
        sig["stock_code"].str.endswith(".SH"),
        "SHSE." + sig["stock_code"].str.split(".").str[0],
        "SZSE." + sig["stock_code"].str.split(".").str[0],
    )
    sig["target_pct"] = target_each
    sig["holding_days"] = 2
    sig["max_holding_days"] = 4
    sig["score_exit_entry_ratio"] = 0.90
    sig["min_holding_days_before_score_exit"] = 2
    sig["score_continue_entry_ratio"] = 1.00
    sig["buy_day_market_available"] = True
    sig["buy_day_hard_gate_complete"] = True
    sig["buy_day_st_rejected"] = False
    sig["buy_day_open_limit_up_rejected"] = False
    sig["latest_market_date"] = "20260702"
    path = OUT_DIR / f"lowgap_multitop_{name}.csv"
    sig.to_csv(path, index=False, encoding="utf-8-sig")
    counts = sig.groupby("buy_date").size()
    return {
        "name": name,
        "signal_file": str(path),
        "topn": topn,
        "target_each": target_each,
        "target_sum": topn * target_each,
        "max_positions": max_positions,
        "signal_rows": int(len(sig)),
        "signal_buy_days": int(sig["buy_date"].nunique()),
        "days_below_topn": int((counts < topn).sum()),
        "avg_names_per_day": float(counts.mean()) if len(counts) else None,
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
        str(int(row["max_positions"])),
        "--holding-days",
        "2",
        "--target-position-pct",
        f"{float(row['target_each']):.3f}",
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
        "0.090",
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
        manifest = [export_case(con, *case) for case in CASES]
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
        "target_sum",
        "days_below_topn",
    ]].to_string(index=False))
    print(SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
