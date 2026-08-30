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
OUT_DIR = REPORT_DIR / "variants_lowgap_adaptive_quality_fine4"
RUN_DIR = REPORT_DIR / "juejin_runs_lowgap_adaptive_quality_fine4"
SUMMARY_CSV = REPORT_DIR / "juejin_lowgap_adaptive_quality_fine4_summary.csv"
RAW_CSV = REPORT_DIR / "juejin_lowgap_adaptive_quality_fine4_raw.csv"
MANIFEST_CSV = REPORT_DIR / "variants_lowgap_adaptive_quality_fine4_manifest.csv"


# name, add for signal_pct_chg (0,2], add for >2, add for <=-2, add for (-2,0], atr24 add
CASES = [
    ("i01_s24_p04_n03_z03_a18", 0.24, 0.04, -0.03, 0.03, -0.18),
    ("i02_s26_p04_n03_z03_a18", 0.26, 0.04, -0.03, 0.03, -0.18),
    ("i03_s28_p04_n03_z03_a18", 0.28, 0.04, -0.03, 0.03, -0.18),
    ("i04_s30_p04_n03_z03_a18", 0.30, 0.04, -0.03, 0.03, -0.18),
    ("i05_s24_p06_n03_z03_a18", 0.24, 0.06, -0.03, 0.03, -0.18),
    ("i06_s28_p06_n03_z03_a18", 0.28, 0.06, -0.03, 0.03, -0.18),
    ("i07_s24_p04_n00_z03_a18", 0.24, 0.04, 0.00, 0.03, -0.18),
    ("i08_s28_p04_n00_z03_a18", 0.28, 0.04, 0.00, 0.03, -0.18),
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


def base_gap_target(gap: float) -> float:
    if gap <= -2.5:
        return 0.57
    if gap <= -1.5:
        return 0.32
    if gap <= 0.0:
        return 0.62
    return 0.40


def clipped(value: float, low: float = 0.05, high: float = 0.70) -> float:
    return float(max(low, min(high, value)))


def load_base_signal() -> pd.DataFrame:
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    try:
        df = con.execute(
            """
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
            WHERE pick_rank <= 1
            ORDER BY buy_date, pick_rank
            """
        ).fetchdf()
    finally:
        con.close()
    df["symbol"] = np.where(
        df["stock_code"].str.endswith(".SH"),
        "SHSE." + df["stock_code"].str.split(".").str[0],
        "SZSE." + df["stock_code"].str.split(".").str[0],
    )
    return df


def target_for(row: pd.Series, case: tuple) -> float:
    _name, add_0_2, add_gt2, add_le_m2, add_m2_0, add_atr24 = case
    target = base_gap_target(float(row["buy_open_gap_raw_pct"]))
    sigchg = float(row["signal_pct_chg"])
    atr = float(row["buy_atr_qfq"])
    if 0.0 < sigchg <= 2.0:
        target += add_0_2
    elif sigchg > 2.0:
        target += add_gt2
    elif sigchg <= -2.0:
        target += add_le_m2
    else:
        target += add_m2_0
    if 2.0 <= atr < 4.0:
        target += add_atr24
    return clipped(target)


def export_case(base: pd.DataFrame, case: tuple) -> dict:
    name = case[0]
    df = base.copy()
    df["target_pct"] = [target_for(row, case) for _, row in df.iterrows()]
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
        "avg_target": float(df["target_pct"].mean()),
        "min_target": float(df["target_pct"].min()),
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
    base = load_base_signal()
    manifest = [export_case(base, case) for case in CASES]
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
        "avg_target",
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
