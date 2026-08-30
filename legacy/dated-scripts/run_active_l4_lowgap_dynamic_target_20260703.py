from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
BASE_SIGNAL = REPORT_DIR / "variants_next_open_gap_decision" / "nextopen_gap_m300_p150_lowgap_h2.csv"
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
OUT_DIR = REPORT_DIR / "variants_lowgap_dynamic_target"
RUN_DIR = REPORT_DIR / "juejin_runs_lowgap_dynamic_target"
MANIFEST_CSV = REPORT_DIR / "variants_lowgap_dynamic_target_manifest.csv"
RAW_CSV = REPORT_DIR / "juejin_lowgap_dynamic_target_raw.csv"
SUMMARY_CSV = REPORT_DIR / "juejin_lowgap_dynamic_target_summary.csv"


CASES = [
    # name, strong_deep, weak_mid, rebound_mild, default
    ("dyn_a_deep60_mid35_mild58_def45", 0.60, 0.35, 0.58, 0.45),
    ("dyn_b_deep62_mid30_mild60_def42", 0.62, 0.30, 0.60, 0.42),
    ("dyn_c_deep65_mid25_mild60_def40", 0.65, 0.25, 0.60, 0.40),
    ("dyn_d_deep58_mid40_mild55_def45", 0.58, 0.40, 0.55, 0.45),
    ("dyn_e_deep60_mid45_mild55_def50", 0.60, 0.45, 0.55, 0.50),
    ("dyn_f_deep55_mid35_mild60_def42", 0.55, 0.35, 0.60, 0.42),
    ("dyn_g_deep68_mid25_mild62_def38", 0.68, 0.25, 0.62, 0.38),
    ("dyn_h_deep62_mid40_mild62_def45", 0.62, 0.40, 0.62, 0.45),
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


def assign_target(gap: float, strong_deep: float, weak_mid: float, rebound_mild: float, default: float) -> float:
    if gap <= -2.5:
        return strong_deep
    if -2.5 < gap <= -1.5:
        return weak_mid
    if -1.5 < gap <= 0.0:
        return rebound_mild
    return default


def export_case(base: pd.DataFrame, case: tuple) -> dict:
    name, strong_deep, weak_mid, rebound_mild, default = case
    df = base.copy()
    df["target_pct"] = [
        assign_target(float(g), strong_deep, weak_mid, rebound_mild, default)
        for g in df["buy_open_gap_raw_pct"]
    ]
    df["holding_days"] = 2
    df["max_holding_days"] = 4
    df["score_exit_entry_ratio"] = 0.90
    df["min_holding_days_before_score_exit"] = 2
    df["score_continue_entry_ratio"] = 1.00
    path = OUT_DIR / f"lowgap_dynamic_target_{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "name": name,
        "signal_file": str(path),
        "strong_deep": strong_deep,
        "weak_mid": weak_mid,
        "rebound_mild": rebound_mild,
        "default": default,
        "avg_target": float(df["target_pct"].mean()),
        "max_target": float(df["target_pct"].max()),
        "signal_rows": int(len(df)),
        "signal_buy_days": int(df["buy_date"].nunique()),
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
    base = pd.read_csv(BASE_SIGNAL)
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
