from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
BASE_SIGNAL = REPORT_DIR / "variants_lowgap_adaptive_quality_fine5" / "j08_s34_p04_n03_z03_a22.csv"
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
OUT_DIR = REPORT_DIR / "variants_lowgap_j08_sell_rule_grid"
RUN_DIR = REPORT_DIR / "juejin_runs_lowgap_j08_sell_rule_grid"
MANIFEST_CSV = REPORT_DIR / "variants_lowgap_j08_sell_rule_grid_manifest.csv"
RAW_CSV = REPORT_DIR / "juejin_lowgap_j08_sell_rule_grid_raw.csv"
SUMMARY_CSV = REPORT_DIR / "juejin_lowgap_j08_sell_rule_grid_summary.csv"


# name, holding_days, max_holding_days, score_exit, min_score_exit, score_continue, light_stop, min_light_stop
CASES = [
    ("base_h2_m4_e90_c100_ls09", 2, 4, 0.90, 2, 1.00, 0.09, 1),
    ("h1_m3_e90_c100_ls09", 1, 3, 0.90, 1, 1.00, 0.09, 1),
    ("h1_m4_e90_c100_ls09", 1, 4, 0.90, 1, 1.00, 0.09, 1),
    ("h2_m3_e90_c100_ls09", 2, 3, 0.90, 2, 1.00, 0.09, 1),
    ("h2_m5_e90_c100_ls09", 2, 5, 0.90, 2, 1.00, 0.09, 1),
    ("h3_m5_e90_c100_ls09", 3, 5, 0.90, 2, 1.00, 0.09, 1),
    ("h2_m4_e85_c100_ls09", 2, 4, 0.85, 2, 1.00, 0.09, 1),
    ("h2_m4_e88_c100_ls09", 2, 4, 0.88, 2, 1.00, 0.09, 1),
    ("h2_m4_e92_c100_ls09", 2, 4, 0.92, 2, 1.00, 0.09, 1),
    ("h2_m4_e95_c100_ls09", 2, 4, 0.95, 2, 1.00, 0.09, 1),
    ("h2_m4_e90_c098_ls09", 2, 4, 0.90, 2, 0.98, 0.09, 1),
    ("h2_m4_e90_c102_ls09", 2, 4, 0.90, 2, 1.02, 0.09, 1),
    ("h2_m4_e90_c105_ls09", 2, 4, 0.90, 2, 1.05, 0.09, 1),
    ("h2_m4_e90_c100_ls07", 2, 4, 0.90, 2, 1.00, 0.07, 1),
    ("h2_m4_e90_c100_ls08", 2, 4, 0.90, 2, 1.00, 0.08, 1),
    ("h2_m4_e90_c100_ls10", 2, 4, 0.90, 2, 1.00, 0.10, 1),
    ("h2_m4_e90_c100_ls11", 2, 4, 0.90, 2, 1.00, 0.11, 1),
    ("h1_m3_e92_c100_ls08", 1, 3, 0.92, 1, 1.00, 0.08, 1),
    ("h2_m3_e92_c100_ls08", 2, 3, 0.92, 2, 1.00, 0.08, 1),
    ("h2_m5_e88_c102_ls10", 2, 5, 0.88, 2, 1.02, 0.10, 1),
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


def export_case(base: pd.DataFrame, case: tuple) -> dict:
    name, holding, max_holding, score_exit, min_score_exit, score_continue, light_stop, min_light_stop = case
    df = base.copy()
    df["holding_days"] = holding
    df["max_holding_days"] = max_holding
    df["score_exit_entry_ratio"] = score_exit
    df["min_holding_days_before_score_exit"] = min_score_exit
    df["score_continue_entry_ratio"] = score_continue
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "name": name,
        "signal_file": str(path),
        "holding_days": holding,
        "max_holding_days": max_holding,
        "score_exit": score_exit,
        "min_score_exit": min_score_exit,
        "score_continue": score_continue,
        "light_stop": light_stop,
        "min_light_stop": min_light_stop,
        "avg_target": float(df["target_pct"].mean()),
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
        str(row["holding_days"]),
        "--target-position-pct",
        "0.70",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        str(row["score_exit"]),
        "--min-holding-days-before-score-exit",
        str(row["min_score_exit"]),
        "--score-continue-entry-ratio",
        str(row["score_continue"]),
        "--max-holding-days",
        str(row["max_holding_days"]),
        "--light-stop-loss-pct",
        str(row["light_stop"]),
        "--min-holding-days-before-light-stop",
        str(row["min_light_stop"]),
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
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "open_count",
        "close_count",
        "win_ratio",
        "avg_target",
    ]].to_string(index=False))
    print(SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
