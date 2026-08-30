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
OUT_DIR = REPORT_DIR / "variants_lowgap_score_exit"
RUN_DIR = REPORT_DIR / "juejin_runs_lowgap_score_exit"
MANIFEST_CSV = REPORT_DIR / "variants_lowgap_score_exit_manifest.csv"
RAW_CSV = REPORT_DIR / "juejin_lowgap_score_exit_raw.csv"
SUMMARY_CSV = REPORT_DIR / "juejin_lowgap_score_exit_summary.csv"


CASES = [
    # name, min_hold, max_hold, continue_ratio, early_exit_ratio, target, light_stop
    ("h1_m2_c090_e090_t50_ls08", 1, 2, 0.90, 0.90, 0.50, 0.08),
    ("h1_m2_c095_e090_t50_ls08", 1, 2, 0.95, 0.90, 0.50, 0.08),
    ("h1_m3_c090_e090_t50_ls08", 1, 3, 0.90, 0.90, 0.50, 0.08),
    ("h1_m3_c095_e090_t50_ls08", 1, 3, 0.95, 0.90, 0.50, 0.08),
    ("h1_m3_c100_e090_t50_ls08", 1, 3, 1.00, 0.90, 0.50, 0.08),
    ("h1_m4_c095_e090_t50_ls08", 1, 4, 0.95, 0.90, 0.50, 0.08),
    ("h2_m3_c090_e090_t50_ls08", 2, 3, 0.90, 0.90, 0.50, 0.08),
    ("h2_m3_c095_e090_t50_ls08", 2, 3, 0.95, 0.90, 0.50, 0.08),
    ("h2_m3_c100_e090_t50_ls08", 2, 3, 1.00, 0.90, 0.50, 0.08),
    ("h2_m4_c090_e090_t50_ls08", 2, 4, 0.90, 0.90, 0.50, 0.08),
    ("h2_m4_c095_e090_t50_ls08", 2, 4, 0.95, 0.90, 0.50, 0.08),
    ("h2_m4_c100_e090_t50_ls08", 2, 4, 1.00, 0.90, 0.50, 0.08),
    ("h2_m3_c095_e095_t50_ls08", 2, 3, 0.95, 0.95, 0.50, 0.08),
    ("h2_m3_c095_e085_t50_ls08", 2, 3, 0.95, 0.85, 0.50, 0.08),
    ("h2_m3_c095_e090_t45_ls08", 2, 3, 0.95, 0.90, 0.45, 0.08),
    ("h2_m3_c095_e090_t50_ls075", 2, 3, 0.95, 0.90, 0.50, 0.075),
    ("h2_m3_c095_e090_t50_ls085", 2, 3, 0.95, 0.90, 0.50, 0.085),
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
    name, min_hold, max_hold, continue_ratio, early_exit_ratio, target, light_stop = case
    df = base.copy()
    df["target_pct"] = target
    df["holding_days"] = min_hold
    df["max_holding_days"] = max_hold
    df["score_exit_entry_ratio"] = early_exit_ratio
    df["min_holding_days_before_score_exit"] = min_hold
    df["score_continue_entry_ratio"] = continue_ratio
    path = OUT_DIR / f"lowgap_scoreexit_{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "name": name,
        "signal_file": str(path),
        "min_hold": min_hold,
        "max_hold": max_hold,
        "continue_ratio": continue_ratio,
        "early_exit_ratio": early_exit_ratio,
        "target": target,
        "light_stop": light_stop,
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
        str(int(row["min_hold"])),
        "--target-position-pct",
        f"{float(row['target']):.2f}",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        f"{float(row['early_exit_ratio']):.3f}",
        "--min-holding-days-before-score-exit",
        str(int(row["min_hold"])),
        "--score-continue-entry-ratio",
        f"{float(row['continue_ratio']):.3f}",
        "--max-holding-days",
        str(int(row["max_hold"])),
        "--light-stop-loss-pct",
        f"{float(row['light_stop']):.3f}",
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
