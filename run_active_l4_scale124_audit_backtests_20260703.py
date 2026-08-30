from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
SIGNAL_FILE = REPORT_DIR / "variants_refill_best_position_scale_fine" / "scale124_cap82.csv"
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
RUN_DIR = REPORT_DIR / "juejin_runs_scale124_audit"
RAW_CSV = REPORT_DIR / "juejin_scale124_audit_raw.csv"
SUMMARY_CSV = REPORT_DIR / "juejin_scale124_audit_summary.csv"


CASES = [
    ("full_slip0015", None, None, 0.0015),
    ("full_slip0030", None, None, 0.0030),
    ("full_slip0040", None, None, 0.0040),
    ("full_slip0065", None, None, 0.0065),
    ("slice_2022", "2022-06-07 09:00:00", "2022-12-31 15:30:00", 0.0015),
    ("slice_2023", "2023-01-01 09:00:00", "2023-12-31 15:30:00", 0.0015),
    ("slice_2024", "2024-01-01 09:00:00", "2024-12-31 15:30:00", 0.0015),
    ("slice_2025", "2025-01-01 09:00:00", "2025-12-31 15:30:00", 0.0015),
    ("slice_2026", "2026-01-01 09:00:00", "2026-07-10 15:30:00", 0.0015),
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


def run_case(case: tuple) -> dict:
    name, start, end, slip = case
    log_file = RUN_DIR / f"{name}.log"
    cmd = [
        str(PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(SIGNAL_FILE),
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--holding-days",
        "2",
        "--target-position-pct",
        "0.82",
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
        str(slip),
    ]
    if start:
        cmd.extend(["--backtest-start", start])
    if end:
        cmd.extend(["--backtest-end", end])
    print(f"RUN {name}", flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=180)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else completed.stdout + completed.stderr
    indicator = _indicator_from_log(text)
    out = {
        "name": name,
        "start": start,
        "end": end,
        "slippage": slip,
        "log_file": str(log_file),
        "returncode": completed.returncode,
        "has_indicator": bool(indicator),
    }
    out.update(indicator)
    return out


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case) for case in CASES]
    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    summary = raw.sort_values(["name"])
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(summary[[
        "name",
        "slippage",
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
