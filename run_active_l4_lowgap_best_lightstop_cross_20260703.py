from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
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
OUT_DIR = REPORT_DIR / "variants_lowgap_best_lightstop_cross"
RUN_DIR = REPORT_DIR / "juejin_runs_lowgap_best_lightstop_cross"
RAW_CSV = REPORT_DIR / "juejin_lowgap_best_lightstop_cross_raw.csv"
SUMMARY_CSV = REPORT_DIR / "juejin_lowgap_best_lightstop_cross_summary.csv"


BASE_SIGNALS = {
    "j04": REPORT_DIR / "variants_lowgap_adaptive_quality_fine5" / "j04_s38_p04_n03_z03_a18.csv",
    "j08": REPORT_DIR / "variants_lowgap_adaptive_quality_fine5" / "j08_s34_p04_n03_z03_a22.csv",
}
LIGHT_STOPS = [0.075, 0.08, 0.085, 0.09]


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


def export_signal(base_name: str, base_file: Path, light_stop: float) -> dict:
    df = pd.read_csv(base_file)
    df["holding_days"] = 2
    df["max_holding_days"] = 4
    df["score_exit_entry_ratio"] = 0.90
    df["min_holding_days_before_score_exit"] = 2
    df["score_continue_entry_ratio"] = 1.00
    name = f"{base_name}_ls{str(light_stop).replace('.', 'p')}"
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "name": name,
        "base": base_name,
        "signal_file": str(path),
        "light_stop": light_stop,
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
        str(row["light_stop"]),
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
    rows = [
        export_signal(base_name, base_file, light_stop)
        for base_name, base_file in BASE_SIGNALS.items()
        for light_stop in LIGHT_STOPS
    ]
    results = [run_juejin(row) for row in rows]
    raw = pd.DataFrame(results)
    raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    summary = raw.sort_values(["pnl_ratio_annual", "sharp_ratio", "max_drawdown"], ascending=[False, False, True])
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(summary[[
        "name",
        "base",
        "light_stop",
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
