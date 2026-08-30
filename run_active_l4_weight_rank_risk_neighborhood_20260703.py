from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_MANIFEST = REPORT_DIR / "variants_0939_weight_rank_ls08_manifest.csv"
RUN_DIR = REPORT_DIR / "juejin_runs_0939_weight_rank_risk_neighborhood"
SUMMARY_CSV = REPORT_DIR / "juejin_0939_weight_rank_risk_neighborhood_summary.csv"
RAW_CSV = REPORT_DIR / "juejin_0939_weight_rank_risk_neighborhood_raw.csv"
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

FOCUS = {
    "0939_w25_25_00_50_rank95_amt50000_top1_t50_h2_ls08",
    "0939_w10_10_20_60_rank95_amt50000_top1_t50_h2_ls08",
}
TARGETS = [0.40, 0.45, 0.50]
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


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SOURCE_MANIFEST)
    manifest = manifest[manifest["name"].isin(FOCUS)].copy()
    rows = []
    for _, source in manifest.iterrows():
        for target in TARGETS:
            for light_stop in LIGHT_STOPS:
                case_name = f"{source['name']}_t{int(target * 100):02d}_ls{int(light_stop * 1000):03d}"
                log_file = RUN_DIR / f"{case_name}.log"
                cmd = [
                    str(PYTHON),
                    str(MAIN / "run_juejin_signal_backtest.py"),
                    "--strategy-dir",
                    str(STRATEGY_DIR),
                    "--signal-file",
                    str(source["signal_file"]),
                    "--log-file",
                    str(log_file),
                    "--max-positions",
                    "1",
                    "--holding-days",
                    "2",
                    "--target-position-pct",
                    f"{target:.2f}",
                    "--score-db",
                    str(source["score_db"]),
                    "--score-table",
                    "score",
                    "--market-db",
                    str(MARKET_DB),
                    "--score-exit-entry-ratio",
                    "9.99",
                    "--min-holding-days-before-score-exit",
                    "2",
                    "--score-continue-entry-ratio",
                    "9.99",
                    "--max-holding-days",
                    "2",
                    "--light-stop-loss-pct",
                    f"{light_stop:.3f}",
                    "--min-holding-days-before-light-stop",
                    "1",
                    "--backtest-adjust",
                    "none",
                    "--backtest-slippage-ratio",
                    "0.0015",
                ]
                print(f"RUN {case_name}", flush=True)
                completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=180)
                text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else completed.stdout + completed.stderr
                indicator = _indicator_from_log(text)
                row = {
                    "name": case_name,
                    "source_name": source["name"],
                    "target": target,
                    "light_stop": light_stop,
                    "signal_file": source["signal_file"],
                    "score_db": source["score_db"],
                    "log_file": str(log_file),
                    "returncode": completed.returncode,
                    "has_indicator": bool(indicator),
                    "signal_rows": source.get("signal_rows"),
                    "signal_buy_days": source.get("signal_buy_days"),
                }
                row.update(indicator)
                rows.append(row)
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
