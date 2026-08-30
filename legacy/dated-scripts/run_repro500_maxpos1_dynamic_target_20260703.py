from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_SIGNAL = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_repro500_p435_s96_p10d70_v20260702"
    / "signals"
    / "full_history_repro500_p435_s96_p10d70.csv"
)
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant" / "main" / "run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

OUT_DIR = REPORT_DIR / "repro500_maxpos1_dynamic_target"
LOG_DIR = OUT_DIR / "juejin_runs"


CASES = [
    {"name": "m1_dyn70_base44_weak35", "deep": 0.70, "strong": 0.60, "base": 0.44, "weak": 0.35},
    {"name": "m1_dyn80_base43_weak32", "deep": 0.80, "strong": 0.65, "base": 0.43, "weak": 0.32},
    {"name": "m1_dyn85_base42_weak28", "deep": 0.85, "strong": 0.68, "base": 0.42, "weak": 0.28},
    {"name": "m1_dyn90_base40_weak25", "deep": 0.90, "strong": 0.70, "base": 0.40, "weak": 0.25},
    {"name": "m1_dyn75_base45_weak40", "deep": 0.75, "strong": 0.62, "base": 0.45, "weak": 0.40},
    {"name": "m1_dyn65_base50_weak38", "deep": 0.65, "strong": 0.58, "base": 0.50, "weak": 0.38},
]


def _target(row: pd.Series, case: dict) -> float:
    pct = float(row["pct_chg"])
    gap = float(row["buy_open_gap_pct"])
    amount = float(row["amount"])
    turnover = float(row["turnover_rate"])
    pred10 = float(row["pred_10d"])

    high_quality = (pct <= -5.0 and gap <= -1.0) or (pct <= -4.0 and gap <= 0 and (amount >= 200000 or turnover >= 3))
    strong_quality = pct <= -3.0 and gap <= 0.5 and pred10 >= 0.95
    weak = pct > -2.0 or gap > 0.8 or pred10 < 0.9
    if high_quality:
        return float(case["deep"])
    if strong_quality:
        return float(case["strong"])
    if weak:
        return float(case["weak"])
    return float(case["base"])


def _extract(stdout: str) -> dict:
    payload = json.loads(stdout)
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        try:
            return ast.literal_eval(indicator)
        except Exception:
            return {}
    return indicator or {}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    src = src.sort_values(["signal_date", "rank", "stock_code"]).groupby("signal_date", as_index=False).head(1).copy()
    src["rank"] = 1

    rows = []
    for case in CASES:
        x = src.copy()
        x["target_pct_float"] = x.apply(lambda row: _target(row, case), axis=1)
        x["target_pct"] = x["target_pct_float"].map(lambda v: f"{v:.5f}")
        x["strategy_variant"] = case["name"]
        x["filter_name"] = case["name"]
        case_dir = OUT_DIR / "signals" / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        signal_file = case_dir / "signals.csv"
        x.drop(columns=["target_pct_float"]).to_csv(signal_file, index=False, encoding="utf-8")

        log_file = LOG_DIR / f"{case['name']}.log"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            "1",
            "--max-holding-days",
            "1",
            "--score-exit-entry-ratio",
            "0.96",
            "--min-holding-days-before-score-exit",
            "1",
            "--score-continue-entry-ratio",
            "9.99",
            "--light-stop-loss-pct",
            "0.05",
            "--min-holding-days-before-light-stop",
            "1",
            "--take-profit-pct",
            "0.08",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            "score",
            "--market-db",
            str(MARKET_DB),
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        print("RUN", case["name"], flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (LOG_DIR / f"{case['name']}.runner.json").write_text(proc.stdout, encoding="utf-8")
        indicator = _extract(proc.stdout) if proc.returncode == 0 else {}
        rows.append(
            {
                "name": case["name"],
                "returncode": proc.returncode,
                "avg_target": float(x["target_pct"].astype(float).mean()),
                "max_target": float(x["target_pct"].astype(float).max()),
                "signal_rows": int(len(x)),
                "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                "sharp_ratio": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
                "signal_file": str(signal_file),
                "log_file": str(log_file),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=False)
    out_csv = OUT_DIR / "maxpos1_dynamic_target_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(out_csv)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
