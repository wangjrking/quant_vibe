from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
MAIN = ROOT / "quant/main"
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
SIGNAL_DIR = OUT_DIR / "scale124_exit_column_signals"
RUN_DIR = OUT_DIR / "scale124_exit_column_grid"
SUMMARY = OUT_DIR / "scale124_exit_column_grid_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, holding_days, max_holding_days, score_exit_ratio, min_score_hold, stop_loss, take_profit, target_scale, target_cap
    ("base_cols", 2, 4, 0.90, 2, 0.08, None, 1.00, 0.82),
    ("h2m3_e095_sl06_tp08", 2, 3, 0.95, 1, 0.06, 0.08, 1.00, 0.82),
    ("h2m2_e095_sl05_tp08", 2, 2, 0.95, 1, 0.05, 0.08, 1.00, 0.82),
    ("h1m2_e095_sl05_tp08", 1, 2, 0.95, 1, 0.05, 0.08, 1.00, 0.82),
    ("h1m1_e999_sl05_tp08", 1, 1, 9.99, 1, 0.05, 0.08, 1.00, 0.82),
    ("h2m3_e092_sl05_tp07", 2, 3, 0.92, 1, 0.05, 0.07, 1.00, 0.82),
    ("h2m2_e092_sl05_tp07", 2, 2, 0.92, 1, 0.05, 0.07, 1.00, 0.82),
    ("h2m4_e088_sl06_tp10", 2, 4, 0.88, 2, 0.06, 0.10, 1.00, 0.82),
    ("h2m5_e088_sl08_tp12", 2, 5, 0.88, 2, 0.08, 0.12, 1.00, 0.82),
    ("h2m4_e090_sl04_notp", 2, 4, 0.90, 2, 0.04, None, 1.00, 0.82),
    ("h2m4_e090_sl08_tp08", 2, 4, 0.90, 2, 0.08, 0.08, 1.00, 0.82),
    ("h2m4_e090_sl08_scale110_cap82", 2, 4, 0.90, 2, 0.08, None, 1.10, 0.82),
    ("h2m4_e090_sl08_scale090_cap75", 2, 4, 0.90, 2, 0.08, None, 0.90, 0.75),
]


def parse_indicator(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if not isinstance(indicator, str):
        return indicator or {}
    text = re.sub(r"datetime\.datetime\(.*?\)\)", "'datetime'", indicator)
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_signal(case: tuple) -> tuple[Path, dict[str, object]]:
    name, hold, max_hold, exit_ratio, min_score_hold, stop_loss, take_profit, scale, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    df["holding_days"] = int(hold)
    df["max_holding_days"] = int(max_hold)
    df["score_exit_entry_ratio"] = float(exit_ratio)
    df["min_holding_days_before_score_exit"] = int(min_score_hold)
    df["score_continue_entry_ratio"] = 1.0
    df["signal_stop_loss_pct"] = float(stop_loss)
    if take_profit is None:
        if "signal_take_profit_pct" in df.columns:
            df = df.drop(columns=["signal_take_profit_pct"])
    else:
        df["signal_take_profit_pct"] = float(take_profit)
    df["target_pct"] = (df["target_pct"].astype(float) * float(scale)).clip(lower=0.05, upper=float(cap))
    out = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    meta = {
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(df["target_pct"].mean()),
        "max_target": float(df["target_pct"].max()),
    }
    return out, meta


def run_case(case: tuple) -> dict[str, object]:
    name, hold, max_hold, exit_ratio, min_score_hold, stop_loss, take_profit, scale, cap = case
    signal_file, meta = build_signal(case)
    log_file = RUN_DIR / f"{name}.log"
    runner_json = RUN_DIR / f"{name}.runner.json"
    if runner_json.exists() and log_file.exists():
        stdout = runner_json.read_text(encoding="utf-8", errors="ignore")
        returncode = 0
    else:
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
            str(hold),
            "--target-position-pct",
            str(cap),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            "score",
            "--market-db",
            str(MARKET_DB),
            "--score-exit-entry-ratio",
            str(exit_ratio),
            "--min-holding-days-before-score-exit",
            str(min_score_hold),
            "--score-continue-entry-ratio",
            "1.000",
            "--max-holding-days",
            str(max_hold),
            "--light-stop-loss-pct",
            str(stop_loss),
            "--min-holding-days-before-light-stop",
            "1",
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        if take_profit is not None:
            cmd.extend(["--take-profit-pct", str(take_profit)])
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")
    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "holding_days": hold,
        "max_holding_days": max_hold,
        "score_exit_entry_ratio": exit_ratio,
        "min_score_hold": min_score_hold,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "target_scale": scale,
        "target_cap": cap,
        **meta,
        "returncode": returncode,
        "log_file": str(log_file),
        "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
        "sharp_ratio": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "pnl_ratio": indicator.get("pnl_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "win_ratio": indicator.get("win_ratio"),
        "calmar_ratio": indicator.get("calmar_ratio"),
    }


def main() -> int:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        print(f"RUN {case[0]}", flush=True)
        rows.append(run_case(case))
    rows = sorted(
        rows,
        key=lambda r: (
            float(r.get("sharp_ratio") or -999),
            float(r.get("pnl_ratio_annual") or -999),
        ),
        reverse=True,
    )
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
