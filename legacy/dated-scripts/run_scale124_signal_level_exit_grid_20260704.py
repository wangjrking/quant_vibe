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
BASE_SIGNAL = OUT_DIR / "scale124_signal_mv_loss_bucket_refine_signals/sig025_mid025_missing050_s145.csv"
SIGNAL_DIR = OUT_DIR / "scale124_signal_level_exit_signals"
RUN_DIR = OUT_DIR / "scale124_signal_level_exit_grid"
SUMMARY = OUT_DIR / "scale124_signal_level_exit_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, holding_days, max_holding_days, score_exit_ratio, min_score_hold, continue_ratio, light_stop
    ("base_signal_columns", None, None, None, None, None, None),
    ("sig_h1_mh2_e090_m1_c100_ls08", 1, 2, 0.90, 1, 1.00, 0.08),
    ("sig_h1_mh3_e090_m1_c100_ls08", 1, 3, 0.90, 1, 1.00, 0.08),
    ("sig_h1_mh3_e095_m1_c100_ls08", 1, 3, 0.95, 1, 1.00, 0.08),
    ("sig_h1_mh3_e090_m1_c102_ls08", 1, 3, 0.90, 1, 1.02, 0.08),
    ("sig_h2_mh3_e090_m1_c100_ls08", 2, 3, 0.90, 1, 1.00, 0.08),
    ("sig_h2_mh4_e085_m2_c100_ls08", 2, 4, 0.85, 2, 1.00, 0.08),
    ("sig_h2_mh4_e095_m2_c100_ls08", 2, 4, 0.95, 2, 1.00, 0.08),
    ("sig_h2_mh4_e090_m2_c102_ls08", 2, 4, 0.90, 2, 1.02, 0.08),
    ("sig_h2_mh5_e090_m2_c100_ls08", 2, 5, 0.90, 2, 1.00, 0.08),
    ("sig_h3_mh5_e090_m2_c100_ls08", 3, 5, 0.90, 2, 1.00, 0.08),
    ("sig_h2_mh4_e090_m2_c100_ls06", 2, 4, 0.90, 2, 1.00, 0.06),
    ("sig_h2_mh4_e090_m2_c100_ls10", 2, 4, 0.90, 2, 1.00, 0.10),
]


def parse_indicator(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if isinstance(indicator, dict):
        return indicator
    if not isinstance(indicator, str):
        return {}
    parsed: dict[str, object] = {}
    for key in [
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "pnl_ratio",
        "open_count",
        "close_count",
        "win_ratio",
        "calmar_ratio",
    ]:
        match = re.search(rf"'{key}': ([0-9eE+\-.]+)", indicator)
        if match:
            value = float(match.group(1))
            parsed[key] = int(value) if key.endswith("_count") else value
    if parsed:
        return parsed
    text = re.sub(r"datetime\.datetime\(.*?\)", "'datetime'", indicator)
    try:
        parsed_any = ast.literal_eval(text)
    except Exception:
        return {}
    return parsed_any if isinstance(parsed_any, dict) else {}


def build_signal(case: tuple[str, int | None, int | None, float | None, int | None, float | None, float | None]) -> tuple[Path, dict[str, object]]:
    name, holding_days, max_holding_days, score_exit_ratio, min_score_hold, continue_ratio, light_stop = case
    df = pd.read_csv(BASE_SIGNAL)
    if holding_days is not None:
        df["holding_days"] = holding_days
    if max_holding_days is not None:
        df["max_holding_days"] = max_holding_days
    if score_exit_ratio is not None:
        df["score_exit_entry_ratio"] = score_exit_ratio
    if min_score_hold is not None:
        df["min_holding_days_before_score_exit"] = min_score_hold
    if continue_ratio is not None:
        df["score_continue_entry_ratio"] = continue_ratio
    df = df.drop(columns=["signal_stop_loss_pct"], errors="ignore")
    if light_stop is not None:
        df["signal_stop_loss_pct"] = light_stop
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(pd.to_numeric(df["target_pct"], errors="coerce").mean()),
    }


def run_case(case: tuple[str, int | None, int | None, float | None, int | None, float | None, float | None]) -> dict[str, object]:
    name, holding_days, max_holding_days, score_exit_ratio, min_score_hold, continue_ratio, light_stop = case
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
            "0.0015",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")
    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "holding_days_col": holding_days,
        "max_holding_days_col": max_holding_days,
        "score_exit_ratio_col": score_exit_ratio,
        "min_score_hold_col": min_score_hold,
        "continue_ratio_col": continue_ratio,
        "light_stop_col": light_stop,
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
    rows.sort(
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
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
