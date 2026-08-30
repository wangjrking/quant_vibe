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
BASE_SIGNAL = OUT_DIR / "top2_overheat_mv_quality_signals/over80_m50_s116.csv"
SIGNAL_DIR = OUT_DIR / "over80_signal_exit_frequency_signals"
RUN_DIR = OUT_DIR / "over80_signal_exit_frequency_grid"
SUMMARY = OUT_DIR / "over80_signal_exit_frequency_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, holding_days, max_holding_days, exit_ratio, min_exit_days, continue_ratio, light_stop, min_light_days, scale
    ("base", 2, 4, 0.90, 2, 1.00, 0.080, 1, 1.00),
    ("h1_exit085_m1_cnone_ls06", 1, 2, 0.85, 1, None, 0.060, 1, 1.00),
    ("h1_exit090_m1_cnone_ls08", 1, 2, 0.90, 1, None, 0.080, 1, 1.00),
    ("h1_exit095_m1_cnone_ls08_s110", 1, 2, 0.95, 1, None, 0.080, 1, 1.10),
    ("h2_exit085_m1_c100_ls06", 2, 4, 0.85, 1, 1.00, 0.060, 1, 1.00),
    ("h2_exit090_m1_c100_ls06", 2, 4, 0.90, 1, 1.00, 0.060, 1, 1.00),
    ("h2_exit095_m1_c100_ls08", 2, 4, 0.95, 1, 1.00, 0.080, 1, 1.00),
    ("h2_exit095_m2_c102_ls08", 2, 5, 0.95, 2, 1.02, 0.080, 1, 1.00),
    ("h3_exit090_m2_c100_ls08", 3, 5, 0.90, 2, 1.00, 0.080, 1, 1.00),
    ("h3_exit095_m2_c102_ls08", 3, 6, 0.95, 2, 1.02, 0.080, 1, 1.00),
    ("h2_exit088_m1_c098_ls05_s105", 2, 4, 0.88, 1, 0.98, 0.050, 1, 1.05),
    ("h2_exit092_m1_c099_ls06_s105", 2, 4, 0.92, 1, 0.99, 0.060, 1, 1.05),
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


def build_signal(case: tuple[str, int, int, float, int, float | None, float, int, float]) -> tuple[Path, dict[str, object]]:
    name, holding_days, max_holding_days, exit_ratio, min_exit_days, continue_ratio, light_stop, min_light_days, scale = case
    df = pd.read_csv(BASE_SIGNAL)
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce")
    df["target_pct"] = (df["target_pct"] * scale).clip(lower=0.03, upper=0.82)

    df["holding_days"] = holding_days
    df["max_holding_days"] = max_holding_days
    df["score_exit_entry_ratio"] = exit_ratio
    df["signal_score_exit_entry_ratio"] = exit_ratio
    df["min_holding_days_before_score_exit"] = min_exit_days
    df["signal_min_holding_days_before_score_exit"] = min_exit_days
    df["score_continue_entry_ratio"] = "" if continue_ratio is None else continue_ratio
    df["signal_score_continue_entry_ratio"] = "" if continue_ratio is None else continue_ratio
    df["signal_stop_loss_pct"] = light_stop
    df["min_holding_days_before_light_stop"] = min_light_days

    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    counts = df.groupby("buy_date").size()
    target_sum = df.groupby("buy_date")["target_pct"].sum()
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "days_below_topn": int((counts < 2).sum()) if len(counts) else 0,
        "avg_names_per_day": float(counts.mean()) if len(counts) else None,
        "avg_target": float(df["target_pct"].mean()) if len(df) else None,
        "avg_target_sum_per_day": float(target_sum.mean()) if len(target_sum) else None,
        "max_target_sum_per_day": float(target_sum.max()) if len(target_sum) else None,
    }


def run_case(case: tuple[str, int, int, float, int, float | None, float, int, float]) -> dict[str, object]:
    name, holding_days, max_holding_days, exit_ratio, min_exit_days, continue_ratio, light_stop, min_light_days, scale = case
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
            "2",
            "--holding-days",
            str(holding_days),
            "--target-position-pct",
            "0.82",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            "score",
            "--market-db",
            str(MARKET_DB),
            "--score-exit-entry-ratio",
            str(exit_ratio),
            "--min-holding-days-before-score-exit",
            str(min_exit_days),
            "--max-holding-days",
            str(max_holding_days),
            "--light-stop-loss-pct",
            str(light_stop),
            "--min-holding-days-before-light-stop",
            str(min_light_days),
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        if continue_ratio is not None:
            cmd.extend(["--score-continue-entry-ratio", str(continue_ratio)])
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")
    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "exit_ratio": exit_ratio,
        "min_exit_days": min_exit_days,
        "continue_ratio": continue_ratio,
        "light_stop": light_stop,
        "min_light_days": min_light_days,
        "scale": scale,
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
