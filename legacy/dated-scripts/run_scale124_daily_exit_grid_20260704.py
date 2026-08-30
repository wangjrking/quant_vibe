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
RUN_DIR = OUT_DIR / "scale124_daily_exit_grid"
SUMMARY = OUT_DIR / "scale124_daily_exit_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, holding_days, min_score_hold, continue_ratio, max_holding_days, score_exit_ratio, light_stop, min_light_hold
    ("base_h2_m2_c100_mh4_e090_ls08", 2, 2, 1.00, 4, 0.90, 0.080, 1),
    ("h1_m1_c100_mh3_e090_ls08", 1, 1, 1.00, 3, 0.90, 0.080, 1),
    ("h1_m1_c102_mh3_e090_ls08", 1, 1, 1.02, 3, 0.90, 0.080, 1),
    ("h1_m1_c105_mh3_e090_ls08", 1, 1, 1.05, 3, 0.90, 0.080, 1),
    ("h1_m1_c100_mh2_e090_ls08", 1, 1, 1.00, 2, 0.90, 0.080, 1),
    ("h1_m1_c100_mh4_e090_ls08", 1, 1, 1.00, 4, 0.90, 0.080, 1),
    ("h2_m1_c100_mh4_e090_ls08", 2, 1, 1.00, 4, 0.90, 0.080, 1),
    ("h2_m1_c100_mh4_e095_ls08", 2, 1, 1.00, 4, 0.95, 0.080, 1),
    ("h2_m1_c102_mh4_e095_ls08", 2, 1, 1.02, 4, 0.95, 0.080, 1),
    ("h2_m2_c100_mh3_e090_ls08", 2, 2, 1.00, 3, 0.90, 0.080, 1),
    ("h2_m2_c102_mh4_e090_ls08", 2, 2, 1.02, 4, 0.90, 0.080, 1),
    ("h2_m2_c105_mh4_e090_ls08", 2, 2, 1.05, 4, 0.90, 0.080, 1),
    ("h2_m2_c100_mh5_e090_ls08", 2, 2, 1.00, 5, 0.90, 0.080, 1),
    ("h2_m2_c100_mh4_e085_ls08", 2, 2, 1.00, 4, 0.85, 0.080, 1),
    ("h2_m2_c100_mh4_e095_ls08", 2, 2, 1.00, 4, 0.95, 0.080, 1),
    ("h2_m2_c100_mh4_e090_ls06", 2, 2, 1.00, 4, 0.90, 0.060, 1),
    ("h2_m2_c100_mh4_e090_ls10", 2, 2, 1.00, 4, 0.90, 0.100, 1),
    ("h2_m2_c100_mh4_e090_nols", 2, 2, 1.00, 4, 0.90, None, 1),
    ("h3_m2_c100_mh5_e090_ls08", 3, 2, 1.00, 5, 0.90, 0.080, 1),
    ("h3_m2_c102_mh5_e090_ls08", 3, 2, 1.02, 5, 0.90, 0.080, 1),
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


def signal_meta() -> dict[str, object]:
    df = pd.read_csv(BASE_SIGNAL, usecols=["buy_date", "target_pct"])
    return {
        "signal_file": str(BASE_SIGNAL),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(pd.to_numeric(df["target_pct"], errors="coerce").mean()),
    }


def run_case(case: tuple[str, int, int, float, int, float, float | None, int]) -> dict[str, object]:
    name, holding_days, min_score_hold, continue_ratio, max_holding_days, score_exit_ratio, light_stop, min_light_hold = case
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
            str(BASE_SIGNAL),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
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
            str(score_exit_ratio),
            "--min-holding-days-before-score-exit",
            str(min_score_hold),
            "--score-continue-entry-ratio",
            str(continue_ratio),
            "--max-holding-days",
            str(max_holding_days),
            "--min-holding-days-before-light-stop",
            str(min_light_hold),
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        if light_stop is not None:
            cmd.extend(["--light-stop-loss-pct", str(light_stop)])
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")
    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "holding_days": holding_days,
        "min_score_hold": min_score_hold,
        "continue_ratio": continue_ratio,
        "max_holding_days": max_holding_days,
        "score_exit_ratio": score_exit_ratio,
        "light_stop": light_stop,
        "min_light_hold": min_light_hold,
        **signal_meta(),
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
