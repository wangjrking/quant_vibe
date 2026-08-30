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
SIGNAL_DIR = OUT_DIR / "over80_lowtarget_missingmv_signals"
RUN_DIR = OUT_DIR / "over80_lowtarget_missingmv_grid"
SUMMARY = OUT_DIR / "over80_lowtarget_missingmv_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, scale, lowtarget_min, lowtarget_mult, missingmv_mult, cap
    ("base", 1.00, 0.00, 1.00, 1.00, 0.82),
    ("drop_low20_s110", 1.10, 0.20, 0.00, 1.00, 0.82),
    ("drop_low20_s120", 1.20, 0.20, 0.00, 1.00, 0.82),
    ("drop_low20_s130", 1.30, 0.20, 0.00, 1.00, 0.82),
    ("half_low20_s110", 1.10, 0.20, 0.50, 1.00, 0.82),
    ("half_low20_s120", 1.20, 0.20, 0.50, 1.00, 0.82),
    ("drop_missing_s108", 1.08, 0.00, 1.00, 0.00, 0.82),
    ("drop_missing_s116", 1.16, 0.00, 1.00, 0.00, 0.82),
    ("drop_low20_missing_s120", 1.20, 0.20, 0.00, 0.00, 0.82),
    ("drop_low20_missing_s135", 1.35, 0.20, 0.00, 0.00, 0.82),
    ("half_low20_missing_s125", 1.25, 0.20, 0.50, 0.00, 0.82),
    ("drop_low25_missing_s135", 1.35, 0.25, 0.00, 0.00, 0.82),
    ("drop_low30_missing_s145", 1.45, 0.30, 0.00, 0.00, 0.82),
    ("drop_low20_missing_cap75_s145", 1.45, 0.20, 0.00, 0.00, 0.75),
    ("drop_low20_missing_cap70_s160", 1.60, 0.20, 0.00, 0.00, 0.70),
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


def build_signal(case: tuple[str, float, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, scale, lowtarget_min, lowtarget_mult, missingmv_mult, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["target_pct", "signal_total_mv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    mult = pd.Series(1.0, index=df.index)
    mult.loc[df["target_pct"] <= lowtarget_min] *= lowtarget_mult
    mult.loc[df["signal_total_mv"].isna()] *= missingmv_mult
    df["target_pct"] = (df["target_pct"] * mult * scale).clip(lower=0.03, upper=cap)
    df = df.loc[df["target_pct"] > 0.031].copy()
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    counts = df.groupby("buy_date").size()
    target_sum = df.groupby("buy_date")["target_pct"].sum()
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "dropped_rows": int(before - len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "days_below_topn": int((counts < 2).sum()) if len(counts) else 0,
        "avg_names_per_day": float(counts.mean()) if len(counts) else None,
        "avg_target": float(df["target_pct"].mean()) if len(df) else None,
        "avg_target_sum_per_day": float(target_sum.mean()) if len(target_sum) else None,
        "max_target_sum_per_day": float(target_sum.max()) if len(target_sum) else None,
    }


def run_case(case: tuple[str, float, float, float, float, float]) -> dict[str, object]:
    name, scale, lowtarget_min, lowtarget_mult, missingmv_mult, cap = case
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
        "scale": scale,
        "lowtarget_min": lowtarget_min,
        "lowtarget_mult": lowtarget_mult,
        "missingmv_mult": missingmv_mult,
        "cap": cap,
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
