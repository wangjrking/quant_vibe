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
SIGNAL_DIR = OUT_DIR / "scale124_open_gap_weight_signals"
RUN_DIR = OUT_DIR / "scale124_open_gap_weight_grid"
SUMMARY = OUT_DIR / "scale124_open_gap_weight_grid_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


BASE_SIG5_8_MULT = 0.45
BASE_SIG_GT8_MULT = 0.55
BASE_MID_MV_MULT = 0.45
BASE_GLOBAL_SCALE = 1.20
BASE_CAP = 0.82

CASES = [
    # name, gap_le_2p7, gap_mid_2p7_1p5, gap_gt_1p0, gap_gt_0p6, global_scale, cap
    ("base_fine", 1.00, 1.00, 1.00, 1.00, 1.00, 0.82),
    ("down_mild_gt1p0_070", 1.00, 1.00, 0.70, 1.00, 1.04, 0.82),
    ("down_mild_gt1p0_050", 1.00, 1.00, 0.50, 1.00, 1.08, 0.82),
    ("down_mild_gt0p6_050", 1.00, 1.00, 1.00, 0.50, 1.06, 0.82),
    ("down_deep_le2p7_070", 0.70, 1.00, 1.00, 1.00, 1.06, 0.82),
    ("down_deep_le2p7_050", 0.50, 1.00, 1.00, 1.00, 1.10, 0.82),
    ("down_mid_2p7_1p5_085", 1.00, 0.85, 1.00, 1.00, 1.06, 0.82),
    ("down_mid_2p7_1p5_070", 1.00, 0.70, 1.00, 1.00, 1.10, 0.82),
    ("barbell_down_extremes_070", 0.70, 1.00, 0.70, 1.00, 1.10, 0.82),
    ("barbell_down_extremes_050", 0.50, 1.00, 0.50, 1.00, 1.15, 0.82),
    ("prefer_mid_gap", 0.70, 1.15, 0.70, 1.00, 1.08, 0.82),
    ("prefer_deep_gap", 1.15, 0.95, 0.70, 1.00, 1.05, 0.82),
    ("prefer_mild_gap", 0.70, 0.95, 1.15, 1.00, 1.05, 0.82),
    ("cap75_down_mild", 1.00, 1.00, 0.60, 1.00, 1.15, 0.75),
    ("cap75_prefer_mid", 0.70, 1.15, 0.70, 1.00, 1.15, 0.75),
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


def build_signal(case: tuple[str, float, float, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, gap_le_2p7, gap_mid_2p7_1p5, gap_gt_1p0, gap_gt_0p6, global_scale, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["buy_total_mv", "signal_pct_chg", "buy_open_gap_raw_pct", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mult = pd.Series(1.0, index=df.index)
    mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= BASE_SIG5_8_MULT
    mult.loc[df["signal_pct_chg"] > 8.0] *= BASE_SIG_GT8_MULT
    mult.loc[(df["buy_total_mv"] > 300000) & (df["buy_total_mv"] <= 500000)] *= BASE_MID_MV_MULT

    gap = df["buy_open_gap_raw_pct"]
    mult.loc[gap <= -2.7] *= gap_le_2p7
    mult.loc[(gap > -2.7) & (gap <= -1.5)] *= gap_mid_2p7_1p5
    mult.loc[gap > -1.0] *= gap_gt_1p0
    mult.loc[gap > -0.6] *= gap_gt_0p6

    df["target_pct"] = (df["target_pct"] * mult * BASE_GLOBAL_SCALE * global_scale).clip(lower=0.05, upper=cap)
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(df["target_pct"].mean()),
        "max_target": float(df["target_pct"].max()),
        "gap_le_2p7_rows": int((gap <= -2.7).sum()),
        "gap_gt_1p0_rows": int((gap > -1.0).sum()),
    }


def run_case(case: tuple[str, float, float, float, float, float, float]) -> dict[str, object]:
    name, gap_le_2p7, gap_mid_2p7_1p5, gap_gt_1p0, gap_gt_0p6, global_scale, cap = case
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
            str(cap),
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
        "gap_le_2p7_mult": gap_le_2p7,
        "gap_mid_2p7_1p5_mult": gap_mid_2p7_1p5,
        "gap_gt_1p0_mult": gap_gt_1p0,
        "gap_gt_0p6_mult": gap_gt_0p6,
        "global_scale": global_scale,
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
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
