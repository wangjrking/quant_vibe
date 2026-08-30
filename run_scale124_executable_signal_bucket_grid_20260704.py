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
SIGNAL_DIR = OUT_DIR / "scale124_executable_signal_bucket_signals"
RUN_DIR = OUT_DIR / "scale124_executable_signal_bucket_grid"
SUMMARY = OUT_DIR / "scale124_executable_signal_bucket_grid_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, sig5_8_mult, sig_gt8_mult, mv_source, mv_mid_mult, open_gap_mult, global_scale, cap
    ("base_prior_buy_mv", 0.45, 0.55, "buy_total_mv", 0.45, 1.00, 1.20, 0.82),
    ("no_mv_bucket_s120", 0.45, 0.55, "none", 1.00, 1.00, 1.20, 0.82),
    ("no_mv_bucket_s130", 0.45, 0.55, "none", 1.00, 1.00, 1.30, 0.82),
    ("signal_mv_mid045_s120", 0.45, 0.55, "signal_total_mv", 0.45, 1.00, 1.20, 0.82),
    ("signal_mv_mid055_s120", 0.45, 0.55, "signal_total_mv", 0.55, 1.00, 1.20, 0.82),
    ("signal_mv_mid065_s120", 0.45, 0.55, "signal_total_mv", 0.65, 1.00, 1.20, 0.82),
    ("signal_mv_mid045_s130", 0.45, 0.55, "signal_total_mv", 0.45, 1.00, 1.30, 0.82),
    ("signal_mv_mid055_s130", 0.45, 0.55, "signal_total_mv", 0.55, 1.00, 1.30, 0.82),
    ("signal_mv_mid045_opengap_s125", 0.45, 0.55, "signal_total_mv", 0.45, 0.80, 1.25, 0.82),
    ("signal_mv_mid055_opengap_s125", 0.45, 0.55, "signal_total_mv", 0.55, 0.80, 1.25, 0.82),
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


def build_signal(case: tuple[str, float, float, str, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, sig5_8_mult, sig_gt8_mult, mv_source, mv_mid_mult, open_gap_mult, global_scale, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in [
        "buy_total_mv",
        "signal_total_mv",
        "signal_pct_chg",
        "buy_open_gap_raw_pct",
        "target_pct",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mult = pd.Series(1.0, index=df.index)
    mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= sig5_8_mult
    mult.loc[df["signal_pct_chg"] > 8.0] *= sig_gt8_mult
    if mv_source != "none":
        mv = df[mv_source]
        mult.loc[(mv > 300000) & (mv <= 500000)] *= mv_mid_mult
    mult.loc[df["buy_open_gap_raw_pct"] > -1.0] *= open_gap_mult
    df["target_pct"] = (df["target_pct"] * mult * global_scale).clip(lower=0.05, upper=cap)
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(df["target_pct"].mean()),
        "max_target": float(df["target_pct"].max()),
        "downweighted_rows": int((mult != 1.0).sum()),
    }


def run_case(case: tuple[str, float, float, str, float, float, float, float]) -> dict[str, object]:
    name, sig5_8_mult, sig_gt8_mult, mv_source, mv_mid_mult, open_gap_mult, global_scale, cap = case
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
        "sig5_8_mult": sig5_8_mult,
        "sig_gt8_mult": sig_gt8_mult,
        "mv_source": mv_source,
        "mv_mid_mult": mv_mid_mult,
        "open_gap_mult": open_gap_mult,
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
