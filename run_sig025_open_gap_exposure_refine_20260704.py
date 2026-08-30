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
SIGNAL_DIR = OUT_DIR / "sig025_open_gap_exposure_refine_signals"
RUN_DIR = OUT_DIR / "sig025_open_gap_exposure_refine_grid"
SUMMARY = OUT_DIR / "sig025_open_gap_exposure_refine_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, mode, scale, cap, extra_sig58_mult
    ("base", "base", 1.00, 0.82, 1.00),
    ("drop_flat_s080_cap82", "drop_flat", 0.80, 0.82, 1.00),
    ("drop_flat_s085_cap82", "drop_flat", 0.85, 0.82, 1.00),
    ("drop_flat_s090_cap82", "drop_flat", 0.90, 0.82, 1.00),
    ("drop_flat_s095_cap82", "drop_flat", 0.95, 0.82, 1.00),
    ("drop_flat_s100_cap82", "drop_flat", 1.00, 0.82, 1.00),
    ("drop_flat_s105_cap82", "drop_flat", 1.05, 0.82, 1.00),
    ("drop_flat_s090_cap75", "drop_flat", 0.90, 0.75, 1.00),
    ("drop_flat_s100_cap75", "drop_flat", 1.00, 0.75, 1.00),
    ("drop_flat_s110_cap75", "drop_flat", 1.10, 0.75, 1.00),
    ("drop_flat_s100_cap70", "drop_flat", 1.00, 0.70, 1.00),
    ("soft_flat_s105_cap82", "soft_flat", 1.05, 0.82, 1.00),
    ("soft_flat_s115_cap82", "soft_flat", 1.15, 0.82, 1.00),
    ("soft_flat_s125_cap82", "soft_flat", 1.25, 0.82, 1.00),
    ("sig58_half_s105_cap82", "sig58_half", 1.05, 0.82, 0.50),
    ("sig58_half_s115_cap82", "sig58_half", 1.15, 0.82, 0.50),
    ("sig58_half_s125_cap82", "sig58_half", 1.25, 0.82, 0.50),
    ("soft_flat_sig58_half_s110_cap82", "soft_flat_sig58", 1.10, 0.82, 0.50),
    ("soft_flat_sig58_half_s120_cap82", "soft_flat_sig58", 1.20, 0.82, 0.50),
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


def build_signal(case: tuple[str, str, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, mode, scale, cap, extra_sig58_mult = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["buy_open_gap_raw_pct", "signal_pct_chg", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    original_rows = len(df)
    if mode == "drop_flat":
        df = df.loc[df["buy_open_gap_raw_pct"] <= -0.5].copy()
    mult = pd.Series(1.0, index=df.index)
    if mode in {"soft_flat", "soft_flat_sig58"}:
        mult.loc[df["buy_open_gap_raw_pct"] > -0.5] *= 0.50
    if mode in {"sig58_half", "soft_flat_sig58"}:
        mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= extra_sig58_mult
    df["target_pct"] = (df["target_pct"] * mult * scale).clip(lower=0.03, upper=cap)
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "dropped_rows": int(original_rows - len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(df["target_pct"].mean()),
        "downweighted_rows": int((mult < 1.0).sum()),
        "cap": cap,
    }


def run_case(case: tuple[str, str, float, float, float]) -> dict[str, object]:
    name, mode, scale, cap, extra_sig58_mult = case
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
        "mode": mode,
        "scale": scale,
        "extra_sig58_mult": extra_sig58_mult,
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
