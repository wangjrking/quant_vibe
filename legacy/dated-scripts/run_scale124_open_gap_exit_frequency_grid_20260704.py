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
SIGNAL_DIR = OUT_DIR / "scale124_open_gap_exit_frequency_signals"
RUN_DIR = OUT_DIR / "scale124_open_gap_exit_frequency_grid"
SUMMARY = OUT_DIR / "scale124_open_gap_exit_frequency_grid_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, risk_hold1, gap_deep_hold1, gap_mild_hold1, sig_up_hold1, stop_loss, continue_ratio, global_scale, cap
    ("base_fine", False, False, False, False, 0.08, 1.00, 1.20, 0.82),
    ("all_hold1_no_continue", True, False, False, False, 0.08, 9.99, 1.20, 0.82),
    ("risk_hold1_no_continue", True, False, False, True, 0.08, 9.99, 1.20, 0.82),
    ("deep_gap_hold1_no_continue", False, True, False, False, 0.08, 9.99, 1.20, 0.82),
    ("mild_gap_hold1_no_continue", False, False, True, False, 0.08, 9.99, 1.20, 0.82),
    ("sig_up_hold1_no_continue", False, False, False, True, 0.08, 9.99, 1.20, 0.82),
    ("risk_hold1_stop06", True, False, False, True, 0.06, 9.99, 1.20, 0.82),
    ("risk_hold1_stop04", True, False, False, True, 0.04, 9.99, 1.20, 0.82),
    ("deep_mild_hold1_stop06", False, True, True, False, 0.06, 9.99, 1.20, 0.82),
    ("risk_hold1_scale130", True, False, False, True, 0.06, 9.99, 1.30, 0.82),
    ("base_stop06", False, False, False, False, 0.06, 1.00, 1.20, 0.82),
    ("base_stop04", False, False, False, False, 0.04, 1.00, 1.20, 0.82),
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


def apply_best_weight(df: pd.DataFrame, global_scale: float, cap: float) -> pd.DataFrame:
    for col in ["buy_total_mv", "signal_pct_chg", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mult = pd.Series(1.0, index=df.index)
    mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= 0.45
    mult.loc[df["signal_pct_chg"] > 8.0] *= 0.55
    mult.loc[(df["buy_total_mv"] > 300000) & (df["buy_total_mv"] <= 500000)] *= 0.45
    df["target_pct"] = (df["target_pct"] * mult * global_scale).clip(lower=0.05, upper=cap)
    return df


def build_signal(case: tuple[str, bool, bool, bool, bool, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, risk_hold1, gap_deep_hold1, gap_mild_hold1, sig_up_hold1, stop_loss, continue_ratio, global_scale, cap = case
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["buy_open_gap_raw_pct", "signal_pct_chg", "buy_total_mv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = apply_best_weight(df, global_scale, cap)
    df["holding_days"] = 2
    df["score_continue_entry_ratio"] = 1.0
    df["signal_stop_loss_pct"] = stop_loss

    risk_mask = pd.Series(False, index=df.index)
    if risk_hold1:
        risk_mask |= ((df["signal_pct_chg"] > 5.0) | ((df["buy_total_mv"] > 300000) & (df["buy_total_mv"] <= 500000)))
    if gap_deep_hold1:
        risk_mask |= df["buy_open_gap_raw_pct"] <= -2.7
    if gap_mild_hold1:
        risk_mask |= df["buy_open_gap_raw_pct"] > -1.0
    if sig_up_hold1:
        risk_mask |= df["signal_pct_chg"] > 5.0
    df.loc[risk_mask, "holding_days"] = 1
    df.loc[risk_mask, "score_continue_entry_ratio"] = continue_ratio

    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "hold1_rows": int(risk_mask.sum()),
        "avg_target": float(df["target_pct"].mean()),
        "max_target": float(df["target_pct"].max()),
    }


def run_case(case: tuple[str, bool, bool, bool, bool, float, float, float, float]) -> dict[str, object]:
    name, risk_hold1, gap_deep_hold1, gap_mild_hold1, sig_up_hold1, stop_loss, continue_ratio, global_scale, cap = case
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
            str(stop_loss),
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
        "risk_hold1": risk_hold1,
        "gap_deep_hold1": gap_deep_hold1,
        "gap_mild_hold1": gap_mild_hold1,
        "sig_up_hold1": sig_up_hold1,
        "stop_loss": stop_loss,
        "continue_ratio": continue_ratio,
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
