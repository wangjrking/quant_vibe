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
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "repro500_composite_supplement"
SIGNAL_DIR = OUT_DIR / "signals"
LOG_DIR = OUT_DIR / "juejin_runs"
SUMMARY_FILE = OUT_DIR / "composite_supplement_summary.csv"

PRIMARY = (
    REPORT_DIR
    / "repro500_weak_sample_attenuation/signals/base0p435_weak0p385_strong0p435_late_or_crowded/signals.csv"
)
SECONDARY_SCALE = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
SECONDARY_LOWGAP = REPORT_DIR / "variants_lowgap_adaptive_quality_fine5/j04_s38_p04_n03_z03_a18.csv"

STRATEGY_DIR = (
    ROOT
    / "quant/data_file/reports/strategy_agent_latest_l4_weight_candidates_20260702/postrank_open_filter_candidates/code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant/main/run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = []
for source_name in ["scale124", "lowgap"]:
    for mode in ["empty_days", "low_open_empty_days", "low_open_all_days"]:
        for factor in [0.08, 0.12, 0.16, 0.20]:
            for cap in [0.70, 0.85, 0.95]:
                CASES.append(
                    {
                        "name": f"{source_name}_{mode}_f{factor:.2f}_cap{cap:.2f}".replace(".", "p"),
                        "source_name": source_name,
                        "mode": mode,
                        "factor": factor,
                        "cap": cap,
                    }
                )


def load_sources(case: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = pd.read_csv(PRIMARY, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    secondary_path = SECONDARY_SCALE if case["source_name"] == "scale124" else SECONDARY_LOWGAP
    secondary = pd.read_csv(secondary_path, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    return primary, secondary


def build_signal(case: dict) -> tuple[Path, int, int, float]:
    primary, secondary = load_sources(case)
    primary = primary.copy()
    secondary = secondary.copy()
    primary["target_pct"] = primary["target_pct"].astype(float)
    secondary["target_pct"] = secondary["target_pct"].astype(float) * float(case["factor"])

    primary_days = set(primary["signal_date"])
    if case["mode"] in {"empty_days", "low_open_empty_days"}:
        secondary = secondary[~secondary["signal_date"].isin(primary_days)].copy()
    if case["mode"] in {"low_open_empty_days", "low_open_all_days"}:
        if "buy_open_gap_raw_pct" not in secondary.columns:
            secondary = secondary.iloc[0:0].copy()
        else:
            secondary = secondary[secondary["buy_open_gap_raw_pct"].astype(float) <= -1.0].copy()

    primary["_priority"] = 0
    secondary["_priority"] = 1
    common_cols = [c for c in primary.columns if c in secondary.columns]
    combined = pd.concat([primary[common_cols], secondary[common_cols]], ignore_index=True)
    combined = combined.drop_duplicates(["signal_date", "stock_code"], keep="first")
    combined["target_pct"] = combined["target_pct"].astype(float)
    combined = combined.sort_values(["signal_date", "_priority", "rank", "stock_code"]).copy()

    rows = []
    for _, day in combined.groupby("signal_date", sort=True):
        total = 0.0
        for _, row in day.iterrows():
            target = float(row["target_pct"])
            if total + target > float(case["cap"]):
                remaining = float(case["cap"]) - total
                if remaining < 0.05:
                    continue
                target = remaining
            row = row.copy()
            row["target_pct"] = target
            rows.append(row)
            total += target
            if total >= float(case["cap"]) - 1e-9:
                break
    out = pd.DataFrame(rows)
    out = out.drop(columns=[c for c in ["_priority"] if c in out.columns])
    out["target_pct"] = out["target_pct"].map(lambda x: f"{float(x):.5f}")
    out["holding_days"] = "1"
    out["max_holding_days"] = "1"
    out["score_exit_entry_ratio"] = "0.96000"
    out["signal_stop_loss_pct"] = "0.05000"
    out["signal_take_profit_pct"] = "0.08000"
    out["strategy_variant"] = case["name"]
    out["filter_name"] = case["name"]
    case_dir = SIGNAL_DIR / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    signal_file = case_dir / "signals.csv"
    out.to_csv(signal_file, index=False, encoding="utf-8")
    return signal_file, int(len(out)), int(out["signal_date"].nunique()), float(out["target_pct"].astype(float).mean())


def parse_indicator(stdout: str) -> dict:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        try:
            return ast.literal_eval(re.sub(r"datetime\.datetime\(.*?\)\)", "'datetime'", indicator))
        except Exception:
            return {}
    return indicator or {}


def run_case(signal_file: Path, case: dict) -> tuple[int, dict, Path]:
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
        "3",
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
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (LOG_DIR / f"{case['name']}.runner.json").write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, parse_indicator(proc.stdout), log_file


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    # Run a focused subset: broad enough to test complementarity, small enough to finish.
    focus = [
        c
        for c in CASES
        if c["factor"] in {0.08, 0.12, 0.16}
        and c["cap"] in {0.70, 0.85}
        and c["mode"] in {"empty_days", "low_open_empty_days", "low_open_all_days"}
    ]
    for case in focus:
        signal_file, signal_rows, signal_days, avg_target = build_signal(case)
        print("RUN", case["name"], flush=True)
        returncode, indicator, log_file = run_case(signal_file, case)
        rows.append(
            {
                **case,
                "signal_rows": signal_rows,
                "signal_days": signal_days,
                "avg_target": avg_target,
                "returncode": returncode,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                "sharp_ratio": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
            }
        )
    rows = sorted(rows, key=lambda x: (x.get("sharp_ratio") or -999, x.get("pnl_ratio_annual") or -999), reverse=True)
    with SUMMARY_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[:20], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
