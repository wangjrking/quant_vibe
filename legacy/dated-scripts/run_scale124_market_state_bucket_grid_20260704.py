from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
MAIN = ROOT / "quant/main"
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
SIGNAL_DIR = OUT_DIR / "scale124_market_state_bucket_signals"
RUN_DIR = OUT_DIR / "scale124_market_state_bucket_grid"
SUMMARY = OUT_DIR / "scale124_market_state_bucket_grid_summary.csv"
BASE_SIGNAL = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, mode, up_min, down5_max, median_min, weak_mult, strong_up, strong_mult, global_scale
    ("base_signal_mv", "keep", 0.0, 1.0, -99.0, 1.0, 1.0, 1.0, 1.20),
    ("skip_up_lt35", "skip", 0.35, 1.0, -99.0, 1.0, 1.0, 1.0, 1.25),
    ("skip_up_lt40", "skip", 0.40, 1.0, -99.0, 1.0, 1.0, 1.0, 1.30),
    ("skip_up_lt45", "skip", 0.45, 1.0, -99.0, 1.0, 1.0, 1.0, 1.35),
    ("skip_median_lt_m05", "skip", 0.0, 1.0, -0.5, 1.0, 1.0, 1.0, 1.30),
    ("skip_median_lt_0", "skip", 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.40),
    ("skip_down5_gt08", "skip", 0.0, 0.08, -99.0, 1.0, 1.0, 1.0, 1.25),
    ("skip_down5_gt12", "skip", 0.0, 0.12, -99.0, 1.0, 1.0, 1.0, 1.22),
    ("downweight_weak_up40", "downweight", 0.40, 1.0, -99.0, 0.45, 0.55, 1.10, 1.22),
    ("downweight_weak_up45", "downweight", 0.45, 1.0, -99.0, 0.50, 0.60, 1.08, 1.22),
    ("downweight_weak_combo", "downweight", 0.40, 0.12, -0.5, 0.40, 0.55, 1.12, 1.25),
    ("downweight_weak_up45_s145", "downweight", 0.45, 1.0, -99.0, 0.50, 0.60, 1.08, 1.45),
    ("downweight_weak_up40_s145", "downweight", 0.40, 1.0, -99.0, 0.45, 0.55, 1.10, 1.45),
    ("downweight_weak_combo_s150", "downweight", 0.40, 0.12, -0.5, 0.40, 0.55, 1.12, 1.50),
    ("skip_combo_strict", "skip", 0.40, 0.12, -0.5, 1.0, 1.0, 1.0, 1.35),
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


def load_breadth() -> pd.DataFrame:
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        return con.execute(
            """
            SELECT
                trade_date AS signal_date,
                avg(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio,
                avg(CASE WHEN pct_chg <= -5 THEN 1.0 ELSE 0.0 END) AS down5_ratio,
                median(pct_chg) AS median_pct,
                avg(pct_chg) AS avg_pct
            FROM STOCK_DAILY_DATA
            WHERE stock_code NOT LIKE '%.BJ'
              AND pct_chg IS NOT NULL
              AND amount IS NOT NULL
              AND coalesce(ST_TYPE, '') IN ('', '0')
              AND coalesce(ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(name, '') NOT LIKE 'ST%'
              AND coalesce(name, '') NOT LIKE '*ST%'
            GROUP BY trade_date
            """
        ).fetchdf()
    finally:
        con.close()


def build_base_signal() -> pd.DataFrame:
    df = pd.read_csv(BASE_SIGNAL)
    df["signal_date"] = df["signal_date"].astype(str)
    df["buy_date"] = df["buy_date"].astype(str)
    for col in ["signal_total_mv", "signal_pct_chg", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mult = pd.Series(1.0, index=df.index)
    mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= 0.45
    mult.loc[df["signal_pct_chg"] > 8.0] *= 0.55
    mult.loc[(df["signal_total_mv"] > 300000) & (df["signal_total_mv"] <= 500000)] *= 0.45
    df["target_pct"] = (df["target_pct"] * mult).clip(lower=0.05, upper=0.82)
    return df


def build_signal(case: tuple[str, str, float, float, float, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, mode, up_min, down5_max, median_min, weak_mult, strong_up, strong_mult, global_scale = case
    df = build_base_signal().merge(load_breadth(), on="signal_date", how="left")
    weak = (df["up_ratio"] < up_min) | (df["down5_ratio"] > down5_max) | (df["median_pct"] < median_min)
    strong = (df["up_ratio"] >= strong_up) & (df["down5_ratio"] <= down5_max) & (df["median_pct"] >= median_min)
    before_rows = len(df)
    if mode == "skip":
        df = df.loc[~weak].copy()
    elif mode == "downweight":
        state_mult = pd.Series(1.0, index=df.index)
        state_mult.loc[weak] *= weak_mult
        state_mult.loc[strong] *= strong_mult
        df["target_pct"] = (df["target_pct"] * state_mult).clip(lower=0.05, upper=0.82)
    df["target_pct"] = (df["target_pct"] * global_scale).clip(lower=0.05, upper=0.82)
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "removed_rows": int(before_rows - len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "avg_target": float(df["target_pct"].mean()) if len(df) else None,
        "weak_days": int(df.loc[weak.reindex(df.index, fill_value=False), "signal_date"].nunique()) if len(df) else 0,
    }


def run_case(case: tuple[str, str, float, float, float, float, float, float, float]) -> dict[str, object]:
    name, mode, up_min, down5_max, median_min, weak_mult, strong_up, strong_mult, global_scale = case
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
        "up_min": up_min,
        "down5_max": down5_max,
        "median_min": median_min,
        "weak_mult": weak_mult,
        "strong_up": strong_up,
        "strong_mult": strong_mult,
        "global_scale": global_scale,
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
