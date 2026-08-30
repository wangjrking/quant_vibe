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
BASE_SIGNAL = OUT_DIR / "scale124_signal_mv_loss_bucket_refine_signals/sig025_mid025_missing050_s145.csv"
SIGNAL_DIR = OUT_DIR / "sig025_open_gap_market_state_signals"
RUN_DIR = OUT_DIR / "sig025_open_gap_market_state_grid"
SUMMARY = OUT_DIR / "sig025_open_gap_market_state_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, scale, up_min, down5_max, median_min, weak_mult, strong_up, strong_mult
    ("drop_flat_s090_base", 0.90, 0.0, 1.0, -99.0, 1.0, 1.0, 1.0),
    ("drop_flat_s095_base", 0.95, 0.0, 1.0, -99.0, 1.0, 1.0, 1.0),
    ("weak_up40_m70_s100", 1.00, 0.40, 1.0, -99.0, 0.70, 0.60, 1.05),
    ("weak_up40_m60_s105", 1.05, 0.40, 1.0, -99.0, 0.60, 0.60, 1.08),
    ("weak_up40_m60_s106", 1.06, 0.40, 1.0, -99.0, 0.60, 0.60, 1.08),
    ("weak_up40_m60_s107", 1.07, 0.40, 1.0, -99.0, 0.60, 0.60, 1.08),
    ("weak_up40_m62_s104", 1.04, 0.40, 1.0, -99.0, 0.62, 0.60, 1.08),
    ("weak_up40_m62_s105", 1.05, 0.40, 1.0, -99.0, 0.62, 0.60, 1.08),
    ("weak_up40_m62_s106", 1.06, 0.40, 1.0, -99.0, 0.62, 0.60, 1.08),
    ("weak_up40_m65_s102", 1.02, 0.40, 1.0, -99.0, 0.65, 0.60, 1.07),
    ("weak_up40_m65_s103", 1.03, 0.40, 1.0, -99.0, 0.65, 0.60, 1.07),
    ("weak_up40_m65_s104", 1.04, 0.40, 1.0, -99.0, 0.65, 0.60, 1.07),
    ("weak_up40_m68_s101", 1.01, 0.40, 1.0, -99.0, 0.68, 0.60, 1.06),
    ("weak_up40_m68_s102", 1.02, 0.40, 1.0, -99.0, 0.68, 0.60, 1.06),
    ("weak_up45_m70_s105", 1.05, 0.45, 1.0, -99.0, 0.70, 0.60, 1.08),
    ("weak_up45_m60_s110", 1.10, 0.45, 1.0, -99.0, 0.60, 0.60, 1.10),
    ("weak_median0_m70_s105", 1.05, 0.0, 1.0, 0.0, 0.70, 0.60, 1.08),
    ("weak_median_m05_m70_s105", 1.05, 0.0, 1.0, -0.5, 0.70, 0.60, 1.08),
    ("weak_down5_12_m70_s105", 1.05, 0.0, 0.12, -99.0, 0.70, 0.60, 1.08),
    ("weak_combo_m70_s110", 1.10, 0.40, 0.12, -0.5, 0.70, 0.60, 1.10),
    ("weak_combo_m60_s115", 1.15, 0.40, 0.12, -0.5, 0.60, 0.60, 1.12),
    ("weak_combo_m50_s120", 1.20, 0.40, 0.12, -0.5, 0.50, 0.60, 1.15),
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


def load_breadth() -> pd.DataFrame:
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                CAST(trade_date AS VARCHAR) AS signal_date,
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
        return df
    finally:
        con.close()


def build_signal(case: tuple[str, float, float, float, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, scale, up_min, down5_max, median_min, weak_mult, strong_up, strong_mult = case
    df = pd.read_csv(BASE_SIGNAL)
    df["signal_date"] = df["signal_date"].astype(str)
    for col in ["buy_open_gap_raw_pct", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    original_rows = len(df)
    df = df.loc[df["buy_open_gap_raw_pct"] <= -0.5].copy()
    df = df.merge(load_breadth(), on="signal_date", how="left")
    weak = (df["up_ratio"] < up_min) | (df["down5_ratio"] > down5_max) | (df["median_pct"] < median_min)
    strong = (df["up_ratio"] >= strong_up) & (df["down5_ratio"] <= down5_max) & (df["median_pct"] >= median_min)
    mult = pd.Series(1.0, index=df.index)
    mult.loc[weak] *= weak_mult
    mult.loc[strong] *= strong_mult
    df["target_pct"] = (df["target_pct"] * mult * scale).clip(lower=0.03, upper=0.82)
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "dropped_rows": int(original_rows - len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "avg_target": float(df["target_pct"].mean()),
        "weak_rows": int(weak.sum()),
        "strong_rows": int(strong.sum()),
        "missing_breadth_rows": int(df["up_ratio"].isna().sum()),
    }


def run_case(case: tuple[str, float, float, float, float, float, float, float]) -> dict[str, object]:
    name, scale, up_min, down5_max, median_min, weak_mult, strong_up, strong_mult = case
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
        "scale": scale,
        "up_min": up_min,
        "down5_max": down5_max,
        "median_min": median_min,
        "weak_mult": weak_mult,
        "strong_up": strong_up,
        "strong_mult": strong_mult,
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
