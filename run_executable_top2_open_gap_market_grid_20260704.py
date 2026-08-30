from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
MAIN = ROOT / "quant/main"
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
SIGNAL_DIR = OUT_DIR / "executable_top2_open_gap_market_signals"
RUN_DIR = OUT_DIR / "executable_top2_open_gap_market_grid"
SUMMARY = OUT_DIR / "executable_top2_open_gap_market_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    # name, topn, row_scale, cap, weak_mult, strong_mult
    ("top2_s090_cap50_w62", 2, 0.90, 0.50, 0.62, 1.08),
    ("top2_s100_cap55_w62", 2, 1.00, 0.55, 0.62, 1.08),
    ("top2_s110_cap60_w62", 2, 1.10, 0.60, 0.62, 1.08),
    ("top2_s120_cap65_w62", 2, 1.20, 0.65, 0.62, 1.08),
    ("top2_s130_cap70_w62", 2, 1.30, 0.70, 0.62, 1.08),
    ("top2_s140_cap75_w62", 2, 1.40, 0.75, 0.62, 1.08),
    ("top2_s150_cap82_w62", 2, 1.50, 0.82, 0.62, 1.08),
    ("top2_s120_cap65_w70", 2, 1.20, 0.65, 0.70, 1.05),
    ("top2_s135_cap75_w70", 2, 1.35, 0.75, 0.70, 1.05),
    ("top2_s150_cap82_w70", 2, 1.50, 0.82, 0.70, 1.05),
    ("top3_s080_cap38_w62", 3, 0.80, 0.38, 0.62, 1.08),
    ("top3_s100_cap45_w62", 3, 1.00, 0.45, 0.62, 1.08),
    ("top3_s120_cap55_w62", 3, 1.20, 0.55, 0.62, 1.08),
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
        return con.execute(
            """
            SELECT
                CAST(trade_date AS VARCHAR) AS signal_date,
                avg(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio
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


def base_target(row: pd.Series) -> float:
    gap = float(row["buy_open_gap_raw_pct"])
    sigchg = float(row["signal_pct_chg"])
    atr = float(row["signal_atr_qfq"])
    if gap <= -2.5:
        target = 0.57
    elif gap <= -1.5:
        target = 0.32
    elif gap <= -0.5:
        target = 0.62
    else:
        target = 0.25
    if 0.0 < sigchg <= 2.0:
        target += 0.20
    elif sigchg > 2.0:
        target += 0.02
    elif sigchg <= -2.0:
        target -= 0.03
    if pd.notna(atr) and 2.0 <= atr < 4.0:
        target -= 0.14
    return float(max(0.03, min(0.70, target)))


def export_signal(case: tuple[str, int, float, float, float, float]) -> tuple[Path, dict[str, object]]:
    name, topn, row_scale, cap, weak_mult, strong_mult = case
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    try:
        df = con.execute(
            f"""
            WITH picked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY buy_date
                        ORDER BY buy_open_gap_raw_pct ASC, pred_prob DESC, stock_code
                    ) AS pick_rank
                FROM active_l4_wide
                WHERE score_pct_rank >= 0.95
                  AND signal_amount >= 50000
                  AND buy_amount >= 50000
                  AND ret_h2 IS NOT NULL
                  AND buy_open_gap_raw_pct >= -3.0
                  AND buy_open_gap_raw_pct <= -0.5
                  AND NOT (buy_open_gap_raw_pct > -2.5 AND buy_open_gap_raw_pct <= -1.5)
                  AND NOT (signal_atr_qfq >= 2.0 AND signal_atr_qfq < 4.0)
            )
            SELECT
                signal_date,
                buy_date,
                stock_code,
                name,
                market,
                pick_rank AS rank,
                pred_prob,
                score_pct_rank,
                signal_pct_chg,
                signal_open_gap_raw_pct,
                buy_open_gap_raw_pct,
                signal_amount,
                buy_amount,
                signal_turnover_rate,
                buy_turnover_rate,
                signal_total_mv,
                buy_total_mv,
                signal_atr_qfq,
                buy_atr_qfq
            FROM picked
            WHERE pick_rank <= {int(topn)}
            ORDER BY buy_date, pick_rank
            """
        ).fetchdf()
    finally:
        con.close()

    df["signal_date"] = df["signal_date"].astype(str)
    df = df.merge(load_breadth(), on="signal_date", how="left")
    df["symbol"] = np.where(
        df["stock_code"].str.endswith(".SH"),
        "SHSE." + df["stock_code"].str.split(".").str[0],
        "SZSE." + df["stock_code"].str.split(".").str[0],
    )
    df["target_pct"] = [base_target(row) for _, row in df.iterrows()]
    for col in ["signal_total_mv", "signal_pct_chg", "target_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mult = pd.Series(1.0, index=df.index)
    mult.loc[(df["signal_pct_chg"] > 5.0) & (df["signal_pct_chg"] <= 8.0)] *= 0.25
    mult.loc[df["signal_pct_chg"] > 8.0] *= 0.45
    mult.loc[(df["signal_total_mv"] > 300000) & (df["signal_total_mv"] <= 500000)] *= 0.25
    mult.loc[df["signal_total_mv"].isna()] *= 0.50
    mult.loc[df["up_ratio"] < 0.40] *= weak_mult
    mult.loc[df["up_ratio"] >= 0.60] *= strong_mult
    df["target_pct"] = (df["target_pct"] * mult * row_scale).clip(lower=0.03, upper=cap)
    df["holding_days"] = 2
    df["max_holding_days"] = 4
    df["score_exit_entry_ratio"] = 0.90
    df["min_holding_days_before_score_exit"] = 2
    df["score_continue_entry_ratio"] = 1.00
    df["signal_stop_loss_pct"] = 0.08
    df["buy_day_market_available"] = True
    df["buy_day_hard_gate_complete"] = True
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["latest_market_date"] = "20260702"
    signal_file = SIGNAL_DIR / f"{name}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    counts = df.groupby("buy_date").size()
    target_sum = df.groupby("buy_date")["target_pct"].sum()
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "days_below_topn": int((counts < topn).sum()),
        "avg_names_per_day": float(counts.mean()) if len(counts) else None,
        "avg_target": float(df["target_pct"].mean()),
        "avg_target_sum_per_day": float(target_sum.mean()) if len(target_sum) else None,
        "max_target_sum_per_day": float(target_sum.max()) if len(target_sum) else None,
        "weak_rows": int((df["up_ratio"] < 0.40).sum()),
    }


def run_case(case: tuple[str, int, float, float, float, float]) -> dict[str, object]:
    name, topn, row_scale, cap, weak_mult, strong_mult = case
    signal_file, meta = export_signal(case)
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
            str(topn),
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
        "topn": topn,
        "row_scale": row_scale,
        "cap": cap,
        "weak_mult": weak_mult,
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
