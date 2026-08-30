from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_SIGNAL = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_repro500_p435_s96_p10d70_v20260702"
    / "signals"
    / "full_history_repro500_p435_s96_p10d70.csv"
)
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant" / "main" / "run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

OUT_DIR = REPORT_DIR / "repro500_quality_scaled_candidates"
LOG_DIR = OUT_DIR / "juejin_runs"


CASES = [
    {
        "name": "qs_deep55_base43_weak31",
        "base": 0.430,
        "deep": 0.550,
        "strong": 0.500,
        "weak": 0.310,
        "very_weak": 0.220,
        "cap": 0.560,
        "weak_gap": 0.0,
        "weak_pct": -2.0,
    },
    {
        "name": "qs_deep60_base425_weak28",
        "base": 0.425,
        "deep": 0.600,
        "strong": 0.520,
        "weak": 0.280,
        "very_weak": 0.180,
        "cap": 0.620,
        "weak_gap": 0.0,
        "weak_pct": -2.0,
    },
    {
        "name": "qs_deep65_base42_weak25",
        "base": 0.420,
        "deep": 0.650,
        "strong": 0.540,
        "weak": 0.250,
        "very_weak": 0.150,
        "cap": 0.660,
        "weak_gap": 0.0,
        "weak_pct": -2.0,
    },
    {
        "name": "qs_liq_deep60_base42_weak24",
        "base": 0.420,
        "deep": 0.600,
        "strong": 0.520,
        "weak": 0.240,
        "very_weak": 0.120,
        "cap": 0.620,
        "weak_gap": 0.0,
        "weak_pct": -2.0,
        "require_liq_for_deep": True,
    },
    {
        "name": "qs_gap_deep58_base43_weak26",
        "base": 0.430,
        "deep": 0.580,
        "strong": 0.520,
        "weak": 0.260,
        "very_weak": 0.160,
        "cap": 0.600,
        "weak_gap": 0.25,
        "weak_pct": -2.0,
        "gap_bonus": True,
    },
    {
        "name": "qs_sharpe_bias_deep52_base41_weak18",
        "base": 0.410,
        "deep": 0.520,
        "strong": 0.470,
        "weak": 0.180,
        "very_weak": 0.080,
        "cap": 0.540,
        "weak_gap": -0.2,
        "weak_pct": -2.5,
    },
]


def _assign_target(row: pd.Series, case: dict) -> float:
    pct = float(row["pct_chg"])
    gap = float(row["buy_open_gap_pct"])
    amount = float(row["amount"])
    turnover = float(row["turnover_rate"])
    pred10 = float(row["pred_10d"])
    pred1 = float(row["pred_1d"])

    liquid = amount >= 200000 or turnover >= 3.0
    very_liquid = amount >= 500000 or turnover >= 8.0
    deep_signal = pct <= -5.0
    strong_signal = pct <= -3.0
    deep_open = gap <= -1.0
    flat_or_down_open = gap <= 0.0
    weak = pct > float(case["weak_pct"]) or gap > float(case["weak_gap"]) or pred1 < -0.002
    very_weak = pct > -1.75 or gap > 1.0 or pred10 < 0.9

    if deep_signal and deep_open and (very_liquid or not case.get("require_liq_for_deep")):
        return float(case["deep"])
    if (deep_signal or strong_signal) and flat_or_down_open and liquid:
        return float(case["strong"])
    if case.get("gap_bonus") and deep_open and liquid and pred10 >= 0.95:
        return min(float(case["strong"]), float(case["cap"]))
    if very_weak:
        return float(case["very_weak"])
    if weak:
        return float(case["weak"])
    return float(case["base"])


def _extract_indicator(stdout: str) -> dict:
    payload = json.loads(stdout)
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        try:
            return ast.literal_eval(indicator)
        except Exception:
            return {}
    return indicator or {}


def _local_stats(df: pd.DataFrame) -> dict:
    x = df.copy()
    x["weighted_ret"] = x["target_pct_float"] * x["ret_open_to_next_open"]
    daily = x.groupby("buy_date")["weighted_ret"].sum().sort_index()
    nav = (1.0 + daily).cumprod()
    peak = nav.cummax()
    std = daily.std(ddof=1)
    return {
        "signal_rows": int(len(x)),
        "buy_days": int(daily.size),
        "avg_target": float(x["target_pct_float"].mean()),
        "max_target": float(x["target_pct_float"].max()),
        "local_annual_proxy": float((1.0 + daily.mean()) ** 252 - 1.0) if len(daily) else 0.0,
        "local_sharpe_proxy": float(daily.mean() / std * (252**0.5)) if std else None,
        "local_mdd_proxy": float(-(nav / peak - 1.0).min()) if len(daily) else 0.0,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})

    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        enriched = con.execute(
            """
            WITH cal AS (
              SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
              FROM (SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA)
            )
            SELECT
              sig.*,
              sig_md.pct_chg AS signal_pct_chg_check,
              sell_md.open_qfq / NULLIF(buy_md.open_qfq, 0) - 1 AS ret_open_to_next_open
            FROM sig
            JOIN cal ON cal.trade_date = sig.buy_date
            JOIN STOCK_DAILY_DATA sig_md
              ON sig_md.trade_date = sig.signal_date AND sig_md.stock_code = sig.stock_code
            JOIN STOCK_DAILY_DATA buy_md
              ON buy_md.trade_date = sig.buy_date AND buy_md.stock_code = sig.stock_code
            JOIN STOCK_DAILY_DATA sell_md
              ON sell_md.trade_date = cal.next_trade_date AND sell_md.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    rows = []
    for case in CASES:
        x = enriched.copy()
        x["target_pct_float"] = x.apply(lambda row: _assign_target(row, case), axis=1).clip(0.01, float(case["cap"]))
        x["target_pct"] = x["target_pct_float"].map(lambda v: f"{v:.5f}")
        x["strategy_variant"] = case["name"]
        x["filter_name"] = case["name"]

        stats = _local_stats(x)
        case_dir = OUT_DIR / "signals" / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        signal_file = case_dir / "signals.csv"
        x.drop(columns=["target_pct_float", "signal_pct_chg_check", "ret_open_to_next_open"]).to_csv(
            signal_file, index=False, encoding="utf-8"
        )

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
        print("RUN", case["name"], flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (LOG_DIR / f"{case['name']}.runner.json").write_text(proc.stdout, encoding="utf-8")
        indicator = _extract_indicator(proc.stdout) if proc.returncode == 0 else {}
        rows.append(
            {
                "name": case["name"],
                "returncode": proc.returncode,
                **stats,
                "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                "sharp_ratio": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
                "signal_file": str(signal_file),
                "log_file": str(log_file),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=False)
    out_csv = OUT_DIR / "quality_scaled_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(out_csv)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
