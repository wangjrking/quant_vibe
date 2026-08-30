from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
SOURCE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_next_open_frequency_20260719"
SIGNAL_DIR = REPORT_DIR / "signals"
SCORE_DIR = REPORT_DIR / "score_assets"
LOG_DIR = REPORT_DIR / "logs"
POOL_DB = SOURCE_DIR / "current_formal_l4_rank_pool.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_high_return_frs_scale090_cap090_v20260716" / "code_snapshot"

BLENDS = {
    "b1_100": {"rank_1d": 1.0, "rank_3d": 0.0, "rank_5d": 0.0, "rank_10d": 0.0},
    "b1_70_10_30": {"rank_1d": 0.7, "rank_3d": 0.0, "rank_5d": 0.0, "rank_10d": 0.3},
    "b1_50_5_10_10_40": {"rank_1d": 0.5, "rank_3d": 0.0, "rank_5d": 0.1, "rank_10d": 0.4},
    "b1_30_3_10_5_20_10_40": {"rank_1d": 0.3, "rank_3d": 0.1, "rank_5d": 0.2, "rank_10d": 0.4},
    "b1_20_5_20_10_60": {"rank_1d": 0.2, "rank_3d": 0.0, "rank_5d": 0.2, "rank_10d": 0.6},
}

REGIMES = {
    "broad": "signal_pct_chg_raw <= 2.0",
    "no_chase": "signal_pct_chg_raw <= 0.5",
    "trend": "ma_qfq_5 >= ma_qfq_20 and close_qfq >= ma_qfq_20 and signal_pct_chg_raw <= 2.0",
    "trend_no_chase": "ma_qfq_5 >= ma_qfq_20 and close_qfq >= ma_qfq_20 and signal_pct_chg_raw <= 0.5",
}


def period_stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, mdd


def symbol(stock_code: str) -> str:
    return ("SHSE." if stock_code.endswith(".SH") else "SZSE.") + stock_code[:6]


def parse_indicator(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result = {}
            for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"]:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            for key in ["open_count", "close_count"]:
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    result[key] = int(match.group(1))
            return result
    return {}


def load_base() -> pd.DataFrame:
    con = duckdb.connect(str(POOL_DB), read_only=True)
    try:
        con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
        return con.execute(
            """
            SELECT p.*, s.close_qfq, s.ma_qfq_5, s.ma_qfq_20
            FROM current_formal_l4_rank_pool p
            JOIN marketdb.STOCK_DAILY_DATA s
              ON s.trade_date = p.trade_date AND s.stock_code = p.stock_code
            WHERE p.amount >= 90000
              AND p.total_mv >= 200000
            """
        ).fetchdf()
    finally:
        con.close()


def select_case(base: pd.DataFrame, blend: str, regime: str, topn: int, model_min: float) -> pd.DataFrame:
    work = base.copy()
    work["model_score"] = sum(float(weight) * work[column] for column, weight in BLENDS[blend].items())
    work = work[(work["model_score"] >= model_min)].copy()
    work = work.query(REGIMES[regime]).copy()
    work["select_score"] = work["model_score"] + 0.005 * (work["amount_rank"] + work["mv_rank"]) / 2.0
    work = work.sort_values(["trade_date", "select_score", "stock_code"], ascending=[True, False, True])
    work["pick_rank"] = work.groupby("trade_date").cumcount() + 1
    return work[work["pick_rank"] <= topn].copy()


def local_metrics(selected: pd.DataFrame, topn: int, all_dates: pd.Index) -> dict:
    mature = selected.dropna(subset=["next_open_return_raw"]).copy()
    mature["weighted_return"] = 0.90 / topn * (mature["next_open_return_raw"] - 0.006)
    daily = mature.groupby("buy_date")["weighted_return"].sum().reindex(all_dates, fill_value=0.0).sort_index()
    early = daily[daily.index.astype(str) < "20250101"]
    late = daily[daily.index.astype(str) >= "20250101"]
    full_stats = period_stats(daily)
    early_stats = period_stats(early)
    late_stats = period_stats(late)
    dated = daily.copy()
    dated.index = pd.to_datetime(dated.index)
    yearly = [period_stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 40]
    return {
        "proxy_annual": full_stats[0], "proxy_sharpe": full_stats[1], "proxy_mdd": full_stats[2],
        "early_annual": early_stats[0], "early_sharpe": early_stats[1],
        "late_annual": late_stats[0], "late_sharpe": late_stats[1],
        "worst_year_annual": min((item[0] for item in yearly), default=0.0),
        "worst_year_sharpe": min((item[1] for item in yearly), default=0.0),
        "signal_days": int(mature["trade_date"].nunique()), "rows": int(len(mature)),
    }


def write_score_asset(base: pd.DataFrame, blend: str) -> Path:
    path = SCORE_DIR / f"{blend}.duckdb"
    path.unlink(missing_ok=True)
    score = pd.DataFrame({"trade_date": base["trade_date"], "stock_code": base["stock_code"]})
    score["pred_prob"] = sum(float(weight) * base[column] for column, weight in BLENDS[blend].items())
    con = duckdb.connect(str(path))
    try:
        con.register("score_frame", score)
        con.execute("CREATE TABLE blended_rank_score AS SELECT * FROM score_frame")
    finally:
        con.close()
    return path


def write_signal(selected: pd.DataFrame, name: str, topn: int) -> Path:
    work = selected.copy()
    work["signal_date"] = work["trade_date"].astype(str)
    work["symbol"] = work["stock_code"].map(symbol)
    work["rank"] = work["pick_rank"].astype(int)
    work["pred_prob"] = work["model_score"]
    work["entry_score"] = work["model_score"]
    work["target_pct"] = 0.90 / topn
    work["holding_days"] = 1
    work["max_holding_days"] = 1
    work["score_exit_entry_ratio"] = 9.99
    work["min_holding_days_before_score_exit"] = 99
    work["score_continue_entry_ratio"] = 9.99
    work["strategy_variant"] = name
    work["filter_name"] = "current_formal_l4_next_open_frequency"
    work["entry_weight_name"] = "equal_weight_90pct"
    work["dynamic_hold_name"] = "fixed_h1"
    work["buy_day_market_available"] = True
    work["buy_day_hard_gate_complete"] = True
    work["buy_day_st_rejected"] = False
    work["buy_day_open_limit_up_rejected"] = False
    work["latest_market_date"] = str(work["buy_date"].max())
    work["buy_open_gap_pct"] = work["buy_open_gap_raw_pct"]
    columns = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "rank_1d", "rank_3d", "rank_5d", "rank_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "signal_pct_chg_raw", "target_pct",
        "holding_days", "max_holding_days", "score_exit_entry_ratio", "min_holding_days_before_score_exit",
        "score_continue_entry_ratio", "strategy_variant", "filter_name", "entry_weight_name", "dynamic_hold_name",
        "buy_day_market_available", "buy_day_hard_gate_complete", "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected", "latest_market_date", "buy_open_gap_pct", "buy_open_gap_raw_pct",
    ]
    path = SIGNAL_DIR / f"{name}.csv"
    work[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_juejin(row: pd.Series, signal_file: Path, score_db: Path) -> dict:
    log_file = LOG_DIR / f"{row['name']}.log"
    env = os.environ.copy()
    env.update({"GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0", "GM_OPEN_DAILY_SCORE_EXIT": "0"})
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY_DIR), "--signal-file", str(signal_file),
        "--log-file", str(log_file), "--max-positions", str(int(row["topn"])), "--holding-days", "1",
        "--max-holding-days", "1", "--score-db", str(score_db), "--score-table", "blended_rank_score",
        "--market-db", str(MARKET_DB), "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log_file)
    return {
        **row.to_dict(), "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal_file), "score_db": str(score_db), "log_file": str(log_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(exist_ok=True)
    SCORE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    base = load_base()
    all_dates = pd.Index(sorted(base["buy_date"].dropna().astype(str).unique()))
    selections = {}
    rows = []
    for blend in BLENDS:
        for regime in REGIMES:
            for topn in [1, 3, 5]:
                for model_min in [0.85, 0.90, 0.95, 0.98]:
                    name = f"{blend}_{regime}_m{int(model_min * 100)}_top{topn}_h1"
                    selected = select_case(base, blend, regime, topn, model_min)
                    metrics = local_metrics(selected, topn, all_dates)
                    row = {"name": name, "blend": blend, "regime": regime, "topn": topn, "model_min": model_min, **metrics}
                    row["robust_score"] = min(row["early_sharpe"], row["late_sharpe"]) + 0.25 * row["proxy_sharpe"]
                    rows.append(row)
                    selections[name] = selected
    local = pd.DataFrame(rows).sort_values(["robust_score", "proxy_sharpe"], ascending=False)
    local.to_csv(REPORT_DIR / "local_screen.csv", index=False, encoding="utf-8-sig")
    eligible = local[
        (local["signal_days"] >= 350)
        & (local["early_annual"] > 0)
        & (local["late_annual"] > 0)
        & (local["worst_year_annual"] > -0.10)
    ].head(8)
    score_assets = {blend: write_score_asset(base, blend) for blend in eligible["blend"].unique()}
    results = []
    for _, row in eligible.iterrows():
        signal_file = write_signal(selections[row["name"]], row["name"], int(row["topn"]))
        result = run_juejin(row, signal_file, score_assets[row["blend"]])
        results.append(result)
        pd.DataFrame(results).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last").to_csv(
            REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    (REPORT_DIR / "result.json").write_text(
        json.dumps({"status": "research_only", "screened_cases": len(local), "juejin_results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
