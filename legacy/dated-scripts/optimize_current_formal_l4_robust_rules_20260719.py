from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
SOURCE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719"
SIGNAL_DIR = REPORT_DIR / "signals"
LOG_DIR = REPORT_DIR / "logs"
POOL_DB = SOURCE_DIR / "current_formal_l4_rank_pool.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_high_return_frs_scale090_cap090_v20260716" / "code_snapshot"

BLENDS = {
    "b10": {"rank_1d": 0.0, "rank_3d": 0.0, "rank_5d": 0.0, "rank_10d": 1.0},
    "b510": {"rank_1d": 0.0, "rank_3d": 0.0, "rank_5d": 0.2, "rank_10d": 0.8},
    "ball": {"rank_1d": 0.05, "rank_3d": 0.10, "rank_5d": 0.15, "rank_10d": 0.70},
}

REGIMES = {
    "broad": "signal_pct_chg_raw <= 1.0",
    "pullback": "signal_pct_chg_raw BETWEEN -5.0 AND -0.5",
    "uptrend": "ma_qfq_5 > ma_qfq_20 AND close_qfq > ma_qfq_20 AND signal_pct_chg_raw <= 1.0",
    "trend_pullback": "ma_qfq_5 > ma_qfq_20 AND close_qfq > ma_qfq_20 AND signal_pct_chg_raw BETWEEN -5.0 AND -0.5",
    "stable_trend": "ma_qfq_5 > ma_qfq_20 AND close_qfq > ma_qfq_20 AND atr_qfq / NULLIF(close_qfq, 0) <= 0.06 AND signal_pct_chg_raw BETWEEN -4.0 AND 1.0",
    "momentum_pullback": "macd_qfq > 0 AND close_qfq > ma_qfq_20 AND signal_pct_chg_raw BETWEEN -5.0 AND -0.5",
    "deep_short_confirm": "pred_1d >= -0.002 AND signal_pct_chg_raw BETWEEN -5.0 AND -1.5",
    "support_ma20": "ma_qfq_20 > ma_qfq_60 AND low_qfq <= ma_qfq_20 * 1.01 AND close_qfq >= ma_qfq_20 AND signal_pct_chg_raw BETWEEN -5.0 AND 0.5",
    "support_ma10": "ma_qfq_10 > ma_qfq_20 AND low_qfq <= ma_qfq_10 * 1.01 AND close_qfq >= ma_qfq_10 AND signal_pct_chg_raw BETWEEN -5.0 AND 0.5",
    "trend_rsi": "ma_qfq_20 > ma_qfq_60 AND close_qfq >= ma_qfq_20 AND rsi_qfq_12 BETWEEN 40.0 AND 70.0 AND signal_pct_chg_raw BETWEEN -4.0 AND 1.0",
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


def load_base(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
    return con.execute(
        """
        WITH future_open AS (
            SELECT
                trade_date,
                stock_code,
                lead(open, 5) OVER (PARTITION BY stock_code ORDER BY trade_date) AS open_h5
            FROM marketdb.STOCK_DAILY_DATA
        )
        SELECT
            p.*,
            s.close_qfq,
            s.low_qfq,
            s.ma_qfq_10,
            s.ma_qfq_5,
            s.ma_qfq_20,
            s.ma_qfq_60,
            s.macd_qfq,
            s.rsi_qfq_12,
            f.open_h5,
            f.open_h5 / NULLIF(p.buy_open_raw, 0) - 1.0 AS forward_open_h5_return
        FROM current_formal_l4_rank_pool p
        JOIN marketdb.STOCK_DAILY_DATA s
          ON s.trade_date = p.trade_date AND s.stock_code = p.stock_code
        LEFT JOIN future_open f
          ON f.trade_date = p.buy_date AND f.stock_code = p.stock_code
        WHERE p.amount >= 90000
          AND p.total_mv >= 200000
        """
    ).fetchdf()


def select_case(base: pd.DataFrame, blend: str, regime: str, topn: int, model_min: float) -> pd.DataFrame:
    work = base.copy()
    weights = BLENDS[blend]
    work["model_score"] = sum(float(w) * work[col] for col, w in weights.items())
    work = work.query("model_score >= @model_min")
    expression = REGIMES[regime].replace("NULLIF(close_qfq, 0)", "close_qfq")
    expression = expression.replace(" BETWEEN -5.0 AND -0.5", " >= -5.0 and signal_pct_chg_raw <= -0.5")
    expression = expression.replace(" BETWEEN -5.0 AND -1.5", " >= -5.0 and signal_pct_chg_raw <= -1.5")
    expression = expression.replace(" BETWEEN -4.0 AND 1.0", " >= -4.0 and signal_pct_chg_raw <= 1.0")
    expression = expression.replace(" BETWEEN -5.0 AND 0.5", " >= -5.0 and signal_pct_chg_raw <= 0.5")
    expression = expression.replace("rsi_qfq_12 BETWEEN 40.0 AND 70.0", "rsi_qfq_12 >= 40.0 and rsi_qfq_12 <= 70.0")
    expression = expression.replace(" AND ", " and ").replace(" OR ", " or ")
    work = work.query(expression)
    work["select_score"] = work["model_score"] + 0.01 * (work["amount_rank"] + work["mv_rank"]) / 2.0
    work = work.sort_values(["trade_date", "select_score", "stock_code"], ascending=[True, False, True])
    work["pick_rank"] = work.groupby("trade_date").cumcount() + 1
    return work[work["pick_rank"] <= topn].copy()


def local_metrics(selected: pd.DataFrame, topn: int) -> dict:
    mature = selected.dropna(subset=["forward_open_h5_return"]).copy()
    mature["target_pct"] = 0.90 / topn
    mature["weighted_return"] = mature["target_pct"] * (mature["forward_open_h5_return"] - 0.006)
    date_index = pd.Index(sorted(mature["buy_date"].unique()))

    def calc(part: pd.DataFrame) -> tuple[float, float, float]:
        daily = part.groupby("buy_date")["weighted_return"].sum().reindex(date_index, fill_value=0.0)
        return period_stats(daily)

    early = mature[mature["buy_date"].astype(str) < "20250101"]
    late = mature[mature["buy_date"].astype(str) >= "20250101"]
    full = calc(mature)
    e = calc(early)
    l = calc(late)
    return {
        "proxy_annual": full[0], "proxy_sharpe": full[1], "proxy_mdd": full[2],
        "early_annual": e[0], "early_sharpe": e[1], "late_annual": l[0], "late_sharpe": l[1],
        "signal_days": int(mature["trade_date"].nunique()), "rows": int(len(mature)),
    }


def write_signal(selected: pd.DataFrame, name: str, topn: int, blend: str) -> Path:
    work = selected.copy()
    work["signal_date"] = work["trade_date"].astype(str)
    work["symbol"] = work["stock_code"].map(symbol)
    work["rank"] = work["pick_rank"].astype(int)
    work["pred_prob"] = work["model_score"]
    work["entry_score"] = work["model_score"]
    work["target_pct"] = 0.90 / topn
    work["holding_days"] = 5
    work["max_holding_days"] = 5
    work["score_exit_entry_ratio"] = 9.99
    work["min_holding_days_before_score_exit"] = 99
    work["score_continue_entry_ratio"] = 9.99
    work["strategy_variant"] = name
    work["filter_name"] = "current_formal_l4_online_regime"
    work["entry_weight_name"] = "equal_weight_90pct"
    work["dynamic_hold_name"] = "fixed_h5"
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


def run_juejin(name: str, path: Path, blend: str, topn: int) -> dict:
    log = LOG_DIR / f"{name}.log"
    score_name = {"b10": "b10_100", "b510": "b5_20_10_80", "ball": "b1_05_3_10_5_15_10_70"}[blend]
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY_DIR), "--signal-file", str(path),
        "--log-file", str(log), "--max-positions", str(topn), "--holding-days", "5",
        "--max-holding-days", "5", "--score-exit-entry-ratio", "9.99",
        "--score-continue-entry-ratio", "9.99", "--min-holding-days-before-score-exit", "99",
        "--score-db", str(SOURCE_DIR / "score_assets" / f"{score_name}.duckdb"),
        "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    env = os.environ.copy()
    env.update({"GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0", "GM_OPEN_DAILY_SCORE_EXIT": "0"})
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log)
    return {
        "name": name, "returncode": proc.returncode, "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(path), "log_file": str(log),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(POOL_DB), read_only=True)
    try:
        base = load_base(con)
    finally:
        con.close()
    rows = []
    selections = {}
    for blend in BLENDS:
        for regime in REGIMES:
            for topn in [3, 5]:
                for model_min in [0.85, 0.90, 0.95]:
                    name = f"{blend}_{regime}_m{int(model_min*100)}_top{topn}_h5"
                    selected = select_case(base, blend, regime, topn, model_min)
                    metrics = local_metrics(selected, topn)
                    row = {"name": name, "blend": blend, "regime": regime, "topn": topn, "model_min": model_min, **metrics}
                    row["robust_score"] = min(row["early_sharpe"], row["late_sharpe"]) + 0.25 * row["proxy_sharpe"]
                    rows.append(row)
                    selections[name] = selected
    local = pd.DataFrame(rows).sort_values(["robust_score", "proxy_sharpe"], ascending=False)
    local.to_csv(REPORT_DIR / "local_screen.csv", index=False, encoding="utf-8-sig")
    eligible = local[(local["signal_days"] >= 450) & (local["early_annual"] > 0) & (local["late_annual"] > 0)].head(8)
    results = []
    for _, row in eligible.iterrows():
        path = write_signal(selections[row["name"]], row["name"], int(row["topn"]), row["blend"])
        result = run_juejin(row["name"], path, row["blend"], int(row["topn"]))
        results.append({**row.to_dict(), **result})
        pd.DataFrame(results).to_csv(REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({k: result.get(k) for k in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
        time.sleep(3)
    payload = {"status": "research_only", "screened_cases": len(local), "juejin_cases": results}
    (REPORT_DIR / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
