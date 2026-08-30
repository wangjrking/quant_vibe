from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_score_quantized_20260720"
    / "research_code_snapshot"
)
SCORE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
    / "score_assets"
    / "w10_100.duckdb"
)
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_independent_sleeve_nav_blend_20260720"
)
LOGS = OUT / "logs"
NAVS = OUT / "nav"
INITIAL_CASH = 700_000.0
START = "2022-06-07 09:00:00"
END = "2026-07-17 15:30:00"

SLEEVES = {
    "top1": {
        "signal": ROOT
        / "quant"
        / "data_file"
        / "reports"
        / "strategy_agent_current_formal_l4_top1_open_gap_risk_20260720"
        / "signals"
        / "high_gt4_s00.csv",
        "max_positions": 1,
        "target_pct": 1.0,
        "min_hold": 10,
        "max_hold": 13,
    },
    "top3": {
        "signal": ROOT
        / "quant"
        / "data_file"
        / "reports"
        / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
        / "signals"
        / "r1_88_min12_max13.csv",
        "max_positions": 3,
        "target_pct": 1.0 / 3.0,
        "min_hold": 12,
        "max_hold": 13,
    },
}


def parse_indicator(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result = {}
            keys = ("pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "open_count", "close_count")
            for key in keys:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            return result
    return {}


def parse_nav(path: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"EXPOSURE\s+(\d{8})\s+post_buy\s+invested_pct=[-+0-9.eE]+\s+"
        r"active_positions=\d+\s+market_value=[-+0-9.eE]+\s+nav=([-+0-9.eE]+)"
    )
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({"trade_date": match.group(1), "nav": float(match.group(2))})
    frame = pd.DataFrame(rows).drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    if frame.empty:
        raise RuntimeError(f"no daily NAV rows parsed from {path}")
    frame["normalized_nav"] = frame["nav"] / INITIAL_CASH
    return frame


def run_sleeve(name: str, config: dict) -> dict:
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_LOG_EXPOSURE": "1",
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(config["min_hold"]),
            "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(config["max_hold"]),
            "GM_INDEPENDENT_REPLACE_RATIO": "1.0",
            "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10",
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_EQUITY_DD_RISK_MODE": "0",
        }
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY),
        "--signal-file",
        str(config["signal"]),
        "--log-file",
        str(log),
        "--max-positions",
        str(config["max_positions"]),
        "--holding-days",
        str(config["min_hold"]),
        "--max-holding-days",
        str(config["max_hold"]),
        "--target-position-pct",
        str(config["target_pct"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "blended_rank_score",
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        str(INITIAL_CASH),
        "--backtest-slippage-ratio",
        "0.003",
        "--backtest-start",
        START,
        "--backtest-end",
        END,
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Juejin sleeve {name} failed: {process.stdout}\n{process.stderr}")
    indicator = parse_indicator(log)
    nav = parse_nav(log)
    nav.to_csv(NAVS / f"{name}_daily_nav.csv", index=False, encoding="utf-8-sig")
    return {"name": name, "indicator": indicator, "nav": nav, "log_file": str(log)}


def portfolio_metrics(frame: pd.DataFrame) -> dict:
    values = frame["portfolio_nav"].astype(float)
    returns = values.pct_change().dropna()
    start_date = datetime.strptime(str(frame["trade_date"].iloc[0]), "%Y%m%d")
    end_date = datetime.strptime(str(frame["trade_date"].iloc[-1]), "%Y%m%d")
    years = max((end_date - start_date).days / 365.25, 1.0 / 365.25)
    annual_return = float(values.iloc[-1] ** (1.0 / years) - 1.0)
    annual_return_juejin_linear_proxy = float((values.iloc[-1] - 1.0) / years)
    volatility = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / volatility * math.sqrt(252.0)) if volatility > 0 else None
    drawdown = values / values.cummax() - 1.0
    return {
        "total_return": float(values.iloc[-1] - 1.0),
        "annual_return_compound": annual_return,
        "annual_return_juejin_linear_proxy": annual_return_juejin_linear_proxy,
        "sharpe_daily_zero_rf": sharpe,
        "max_drawdown": float(-drawdown.min()),
        "trade_days": int(len(frame)),
    }


def main() -> None:
    for path in (OUT, LOGS, NAVS):
        path.mkdir(parents=True, exist_ok=True)

    sleeve_results = {name: run_sleeve(name, config) for name, config in SLEEVES.items()}
    nav = sleeve_results["top1"]["nav"].rename(columns={"normalized_nav": "top1_nav"})[
        ["trade_date", "top1_nav"]
    ]
    nav = nav.merge(
        sleeve_results["top3"]["nav"].rename(columns={"normalized_nav": "top3_nav"})[
            ["trade_date", "top3_nav"]
        ],
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )

    rows = []
    for top1_weight in [value / 20.0 for value in range(0, 21)]:
        frame = nav.copy()
        frame["portfolio_nav"] = top1_weight * frame["top1_nav"] + (1.0 - top1_weight) * frame["top3_nav"]
        metrics = portfolio_metrics(frame)
        rows.append({"top1_weight": top1_weight, "top3_weight": 1.0 - top1_weight, **metrics})
    results = pd.DataFrame(rows).sort_values(
        ["sharpe_daily_zero_rf", "annual_return_juejin_linear_proxy"], ascending=False
    )
    results.to_csv(OUT / "nav_blend_results.csv", index=False, encoding="utf-8-sig")
    nav.to_csv(OUT / "aligned_sleeve_nav.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "research_only",
        "method": "independent Juejin sleeve NAVs combined with fixed initial capital weights",
        "backtest_contract": {
            "start": START,
            "end": END,
            "adjust": "none",
            "slippage_ratio_one_side": 0.003,
            "initial_cash_per_sleeve": INITIAL_CASH,
        },
        "component_indicators": {
            name: {**result["indicator"], "log_file": result["log_file"]}
            for name, result in sleeve_results.items()
        },
        "best_by_sharpe": results.iloc[0].to_dict(),
        "target_hits": results[
            (results["annual_return_juejin_linear_proxy"] >= 5.0)
            & (results["sharpe_daily_zero_rf"] >= 4.0)
            & (results["max_drawdown"] <= 0.4)
        ].to_dict("records"),
        "caveat": "Combined metrics are derived from two Juejin-generated daily NAV series; promising weights still require one-account Juejin implementation before admission.",
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
