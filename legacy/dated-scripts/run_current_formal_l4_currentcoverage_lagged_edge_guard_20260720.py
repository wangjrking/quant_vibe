from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
COMMON_PATH = MAIN / "run_current_formal_l4_currentcoverage_entry_exit_grid_20260720.py"
SPEC = importlib.util.spec_from_file_location("currentcoverage_common", COMMON_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

BASE_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_entry_exit_grid_20260720"
    / "signals"
    / "buygap_gt5_scale50.csv"
)
FEATURE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_persistent_edge_20260720"
    / "persistent_edge_features.duckdb"
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
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_lagged_edge_guard_20260720"
)
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"

CASES = [
    {"metric": "mean", "window": window, "threshold": threshold, "bad_scale": bad_scale}
    for window in (20, 40, 60)
    for threshold in (0.0, 0.01)
    for bad_scale in (0.0, 0.5)
] + [
    {"metric": "win_rate", "window": window, "threshold": threshold, "bad_scale": 0.0}
    for window in (20, 40, 60)
    for threshold in (0.50, 0.55)
]


def build_history(base: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    try:
        returns = con.execute(
            "SELECT trade_date, stock_code, ret_open_12 FROM persistent_edge_features"
        ).fetchdf()
    finally:
        con.close()
    frame = base[["signal_date", "stock_code"]].copy()
    frame["signal_date"] = frame["signal_date"].astype(str)
    returns["trade_date"] = returns["trade_date"].astype(str)
    frame = frame.merge(
        returns,
        left_on=["signal_date", "stock_code"],
        right_on=["trade_date", "stock_code"],
        how="left",
        validate="many_to_one",
    )
    market = duckdb.connect(str(COMMON.MARKET_DB), read_only=True)
    try:
        calendar = pd.Index(
            market.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")
            .fetchnumpy()["trade_date"]
            .astype(str)
        )
    finally:
        market.close()
    positions = {date: i for i, date in enumerate(calendar)}
    frame["available_date"] = frame["signal_date"].map(
        lambda date: calendar[positions[date] + 12] if date in positions and positions[date] + 12 < len(calendar) else None
    )
    return frame.dropna(subset=["ret_open_12", "available_date"]).sort_values(
        ["available_date", "signal_date", "stock_code"]
    )


def regime_table(base: pd.DataFrame, history: pd.DataFrame, case: dict) -> pd.DataFrame:
    rows = []
    for signal_date in sorted(base["signal_date"].astype(str).unique()):
        known = history[history["available_date"] <= signal_date].tail(int(case["window"]))
        if len(known) < int(case["window"]):
            metric_value = np.nan
            good = True
        elif case["metric"] == "mean":
            metric_value = float(known["ret_open_12"].mean())
            good = metric_value >= float(case["threshold"])
        else:
            metric_value = float((known["ret_open_12"] > 0).mean())
            good = metric_value >= float(case["threshold"])
        rows.append(
            {
                "signal_date": signal_date,
                "known_observations": int(len(known)),
                "metric_value": metric_value,
                "regime_good": bool(good),
            }
        )
    return pd.DataFrame(rows)


def case_name(case: dict) -> str:
    threshold = str(case["threshold"]).replace(".", "p")
    return (
        f"{case['metric']}_w{case['window']}_t{threshold}"
        f"_badscale{int(float(case['bad_scale']) * 100)}"
    )


def build_signal(base: pd.DataFrame, regimes: pd.DataFrame, case: dict) -> tuple[Path, dict]:
    frame = base.merge(regimes, on="signal_date", how="left", validate="many_to_one")
    bad = ~frame["regime_good"].fillna(True)
    frame.loc[bad, "target_pct"] *= float(case["bad_scale"])
    frame = frame[pd.to_numeric(frame["target_pct"], errors="coerce") > 0].copy()
    name = case_name(case)
    frame["strategy_variant"] = name
    path = SIGNALS / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    meta = {
        "bad_signal_days": int(regimes.loc[~regimes["regime_good"], "signal_date"].nunique()),
        "total_signal_days": int(regimes["signal_date"].nunique()),
        "output_rows": int(len(frame)),
    }
    return path, meta


def run_case(base: pd.DataFrame, history: pd.DataFrame, case: dict) -> dict:
    regimes = regime_table(base, history, case)
    signal, meta = build_signal(base, regimes, case)
    name = case_name(case)
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
            "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_EQUITY_DD_RISK_MODE": "0",
        }
    )
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(COMMON.STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13",
        "--target-position-pct", str(1.0 / 3.0), "--score-db", str(SCORE_DB),
        "--score-table", "blended_rank_score", "--market-db", str(COMMON.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    return {
        "name": name, **case, **meta, "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "signal_sha256": COMMON.sha256(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_SIGNAL)
    base["signal_date"] = base["signal_date"].astype(str)
    history = build_history(base)
    rows: list[dict] = []
    for case in CASES:
        result = run_case(base, history, case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in ("name", "annual_return", "sharpe", "max_drawdown")}, ensure_ascii=False), flush=True)
    payload = {
        "status": "research_only_not_admitted",
        "information_contract": "only returns with available_date <= current signal_date are used",
        "base_signal": str(BASE_SIGNAL), "base_signal_sha256": COMMON.sha256(BASE_SIGNAL),
        "score_db": str(SCORE_DB), "score_db_sha256": COMMON.sha256(SCORE_DB),
        "results": rows,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
