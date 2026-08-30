from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
COMMON_PATH = MAIN / "run_current_formal_l4_currentcoverage_entry_exit_grid_20260720.py"
SPEC = importlib.util.spec_from_file_location("currentcoverage_common", COMMON_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

FEATURE_DB = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_persistent_edge_20260720/persistent_edge_features.duckdb"
SOURCE_SIGNAL = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_currentcoverage_z10_confidence_grid_20260720/signals/baseline.csv"
SCORE_DB = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720/score_assets/w10_100.duckdb"
OUT = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_currentcoverage_lagged_rankic_guard_20260720"
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"

CASES = [
    {"name": "baseline", "rule": "none", "scale": 1.0, "threshold": 0.0},
    {"name": "ic1_20_neg_s50", "rule": "ic1", "scale": 0.5, "threshold": 0.0},
    {"name": "ic1_20_neg_stop", "rule": "ic1", "scale": 0.0, "threshold": 0.0},
    {"name": "ic1_20_lt1pct_s50", "rule": "ic1", "scale": 0.5, "threshold": 0.01},
    {"name": "ic5_20_neg_s50", "rule": "ic5", "scale": 0.5, "threshold": 0.0},
    {"name": "ic5_20_neg_stop", "rule": "ic5", "scale": 0.0, "threshold": 0.0},
    {"name": "either_ic_neg_s50", "rule": "either", "scale": 0.5, "threshold": 0.0},
    {"name": "both_ic_neg_s50", "rule": "both", "scale": 0.5, "threshold": 0.0},
]


def load_state(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        WITH return_ranks AS (
            SELECT trade_date, rank_1d, rank_5d,
                   percent_rank() OVER (
                       PARTITION BY trade_date ORDER BY next_open_return_raw
                   ) AS return_1d_rank,
                   percent_rank() OVER (
                       PARTITION BY trade_date ORDER BY ret_open_5
                   ) AS return_5d_rank,
                   next_open_return_raw, ret_open_5
            FROM persistent_edge_features
        ), daily_ic AS (
            SELECT trade_date,
                   corr(rank_1d, return_1d_rank) FILTER (WHERE next_open_return_raw IS NOT NULL) AS ic1,
                   corr(rank_5d, return_5d_rank) FILTER (WHERE ret_open_5 IS NOT NULL) AS ic5
            FROM return_ranks GROUP BY trade_date
        )
        SELECT trade_date AS signal_date,
               avg(ic1) OVER (
                   ORDER BY trade_date ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING
               ) AS ic1_roll20_mature,
               avg(ic5) OVER (
                   ORDER BY trade_date ROWS BETWEEN 25 PRECEDING AND 6 PRECEDING
               ) AS ic5_roll20_mature
        FROM daily_ic ORDER BY trade_date
        """
    ).fetchdf()


def build_signal(state: pd.DataFrame, case: dict) -> Path:
    frame = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str})
    state = state.copy()
    state["signal_date"] = state["signal_date"].astype(str)
    frame = frame.merge(state, on="signal_date", how="left", validate="many_to_one")
    if case["rule"] == "none":
        weak = pd.Series(False, index=frame.index)
    elif case["rule"] == "ic1":
        weak = frame["ic1_roll20_mature"] < float(case["threshold"])
    elif case["rule"] == "ic5":
        weak = frame["ic5_roll20_mature"] < float(case["threshold"])
    elif case["rule"] == "either":
        weak = (frame["ic1_roll20_mature"] < float(case["threshold"])) | (
            frame["ic5_roll20_mature"] < float(case["threshold"])
        )
    elif case["rule"] == "both":
        weak = (frame["ic1_roll20_mature"] < float(case["threshold"])) & (
            frame["ic5_roll20_mature"] < float(case["threshold"])
        )
    else:
        raise ValueError(case["rule"])
    frame["lagged_rankic_weak"] = weak.fillna(False)
    if float(case["scale"]) == 0.0:
        frame = frame[~frame["lagged_rankic_weak"]].copy()
    elif float(case["scale"]) < 1.0:
        frame.loc[frame["lagged_rankic_weak"], "target_pct"] *= float(case["scale"])
    frame["strategy_variant"] = case["name"]
    frame["entry_weight_name"] = "mature_lagged_rankic_guard_then_buy_gap_gt5_half"
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(state: pd.DataFrame, case: dict) -> dict:
    signal = build_signal(state, case)
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "12", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(COMMON.STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "3",
        "--holding-days", "12", "--max-holding-days", "13", "--target-position-pct", str(1.0 / 3.0),
        "--score-db", str(SCORE_DB), "--score-table", "blended_rank_score", "--market-db", str(COMMON.MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    signal_frame = pd.read_csv(signal)
    return {
        **case, "signal_days": int(signal_frame["signal_date"].nunique()), "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "signal_sha256": COMMON.sha256(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    try:
        state = load_state(con)
    finally:
        con.close()
    state.to_csv(OUT / "mature_lagged_rankic_state.csv", index=False, encoding="utf-8-sig")
    rows: list[dict] = []
    for case in CASES:
        result = run_case(state, case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
            OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in (
            "name", "signal_days", "annual_return", "sharpe", "max_drawdown"
        )}, ensure_ascii=False), flush=True)
    payload = {
        "status": "research_only_not_admitted", "maturity_rule": {
            "1d": "rolling 20 daily cross-sectional rank IC through t-2",
            "5d": "rolling 20 daily cross-sectional rank IC through t-6",
        }, "feature_db": str(FEATURE_DB), "feature_db_sha256": COMMON.sha256(FEATURE_DB),
        "source_signal": str(SOURCE_SIGNAL), "source_signal_sha256": COMMON.sha256(SOURCE_SIGNAL),
        "results": rows,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
