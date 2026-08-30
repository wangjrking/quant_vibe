from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
SOURCE_PATH = MAIN / "run_current_formal_l4_open_gap_sell_neighborhood_20260720.py"
SPEC = importlib.util.spec_from_file_location("top1_common", SOURCE_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

FEATURE_DB = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_persistent_edge_20260720/persistent_edge_features.duckdb"
SOURCE_SIGNAL = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_top1_open_gap_risk_20260720/signals/high_gt4_s00.csv"
STRATEGY = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_score_quantized_20260720/research_code_snapshot"
OUT = ROOT / "quant/data_file/reports/strategy_agent_current_formal_l4_top1_z10_confidence_grid_20260720"
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"

CASES = [
    {"name": "baseline", "low": None, "high": None, "outside_scale": 1.0},
    {"name": "soft_1p8_3p0_s50", "low": 1.8, "high": 3.0, "outside_scale": 0.5},
    {"name": "soft_2p0_3p0_s50", "low": 2.0, "high": 3.0, "outside_scale": 0.5},
    {"name": "soft_2p0_2p8_s50", "low": 2.0, "high": 2.8, "outside_scale": 0.5},
    {"name": "soft_2p1_2p7_s50", "low": 2.1, "high": 2.7, "outside_scale": 0.5},
    {"name": "soft_2p1_2p7_s70", "low": 2.1, "high": 2.7, "outside_scale": 0.7},
    {"name": "hard_1p8_3p0", "low": 1.8, "high": 3.0, "outside_scale": 0.0},
    {"name": "hard_2p0_3p0", "low": 2.0, "high": 3.0, "outside_scale": 0.0},
    {"name": "hard_2p1_2p7", "low": 2.1, "high": 2.7, "outside_scale": 0.0},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_signal(con: duckdb.DuckDBPyConnection, case: dict) -> Path:
    source = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str, "stock_code": str})
    stats = con.execute(
        """
        WITH day_stats AS (
            SELECT trade_date, avg(pred_10d) AS pred10_mean, stddev_pop(pred_10d) AS pred10_std
            FROM persistent_edge_features GROUP BY trade_date
        )
        SELECT p.trade_date AS signal_date, p.stock_code,
               (p.pred_10d - d.pred10_mean) / nullif(d.pred10_std, 0) AS z10
        FROM persistent_edge_features p JOIN day_stats d USING (trade_date)
        """
    ).fetchdf()
    stats["signal_date"] = stats["signal_date"].astype(str)
    frame = source.merge(stats, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    if frame["z10"].isna().any():
        raise RuntimeError(f"missing z10 rows: {int(frame['z10'].isna().sum())}")
    frame["confidence_in_range"] = True
    if case["low"] is not None:
        frame["confidence_in_range"] = frame["z10"].between(float(case["low"]), float(case["high"]), inclusive="both")
        if float(case["outside_scale"]) == 0.0:
            frame = frame[frame["confidence_in_range"]].copy()
    frame["target_pct"] = 1.0
    if case["low"] is not None and float(case["outside_scale"]) > 0.0:
        frame.loc[~frame["confidence_in_range"], "target_pct"] *= float(case["outside_scale"])
    frame["holding_days"] = 10
    frame["max_holding_days"] = 13
    frame["strategy_variant"] = case["name"]
    frame["entry_weight_name"] = "top1_z10_confidence"
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(con: duckdb.DuckDBPyConnection, case: dict) -> dict:
    signal = build_signal(con, case)
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": "10", "GM_INDEPENDENT_REPLACE_MAX_HOLD": "13",
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0", "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(COMMON.RUNNER), "--strategy-dir", str(STRATEGY),
        "--signal-file", str(signal), "--log-file", str(log), "--max-positions", "1",
        "--holding-days", "10", "--max-holding-days", "13", "--target-position-pct", "1.0",
        "--score-db", str(COMMON.SCORE_DB), "--score-table", "blended_rank_score",
        "--market-db", str(COMMON.MARKET_DB), "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003", "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    signal_frame = pd.read_csv(signal)
    return {
        **case, "signal_days": int(signal_frame["signal_date"].nunique()), "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"), "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"), "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"), "close_count": indicator.get("close_count"),
        "signal_file": str(signal), "signal_sha256": sha256(signal), "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    rows: list[dict] = []
    try:
        for case in CASES:
            result = run_case(con, case)
            rows.append(result)
            pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).to_csv(
                OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
            )
            print(json.dumps({key: result.get(key) for key in (
                "name", "signal_days", "annual_return", "sharpe", "max_drawdown"
            )}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    payload = {
        "status": "research_only_not_admitted", "feature_db": str(FEATURE_DB),
        "feature_db_sha256": sha256(FEATURE_DB), "source_signal": str(SOURCE_SIGNAL),
        "source_signal_sha256": sha256(SOURCE_SIGNAL), "results": rows,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
