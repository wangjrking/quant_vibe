from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
COMMON_SOURCE = MAIN / "run_current_formal_l4_open_gap_sell_neighborhood_20260720.py"
SPEC = importlib.util.spec_from_file_location("common", COMMON_SOURCE)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_dynamic_sleeve_switch_20260720"
)
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"
NAV_SOURCE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_independent_sleeve_nav_blend_20260720"
    / "aligned_sleeve_nav.csv"
)
TOP1_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_top1_open_gap_risk_20260720"
    / "signals"
    / "high_gt4_s00.csv"
)
TOP3_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
    / "signals"
    / "r1_88_min12_max13.csv"
)
STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_score_quantized_20260720"
    / "research_code_snapshot"
)
START = "2022-06-07 09:00:00"
END = "2026-07-17 15:30:00"
LOOKBACKS = (20, 25, 30, 35, 40)
HOLD_WINDOWS = ((8, 12), (8, 13), (10, 12), (10, 13), (10, 14), (12, 13), (12, 14))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trailing_return(values: pd.Series, lookback: int) -> pd.Series:
    daily = values.astype(float).pct_change().fillna(0.0)
    return (1.0 + daily).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0


def build_signal(lookback: int) -> tuple[Path, dict]:
    nav = pd.read_csv(NAV_SOURCE, dtype={"trade_date": str})
    nav["top1_trailing_return"] = trailing_return(nav["top1_nav"], lookback)
    nav["top3_trailing_return"] = trailing_return(nav["top3_nav"], lookback)
    nav["use_top1"] = nav["top1_trailing_return"] > nav["top3_trailing_return"]

    # A signal formed after T close may use sleeve NAV through T. The selected
    # row executes at T+1 open, so no T+1 return enters this decision.
    regime = nav[
        ["trade_date", "top1_trailing_return", "top3_trailing_return", "use_top1"]
    ].rename(columns={"trade_date": "signal_date"})
    top1 = pd.read_csv(TOP1_SIGNAL, dtype={"signal_date": str, "buy_date": str})
    top3 = pd.read_csv(TOP3_SIGNAL, dtype={"signal_date": str, "buy_date": str})
    top1["sleeve_source"] = "top1"
    top3["sleeve_source"] = "top3"

    chosen_top1 = top1.merge(regime, on="signal_date", how="inner", validate="many_to_one")
    chosen_top1 = chosen_top1[chosen_top1["use_top1"]].copy()
    chosen_top3 = top3.merge(regime, on="signal_date", how="inner", validate="many_to_one")
    chosen_top3 = chosen_top3[~chosen_top3["use_top1"]].copy()
    combined = pd.concat([chosen_top1, chosen_top3], ignore_index=True)
    combined = combined.sort_values(["signal_date", "rank", "stock_code"]).reset_index(drop=True)
    combined["strategy_variant"] = f"dynamic_sleeve_switch_lb{lookback}"
    combined["filter_name"] = "lagged_shadow_sleeve_relative_momentum"

    output = SIGNALS / f"dynamic_sleeve_switch_lb{lookback}.csv"
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    summary = {
        "lookback": lookback,
        "rows": int(len(combined)),
        "signal_days": int(combined["signal_date"].nunique()),
        "top1_rows": int((combined["sleeve_source"] == "top1").sum()),
        "top3_rows": int((combined["sleeve_source"] == "top3").sum()),
        "min_signal_date": str(combined["signal_date"].min()),
        "max_signal_date": str(combined["signal_date"].max()),
        "duplicate_keys": int(combined.duplicated(["signal_date", "stock_code"]).sum()),
        "target_sum_max": float(combined.groupby("signal_date")["target_pct"].sum().max()),
        "signal_file": str(output),
        "signal_sha256": sha256(output),
    }
    return output, summary


def run_case(signal_file: Path, lookback: int, min_hold: int, max_hold: int) -> dict:
    name = f"lb{lookback}_min{min_hold}_max{max_hold}"
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(min_hold),
            "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(max_hold),
            "GM_INDEPENDENT_REPLACE_RATIO": "1.0",
            "GM_INDEPENDENT_REPLACE_EDGE": "0.0",
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10",
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_EQUITY_DD_RISK_MODE": "0",
        }
    )
    command = [
        sys.executable,
        str(COMMON.RUNNER),
        "--strategy-dir",
        str(STRATEGY),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log),
        "--max-positions",
        "3",
        "--holding-days",
        str(min_hold),
        "--max-holding-days",
        str(max_hold),
        "--target-position-pct",
        "1.0",
        "--score-db",
        str(COMMON.SCORE_DB),
        "--score-table",
        "blended_rank_score",
        "--market-db",
        str(COMMON.MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.003",
        "--backtest-start",
        START,
        "--backtest-end",
        END,
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    return {
        "name": name,
        "lookback": lookback,
        "min_hold": min_hold,
        "max_hold": max_hold,
        "returncode": process.returncode,
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "signal_file": str(signal_file),
        "signal_sha256": sha256(signal_file),
        "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    signal_files = {}
    signal_summaries = []
    for lookback in LOOKBACKS:
        signal_files[lookback], summary = build_signal(lookback)
        signal_summaries.append(summary)

    result_path = OUT / "juejin_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {str(row["name"]) for row in rows if int(row.get("returncode", -1)) == 0}
    for lookback in LOOKBACKS:
        for min_hold, max_hold in HOLD_WINDOWS:
            name = f"lb{lookback}_min{min_hold}_max{max_hold}"
            if name in completed:
                continue
            result = run_case(signal_files[lookback], lookback, min_hold, max_hold)
            rows.append(result)
            pd.DataFrame(rows).sort_values(
                ["annual_return", "sharpe"], ascending=False, na_position="last"
            ).to_csv(result_path, index=False, encoding="utf-8-sig")
            print(json.dumps(result, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    successful = frame[frame["returncode"] == 0].copy()
    target_hits = successful[
        (successful["annual_return"] >= 5.0)
        & (successful["sharpe"] >= 4.0)
        & (successful["max_drawdown"] <= 0.4)
    ]
    payload = {
        "status": "research_only_not_admitted",
        "method": "lagged shadow-sleeve relative momentum, executed in one Juejin account",
        "backtest_contract": {
            "start": START,
            "end": END,
            "adjust": "none",
            "slippage_ratio_one_side": 0.003,
            "max_positions": 3,
            "score_decimals": 10,
            "intraday_risk": False,
            "adaptive_slippage": False,
        },
        "input_sha256": {
            "nav_source": sha256(NAV_SOURCE),
            "top1_signal": sha256(TOP1_SIGNAL),
            "top3_signal": sha256(TOP3_SIGNAL),
            "strategy_main": sha256(STRATEGY / "main.py"),
            "score_db": sha256(COMMON.SCORE_DB),
            "market_db": sha256(COMMON.MARKET_DB),
        },
        "signal_summaries": signal_summaries,
        "best_by_annual_return": (
            successful.sort_values(["annual_return", "sharpe"], ascending=False).iloc[0].to_dict()
            if not successful.empty
            else None
        ),
        "best_by_sharpe": (
            successful.sort_values(["sharpe", "annual_return"], ascending=False).iloc[0].to_dict()
            if not successful.empty
            else None
        ),
        "target_hits": target_hits.to_dict("records"),
        "future_leakage_guard": (
            "Regime at signal_date T uses only shadow sleeve NAV through T; the selected signal executes at T+1 open."
        ),
    }
    (OUT / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
