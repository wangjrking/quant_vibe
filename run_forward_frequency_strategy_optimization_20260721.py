from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_observation_validation_strategy_optimization_20260721.py"
CONFIG = MAIN / "config" / "strategy_research" / "forward_frequency_strategy_optimization_20260721.json"
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_forward_frequency_optimization_20260721"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("observation_validation_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
base.CONFIG = CONFIG
base.OUT = OUT
base.POOL_DB = OUT / "current_l4_pool.duckdb"
base.SIGNALS = OUT / "signals"
base.SCORES = OUT / "score_assets"
base.LOGS = OUT / "juejin_logs"
base.LOCK = OUT / "preregistration_lock.json"
base.SCREEN = OUT / "entry_screen_results.csv"
base.ENTRY_FINALISTS = OUT / "entry_finalists.json"
base.JUEJIN_TUNE = OUT / "juejin_tuning_results.csv"
base.JUEJIN_FINALISTS = OUT / "juejin_finalists_frozen.json"
base.__file__ = str(Path(__file__).resolve())


def freeze() -> None:
    base.freeze()
    lock = base.read_json(base.LOCK)
    lock["base_module"] = str(BASE_SCRIPT)
    lock["base_module_sha256"] = base.sha256(BASE_SCRIPT)
    lock["selection_data_max_date"] = "20251231"
    lock["previous_2026_holdout_status"] = "observed_and_forbidden_for_selection"
    lock["new_unseen_forward_start"] = "20260722"
    base.write_json(base.LOCK, lock)


original_verify_lock = base.verify_lock


def verify_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], Path]:
    result = original_verify_lock()
    lock = result[1]
    if base.sha256(BASE_SCRIPT) != lock.get("base_module_sha256"):
        raise RuntimeError("base module changed after preregistration freeze")
    if lock.get("selection_data_max_date") != "20251231":
        raise RuntimeError("selection date boundary is not frozen")
    return result


base.verify_lock = verify_lock


def run_juejin(
    signal: Path,
    score_db: Path,
    name: str,
    start: str,
    end: str,
    slippage: float,
    exit_case: dict[str, Any],
    market: Path,
    max_positions: int,
    target_position_ceiling: float,
) -> dict[str, Any]:
    if end[:10].replace("-", "") > "20251231":
        raise RuntimeError("2026 and later data are forbidden during candidate selection")
    log = base.LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(exit_case["min_hold"]),
            "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(exit_case["max_hold"]),
            "GM_INDEPENDENT_REPLACE_RATIO": "1.0",
            "GM_INDEPENDENT_REPLACE_EDGE": str(exit_case["replace_edge"]),
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10",
            "GM_MAX_DAILY_BUYS": str(exit_case["max_daily_buys"]),
            "GM_MAX_DAILY_SELLS": str(exit_case["max_daily_sells"]),
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_INDEPENDENT_EARLY_OPEN_GAP_PCT": "",
            "GM_EQUITY_DD_RISK_MODE": "0",
        }
    )
    command = [
        sys.executable,
        str(base.RUNNER),
        "--strategy-dir",
        str(base.STRATEGY),
        "--signal-file",
        str(signal),
        "--log-file",
        str(log),
        "--max-positions",
        str(max_positions),
        "--target-position-pct",
        str(target_position_ceiling),
        "--holding-days",
        str(exit_case["min_hold"]),
        "--max-holding-days",
        str(exit_case["max_hold"]),
        "--score-db",
        str(score_db),
        "--score-table",
        "blended_rank_score",
        "--market-db",
        str(market),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        str(slippage),
        "--backtest-start",
        start,
        "--backtest-end",
        end,
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = base.parse_indicator(log)
    return {
        "returncode": process.returncode,
        "annual": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "mdd": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "log": str(log),
    }


base.run_juejin = run_juejin


def write_forward_contract() -> None:
    frozen = base.read_json(base.JUEJIN_FINALISTS)
    payload = {
        "schema_version": 1,
        "status": "frozen_waiting_for_new_unseen_forward_data",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_data_max_date": "20251231",
        "previous_2026_holdout": "observed_and_not_reusable_for_admission",
        "new_forward_start": "20260722",
        "minimum_forward_trade_days_before_first_review": 20,
        "parameter_changes_after_freeze_allowed": False,
        "production_publication_allowed": False,
        "finalists": frozen["finalists"],
        "preregistration_lock_sha256": base.sha256(base.LOCK),
        "config_sha256": base.sha256(CONFIG),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "base_module_sha256": base.sha256(BASE_SCRIPT),
    }
    base.write_json(OUT / "forward_validation_contract.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "screen", "juejin-tune"), required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "screen":
        base.screen_entries()
    else:
        base.tune_juejin()
        write_forward_contract()


if __name__ == "__main__":
    main()
