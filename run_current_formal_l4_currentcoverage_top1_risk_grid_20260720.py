from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
COMMON_PATH = MAIN / "run_current_formal_l4_currentcoverage_entry_exit_grid_20260720.py"
SPEC = importlib.util.spec_from_file_location("currentcoverage_common", COMMON_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMMON)

SOURCE_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
)
BASE_SIGNAL = SOURCE_DIR / "signals" / "r1_88_min12_max13.csv"
SCORE_DB = SOURCE_DIR / "score_assets" / "w10_100.duckdb"
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_top1_risk_grid_20260720"
)
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"


CASES = [
    {
        "name": f"cap{int(cap * 100)}_min{min_hold}_max13",
        "cap": cap,
        "min_hold": min_hold,
        "max_hold": 13,
        "buy_gap": None,
        "buy_scale": 1.0,
    }
    for cap in (0.60, 0.70, 0.80, 0.90, 1.00)
    for min_hold in (10, 11, 12)
] + [
    {
        "name": f"cap100_min10_max13_gap{int(gap)}_scale{int(scale * 100)}",
        "cap": 1.0,
        "min_hold": 10,
        "max_hold": 13,
        "buy_gap": gap,
        "buy_scale": scale,
    }
    for gap in (4.0, 5.0)
    for scale in (0.50, 0.75)
]


def build_signal(base: pd.DataFrame, case: dict) -> Path:
    frame = base.copy()
    frame["target_pct"] = float(case["cap"])
    if case["buy_gap"] is not None:
        gap = pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce")
        frame.loc[gap > float(case["buy_gap"]), "target_pct"] = (
            float(case["cap"]) * float(case["buy_scale"])
        )
    frame["holding_days"] = int(case["min_hold"])
    frame["max_holding_days"] = int(case["max_hold"])
    frame["strategy_variant"] = case["name"]
    path = SIGNALS / f"{case['name']}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_case(base: pd.DataFrame, case: dict) -> dict:
    signal = build_signal(base, case)
    log = LOGS / f"{case['name']}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INDEPENDENT_REPLACE_MODE": "1",
            "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(case["min_hold"]),
            "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(case["max_hold"]),
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
        str(COMMON.STRATEGY),
        "--signal-file",
        str(signal),
        "--log-file",
        str(log),
        "--max-positions",
        "1",
        "--holding-days",
        str(case["min_hold"]),
        "--max-holding-days",
        str(case["max_hold"]),
        "--target-position-pct",
        str(case["cap"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "blended_rank_score",
        "--market-db",
        str(COMMON.MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.003",
        "--backtest-start",
        "2022-06-07 09:00:00",
        "--backtest-end",
        "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = COMMON.parse_indicator(log)
    return {
        **case,
        "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "signal_sha256": COMMON.sha256(signal),
        "signal_file": str(signal),
        "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_SIGNAL)
    rows: list[dict] = []
    for case in CASES:
        result = run_case(base, case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(
            ["annual_return", "sharpe"], ascending=False, na_position="last"
        ).to_csv(OUT / "juejin_results.csv", index=False, encoding="utf-8-sig")
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("name", "returncode", "annual_return", "sharpe", "max_drawdown")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    payload = {
        "status": "research_only_not_admitted",
        "input": {
            "base_signal": str(BASE_SIGNAL),
            "base_signal_sha256": COMMON.sha256(BASE_SIGNAL),
            "score_db": str(SCORE_DB),
            "score_db_sha256": COMMON.sha256(SCORE_DB),
            "max_signal_date": str(base["signal_date"].max()),
            "max_buy_date": str(base["buy_date"].max()),
        },
        "results": rows,
        "target_hits": [
            row["name"]
            for row in rows
            if row.get("annual_return") is not None
            and row["annual_return"] >= 5.0
            and row.get("sharpe") is not None
            and row["sharpe"] >= 4.0
            and row.get("max_drawdown") is not None
            and row["max_drawdown"] <= 0.40
        ],
    }
    (OUT / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
