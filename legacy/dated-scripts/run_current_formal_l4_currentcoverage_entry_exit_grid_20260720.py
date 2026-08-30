from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SOURCE_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
)
BASE_SIGNAL = SOURCE_DIR / "signals" / "r1_88_min12_max13.csv"
SCORE_DB = SOURCE_DIR / "score_assets" / "w10_100.duckdb"
STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_score_quantized_20260720"
    / "research_code_snapshot"
)
OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_entry_exit_grid_20260720"
)
SIGNALS = OUT / "signals"
LOGS = OUT / "logs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_indicator(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result: dict[str, float | int] = {}
            for key in ("pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"):
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            for key in ("open_count", "close_count"):
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    result[key] = int(match.group(1))
            return result
    return {}


def cases() -> list[dict]:
    rows: list[dict] = []
    for min_hold in (4, 6, 8, 10, 11, 12):
        for max_hold in (13, 16):
            if min_hold < max_hold:
                rows.append(
                    {
                        "family": "sell_frequency",
                        "name": f"freq_min{min_hold}_max{max_hold}",
                        "min_hold": min_hold,
                        "max_hold": max_hold,
                        "edge": 0.0,
                    }
                )
    for min_hold in (8, 10, 12):
        for edge in (0.01, 0.02):
            rows.append(
                {
                    "family": "sell_advantage",
                    "name": f"edge_min{min_hold}_max16_e{int(edge * 100):02d}",
                    "min_hold": min_hold,
                    "max_hold": 16,
                    "edge": edge,
                }
            )
    for gap in (3.0, 4.0, 5.0):
        for scale in (0.0, 0.5, 0.75):
            rows.append(
                {
                    "family": "buy_open_gap",
                    "name": f"buygap_gt{int(gap)}_scale{int(scale * 100)}",
                    "min_hold": 12,
                    "max_hold": 13,
                    "edge": 0.0,
                    "buy_gap": gap,
                    "buy_scale": scale,
                }
            )
    for gap in (-2.0, -3.0, -4.0):
        for early_min in (1, 2):
            rows.append(
                {
                    "family": "held_open_gap_exit",
                    "name": f"heldgap_m{abs(int(gap))}_h{early_min}",
                    "min_hold": 12,
                    "max_hold": 13,
                    "edge": 0.0,
                    "early_open_gap": gap,
                    "early_min_hold": early_min,
                }
            )
    return rows


def build_signal(base: pd.DataFrame, case: dict) -> Path:
    frame = base.copy()
    frame["holding_days"] = int(case["min_hold"])
    frame["max_holding_days"] = int(case["max_hold"])
    if case["family"] == "buy_open_gap":
        gap = pd.to_numeric(frame["buy_open_gap_raw_pct"], errors="coerce")
        high = gap > float(case["buy_gap"])
        frame.loc[high, "target_pct"] = (1.0 / 3.0) * float(case["buy_scale"])
        frame = frame[pd.to_numeric(frame["target_pct"], errors="coerce") > 0].copy()
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
            "GM_INDEPENDENT_REPLACE_EDGE": str(case["edge"]),
            "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10",
            "GM_RESIZE_HELD_ON_SIGNAL": "0",
            "GM_INDEPENDENT_EARLY_OPEN_GAP_PCT": (
                "" if "early_open_gap" not in case else str(case["early_open_gap"])
            ),
            "GM_INDEPENDENT_EARLY_OPEN_GAP_MIN_HOLD": str(case.get("early_min_hold", 1)),
        }
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY),
        "--signal-file",
        str(signal),
        "--log-file",
        str(log),
        "--max-positions",
        "3",
        "--holding-days",
        str(case["min_hold"]),
        "--max-holding-days",
        str(case["max_hold"]),
        "--target-position-pct",
        str(1.0 / 3.0),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "blended_rank_score",
        "--market-db",
        str(MARKET_DB),
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
    indicator = parse_indicator(log)
    return {
        **case,
        "signal_rows": int(len(pd.read_csv(signal))),
        "signal_sha256": sha256(signal),
        "returncode": int(process.returncode),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "signal_file": str(signal),
        "log_file": str(log),
    }


def main() -> None:
    for path in (OUT, SIGNALS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_SIGNAL)
    input_audit = {
        "base_signal": str(BASE_SIGNAL),
        "base_signal_sha256": sha256(BASE_SIGNAL),
        "score_db": str(SCORE_DB),
        "score_db_sha256": sha256(SCORE_DB),
        "rows": int(len(base)),
        "min_signal_date": str(base["signal_date"].min()),
        "max_signal_date": str(base["signal_date"].max()),
        "max_buy_date": str(base["buy_date"].max()),
        "duplicate_keys": int(base.duplicated(["signal_date", "stock_code"]).sum()),
        "bj_rows": int(base["stock_code"].astype(str).str.endswith(".BJ").sum()),
        "next_open_rule": "buy open gap is known at T+1 open; never written back to T signal selection",
        "execution_prices": "raw unadjusted open/pre_close",
        "factor_prices": "explicit qfq fields only",
    }
    rows: list[dict] = []
    for case in cases():
        result = run_case(base, case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(
            ["sharpe", "annual_return"], ascending=False, na_position="last"
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
        "input_audit": input_audit,
        "results": rows,
        "target": {"annual_return": 5.0, "sharpe": 4.0, "max_drawdown": 0.40},
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
