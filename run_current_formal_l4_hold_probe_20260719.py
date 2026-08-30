from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
SOURCE_JSON = REPORT_DIR / "pullback_frequency_local_grid.json"
SIGNAL_DIR = REPORT_DIR / "hold_probe_signals"
LOG_DIR = REPORT_DIR / "hold_probe_logs"
OUT_CSV = REPORT_DIR / "hold_probe_juejin.csv"
OUT_JSON = REPORT_DIR / "hold_probe_juejin.json"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_high_return_frs_scale090_cap090_v20260716" / "code_snapshot"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def parse_indicator(log_file: Path) -> dict:
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        try:
            return ast.literal_eval(line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip())
        except Exception:
            payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
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


def prepare_signal(source_file: Path, case_name: str, hold: int) -> tuple[Path, int]:
    frame = pd.read_csv(source_file, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    frame["holding_days"] = hold
    frame["max_holding_days"] = hold
    frame["score_exit_entry_ratio"] = 9.99
    frame["min_holding_days_before_score_exit"] = 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = f"{case_name}_fixed_h{hold}"
    frame["filter_name"] = f"{case_name}_fixed_h{hold}"
    output = SIGNAL_DIR / f"{case_name}_fixed_h{hold}.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return output, int(frame.groupby("buy_date")["stock_code"].count().max())


def run_case(item: dict, hold: int) -> dict:
    source_file = Path(item["signal_file"])
    signal_file, max_positions = prepare_signal(source_file, str(item["name"]), hold)
    log_file = LOG_DIR / f"{item['name']}_fixed_h{hold}.log"
    command = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", str(max_positions),
        "--holding-days", str(hold),
        "--max-holding-days", str(hold),
        "--score-exit-entry-ratio", "9.99",
        "--score-continue-entry-ratio", "9.99",
        "--min-holding-days-before-score-exit", "99",
        "--score-db", str(item["score_db"]),
        "--score-table", str(item["score_table"]),
        "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none",
        "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00",
        "--backtest-end", "2026-07-17 15:30:00",
    ]
    env = os.environ.copy()
    env.update(
        {
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_STOP_LOSS_PCT": "",
            "GM_TAKE_PROFIT_PCT": "",
            "GM_LIGHT_STOP_LOSS_PCT": "",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
        }
    )
    parse_only = str(os.environ.get("STRATEGY_PARSE_ONLY", "0")).strip().lower() in {"1", "true", "yes"}
    indicator = parse_indicator(log_file) if parse_only else {}
    returncode = 0 if indicator else 1
    if not parse_only:
        process = None
        for attempt in range(1, 4):
            log_file.unlink(missing_ok=True)
            process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
            indicator = parse_indicator(log_file)
            log_text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
            if process.returncode == 0 and indicator:
                break
            if "1026" not in log_text or attempt == 3:
                break
            time.sleep(12)
        assert process is not None
        returncode = int(process.returncode)
        time.sleep(4)
    return {
        "case": f"{item['name']}_fixed_h{hold}",
        "source_case": item["name"],
        "hold": hold,
        "returncode": returncode,
        "signal_file": str(signal_file),
        "score_db": str(item["score_db"]),
        "score_table": str(item["score_table"]),
        "log_file": str(log_file),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
    }


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    candidates = payload["top_manifest"][:4]
    rows = []
    for item in candidates:
        for hold in [2, 3, 5, 10]:
            result = run_case(item, hold)
            rows.append(result)
            print(json.dumps({key: result.get(key) for key in ["case", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(
            {
                "status": "research_only_juejin",
                "execution_contract": {
                    "backtest_adjust": "none",
                    "slippage_each_side": 0.003,
                    "intraday_risk": False,
                    "adaptive_slippage": False,
                    "score_exit": False,
                },
                "results": frame.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
