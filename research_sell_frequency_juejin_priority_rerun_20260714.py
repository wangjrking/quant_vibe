from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
PRIORITY_CSV = REPORT_DIR / "sell_frequency_l1_proxy_priority_for_juejin_20260714.csv"
OUT_CSV = REPORT_DIR / "sell_frequency_priority_juejin_rerun_results_20260714.csv"
OUT_JSON = REPORT_DIR / "sell_frequency_priority_juejin_rerun_results_20260714.json"
STATUS_JSON = REPORT_DIR / "sell_frequency_priority_juejin_rerun_status_20260714.json"
LOG_DIR = REPORT_DIR / "logs" / "sell_frequency_priority_juejin_rerun"

RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"


def can_read_market_db() -> tuple[bool, str | None]:
    try:
        con = duckdb.connect(str(MARKET_DB), read_only=True)
        con.execute("select 1").fetchone()
        con.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


def extract_indicator(log_text: str):
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except Exception:
                return None
    return None


def max_positions(signal_file: Path) -> int:
    df = pd.read_csv(signal_file, encoding="utf-8-sig", dtype={"buy_date": str, "stock_code": str})
    if df.empty:
        return 1
    return int(df.groupby("buy_date")["stock_code"].nunique().max())


def run_case(row: dict) -> dict:
    signal_file = Path(row["signal_file"])
    hold_days = int(row["hold_days_proxy"])
    case = str(row["case"])
    log_file = LOG_DIR / f"{case}_h{hold_days}.log"
    mp = max_positions(signal_file)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--max-positions",
        str(mp),
        "--holding-days",
        str(hold_days),
        "--max-holding-days",
        str(hold_days),
        "--score-exit-entry-ratio",
        "0.96",
        "--min-holding-days-before-score-exit",
        str(min(hold_days, 1)),
        "--score-continue-entry-ratio",
        "9.99",
        "--backtest-slippage-ratio",
        "0.0030",
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    indicator = None
    if log_file.exists():
        indicator = extract_indicator(log_file.read_text(encoding="utf-8", errors="ignore"))
    out = dict(row)
    out.update(
        {
            "max_positions": mp,
            "juejin_holding_days": hold_days,
            "returncode": proc.returncode,
            "log_file": str(log_file),
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }
    )
    if isinstance(indicator, dict):
        out.update(indicator)
    else:
        out["indicator_error"] = "missing"
    return out


def main() -> None:
    ok, err = can_read_market_db()
    if not ok:
        status = {
            "status": "blocked",
            "blocked_step": "market_db_read_precheck",
            "market_db": str(MARKET_DB),
            "error": err,
            "next_action": "wait for L2 integration process to release DuckDB, then rerun this script",
        }
        STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False))
        return

    rows: list[dict] = []
    with PRIORITY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    results: list[dict] = []
    for row in rows:
        result = run_case(row)
        results.append(result)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "hold_days": result.get("juejin_holding_days"),
                    "returncode": result.get("returncode"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    status = {
        "status": "completed",
        "cases": len(results),
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "log_dir": str(LOG_DIR),
    }
    STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
