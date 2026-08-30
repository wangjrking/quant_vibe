from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
LOG_DIR = REPORT_DIR / "logs" / "lbc_slice_stability"
OUT_CSV = REPORT_DIR / "lbc_slice_stability_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "lbc_slice_stability_juejin_results_20260714.json"


CASES = {
    "lbc_q0b150_nondeep": REPORT_DIR / "signals" / "lowbucket_compensation" / "lbc_q0b150_nondeep.csv",
    "lbc_q0b180_pred98": REPORT_DIR / "signals" / "lowbucket_compensation" / "lbc_q0b180_pred98.csv",
    "ogd_gap1x50_deep8up110": REPORT_DIR
    / "signals"
    / "open_gap_deep_rebalance"
    / "ogd_gap1x50_deep8up110.csv",
    "ogd_deep5_up105": REPORT_DIR / "signals" / "open_gap_deep_rebalance" / "ogd_deep5_up105.csv",
}

PERIODS = [
    ("2022h2", "2022-08-12 09:00:00", "2022-12-31 15:30:00"),
    ("2023", "2023-01-01 09:00:00", "2023-12-31 15:30:00"),
    ("2024", "2024-01-01 09:00:00", "2024-12-31 15:30:00"),
    ("2025", "2025-01-01 09:00:00", "2025-12-31 15:30:00"),
    ("2026ytd", "2026-01-01 09:00:00", "2026-07-24 15:30:00"),
]


def extract_indicator(text: str) -> dict:
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(text.splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            out: dict[str, float | int | str] = {"raw_indicator": payload}
            for key in [
                "pnl_ratio",
                "pnl_ratio_annual",
                "sharp_ratio",
                "max_drawdown",
                "risk_ratio",
                "win_ratio",
                "calmar_ratio",
            ]:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    out[key] = float(match.group(1))
            for key in ["open_count", "close_count", "win_count", "lose_count"]:
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    out[key] = int(match.group(1))
            return out
    return {}


def run_one(case: str, signal_file: Path, period: str, start: str, end: str) -> dict:
    log_file = LOG_DIR / f"{case}_{period}.log"
    if log_file.exists() and log_file.stat().st_size > 0:
        ind = extract_indicator(log_file.read_text(encoding="utf-8", errors="ignore"))
        if ind.get("pnl_ratio_annual") is not None:
            return {
                "case": case,
                "period": period,
                "start": start,
                "end": end,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                "returncode": 0,
                **ind,
            }
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
        "--backtest-start",
        start,
        "--backtest-end",
        end,
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    ind = extract_indicator(log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else "")
    return {
        "case": case,
        "period": period,
        "start": start,
        "end": end,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "returncode": proc.returncode,
        **ind,
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case, signal_file in CASES.items():
        for period, start, end in PERIODS:
            row = run_one(case, signal_file, period, start, end)
            results.append(row)
            print(
                json.dumps(
                    {
                        "case": case,
                        "period": period,
                        "pnl_ratio_annual": row.get("pnl_ratio_annual"),
                        "sharp_ratio": row.get("sharp_ratio"),
                        "max_drawdown": row.get("max_drawdown"),
                        "open_count": row.get("open_count"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "rows": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
