from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
)
MANIFEST = REPORT_DIR / "pass_ratio_slice_signal_manifest_20260715.csv"
LOG_DIR = REPORT_DIR / "logs" / "slice_juejin_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_slice_juejin_results_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_slice_juejin_results_20260715.json"

RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_fw_soft_deepdrop_weight_v20260706"
    / "code_snapshot"
)
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def parse_indicator(log_text: str) -> dict | None:
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            return {"indicator_parse_error": payload[:500]}
        return parsed if isinstance(parsed, dict) else {"indicator_value": parsed}
    return None


def classify_failure(stdout: str, stderr: str, log_text: str) -> str:
    blob = "\n".join([stdout or "", stderr or "", log_text or ""])
    if '"status": 1001' in blob or "status=1001" in blob:
        return "GM_STATUS_1001_TERMINAL_UNAVAILABLE"
    if "GmError" in blob:
        return "GM_ERROR"
    if "Traceback" in blob:
        return "PYTHON_TRACEBACK"
    if "GM_BACKTEST_INDICATOR" not in blob:
        return "NO_INDICATOR"
    return "UNKNOWN"


def run_one(row: dict) -> dict:
    case = str(row["case"])
    slice_name = str(row["slice"])
    log_file = LOG_DIR / f"{case}_{slice_name}.log"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(row["slice_signal"]),
        "--log-file",
        str(log_file),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    log_text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    indicator = parse_indicator(log_text)
    result = dict(row)
    result.update(
        {
            "returncode": int(proc.returncode),
            "log_file": str(log_file),
            "failure_reason": "" if proc.returncode == 0 and indicator else classify_failure(proc.stdout, proc.stderr, log_text),
            "stdout_tail": (proc.stdout or "")[-1000:],
            "stderr_tail": (proc.stderr or "")[-1000:],
        }
    )
    if indicator:
        result.update(indicator)
    return result


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    rows: list[dict] = []
    for _, item in manifest.iterrows():
        result = run_one(item.to_dict())
        rows.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "slice": result.get("slice"),
                    "returncode": result.get("returncode"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "failure_reason": result.get("failure_reason"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
