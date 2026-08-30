from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
MAIN = ROOT / "quant/main"
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
BASE_SIGNAL = OUT_DIR / "scale124_signal_mv_loss_bucket_refine_signals/sig025_mid025_missing050_s145.csv"
SIGNAL_DIR = OUT_DIR / "scale124_signal_mv_takeprofit_signals"
RUN_DIR = OUT_DIR / "scale124_signal_mv_takeprofit_grid"
SUMMARY = OUT_DIR / "scale124_signal_mv_takeprofit_summary.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "base_no_takeprofit",
        "tp": None,
        "tp_high_target": None,
        "tp_high_signal": None,
        "tp_low_amount": None,
        "stop_loss": 0.080,
    },
    {"name": "tp08_all", "tp": 0.080, "tp_high_target": None, "tp_high_signal": None, "tp_low_amount": None, "stop_loss": 0.080},
    {"name": "tp10_all", "tp": 0.100, "tp_high_target": None, "tp_high_signal": None, "tp_low_amount": None, "stop_loss": 0.080},
    {"name": "tp12_all", "tp": 0.120, "tp_high_target": None, "tp_high_signal": None, "tp_low_amount": None, "stop_loss": 0.080},
    {"name": "tp15_all", "tp": 0.150, "tp_high_target": None, "tp_high_signal": None, "tp_low_amount": None, "stop_loss": 0.080},
    {"name": "tp20_all", "tp": 0.200, "tp_high_target": None, "tp_high_signal": None, "tp_low_amount": None, "stop_loss": 0.080},
    {
        "name": "tp10_high_target",
        "tp": None,
        "tp_high_target": 0.100,
        "tp_high_signal": None,
        "tp_low_amount": None,
        "stop_loss": 0.080,
    },
    {
        "name": "tp12_high_target",
        "tp": None,
        "tp_high_target": 0.120,
        "tp_high_signal": None,
        "tp_low_amount": None,
        "stop_loss": 0.080,
    },
    {
        "name": "tp10_signal_gt5",
        "tp": None,
        "tp_high_target": None,
        "tp_high_signal": 0.100,
        "tp_low_amount": None,
        "stop_loss": 0.080,
    },
    {
        "name": "tp12_signal_gt5",
        "tp": None,
        "tp_high_target": None,
        "tp_high_signal": 0.120,
        "tp_low_amount": None,
        "stop_loss": 0.080,
    },
    {
        "name": "tp10_low_amount",
        "tp": None,
        "tp_high_target": None,
        "tp_high_signal": None,
        "tp_low_amount": 0.100,
        "stop_loss": 0.080,
    },
    {
        "name": "tp12_high_target_stop10",
        "tp": None,
        "tp_high_target": 0.120,
        "tp_high_signal": None,
        "tp_low_amount": None,
        "stop_loss": 0.100,
    },
]


def parse_indicator(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if isinstance(indicator, dict):
        return indicator
    if not isinstance(indicator, str):
        return {}
    text = indicator
    text = re.sub(r"'created_at': datetime\.datetime\(.*?\), 'updated_at':", "'created_at': 'datetime', 'updated_at':", text)
    text = re.sub(r"'updated_at': datetime\.datetime\(.*?\), 'risk_ratio':", "'updated_at': 'datetime', 'risk_ratio':", text)
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = {}
        for key in [
            "pnl_ratio_annual",
            "sharp_ratio",
            "max_drawdown",
            "pnl_ratio",
            "open_count",
            "close_count",
            "win_ratio",
            "calmar_ratio",
        ]:
            match = re.search(rf"'{key}': ([0-9eE+\-.]+)", indicator)
            if match:
                value = float(match.group(1))
                parsed[key] = int(value) if key.endswith("_count") else value
        return parsed
    return parsed if isinstance(parsed, dict) else {}


def build_signal(case: dict[str, object]) -> tuple[Path, dict[str, object]]:
    df = pd.read_csv(BASE_SIGNAL)
    for col in ["target_pct", "signal_pct_chg", "signal_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["signal_take_profit_pct"], errors="ignore")
    tp = case["tp"]
    if tp is not None:
        df["signal_take_profit_pct"] = float(tp)
    else:
        tp_series = pd.Series(pd.NA, index=df.index, dtype="Float64")
        if case["tp_high_target"] is not None:
            tp_series.loc[df["target_pct"] >= 0.75] = float(case["tp_high_target"])
        if case["tp_high_signal"] is not None:
            tp_series.loc[df["signal_pct_chg"] > 5.0] = float(case["tp_high_signal"])
        if case["tp_low_amount"] is not None:
            tp_series.loc[df["signal_amount"] <= 150000] = float(case["tp_low_amount"])
        if tp_series.notna().any():
            df["signal_take_profit_pct"] = tp_series

    signal_file = SIGNAL_DIR / f"{case['name']}.csv"
    df.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return signal_file, {
        "signal_file": str(signal_file),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "takeprofit_rows": int(df.get("signal_take_profit_pct", pd.Series(dtype=float)).notna().sum()),
        "avg_target": float(df["target_pct"].mean()),
    }


def run_case(case: dict[str, object]) -> dict[str, object]:
    signal_file, meta = build_signal(case)
    name = str(case["name"])
    log_file = RUN_DIR / f"{name}.log"
    runner_json = RUN_DIR / f"{name}.runner.json"
    if runner_json.exists() and log_file.exists():
        stdout = runner_json.read_text(encoding="utf-8", errors="ignore")
        returncode = 0
    else:
        cmd = [
            sys.executable,
            str(RUNNER),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            "2",
            "--target-position-pct",
            "0.82",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            "score",
            "--market-db",
            str(MARKET_DB),
            "--score-exit-entry-ratio",
            "0.900",
            "--min-holding-days-before-score-exit",
            "2",
            "--score-continue-entry-ratio",
            "1.000",
            "--max-holding-days",
            "4",
            "--light-stop-loss-pct",
            str(case["stop_loss"]),
            "--min-holding-days-before-light-stop",
            "1",
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")

    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "tp": case["tp"],
        "tp_high_target": case["tp_high_target"],
        "tp_high_signal": case["tp_high_signal"],
        "tp_low_amount": case["tp_low_amount"],
        "stop_loss": case["stop_loss"],
        **meta,
        "returncode": returncode,
        "log_file": str(log_file),
        "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
        "sharp_ratio": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "pnl_ratio": indicator.get("pnl_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "win_ratio": indicator.get("win_ratio"),
        "calmar_ratio": indicator.get("calmar_ratio"),
    }


def main() -> int:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        print(f"RUN {case['name']}", flush=True)
        rows.append(run_case(case))
    rows.sort(
        key=lambda r: (
            float(r.get("sharp_ratio") or -999),
            float(r.get("pnl_ratio_annual") or -999),
        ),
        reverse=True,
    )
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
