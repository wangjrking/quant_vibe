from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"
BASE_SIGNAL = REPORT_DIR / "open_gap_frequency_signals" / "hp_new70_turn80_top5_s170_cap27_h2m3__gap_rerank_penalty_1p5.csv"
OUT_SIGNAL_DIR = REPORT_DIR / "concentration_probe_signals"
OUT_LOG_DIR = REPORT_DIR / "concentration_probe_logs"
OUT_CSV = REPORT_DIR / "concentration_probe_20260630.csv"


CASES: list[dict[str, Any]] = [
    {"name": "top1_equal60", "topn": 1, "fixed_target": 0.60, "target_cap": 0.60},
    {"name": "top1_equal80", "topn": 1, "fixed_target": 0.80, "target_cap": 0.80},
    {"name": "top1_gap_nochase80", "topn": 1, "fixed_target": 0.80, "target_cap": 0.80, "gap_nochase": True},
    {"name": "top2_equal45", "topn": 2, "fixed_target": 0.45, "target_cap": 0.45},
    {"name": "top2_equal50", "topn": 2, "fixed_target": 0.50, "target_cap": 0.50},
    {"name": "top2_gap_nochase50", "topn": 2, "fixed_target": 0.50, "target_cap": 0.50, "gap_nochase": True},
]


SELL_RULE = {
    "holding_days": 2,
    "max_holding_days": 3,
    "score_exit": 0.98,
    "score_continue": 0.99,
    "min_score_exit_days": 1,
    "day_drop_ratio": 0.995,
    "max_daily_sells": 1,
}


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _target_for_case(row: dict[str, Any], case: dict[str, Any]) -> float:
    target = float(case["fixed_target"])
    gap = _float(row.get("buy_open_gap"))
    if case.get("gap_nochase") and gap is not None:
        if gap >= 0.02:
            target *= 0.50
        elif gap <= -0.08:
            target *= 0.70
        elif -0.03 <= gap <= -0.005:
            target *= 1.10
    return max(0.0, min(target, float(case["target_cap"])))


def _make_signal(case: dict[str, Any], base_rows: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        if int(float(row["rank"])) > int(case["topn"]):
            continue
        item = dict(row)
        item["target_pct"] = f"{_target_for_case(row, case):.5f}"
        item["holding_days"] = str(SELL_RULE["holding_days"])
        item["max_holding_days"] = str(SELL_RULE["max_holding_days"])
        item["score_exit_entry_ratio"] = f"{SELL_RULE['score_exit']:.5f}"
        item["score_continue_entry_ratio"] = f"{SELL_RULE['score_continue']:.5f}"
        item["min_holding_days_before_score_exit"] = str(SELL_RULE["min_score_exit_days"])
        item["strategy_variant"] = str(case["name"])
        item["dynamic_hold_name"] = "h2m3_e098_c099_daydrop995_ms1"
        item["filter_name"] = str(case["name"])
        rows.append(item)
    rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, rows)
    counts = Counter(row["signal_date"] for row in rows)
    return output, {
        "signal_rows": len(rows),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < int(case["topn"])),
    }


def _run(case: dict[str, Any], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case, base_rows)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(SELL_RULE["max_daily_sells"]),
                "GM_LIGHT_STOP_LOSS_PCT": "none",
                "GM_LOG_EXPOSURE": "1",
                "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
                "GM_SYNC_POSITIONS": "1",
                "GM_CASH_BUFFER": "0.99",
                "GM_VERBOSE_TRADES": "1",
                "GM_FORCE_SELL_MARKET_ORDER": "0",
                "GM_FORCE_BUY_MARKET_ORDER": "0",
                "GM_INTRADAY_RISK_MODE": "0",
                "GM_INTRADAY_REPLACE_BUY": "0",
                "GM_EQUITY_DD_RISK_MODE": "0",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(SELL_RULE["min_score_exit_days"]),
                "GM_SCORE_EXIT_ENTRY_RATIO": str(SELL_RULE["score_exit"]),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(SELL_RULE["score_continue"]),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": str(SELL_RULE["day_drop_ratio"]),
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
                "GM_RESIZE_HELD_ON_SIGNAL": "0",
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
                "GM_INDEX_RISK_EXIT_MODE": "0",
                "GM_BREADTH_RISK_EXIT_MODE": "0",
            }
        )
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(int(case["topn"])),
            "--holding-days",
            str(SELL_RULE["holding_days"]),
            "--max-holding-days",
            str(SELL_RULE["max_holding_days"]),
            "--target-position-pct",
            str(float(case["target_cap"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            "2022-06-07 09:00:00",
            "--backtest-end",
            "2026-06-29 15:30:00",
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        **case,
        **signal_meta,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "note": "research-only concentration probe; no production parameter change",
    }


def main() -> None:
    if not BASE_SIGNAL.exists():
        raise FileNotFoundError(BASE_SIGNAL)
    base_rows = list(csv.DictReader(BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="")))
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case, base_rows)
        results.append(row)
        _write_rows(OUT_CSV, results)
        print(row)


if __name__ == "__main__":
    main()

