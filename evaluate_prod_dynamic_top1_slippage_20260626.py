from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

STRATEGY_ID = "prod_dynamic_top1_amt9w_warmup60_v20260626"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / STRATEGY_ID
CODE_SNAPSHOT_DIR = STRATEGY_DIR / "code_snapshot"

SIGNAL_FILE = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round3_20260626" / "signals" / "w84_09_07_amt90_mv20__exec_c097_pos90_dd12.csv"
SCORE_DB = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round3_20260626" / "scores" / "grid_scores.db"
SCORE_TABLE = "score_w84_09_07_amt90_mv20__exec_c097_pos90_dd12"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"

REPORT_DIR = DATA / "reports" / "strategy_agent_prod_dynamic_top1_slippage_eval_20260626"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-25 15:30:00"

SLIPPAGE_CASES = [
    {"name": "s0015", "ratio": 0.0015},
    {"name": "s0030", "ratio": 0.0030},
    {"name": "s0050", "ratio": 0.0050},
    {"name": "s0065", "ratio": 0.0065},
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_FORCE_SELL_MARKET_ORDER": "1",
    "GM_INTRADAY_RISK_MODE": "1",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.085",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.12",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.78",
    "GM_EQUITY_DD_HARD_SCALE": "0.58",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.97",
}


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


def _avg_invested_pct(log_file: Path) -> float | None:
    pattern = re.compile(r"EXPOSURE\s+\d{8}\s+post_buy\s+invested_pct=([0-9.]+)")
    values: list[float] = []
    if not log_file.exists():
        return None
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    return sum(values) / len(values) if values else None


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    log_file = REPORT_DIR / "logs" / f"{STRATEGY_ID}_{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        command = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(CODE_SNAPSHOT_DIR),
            "--signal-file",
            str(SIGNAL_FILE),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            "2",
            "--max-holding-days",
            "3",
            "--target-position-pct",
            "0.9",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            BACKTEST_START,
            "--backtest-end",
            BACKTEST_END,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            str(case["ratio"]),
            "--stop-loss-pct",
            "0.06",
            "--take-profit-pct",
            "0.07",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "strategy_id": STRATEGY_ID,
        "slippage_name": case["name"],
        "slippage_ratio": case["ratio"],
        "returncode": returncode,
        "annual_return": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "avg_invested_pct": _avg_invested_pct(log_file),
        "log_file": str(log_file),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 生产策略滑点敏感性评估",
        "",
        f"策略：`{STRATEGY_ID}`",
        "口径：掘金回测，全周期 `2024-06-05` 到 `2026-06-25`，仅调整成交滑点比例。",
        "",
        "| 滑点 | 年化 | 累计收益 | 夏普 | 最大回撤 | 开仓 | 平仓 | 平均持仓率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['slippage_ratio']):.4f} | {float(row['annual_return']):.2%} | "
            f"{float(row['pnl_ratio']):.2%} | {float(row['sharpe']):.3f} | "
            f"{float(row['max_drawdown']):.2%} | {int(float(row['open_count']))} | "
            f"{int(float(row['close_count']))} | {float(row['avg_invested_pct']):.2%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = [_run_case(case) for case in SLIPPAGE_CASES]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORT_DIR / "slippage_sensitivity.csv", rows)
    _write_json = lambda p, o: p.write_text(json.dumps(o, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_json(REPORT_DIR / "slippage_sensitivity.json", rows)
    (REPORT_DIR / "slippage_sensitivity.md").write_text(_build_report(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
