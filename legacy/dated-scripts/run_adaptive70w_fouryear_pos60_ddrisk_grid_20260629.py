from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
SIGNAL_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_strict_neighbor_20260629"
CASE_KEY = os.environ.get(
    "STRATEGY_CASE_KEY",
    "w72_23_05_amt150_mv30__hold3m5_c097_e096_ddtight_pos65__strict_amt150000_mv300000_pc150_pos60",
)
TARGET_PCT = os.environ.get("STRATEGY_TARGET_PCT", "0.60")
OUT_DIR = Path(os.environ.get(
    "STRATEGY_DDRISK_OUT_DIR",
    str(DATA / "reports" / "strategy_agent_adaptive70w_fouryear_pos60_ddrisk_20260629"),
))
REPORT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_fullwindow_20260629"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
SIGNAL_FILE = SIGNAL_DIR / "signals" / f"{CASE_KEY}.csv"
SCORE_TABLE = f"score_{CASE_KEY}"

SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-26 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-26 15:30:00"),
    ("recent60", "2026-03-25 09:00:00", "2026-06-26 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-26 15:30:00"),
]

VARIANTS = [
    {
        "name": "dd_base",
        "mode": "1",
        "soft_trigger": "0.06",
        "hard_trigger": "0.10",
        "recover": "0.03",
        "soft_scale": "0.65",
        "hard_scale": "0.45",
    },
    {
        "name": "dd_loose",
        "mode": "1",
        "soft_trigger": "0.08",
        "hard_trigger": "0.14",
        "recover": "0.04",
        "soft_scale": "0.80",
        "hard_scale": "0.60",
    },
    {
        "name": "dd_mid_a",
        "mode": "1",
        "soft_trigger": "0.07",
        "hard_trigger": "0.12",
        "recover": "0.035",
        "soft_scale": "0.75",
        "hard_scale": "0.55",
    },
    {
        "name": "dd_mid_b",
        "mode": "1",
        "soft_trigger": "0.08",
        "hard_trigger": "0.12",
        "recover": "0.04",
        "soft_scale": "0.75",
        "hard_scale": "0.55",
    },
    {
        "name": "dd_mid_c",
        "mode": "1",
        "soft_trigger": "0.07",
        "hard_trigger": "0.14",
        "recover": "0.035",
        "soft_scale": "0.80",
        "hard_scale": "0.60",
    },
    {
        "name": "dd_mild",
        "mode": "1",
        "soft_trigger": "0.10",
        "hard_trigger": "0.16",
        "recover": "0.05",
        "soft_scale": "0.85",
        "hard_scale": "0.70",
    },
    {
        "name": "dd_off",
        "mode": "0",
        "soft_trigger": "0.06",
        "hard_trigger": "0.10",
        "recover": "0.03",
        "soft_scale": "1.00",
        "hard_scale": "1.00",
    },
]


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


def _base_env(variant: dict[str, str], sell_mult: str) -> dict[str, str]:
    return {
        "GM_OPEN_DAILY_SCORE_EXIT": "1",
        "GM_MAX_DAILY_SELLS": "0",
        "GM_LIGHT_STOP_LOSS_PCT": "none",
        "GM_LOG_EXPOSURE": "1",
        "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
        "GM_SYNC_POSITIONS": "1",
        "GM_CASH_BUFFER": "0.99",
        "GM_VERBOSE_TRADES": "1",
        "GM_FORCE_SELL_MARKET_ORDER": "0",
        "GM_FORCE_BUY_MARKET_ORDER": "0",
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_INTRADAY_REPLACE_BUY": "0",
        "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
        "GM_EQUITY_DD_RISK_MODE": variant["mode"],
        "GM_EQUITY_DD_RESIZE_EXISTING": "0",
        "GM_EQUITY_DD_SOFT_TRIGGER": variant["soft_trigger"],
        "GM_EQUITY_DD_HARD_TRIGGER": variant["hard_trigger"],
        "GM_EQUITY_DD_RECOVER_TRIGGER": variant["recover"],
        "GM_EQUITY_DD_SOFT_SCALE": variant["soft_scale"],
        "GM_EQUITY_DD_HARD_SCALE": variant["hard_scale"],
        "GM_EQUITY_DD_STRICT_WHEN_DRAWDOWN": os.environ.get("STRATEGY_DD_STRICT_WHEN_DRAWDOWN", "0"),
        "GM_EQUITY_DD_STRICT_TRIGGER": os.environ.get("STRATEGY_DD_STRICT_TRIGGER", variant["soft_trigger"]),
        "GM_EQUITY_DD_STRICT_SOFT_TRIGGER": os.environ.get("STRATEGY_DD_STRICT_SOFT_TRIGGER", "0.06"),
        "GM_EQUITY_DD_STRICT_HARD_TRIGGER": os.environ.get("STRATEGY_DD_STRICT_HARD_TRIGGER", "0.10"),
        "GM_EQUITY_DD_STRICT_RECOVER_TRIGGER": os.environ.get("STRATEGY_DD_STRICT_RECOVER_TRIGGER", "0.03"),
        "GM_EQUITY_DD_STRICT_SOFT_SCALE": os.environ.get("STRATEGY_DD_STRICT_SOFT_SCALE", "0.65"),
        "GM_EQUITY_DD_STRICT_HARD_SCALE": os.environ.get("STRATEGY_DD_STRICT_HARD_SCALE", "0.45"),
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.97",
        "GM_SCORE_EXIT_ENTRY_RATIO": "0.96",
        "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
        "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
        "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
        "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
        "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
        "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
        "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
        "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": sell_mult,
    }


def _run(variant: dict[str, str], sell_mult: str, tag: str, start: str, end: str) -> dict[str, Any]:
    mult_name = sell_mult.replace(".", "p")
    log_file = OUT_DIR / "logs" / f"{CASE_KEY}__{variant['name']}__sell{mult_name}__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(_base_env(variant, sell_mult))
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir", str(STRATEGY_DIR),
            "--signal-file", str(SIGNAL_FILE),
            "--log-file", str(log_file),
            "--max-positions", "1",
            "--holding-days", "3",
            "--max-holding-days", "5",
            "--target-position-pct", TARGET_PCT,
            "--score-db", str(SCORE_DB),
            "--score-table", SCORE_TABLE,
            "--market-db", str(MARKET_DB),
            "--backtest-start", start,
            "--backtest-end", end,
            "--backtest-adjust", "none",
            "--backtest-initial-cash", "600000",
            "--backtest-slippage-ratio", "0",
            "--stop-loss-pct", "0.05",
            "--take-profit-pct", "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "variant": variant["name"],
        "sell_mult": sell_mult,
        "slice": tag,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def main() -> None:
    rows = []
    for variant in VARIANTS:
        for sell_mult in ["0.0", "0.25"]:
            for tag, start, end in SLICES:
                rows.append(_run(variant, sell_mult, tag, start, end))
                _write_rows(OUT_DIR / "ddrisk_detail.csv", rows)
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        item = summary.setdefault((row["variant"], row["sell_mult"]), {"variant": row["variant"], "sell_mult": row["sell_mult"]})
        for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count", "log_file"]:
            item[f"{row['slice']}_{key}"] = row.get(key)
    _write_rows(OUT_DIR / "ddrisk_summary.csv", list(summary.values()))
    print("wrote", OUT_DIR / "ddrisk_summary.csv")


if __name__ == "__main__":
    main()

