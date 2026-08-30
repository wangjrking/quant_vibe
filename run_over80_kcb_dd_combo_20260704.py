from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
RUN_DIR = OUT_DIR / "over80_kcb_dd_combo_grid"
SUMMARY = OUT_DIR / "over80_kcb_dd_combo_summary.csv"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = [
    (
        "kcb_boost130_no_dd",
        OUT_DIR / "over80_market_board_signals/kcb_boost130_s105.csv",
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "kcb_boost130_dd_08_14",
        OUT_DIR / "over80_market_board_signals/kcb_boost130_s105.csv",
        0.08,
        0.14,
        0.04,
        0.85,
        0.65,
    ),
    (
        "kcb_boost160_dd_08_14",
        OUT_DIR / "over80_market_board_signals/kcb_boost160_s108.csv",
        0.08,
        0.14,
        0.04,
        0.85,
        0.65,
    ),
    (
        "main_down80_kcb180_dd_08_14",
        OUT_DIR / "over80_market_board_signals/main_down80_kcb180_s115.csv",
        0.08,
        0.14,
        0.04,
        0.85,
        0.65,
    ),
]


def parse_indicator(log_text: str) -> dict[str, object]:
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker not in line:
            continue
        raw = line.split(marker, 1)[1].strip()
        parsed: dict[str, object] = {}
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
            match = re.search(rf"'{key}': ([0-9eE+\-.]+)", raw)
            if match:
                value = float(match.group(1))
                parsed[key] = int(value) if key.endswith("_count") else value
        if parsed:
            return parsed
        text = re.sub(r"datetime\.datetime\(.*?\)", "'datetime'", raw)
        try:
            parsed_any = ast.literal_eval(text)
        except Exception:
            return {}
        return parsed_any if isinstance(parsed_any, dict) else {}
    return {}


def run_case(case: tuple[str, Path, float | None, float | None, float | None, float | None, float | None]) -> dict[str, object]:
    name, signal_file, soft, hard, recover, soft_scale, hard_scale = case
    log_file = RUN_DIR / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "GM_SIGNAL_FILE": str(signal_file),
            "GM_MAX_POSITIONS": "2",
            "GM_HOLDING_DAYS": "2",
            "GM_TARGET_POSITION_PCT": "0.82",
            "GM_SCORE_DB": str(SCORE_DB),
            "GM_SCORE_TABLE": "score",
            "GM_MARKET_DB": str(MARKET_DB),
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.900",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.000",
            "GM_MAX_HOLDING_DAYS": "4",
            "GM_LIGHT_STOP_LOSS_PCT": "0.080",
            "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "1",
            "GM_BACKTEST_ADJUST": "none",
            "GM_BACKTEST_SLIPPAGE_RATIO": "0.0015",
            "GM_BACKTEST_START": "2022-06-07 09:00:00",
            "GM_BACKTEST_END": "2026-07-10 15:30:00",
        }
    )
    if soft is None:
        env["GM_EQUITY_DD_RISK_MODE"] = "0"
    else:
        env.update(
            {
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_SOFT_TRIGGER": str(soft),
                "GM_EQUITY_DD_HARD_TRIGGER": str(hard),
                "GM_EQUITY_DD_RECOVER_TRIGGER": str(recover),
                "GM_EQUITY_DD_SOFT_SCALE": str(soft_scale),
                "GM_EQUITY_DD_HARD_SCALE": str(hard_scale),
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
            }
        )
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(STRATEGY_DIR),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    indicator = parse_indicator(log_file.read_text(encoding="utf-8", errors="ignore"))
    return {
        "name": name,
        "signal_file": str(signal_file),
        "soft": soft,
        "hard": hard,
        "recover": recover,
        "soft_scale": soft_scale,
        "hard_scale": hard_scale,
        "returncode": proc.returncode,
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
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        print(f"RUN {case[0]}", flush=True)
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
