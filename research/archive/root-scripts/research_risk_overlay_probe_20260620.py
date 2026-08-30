from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import (
    JUEJIN_PYTHON,
    MAIN,
    MARKET_DB,
    PRED_DB,
    STRATEGY_DIR,
    TABLE_10D,
    _safe,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260620"
)
SIGNAL_FILE = (
    BASE_REPORT_DIR
    / "latest_10d_confirm5d_close_rate_refine_20260620"
    / "signals"
    / "tk5_h5_mv200_5d50_cr0p965_1p085.csv"
)
REPORT_DIR = BASE_REPORT_DIR / "latest_10d_risk_overlay_probe_20260620"


def _extract_indicator(log_file: Path) -> dict | None:
    marker = "GM_BACKTEST_INDICATOR:"
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats() -> dict:
    with SIGNAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _run(params: dict, log_file: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_MAX_DAILY_SELLS": "1",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_INDEX_RISK_EXIT_MODE": str(params.get("index_risk_mode", 0)),
            "GM_INDEX_RISK_CC_THRESHOLD": str(params.get("index_cc_threshold", -0.04)),
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": str(params.get("index_intraday_threshold", -0.04)),
            "GM_INDEX_RISK_BUY_SCALE": str(params.get("index_buy_scale", 1.0)),
            "GM_EQUITY_DD_RISK_MODE": str(params.get("equity_dd_mode", 0)),
            "GM_EQUITY_DD_SOFT_TRIGGER": str(params.get("equity_soft_trigger", 0.08)),
            "GM_EQUITY_DD_HARD_TRIGGER": str(params.get("equity_hard_trigger", 0.14)),
            "GM_EQUITY_DD_RECOVER_TRIGGER": str(params.get("equity_recover_trigger", 0.04)),
            "GM_EQUITY_DD_SOFT_SCALE": str(params.get("equity_soft_scale", 0.85)),
            "GM_EQUITY_DD_HARD_SCALE": str(params.get("equity_hard_scale", 0.65)),
            "GM_EQUITY_DD_RESIZE_EXISTING": str(params.get("equity_resize_existing", 0)),
        }
    )
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(SIGNAL_FILE),
        "--log-file",
        str(log_file),
        "--max-positions",
        "5",
        "--holding-days",
        "5",
        "--max-holding-days",
        "5",
        "--target-position-pct",
        "0.98",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return int(payload.get("returncode", proc.returncode))
        except Exception:
            pass
    return proc.returncode


def _row(params: dict, log_file: Path, returncode: int) -> dict:
    indicator = _extract_indicator(log_file)
    row = {
        **params,
        "returncode": returncode,
        "signal_file": str(SIGNAL_FILE),
        "log_file": str(log_file),
        **_signal_stats(),
        **_exposure_stats(log_file),
    }
    if indicator:
        row.update(
            {
                "annual": indicator.get("pnl_ratio_annual"),
                "sharpe": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
                "calmar_ratio": indicator.get("calmar_ratio"),
            }
        )
    return row


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    grid = []
    for threshold in [-0.02, -0.03, -0.04]:
        for scale in [0.50, 0.70, 0.85]:
            grid.append(
                {
                    "risk_case": "index",
                    "index_risk_mode": 1,
                    "index_cc_threshold": threshold,
                    "index_intraday_threshold": threshold,
                    "index_buy_scale": scale,
                    "equity_dd_mode": 0,
                    "equity_resize_existing": 0,
                }
            )
    for soft, hard, soft_scale, hard_scale, resize in [
        (0.08, 0.14, 0.90, 0.75, 0),
        (0.10, 0.16, 0.90, 0.75, 0),
        (0.08, 0.14, 0.85, 0.65, 0),
        (0.10, 0.16, 0.85, 0.65, 0),
        (0.08, 0.14, 0.90, 0.75, 1),
        (0.10, 0.16, 0.90, 0.75, 1),
    ]:
        grid.append(
            {
                "risk_case": "equity",
                "index_risk_mode": 0,
                "index_cc_threshold": -0.04,
                "index_intraday_threshold": -0.04,
                "index_buy_scale": 1.0,
                "equity_dd_mode": 1,
                "equity_soft_trigger": soft,
                "equity_hard_trigger": hard,
                "equity_recover_trigger": 0.04,
                "equity_soft_scale": soft_scale,
                "equity_hard_scale": hard_scale,
                "equity_resize_existing": resize,
            }
        )
    results = []
    for idx, params in enumerate(grid, start=1):
        if params["risk_case"] == "index":
            label = (
                f"idx{idx:02d}_cc{_safe(params['index_cc_threshold'])}"
                f"_sc{_safe(params['index_buy_scale'])}"
            )
        else:
            label = (
                f"eq{idx:02d}_s{_safe(params['equity_soft_trigger'])}"
                f"_h{_safe(params['equity_hard_trigger'])}"
                f"_ss{_safe(params['equity_soft_scale'])}"
                f"_hs{_safe(params['equity_hard_scale'])}"
                f"_rz{params['equity_resize_existing']}"
            )
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        returncode = 0 if log_file.exists() else _run(params, log_file)
        row = _row(params, log_file, returncode)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str))

    fieldnames = []
    for row in results:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    for row in results:
        for field in fieldnames:
            row.setdefault(field, None)
    outputs = {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_qualified_annual2_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0 and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for name, data in outputs.items():
        with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
