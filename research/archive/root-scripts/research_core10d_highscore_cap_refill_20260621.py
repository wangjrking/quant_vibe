from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from research_list_age_current_best_probe_20260620 import (
    END_DATE,
    JUEJIN_PYTHON,
    MAIN,
    MARKET_DB,
    PRED_DB,
    START_DATE,
    STRATEGY_DIR,
    TABLE_10D,
    _extract_indicator,
    _exposure_stats,
    _load_rows,
    _safe,
    _signal_stats,
    _to_float,
)
from selection_module import SelectionConfig


REPORT_DIR = (
    Path(__file__).resolve().parents[2]
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "core10d_highscore_cap_refill_20260621"
)


GRID = []
for max_pred_prob in (0.40, 0.45, 0.50, 0.55, 0.60):
    GRID.append(
        {
            "max_pred_prob": max_pred_prob,
            "min_pred_prob": None,
            "confirm_rank_min": 0.50,
            "min_close_rate": 0.965,
            "max_close_rate": 1.085,
            "max_total_mv": 200000.0,
            "top_k": 5,
            "holding_days": 5,
            "target_total_pct": 0.98,
            "weight_mode": "equal",
        }
    )
for max_pred_prob in (0.45, 0.50, 0.55):
    for min_pred_prob in (0.01, 0.02):
        GRID.append(
            {
                "max_pred_prob": max_pred_prob,
                "min_pred_prob": min_pred_prob,
                "confirm_rank_min": 0.50,
                "min_close_rate": 0.965,
                "max_close_rate": 1.085,
                "max_total_mv": 200000.0,
                "top_k": 5,
                "holding_days": 5,
                "target_total_pct": 0.98,
                "weight_mode": "equal",
            }
        )
for confirm_rank_min in (0.45, 0.55, 0.60):
    GRID.append(
        {
            "max_pred_prob": 0.50,
            "min_pred_prob": None,
            "confirm_rank_min": confirm_rank_min,
            "min_close_rate": 0.965,
            "max_close_rate": 1.085,
            "max_total_mv": 200000.0,
            "top_k": 5,
            "holding_days": 5,
            "target_total_pct": 0.98,
            "weight_mode": "equal",
        }
    )
for min_close_rate in (0.955, 0.96, 0.9675, 0.97):
    GRID.append(
        {
            "max_pred_prob": 0.50,
            "min_pred_prob": None,
            "confirm_rank_min": 0.50,
            "min_close_rate": min_close_rate,
            "max_close_rate": 1.085,
            "max_total_mv": 200000.0,
            "top_k": 5,
            "holding_days": 5,
            "target_total_pct": 0.98,
            "weight_mode": "equal",
        }
    )


def _filtered_rows(
    rows: list[dict],
    *,
    max_pred_prob: float,
    min_pred_prob: float | None,
    confirm_rank_min: float,
    min_close_rate: float,
    max_close_rate: float,
) -> list[dict]:
    selected = []
    for row in rows:
        pred_prob = _to_float(row.get("pred_prob"))
        if pred_prob is None or pred_prob > max_pred_prob:
            continue
        if min_pred_prob is not None and pred_prob < min_pred_prob:
            continue
        rank5d = _to_float(row.get("rank5d"))
        if rank5d is None or rank5d < confirm_rank_min:
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < min_close_rate or close_rate > max_close_rate:
            continue
        selected.append(row)
    return selected


def _write_signal(rows: list[dict], signal_file: Path, *, params: dict) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    config = SelectionConfig(
        top_k=int(params["top_k"]),
        pred_col="pred_prob",
        min_pred_prob=None,
        max_atr_ratio=None,
        max_total_mv=float(params["max_total_mv"]),
        max_per_industry=999,
        exclude_bj=True,
        exclude_st=True,
        exclude_delisting=True,
        exclude_current_limit=True,
    )
    signals = build_gm_signal_rows(
        rows,
        config=config,
        market_rows_by_trade_date=market_rows,
        holding_days=int(params["holding_days"]),
        max_positions=5,
        weight_mode=str(params["weight_mode"]),
        target_total_pct=float(params["target_total_pct"]),
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        signal["rank5d"] = source.get("rank5d")
        signal["close_rate"] = source.get("close_rate")
        signal["total_mv"] = source.get("total_mv")
        signal["list_age_days"] = source.get("list_age_days")
    write_gm_signals_csv(signals, signal_file)


def _run_backtest(signal_file: Path, log_file: Path, *, params: dict) -> int:
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
        }
    )
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        "5",
        "--holding-days",
        str(params["holding_days"]),
        "--max-holding-days",
        str(params["holding_days"]),
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
    log_file.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return int(payload.get("returncode", proc.returncode))
        except Exception:
            pass
    return proc.returncode


def _result_row(params: dict, signal_file: Path, log_file: Path, returncode: int) -> dict:
    indicator = _extract_indicator(log_file)
    row = {
        **params,
        "slippage_ratio": 0.0015,
        "returncode": returncode,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        **_signal_stats(signal_file),
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
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    results = []
    for params in GRID:
        label = (
            f"cap{_safe(params['max_pred_prob'])}_min{_safe(params['min_pred_prob'])}"
            f"_c{_safe(params['confirm_rank_min'])}"
            f"_cr{_safe(params['min_close_rate'])}_{_safe(params['max_close_rate'])}"
        )
        signal_file = REPORT_DIR / "signals" / f"{label}.csv"
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        if not log_file.exists():
            filtered = _filtered_rows(
                rows,
                max_pred_prob=float(params["max_pred_prob"]),
                min_pred_prob=params["min_pred_prob"],
                confirm_rank_min=float(params["confirm_rank_min"]),
                min_close_rate=float(params["min_close_rate"]),
                max_close_rate=float(params["max_close_rate"]),
            )
            _write_signal(filtered, signal_file, params=params)
            returncode = _run_backtest(signal_file, log_file, params=params)
        else:
            returncode = 0
        row = _result_row(params, signal_file, log_file, returncode)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)

    fieldnames = list(results[0].keys())
    outputs = {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_qualified_target_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0
            and float(row.get("sharpe") or -999) >= 3.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for name, data in outputs.items():
        with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
