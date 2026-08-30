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
    / "strategy_agent_model_application_20260620"
    / "latest_10d_confirm_mv_boundary_refine_20260620"
)


def _filtered_rows(
    rows: list[dict],
    *,
    confirm_rank_min: float,
    min_close_rate: float,
    max_close_rate: float,
) -> list[dict]:
    selected = []
    for row in rows:
        rank5d = _to_float(row.get("rank5d"))
        if rank5d is None or rank5d < confirm_rank_min:
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < min_close_rate or close_rate > max_close_rate:
            continue
        selected.append(row)
    return selected


def _write_signal(rows: list[dict], signal_file: Path, *, max_total_mv: float, top_k: int = 5) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    config = SelectionConfig(
        top_k=top_k,
        pred_col="pred_prob",
        min_pred_prob=None,
        max_atr_ratio=None,
        max_total_mv=max_total_mv,
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
        holding_days=5,
        max_positions=5,
        weight_mode="equal",
        target_total_pct=0.98,
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        signal["rank5d"] = source.get("rank5d")
        signal["close_rate"] = source.get("close_rate")
        signal["total_mv"] = source.get("total_mv")
        signal["list_age_days"] = source.get("list_age_days")
    write_gm_signals_csv(signals, signal_file)


def _run_backtest(signal_file: Path, log_file: Path) -> int:
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


def _result_row(params: dict, signal_file: Path, log_file: Path, returncode: int) -> dict:
    indicator = _extract_indicator(log_file)
    row = {
        **params,
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
    grid = []
    for confirm_rank_min in [0.45, 0.50, 0.55, 0.60]:
        for max_total_mv in [150000.0, 180000.0, 200000.0, 250000.0]:
            grid.append(
                {
                    "confirm_rank_min": confirm_rank_min,
                    "min_close_rate": 0.965,
                    "max_close_rate": 1.085,
                    "max_total_mv": max_total_mv,
                    "top_k": 5,
                    "holding_days": 5,
                    "target_total_pct": 0.98,
                    "open_daily_score_exit": 0,
                    "max_daily_sells": 1,
                }
            )

    results = []
    for params in grid:
        label = (
            f"c{_safe(params['confirm_rank_min'])}_mv{int(params['max_total_mv'] / 1000)}"
            f"_cr{_safe(params['min_close_rate'])}_{_safe(params['max_close_rate'])}"
        )
        signal_file = REPORT_DIR / "signals" / f"{label}.csv"
        log_file = REPORT_DIR / "logs" / f"{label}.log"
        if not log_file.exists():
            filtered = _filtered_rows(
                rows,
                confirm_rank_min=float(params["confirm_rank_min"]),
                min_close_rate=float(params["min_close_rate"]),
                max_close_rate=float(params["max_close_rate"]),
            )
            _write_signal(filtered, signal_file, max_total_mv=float(params["max_total_mv"]), top_k=5)
            returncode = _run_backtest(signal_file, log_file)
        else:
            returncode = 0
        row = _result_row(params, signal_file, log_file, returncode)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str))

    fieldnames = list(results[0].keys())
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
