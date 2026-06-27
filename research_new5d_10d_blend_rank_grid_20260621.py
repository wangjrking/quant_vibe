from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import research_10d_with_new5d_confirm_probe_20260621 as base
from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR, TABLE_10D
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_10d_blend_rank_grid_20260621"
)


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none")


GRID: list[dict] = []
for weight_5d in (0.20, 0.35, 0.50):
    for rank5d_min in (0.35, 0.45):
        for max_pred_prob in (None, 0.60):
            for holding_days in (5, 7):
                GRID.append(
                    {
                        "name": (
                            f"w5d{_safe(weight_5d)}_r5min{_safe(rank5d_min)}"
                            f"_cap{_safe(max_pred_prob)}_h{holding_days}"
                        ),
                        "weight_5d": weight_5d,
                        "rank5d_min": rank5d_min,
                        "max_pred_prob": max_pred_prob,
                        "min_close_rate": 0.955,
                        "max_close_rate": 1.085,
                        "max_total_mv": 200000.0,
                        "min_amount": None,
                        "top_k": 5,
                        "holding_days": holding_days,
                        "max_positions": 6,
                        "target_total_pct": 0.98,
                    }
                )


def _to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_rank10d(rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["trade_date"]), []).append(row)
    for day_rows in grouped.values():
        day_rows.sort(key=lambda row: (-float(row.get("pred_prob") or -999), str(row.get("stock_code", ""))))
        denom = max(len(day_rows) - 1, 1)
        for idx, row in enumerate(day_rows):
            row["rank10d"] = 1.0 - (idx / denom)


def _prepare_rows(params: dict, all_rows: list[dict]) -> list[dict]:
    weight_5d = float(params["weight_5d"])
    weight_10d = 1.0 - weight_5d
    selected = []
    for row in all_rows:
        pred_prob = _to_float(row.get("pred_prob"))
        rank10d = _to_float(row.get("rank10d"))
        rank5d = _to_float(row.get("rank5d_new"))
        if pred_prob is None or rank10d is None or rank5d is None:
            continue
        if rank5d < float(params["rank5d_min"]):
            continue
        max_pred_prob = params.get("max_pred_prob")
        if max_pred_prob is not None and pred_prob > float(max_pred_prob):
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < float(params["min_close_rate"]) or close_rate > float(params["max_close_rate"]):
            continue
        total_mv = _to_float(row.get("total_mv"))
        if total_mv is None or total_mv > float(params["max_total_mv"]):
            continue
        min_amount = params.get("min_amount")
        amount = _to_float(row.get("amount"))
        if min_amount is not None and (amount is None or amount < float(min_amount)):
            continue
        enriched = dict(row)
        enriched["blend_score"] = weight_10d * rank10d + weight_5d * rank5d
        selected.append(enriched)
    return selected


def _write_signal(rows: list[dict], signal_file: Path, *, params: dict) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, base.START_DATE, base.END_DATE)
    config = SelectionConfig(
        top_k=int(params["top_k"]),
        pred_col="blend_score",
        min_pred_prob=None,
        max_atr_ratio=None,
        min_amount=params.get("min_amount"),
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
        max_positions=int(params["max_positions"]),
        weight_mode="equal",
        target_total_pct=float(params["target_total_pct"]),
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        signal["rank10d"] = source.get("rank10d")
        signal["rank5d_new"] = source.get("rank5d_new")
        signal["raw_10d_pred_prob"] = source.get("pred_prob")
        signal["close_rate"] = source.get("close_rate")
        signal["amount"] = source.get("amount")
        signal["total_mv"] = source.get("total_mv")
        signal["list_age_days"] = source.get("list_age_days")
    write_gm_signals_csv(signals, signal_file)


def _extract_indicator(log_file: Path) -> dict | None:
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


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
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


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _run(signal_file: Path, log_file: Path, params: dict) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
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
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "999",
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
        str(params["max_positions"]),
        "--holding-days",
        "5",
        "--max-holding-days",
        "8",
        "--target-position-pct",
        str(float(params["target_total_pct"]) / float(params["max_positions"])),
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--score-continue-entry-ratio",
        "999",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = base._load_10d_rows_with_new5d_rank()
    _add_rank10d(rows)
    results = []
    for params in GRID:
        signal_file = REPORT_DIR / "signals" / f"{params['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{params['name']}.log"
        if not signal_file.exists():
            _write_signal(_prepare_rows(params, rows), signal_file, params=params)
        returncode = _run(signal_file, log_file, params)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **params,
            "table_10d": TABLE_10D,
            "table_5d": base.TABLE_5D_NEW,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)

    fieldnames = list(results[0].keys())
    outputs = {
        "summary.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
        "summary_qualified_target_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 2.0
            and float(row.get("sharpe") or -999) >= 3.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for filename, data in outputs.items():
        with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
