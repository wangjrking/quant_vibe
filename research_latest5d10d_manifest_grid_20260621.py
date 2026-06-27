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
from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "latest5d10d_direction_probe_20260621"
)


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none")


GRID: list[dict] = []
for score_mode in ("rank5d", "inv10d", "blend5d80", "blend5d80_inv10d"):
    for rank5d_min in (0.0,):
        for mv_name, min_total_mv, max_total_mv in (
            ("mvlt200", None, 200000.0),
            ("mv120_200", 120000.0, 200000.0),
        ):
            for max_positions in (8, 9):
                continue_ratio, max_holding_days = (1.0, 10)
                GRID.append(
                    {
                        "name": (
                            f"{score_mode}_r5ge{_safe(rank5d_min)}_{mv_name}"
                            f"_mp{max_positions}_listage60_cont{_safe(continue_ratio)}"
                        ),
                        "score_mode": score_mode,
                        "rank5d_min": rank5d_min,
                        "min_total_mv": min_total_mv,
                        "max_total_mv": max_total_mv,
                        "min_close_rate": 0.955,
                        "max_close_rate": 1.085,
                        "min_list_age_days": 60,
                        "top_k": 5,
                        "holding_days": 7,
                        "max_positions": max_positions,
                        "target_total_pct": 0.98,
                        "continue_ratio": continue_ratio,
                        "max_holding_days": max_holding_days,
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


def _score_value(row: dict, score_mode: str) -> float | None:
    rank10d = _to_float(row.get("rank10d"))
    rank5d = _to_float(row.get("rank5d_new"))
    if rank10d is None or rank5d is None:
        return None
    if score_mode == "rank10d":
        return rank10d
    if score_mode == "rank5d":
        return rank5d
    if score_mode == "inv10d":
        return 1.0 - rank10d
    if score_mode == "blend20":
        return 0.80 * rank10d + 0.20 * rank5d
    if score_mode == "blend35":
        return 0.65 * rank10d + 0.35 * rank5d
    if score_mode == "blend5d80":
        return 0.20 * rank10d + 0.80 * rank5d
    if score_mode == "blend5d80_inv10d":
        return 0.20 * (1.0 - rank10d) + 0.80 * rank5d
    raise ValueError(f"unsupported score_mode: {score_mode}")


def _prepare_rows(all_rows: list[dict], params: dict) -> list[dict]:
    selected: list[dict] = []
    for row in all_rows:
        rank5d = _to_float(row.get("rank5d_new"))
        if rank5d is None or rank5d < float(params["rank5d_min"]):
            continue
        total_mv = _to_float(row.get("total_mv"))
        min_total_mv = params.get("min_total_mv")
        max_total_mv = params.get("max_total_mv")
        if min_total_mv is not None and (total_mv is None or total_mv < float(min_total_mv)):
            continue
        if max_total_mv is not None and (total_mv is None or total_mv > float(max_total_mv)):
            continue
        close_rate = _to_float(row.get("close_rate"))
        if close_rate is None or close_rate < float(params["min_close_rate"]) or close_rate > float(params["max_close_rate"]):
            continue
        list_age_days = _to_float(row.get("list_age_days"))
        if list_age_days is None or list_age_days < float(params["min_list_age_days"]):
            continue
        score = _score_value(row, str(params["score_mode"]))
        if score is None:
            continue
        enriched = dict(row)
        enriched["strategy_score"] = score
        selected.append(enriched)
    return selected


def _write_signal(rows: list[dict], signal_file: Path, *, params: dict) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, base.START_DATE, base.END_DATE)
    config = SelectionConfig(
        top_k=int(params["top_k"]),
        pred_col="strategy_score",
        min_pred_prob=None,
        max_atr_ratio=None,
        max_total_mv=float(params["max_total_mv"]) if params.get("max_total_mv") is not None else None,
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
        for field in ("rank10d", "rank5d_new", "strategy_score", "close_rate", "total_mv", "list_age_days"):
            signal[field] = source.get(field)
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


def _run_backtest(signal_file: Path, log_file: Path, *, params: dict) -> int:
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
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(params["continue_ratio"] if params.get("continue_ratio") is not None else 999),
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
        str(params["holding_days"]),
        "--max-holding-days",
        str(params["max_holding_days"]),
        "--target-position-pct",
        "0.20",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        base.TABLE_10D,
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
            _write_signal(_prepare_rows(rows, params), signal_file, params=params)
        returncode = _run_backtest(signal_file, log_file, params=params)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **params,
            "table_10d": base.TABLE_10D,
            "table_5d_confirm": base.TABLE_5D_NEW,
            "slippage_ratio": 0.0015,
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
