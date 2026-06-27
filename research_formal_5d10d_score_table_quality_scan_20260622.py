from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from export_formal_5d10d_l5_signals import (
    _feature_lookup,
    _load_rows,
    _pre_filter_rows,
    _signal_day_features,
    _to_float,
)
from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_score_table_quality_scan_20260622"
)
PROD_DIR = MAIN / "strategy_library" / "production" / "prod_formal_5d10d_best_v20260621"
SCORE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_day_pool_replace_20260621"
    / "weak_day_pool_scores.db"
)
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
    / "fusion_5d10d.db"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
START_DATE = "20240604"
END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


SCORE_TABLES = {
    "current_f60_liq": "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5",
    "fallback150": "weak_fallback150_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_10000p0_0p3",
    "primary80": "weak_primary80_primary_count_le1_prank_10d80_5d20_frank_10d60_5d40_10000p0_0p3",
    "pmv100": "weak_pmv100_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_10000p0_0p3",
    "f50_liq": "weak_f50_liq_primary_count_le1_prank_10d90_5d10_frank_10d50_5d50_20000p0_0p5",
}

BASE_SELECTION = {
    "exclude_bj": True,
    "exclude_st": True,
    "exclude_delisting": True,
    "exclude_current_limit": True,
    "top_k_after_base_pool": 5,
    "rank_max": 5,
    "min_pred_10d": 0.0,
}

RANK_TARGET = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}

VARIANTS: list[dict] = []
for label in ["current_f60_liq", "fallback150", "primary80", "pmv100", "f50_liq"]:
    for primary_count_min in [1, 2]:
        for avg_min in [1.95, 2.00, 2.03, 2.05, 2.08]:
            VARIANTS.append(
                {
                    "name": f"{label}_pc{primary_count_min}_avg{str(avg_min).replace('.', 'p')}",
                    "score_label": label,
                    "score_table": SCORE_TABLES[label],
                    "primary_count_min": primary_count_min,
                    "post_filter_avg_pred_min": avg_min,
                    "holding_days": 7,
                    "max_holding_days": 10,
                    "max_positions": 5,
                    "target_position_pct": 0.21,
                }
            )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_signals(variant: dict, output: Path) -> None:
    rows = _pre_filter_rows(
        _load_rows(SCORE_DB, variant["score_table"], FUSION_DB, START_DATE, END_DATE),
        BASE_SELECTION,
    )
    feature_by_key = _feature_lookup(rows)
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    base_signals = build_gm_signal_rows(
        rows,
        config=SelectionConfig(
            top_k=int(BASE_SELECTION["top_k_after_base_pool"]),
            pred_col="pred_prob",
            min_pred_prob=None,
            min_pred_quantile=None,
            max_atr_ratio=None,
            min_amount=None,
            min_turnover_rate=None,
            max_total_mv=None,
            max_per_industry=999999,
            exclude_bj=True,
            exclude_st=True,
            exclude_delisting=True,
            exclude_current_limit=True,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=int(variant["holding_days"]),
        max_positions=int(variant["max_positions"]),
        weight_mode="equal",
        target_total_pct=1.0,
    )
    features = _signal_day_features(base_signals)
    filtered: list[dict] = []
    for row in base_signals:
        signal_date = str(row.get("signal_date") or "")
        day_features = features.get(signal_date, {})
        rank = int(float(row.get("rank") or 999999))
        feature_row = feature_by_key.get((signal_date, str(row.get("stock_code") or "")), {})
        pred_10d = _to_float(feature_row.get("pred_10d"))
        avg_pred = _to_float(day_features.get("avg_pred"))
        if int(day_features.get("primary_count") or 0) < int(variant["primary_count_min"]):
            continue
        if rank > int(BASE_SELECTION["rank_max"]):
            continue
        if pred_10d is None or pred_10d < float(BASE_SELECTION["min_pred_10d"]):
            continue
        if avg_pred is None or avg_pred < float(variant["post_filter_avg_pred_min"]):
            continue
        output_row = dict(row)
        output_row["target_pct"] = f"{RANK_TARGET.get(rank, 0.0):.5f}"
        output_row["score_exit_entry_ratio"] = "1.0"
        output_row["min_holding_days_before_score_exit"] = "3"
        output_row["max_daily_sells"] = "1"
        output_row["score_continue_entry_ratio"] = "1.02"
        output_row["weak_mode"] = variant["score_label"]
        output_row["pred_10d"] = feature_row.get("pred_10d")
        output_row["pred_5d"] = feature_row.get("pred_5d")
        filtered.append(output_row)
    write_gm_signals_csv(filtered, output)


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
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


def _run_backtest(variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
            "GM_MAX_DAILY_SELLS": "1",
            "GM_STOP_LOSS_PCT": "0.08",
            "GM_TAKE_PROFIT_PCT": "none",
            "GM_LIGHT_STOP_LOSS_PCT": "none",
            "GM_SCORE_EXIT_RANK": "none",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.99",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.90",
            "GM_EQUITY_DD_HARD_SCALE": "0.70",
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
        str(int(variant["max_positions"])),
        "--holding-days",
        str(int(variant["holding_days"])),
        "--max-holding-days",
        str(int(variant["max_holding_days"])),
        "--target-position-pct",
        f"{float(variant['target_position_pct']):.5f}",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        variant["score_table"],
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
        "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sort_by_annual(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("annual") or -999.0),
        float(row.get("sharpe") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _sort_by_sharpe(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _build_signals(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            **variant,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(row)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')} signals={row.get('signal_count')}",
            flush=True,
        )
    _write_rows(REPORT_DIR / "summary.csv", results)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
    target = [
        row
        for row in results
        if float(row.get("annual") or -999.0) >= 3.0
        and float(row.get("sharpe") or -999.0) >= 4.0
        and float(row.get("avg_invested_pct") or -999.0) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_target_hits.csv", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
