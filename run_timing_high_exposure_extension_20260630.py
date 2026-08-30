from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import duckdb
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
SCORE_DB = REPORT_DIR / "scores" / "timing1d_scores.duckdb"
OUT_SIGNAL_DIR = REPORT_DIR / "timing_high_exposure_extension_signals"
OUT_LOG_DIR = REPORT_DIR / "timing_high_exposure_extension_logs"
OUT_CSV = REPORT_DIR / "timing_high_exposure_extension_20260630.csv"
OUT_JSON = REPORT_DIR / "timing_high_exposure_extension_20260630.json"


BASES: dict[str, dict[str, Any]] = {
    "timing55": {
        "signal_file": REPORT_DIR / "timing1d_signals" / "timing_55_30_10_05_g1p4_top5_dyn.csv",
        "score_table": "score_timing_55_30_10_05_g1p4_top5_dyn",
    },
    "timing3545": {
        "signal_file": REPORT_DIR / "timing1d_signals" / "timing_35_45_10_10_g1p4_top5_dyn.csv",
        "score_table": "score_timing_35_45_10_10_g1p4_top5_dyn",
    },
    "skip50": {
        "signal_file": REPORT_DIR / "timing1d_signals" / "timing_base_1d_skip50_top5_dyn.csv",
        "score_table": "score_timing_base_1d_skip50_top5_dyn",
    },
}


CASES: list[dict[str, Any]] = [
    {
        "name": "thex_t55_top5_s260_cap42_h2m3",
        "base": "timing55",
        "topn": 5,
        "target_scale": 2.60,
        "cap": 0.42,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
    },
    {
        "name": "thex_t55_top5_s300_cap48_h2m3",
        "base": "timing55",
        "topn": 5,
        "target_scale": 3.00,
        "cap": 0.48,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
    },
    {
        "name": "thex_t55_top5_s300_cap48_h1m2",
        "base": "timing55",
        "topn": 5,
        "target_scale": 3.00,
        "cap": 0.48,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 1,
        "gap_mode": "none",
    },
    {
        "name": "thex_t55_top3_s320_cap55_h2m3",
        "base": "timing55",
        "topn": 3,
        "target_scale": 3.20,
        "cap": 0.55,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
    },
    {
        "name": "thex_t3545_top5_s260_cap42_h2m3",
        "base": "timing3545",
        "topn": 5,
        "target_scale": 2.60,
        "cap": 0.42,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
    },
    {
        "name": "thex_t3545_top5_s300_cap48_h1m2",
        "base": "timing3545",
        "topn": 5,
        "target_scale": 3.00,
        "cap": 0.48,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 1,
        "gap_mode": "none",
    },
    {
        "name": "thex_skip50_top5_s300_cap48_h2m3_gap",
        "base": "skip50",
        "topn": 5,
        "target_scale": 3.00,
        "cap": 0.48,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "scale",
    },
    {
        "name": "thex_t55_top5_s340_cap55_h2m3_gap",
        "base": "timing55",
        "topn": 5,
        "target_scale": 3.40,
        "cap": 0.55,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "scale",
    },
    {
        "name": "thex_t55_top5_s500_cap80_h2m3_ddon",
        "base": "timing55",
        "topn": 5,
        "target_scale": 5.00,
        "cap": 0.80,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
        "equity_dd_risk_mode": 1,
    },
    {
        "name": "thex_t55_top5_s500_cap80_h2m3_ddoff",
        "base": "timing55",
        "topn": 5,
        "target_scale": 5.00,
        "cap": 0.80,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
        "equity_dd_risk_mode": 0,
    },
    {
        "name": "thex_t55_top1_s700_cap90_h2m3_ddon",
        "base": "timing55",
        "topn": 1,
        "target_scale": 7.00,
        "cap": 0.90,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
        "equity_dd_risk_mode": 1,
    },
    {
        "name": "thex_t55_top1_s700_cap90_h1m2_ddon",
        "base": "timing55",
        "topn": 1,
        "target_scale": 7.00,
        "cap": 0.90,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 1,
        "gap_mode": "none",
        "equity_dd_risk_mode": 1,
    },
    {
        "name": "thex_t55_top5_s1000_cap99_h2m3_ddoff",
        "base": "timing55",
        "topn": 5,
        "target_scale": 10.00,
        "cap": 0.99,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
        "equity_dd_risk_mode": 0,
    },
    {
        "name": "thex_t55_top3_s1000_cap99_h2m3_ddoff",
        "base": "timing55",
        "topn": 3,
        "target_scale": 10.00,
        "cap": 0.99,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "gap_mode": "none",
        "equity_dd_risk_mode": 0,
    },
]


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


def _market_cache(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    keys = {(str(row["stock_code"]), str(row["signal_date"])) for row in rows}
    keys.update((str(row["stock_code"]), str(row.get("buy_date") or "")) for row in rows)
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        out: dict[tuple[str, str], dict[str, float]] = {}
        for stock_code, trade_date in sorted(key for key in keys if key[1]):
            row = con.execute(
                "select open, close, pre_close, list_date from STOCK_DAILY_DATA where stock_code=? and trade_date=?",
                (stock_code, trade_date),
            ).fetchone()
            if not row:
                continue
            open_price, close_price, pre_close, list_date = row
            item: dict[str, float] = {}
            if open_price is not None:
                item["open"] = float(open_price)
            if close_price is not None:
                item["close"] = float(close_price)
            if pre_close is not None:
                item["pre_close"] = float(pre_close)
            if list_date:
                try:
                    signal_dt = datetime_module.datetime.strptime(trade_date, "%Y%m%d")
                    list_dt = datetime_module.datetime.strptime(str(list_date), "%Y%m%d")
                    item["list_age_days"] = float((signal_dt - list_dt).days)
                except ValueError:
                    pass
            out[(stock_code, trade_date)] = item
        return out
    finally:
        con.close()


def _base_target(row: dict[str, Any]) -> float:
    rank_1d = _float(row.get("rank_1d")) or 0.0
    rank_3d = _float(row.get("rank_3d")) or 0.0
    rank_5d = _float(row.get("rank_5d")) or 0.0
    if rank_3d < 0.45 or rank_5d < 0.45:
        target = 0.09
    elif rank_3d < 0.65 or rank_5d < 0.65:
        target = 0.12
    else:
        target = 0.15
    if rank_1d < 0.50:
        target *= 0.4
    return target


def _target_pct(row: dict[str, Any], case: dict[str, Any], market: dict[tuple[str, str], dict[str, float]]) -> float:
    target = _base_target(row) * float(case["target_scale"])
    stock_code = str(row["stock_code"])
    signal_date = str(row["signal_date"])
    buy_date = str(row.get("buy_date") or "")
    atr = _float(row.get("atr_qfq"))
    turnover = _float(row.get("turnover_rate")) or 0.0
    list_age_days = market.get((stock_code, signal_date), {}).get("list_age_days")
    if atr is None or list_age_days is None or list_age_days < 60:
        target *= 0.75
    elif turnover >= 80.0:
        target *= 0.90
    if case.get("gap_mode") == "scale" and buy_date:
        sig = market.get((stock_code, signal_date), {})
        buy = market.get((stock_code, buy_date), {})
        gap = None
        if sig.get("close") and buy.get("open"):
            gap = buy["open"] / sig["close"] - 1.0
        if gap is not None:
            if gap >= 0.03:
                target *= 0.35
            elif gap >= 0.02:
                target *= 0.55
            elif gap <= -0.06:
                target *= 0.55
            elif -0.03 <= gap <= -0.005:
                target *= 1.10
    return max(0.0, min(target, float(case["cap"])))


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = BASES[str(case["base"])]
    base_rows = list(csv.DictReader(base["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    market = _market_cache(base_rows)
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        if int(float(row["rank"])) > int(case["topn"]):
            continue
        item = dict(row)
        item["target_pct"] = f"{_target_pct(row, case, market):.5f}"
        item["holding_days"] = str(int(case["holding_days"]))
        item["max_holding_days"] = str(int(case["max_holding_days"]))
        item["score_exit_entry_ratio"] = f"{float(case['score_exit']):.5f}"
        item["score_continue_entry_ratio"] = f"{float(case['score_continue']):.5f}"
        item["min_holding_days_before_score_exit"] = "1"
        item["strategy_variant"] = str(case["name"])
        item["filter_name"] = str(case["name"])
        item["dynamic_hold_name"] = str(case["name"])
        rows.append(item)
    rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, rows)
    counts = Counter(row["signal_date"] for row in rows)
    return output, {
        "signal_rows": len(rows),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < int(case["topn"])),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        base = BASES[str(case["base"])]
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(int(case["max_daily_sells"])),
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
                "GM_EQUITY_DD_RISK_MODE": str(int(case.get("equity_dd_risk_mode", 1))),
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
                "GM_EQUITY_DD_SOFT_SCALE": "0.80",
                "GM_EQUITY_DD_HARD_SCALE": "0.60",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": str(float(case["score_exit"])),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(case["score_continue"])),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.995",
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
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["max_holding_days"])),
            "--target-position-pct",
            str(float(case["cap"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            str(base["score_table"]),
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
        "score_db": str(SCORE_DB),
        "score_table": BASES[str(case["base"])]["score_table"],
        "log_file": str(log_file),
        "note": "research-only timing high exposure extension; production unchanged",
    }


def main() -> None:
    for base in BASES.values():
        if not base["signal_file"].exists():
            raise FileNotFoundError(base["signal_file"])
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

