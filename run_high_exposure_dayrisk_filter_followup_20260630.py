from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import duckdb
import subprocess
from collections import Counter, defaultdict
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
SCORE_TABLE = "score_timing_55_30_10_05_g1p4_top5_dyn"

OUT_SIGNAL_DIR = REPORT_DIR / "high_exposure_dayrisk_filter_signals"
OUT_LOG_DIR = REPORT_DIR / "high_exposure_dayrisk_filter_logs"
OUT_CSV = REPORT_DIR / "high_exposure_dayrisk_filter_followup_20260630.csv"
OUT_JSON = REPORT_DIR / "high_exposure_dayrisk_filter_followup_20260630.json"


BASES: dict[str, dict[str, Any]] = {
    "top3_full": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top3_s1000_cap99_h2m3_ddoff.csv",
        "topn": 3,
        "max_positions": 3,
    },
    "top5_full": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top5_s1000_cap99_h2m3_ddoff.csv",
        "topn": 5,
        "max_positions": 5,
    },
}


CASES: list[dict[str, Any]] = [
    {
        "name": "hef_top3_drop_scale",
        "base": "top3_full",
        "mode": "drop_scale",
    },
    {
        "name": "hef_top3_deep_skip_drop_scale",
        "base": "top3_full",
        "mode": "deep_skip_drop_scale",
    },
    {
        "name": "hef_top3_combo_soft",
        "base": "top3_full",
        "mode": "combo_soft",
    },
    {
        "name": "hef_top3_combo_hard",
        "base": "top3_full",
        "mode": "combo_hard",
    },
    {
        "name": "hef_top3_dayrisk_soft",
        "base": "top3_full",
        "mode": "dayrisk_soft",
    },
    {
        "name": "hef_top3_dayrisk_combo",
        "base": "top3_full",
        "mode": "dayrisk_combo",
    },
    {
        "name": "hef_top5_combo_soft",
        "base": "top5_full",
        "mode": "combo_soft",
    },
    {
        "name": "hef_top5_combo_hard",
        "base": "top5_full",
        "mode": "combo_hard",
    },
    {
        "name": "hef_top5_dayrisk_soft",
        "base": "top5_full",
        "mode": "dayrisk_soft",
    },
    {
        "name": "hef_top5_dayrisk_combo",
        "base": "top5_full",
        "mode": "dayrisk_combo",
    },
    {
        "name": "hef_top3_combo_soft_resize",
        "base": "top3_full",
        "mode": "combo_soft",
        "resize_held": 1,
    },
    {
        "name": "hef_top3_dayrisk_combo_resize",
        "base": "top3_full",
        "mode": "dayrisk_combo",
        "resize_held": 1,
    },
    {
        "name": "hef_top5_combo_hard_resize",
        "base": "top5_full",
        "mode": "combo_hard",
        "resize_held": 1,
    },
    {
        "name": "hef_top5_dayrisk_combo_resize",
        "base": "top5_full",
        "mode": "dayrisk_combo",
        "resize_held": 1,
    },
    {
        "name": "hef_top3_fast_exit_h1m2_e0995",
        "base": "top3_full",
        "mode": "none",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
        "score_day_drop": 0.99,
    },
    {
        "name": "hef_top3_fast_exit_h1m2_e100",
        "base": "top3_full",
        "mode": "none",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 1.0,
        "score_continue": 1.01,
        "max_daily_sells": 3,
        "score_day_drop": 0.99,
    },
    {
        "name": "hef_top3_fast_exit_combo_h1m2",
        "base": "top3_full",
        "mode": "combo_soft",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
        "score_day_drop": 0.99,
    },
    {
        "name": "hef_top5_fast_exit_h1m2_e0995",
        "base": "top5_full",
        "mode": "none",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
        "score_day_drop": 0.99,
    },
    {
        "name": "hef_top5_fast_exit_combo_h1m2",
        "base": "top5_full",
        "mode": "combo_soft",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
        "score_day_drop": 0.99,
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


def _buy_gap(row: dict[str, Any], market: dict[tuple[str, str], dict[str, float]]) -> float | None:
    stock_code = str(row.get("stock_code") or "")
    signal_date = str(row.get("signal_date") or "")
    buy_date = str(row.get("buy_date") or "")
    sig = market.get((stock_code, signal_date), {})
    buy = market.get((stock_code, buy_date), {})
    sig_close = sig.get("close")
    buy_open = buy.get("open")
    if sig_close is None or buy_open is None or sig_close <= 0:
        return None
    return buy_open / sig_close - 1.0


def _day_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row.get("signal_date") or "")].append(row)
    out: dict[str, dict[str, float]] = {}
    for date, items in by_date.items():
        pct_values = [_float(row.get("pct_chg")) for row in items]
        two_day_values = [_float(row.get("two_day_ret")) for row in items]
        pct_clean = [value for value in pct_values if value is not None]
        two_day_clean = [value for value in two_day_values if value is not None]
        out[date] = {
            "avg_pct_chg": sum(pct_clean) / len(pct_clean) if pct_clean else math.nan,
            "avg_two_day_ret": sum(two_day_clean) / len(two_day_clean) if two_day_clean else math.nan,
            "min_pct_chg": min(pct_clean) if pct_clean else math.nan,
            "min_two_day_ret": min(two_day_clean) if two_day_clean else math.nan,
        }
    return out


def _target_scale(row: dict[str, Any], day: dict[str, float], gap: float | None, mode: str) -> tuple[float, str]:
    pct = _float(row.get("pct_chg"))
    prev_pct = _float(row.get("prev_pct_chg"))
    two_day = _float(row.get("two_day_ret"))
    amount = _float(row.get("amount"))
    total_mv = _float(row.get("total_mv"))
    turnover = _float(row.get("turnover_rate"))
    atr = _float(row.get("atr_qfq"))
    scale = 1.0
    tags: list[str] = []

    def apply(cond: bool, factor: float, tag: str) -> None:
        nonlocal scale
        if cond:
            scale *= factor
            tags.append(tag)

    if mode in {"drop_scale", "deep_skip_drop_scale", "combo_soft", "combo_hard", "dayrisk_combo"}:
        apply(pct is not None and pct <= -4.0, 0.55, "pct_drop4")
        apply(pct is not None and pct <= -8.0, 0.55, "pct_drop8")
        apply(two_day is not None and two_day <= -0.06, 0.60, "twoday_drop6")
        apply(two_day is not None and two_day <= -0.10, 0.60, "twoday_drop10")
        apply(prev_pct is not None and prev_pct <= -8.0 and pct is not None and pct <= -2.0, 0.70, "serial_drop")

    if mode in {"deep_skip_drop_scale", "combo_hard"}:
        if (pct is not None and pct <= -10.0) or (two_day is not None and two_day <= -0.13):
            return 0.0, "skip_deep_drop"

    if mode in {"combo_soft", "combo_hard", "dayrisk_combo"}:
        apply(gap is not None and gap >= 0.03, 0.55, "gap_up3")
        apply(gap is not None and gap >= 0.05, 0.60, "gap_up5")
        apply(gap is not None and gap <= -0.06, 0.65, "gap_down6")
        apply(amount is not None and amount < 300000, 0.70, "amount_lt30w")
        apply(total_mv is not None and total_mv < 500000, 0.75, "mv_lt50w")
        apply(turnover is not None and turnover >= 15.0, 0.80, "turnover_ge15")
        apply(atr is None, 0.80, "atr_missing")

    if mode in {"dayrisk_soft", "dayrisk_combo"}:
        avg_pct = day.get("avg_pct_chg", math.nan)
        avg_two = day.get("avg_two_day_ret", math.nan)
        min_two = day.get("min_two_day_ret", math.nan)
        apply(math.isfinite(avg_pct) and avg_pct <= -4.0, 0.45, "day_avg_pct_drop4")
        apply(math.isfinite(avg_pct) and -4.0 < avg_pct <= -2.0, 0.70, "day_avg_pct_drop2")
        apply(math.isfinite(avg_two) and avg_two <= -0.06, 0.55, "day_avg_twoday_drop6")
        apply(math.isfinite(min_two) and min_two <= -0.12, 0.80, "day_min_twoday_drop12")

    return max(0.0, min(scale, 1.25)), "|".join(tags) if tags else "base"


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = BASES[str(case["base"])]
    rows = list(csv.DictReader(base["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    market = _market_cache(rows)
    day_map = _day_stats(rows)
    output_rows: list[dict[str, Any]] = []
    skipped = 0
    tag_counter: Counter[str] = Counter()
    for row in rows:
        gap = _buy_gap(row, market)
        scale, tag = _target_scale(row, day_map.get(str(row.get("signal_date") or ""), {}), gap, str(case["mode"]))
        tag_counter[tag] += 1
        if scale <= 0:
            skipped += 1
            continue
        item = dict(row)
        base_target = _float(row.get("target_pct")) or 0.0
        item["target_pct"] = f"{min(0.99, max(0.0, base_target * scale)):.5f}"
        item["holding_days"] = str(int(case.get("holding_days", 2)))
        item["max_holding_days"] = str(int(case.get("max_holding_days", 3)))
        item["score_exit_entry_ratio"] = f"{float(case.get('score_exit', 0.98)):.5f}"
        item["score_continue_entry_ratio"] = f"{float(case.get('score_continue', 0.99)):.5f}"
        item["min_holding_days_before_score_exit"] = "1"
        item["strategy_variant"] = str(case["name"])
        item["filter_name"] = str(case["name"])
        item["dynamic_hold_name"] = str(case["name"])
        item["risk_scale"] = f"{scale:.6f}"
        item["risk_tags"] = tag
        item["buy_open_gap"] = "" if gap is None else f"{gap:.8f}"
        output_rows.append(item)

    output_rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, output_rows)
    counts = Counter(row["signal_date"] for row in output_rows)
    return output, {
        "signal_rows": len(output_rows),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < int(base["topn"])),
        "skipped_rows": skipped,
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "risk_tag_top": dict(tag_counter.most_common(8)),
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    base = BASES[str(case["base"])]
    signal_file, signal_meta = _make_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(int(case.get("max_daily_sells", 1))),
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
                "GM_EQUITY_DD_RISK_MODE": "0",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": str(float(case.get("score_exit", 0.98))),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(case.get("score_continue", 0.99))),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": str(float(case.get("score_day_drop", 0.995))),
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
                "GM_RESIZE_HELD_ON_SIGNAL": str(int(case.get("resize_held", 0))),
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
            str(int(base["max_positions"])),
            "--holding-days",
            str(int(case.get("holding_days", 2))),
            "--max-holding-days",
            str(int(case.get("max_holding_days", 3))),
            "--target-position-pct",
            "0.99",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
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
        "score_table": SCORE_TABLE,
        "log_file": str(log_file),
        "note": "research-only high-exposure day-risk filter follow-up; production unchanged",
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

