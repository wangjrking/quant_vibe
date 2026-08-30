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
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"

OUT_SIGNAL_DIR = REPORT_DIR / "bucket_gap_frequency_signals"
OUT_LOG_DIR = REPORT_DIR / "bucket_gap_frequency_logs"
OUT_CSV = REPORT_DIR / "bucket_gap_frequency_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "bucket_gap_frequency_probe_20260630.json"


BASES: list[dict[str, Any]] = [
    {
        "base": "top5_cool2d10_high_annual_sharpe",
        "signal_file": REPORT_DIR / "signals" / "w72_23_05_amt150_mv30_top5_pos15_cool2d10_h2m3_e097_c098_daydrop99.csv",
        "target_position_pct": 0.15,
        "cap": 0.15,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "sell_slippage_mult": 0.0,
    },
    {
        "base": "dyn_mild_09_12_15",
        "signal_file": REPORT_DIR / "dynamic_target_signals" / "dyn_mild_09_12_15.csv",
        "target_position_pct": 0.15,
        "cap": 0.15,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "sell_slippage_mult": 0.0,
    },
]


CASES: list[dict[str, Any]] = [
    {
        "name": "identity_baseline_h2m3",
        "bad_scale": 1.0,
        "moderate_bad_scale": 1.0,
        "good_scale": 1.0,
        "good_cap": 0.15,
        "gap_up_scale": 1.0,
        "gap_down_bad_scale": 1.0,
        "normal_gap_scale": 1.0,
    },
    {
        "name": "bucket_bad35_good115_h2m3",
        "bad_scale": 0.35,
        "moderate_bad_scale": 0.70,
        "good_scale": 1.15,
        "good_cap": 0.18,
    },
    {
        "name": "bucket_bad25_good120_h2m3",
        "bad_scale": 0.25,
        "moderate_bad_scale": 0.60,
        "good_scale": 1.20,
        "good_cap": 0.18,
    },
    {
        "name": "bucket_bad50_only_h2m3",
        "bad_scale": 0.50,
        "moderate_bad_scale": 0.75,
        "good_scale": 1.00,
        "good_cap": 0.15,
    },
    {
        "name": "gap_bad35_chase50_h2m3",
        "gap_up_scale": 0.50,
        "gap_down_bad_scale": 0.35,
        "normal_gap_scale": 1.05,
        "good_cap": 0.17,
    },
    {
        "name": "combo_bucket_gap_h2m3",
        "bad_scale": 0.35,
        "moderate_bad_scale": 0.70,
        "good_scale": 1.15,
        "good_cap": 0.18,
        "gap_up_scale": 0.60,
        "gap_down_bad_scale": 0.35,
    },
    {
        "name": "combo_bucket_gap_h1m2_fast",
        "bad_scale": 0.35,
        "moderate_bad_scale": 0.70,
        "good_scale": 1.15,
        "good_cap": 0.18,
        "gap_up_scale": 0.60,
        "gap_down_bad_scale": 0.35,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 0.995,
        "max_daily_sells": 2,
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


def _load_open_gaps(signal_files: list[Path]) -> dict[tuple[str, str], float]:
    pairs: set[tuple[str, str]] = set()
    for path in signal_files:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                stock_code = str(row.get("stock_code") or "").strip()
                buy_date = str(row.get("buy_date") or "").strip()
                if stock_code and buy_date:
                    pairs.add((stock_code, buy_date))
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        out: dict[tuple[str, str], float] = {}
        for stock_code, buy_date in sorted(pairs):
            row = con.execute(
                """
                SELECT open, pre_close
                FROM STOCK_DAILY_DATA
                WHERE stock_code = ? AND trade_date = ?
                """,
                (stock_code, buy_date),
            ).fetchone()
            if not row:
                continue
            open_price, pre_close = row
            if open_price is None or pre_close in (None, 0):
                continue
            out[(stock_code, buy_date)] = float(open_price) / float(pre_close) - 1.0
        return out
    finally:
        con.close()


def _base_rows(base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(base["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    for row in rows:
        row["holding_days"] = str(int(base["holding_days"]))
        row["max_holding_days"] = str(int(base["max_holding_days"]))
        row["score_exit_entry_ratio"] = f"{float(base['score_exit']):.5f}"
        row["score_continue_entry_ratio"] = f"{float(base['score_continue']):.5f}"
        row["min_holding_days_before_score_exit"] = "1"
    return rows


def _day_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("signal_date") or "")].append(row)
    stats: dict[str, dict[str, float]] = {}
    for signal_date, items in grouped.items():
        pct_values = [_float(row.get("pct_chg")) for row in items]
        two_day_values = [_float(row.get("two_day_ret")) for row in items]
        pct_values = [value for value in pct_values if value is not None]
        two_day_values = [value for value in two_day_values if value is not None]
        stats[signal_date] = {
            "avg_pct_chg": sum(pct_values) / len(pct_values) if pct_values else 0.0,
            "avg_two_day_ret": sum(two_day_values) / len(two_day_values) if two_day_values else 0.0,
        }
    return stats


def _bucket_scale(case: dict[str, Any], day_stat: dict[str, float]) -> tuple[float, str]:
    avg_pct = day_stat["avg_pct_chg"]
    avg_two_day = day_stat["avg_two_day_ret"]
    if avg_two_day <= -0.08 and avg_pct <= -4.0:
        return float(case.get("bad_scale", 1.0)), "deep_fall"
    if avg_two_day <= -0.06 and avg_pct <= -3.0:
        return float(case.get("moderate_bad_scale", 1.0)), "moderate_fall"
    if -0.03 <= avg_two_day <= 0.05 and 0.0 <= avg_pct <= 6.0:
        return float(case.get("good_scale", 1.0)), "constructive"
    return 1.0, "neutral"


def _gap_scale(case: dict[str, Any], gap: float | None, bucket: str) -> tuple[float, str]:
    if gap is None:
        return 1.0, "gap_missing"
    if gap >= 0.03:
        return float(case.get("gap_up_scale", 1.0)), "gap_chase"
    if gap <= -0.05 and bucket in {"deep_fall", "moderate_fall"}:
        return float(case.get("gap_down_bad_scale", 1.0)), "gap_fall_confirm"
    if -0.02 <= gap <= 0.02:
        return float(case.get("normal_gap_scale", 1.0)), "gap_normal"
    return 1.0, "gap_neutral"


def _make_signal(
    base: dict[str, Any],
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    stats: dict[str, dict[str, float]],
    gaps: dict[tuple[str, str], float],
) -> tuple[Path, dict[str, Any]]:
    name = f"{base['base']}__{case['name']}"
    out: list[dict[str, Any]] = []
    scaled_rows = 0
    boosted_rows = 0
    deep_fall_rows = 0
    chase_scaled_rows = 0
    for row in rows:
        signal_date = str(row.get("signal_date") or "")
        stock_code = str(row.get("stock_code") or "")
        buy_date = str(row.get("buy_date") or "")
        day_stat = stats.get(signal_date, {"avg_pct_chg": 0.0, "avg_two_day_ret": 0.0})
        bucket_scale, bucket = _bucket_scale(case, day_stat)
        gap = gaps.get((stock_code, buy_date))
        gap_scale, gap_bucket = _gap_scale(case, gap, bucket)
        scale = bucket_scale * gap_scale
        target = _float(row.get("target_pct")) or float(base["target_position_pct"])
        cap = float(case.get("good_cap", base["cap"]))
        adjusted = min(target * scale, cap)
        item = dict(row)
        item["target_pct"] = f"{adjusted:.5f}"
        item["holding_days"] = str(int(case.get("holding_days", base["holding_days"])))
        item["max_holding_days"] = str(int(case.get("max_holding_days", base["max_holding_days"])))
        item["score_exit_entry_ratio"] = f"{float(case.get('score_exit', base['score_exit'])):.5f}"
        item["score_continue_entry_ratio"] = f"{float(case.get('score_continue', base['score_continue'])):.5f}"
        item["min_holding_days_before_score_exit"] = "1"
        item["strategy_variant"] = name
        item["filter_name"] = str(case["name"])
        item["day_avg_pct_chg"] = f"{day_stat['avg_pct_chg']:.6f}"
        item["day_avg_two_day_ret"] = f"{day_stat['avg_two_day_ret']:.8f}"
        item["day_momentum_bucket"] = bucket
        item["buy_open_gap"] = "" if gap is None else f"{gap:.8f}"
        item["buy_open_gap_bucket"] = gap_bucket
        item["bucket_gap_scale"] = f"{scale:.5f}"
        if abs(scale - 1.0) > 1e-9:
            scaled_rows += 1
        if scale > 1.0:
            boosted_rows += 1
        if bucket == "deep_fall":
            deep_fall_rows += 1
        if gap_bucket == "gap_chase" and scale < 1.0:
            chase_scaled_rows += 1
        out.append(item)

    out.sort(key=lambda item: (str(item.get("signal_date") or ""), int(float(item.get("rank") or 9999)), str(item.get("stock_code") or "")))
    path = OUT_SIGNAL_DIR / f"{name}.csv"
    _write_rows(path, out)
    counts = Counter(row["signal_date"] for row in out)
    return path, {
        "signal_rows": len(out),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < int(base["max_positions"])),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "scaled_rows": scaled_rows,
        "boosted_rows": boosted_rows,
        "deep_fall_rows": deep_fall_rows,
        "chase_scaled_rows": chase_scaled_rows,
    }


def _run(base: dict[str, Any], case: dict[str, Any], rows: list[dict[str, Any]], stats: dict[str, dict[str, float]], gaps: dict[tuple[str, str], float]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(base, case, rows, stats, gaps)
    case_key = f"{base['base']}__{case['name']}"
    log_file = OUT_LOG_DIR / f"{case_key}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(int(case.get("max_daily_sells", base["max_daily_sells"]))),
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
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
                "GM_EQUITY_DD_SOFT_SCALE": "0.80",
                "GM_EQUITY_DD_HARD_SCALE": "0.60",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": str(float(case.get("score_exit", base["score_exit"]))),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(case.get("score_continue", base["score_continue"]))),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.99",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": str(float(base["sell_slippage_mult"])),
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
            str(int(case.get("holding_days", base["holding_days"]))),
            "--max-holding-days",
            str(int(case.get("max_holding_days", base["max_holding_days"]))),
            "--target-position-pct",
            str(float(case.get("good_cap", base["cap"]))),
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
        "base": base["base"],
        "case": case["name"],
        **{key: value for key, value in case.items() if key != "name"},
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
        "log_file": str(log_file),
        "note": "research-only bucket/open-gap frequency probe; production unchanged",
    }


def main() -> None:
    for item in BASES:
        if not item["signal_file"].exists():
            raise FileNotFoundError(item["signal_file"])
    gaps = _load_open_gaps([item["signal_file"] for item in BASES])
    results: list[dict[str, Any]] = []
    for item in BASES:
        rows = _base_rows(item)
        stats = _day_stats(rows)
        for case in CASES:
            result = _run(item, case, rows, stats, gaps)
            results.append(result)
            _write_rows(OUT_CSV, results)
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
