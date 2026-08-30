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
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"

BASE_DIR = REPORT_DIR / "formal_score_frequency_followup_signals"
OUT_SIGNAL_DIR = REPORT_DIR / "formal_next_open_gap_followup_signals"
OUT_LOG_DIR = REPORT_DIR / "formal_next_open_gap_followup_logs"
OUT_CSV = REPORT_DIR / "formal_next_open_gap_followup_20260630.csv"
OUT_JSON = REPORT_DIR / "formal_next_open_gap_followup_20260630.json"


BASES = {
    "top3_h2m3_scoreexit": {
        "path": BASE_DIR / "top3_cool2d18_formal_h2m3_ms1.csv",
        "topn": 3,
        "target_cap": 0.25,
        "holding_days": 2,
        "max_holding_days": 3,
        "open_daily_score_exit": 1,
        "max_daily_sells": 1,
    },
    "top3_h2m3_no_scoreexit": {
        "path": BASE_DIR / "top3_cool2d18_no_scoreexit_h2m3_ms1.csv",
        "topn": 3,
        "target_cap": 0.25,
        "holding_days": 2,
        "max_holding_days": 3,
        "open_daily_score_exit": 0,
        "max_daily_sells": 1,
    },
    "dyn_h1m2_scoreexit": {
        "path": BASE_DIR / "dyn_mild_formal_h1m2_ms1.csv",
        "topn": 5,
        "target_cap": 0.15,
        "holding_days": 1,
        "max_holding_days": 2,
        "open_daily_score_exit": 1,
        "max_daily_sells": 1,
    },
}


CASES: list[dict[str, Any]] = [
    {
        "name": "top3_gap_scale_up2",
        "base": "top3_h2m3_scoreexit",
        "mode": "scale_by_gap",
        "up_cut": 0.02,
        "up_scale": 0.50,
        "deep_down_cut": -0.06,
        "deep_down_scale": 0.50,
        "mid_down_low": -0.03,
        "mid_down_high": 0.0,
        "mid_down_scale": 1.15,
    },
    {
        "name": "top3_gap_scale_up3",
        "base": "top3_h2m3_scoreexit",
        "mode": "scale_by_gap",
        "up_cut": 0.03,
        "up_scale": 0.35,
        "deep_down_cut": -0.06,
        "deep_down_scale": 0.60,
        "mid_down_low": -0.03,
        "mid_down_high": 0.0,
        "mid_down_scale": 1.20,
    },
    {
        "name": "top3_gap_skip_up3",
        "base": "top3_h2m3_scoreexit",
        "mode": "skip_up_gap",
        "up_cut": 0.03,
    },
    {
        "name": "top3_noscore_gap_scale_up2",
        "base": "top3_h2m3_no_scoreexit",
        "mode": "scale_by_gap",
        "up_cut": 0.02,
        "up_scale": 0.50,
        "deep_down_cut": -0.06,
        "deep_down_scale": 0.50,
        "mid_down_low": -0.03,
        "mid_down_high": 0.0,
        "mid_down_scale": 1.15,
    },
    {
        "name": "dyn_gap_scale_up2",
        "base": "dyn_h1m2_scoreexit",
        "mode": "scale_by_gap",
        "up_cut": 0.02,
        "up_scale": 0.50,
        "deep_down_cut": -0.06,
        "deep_down_scale": 0.50,
        "mid_down_low": -0.03,
        "mid_down_high": 0.0,
        "mid_down_scale": 1.15,
    },
    {
        "name": "dyn_gap_skip_up3",
        "base": "dyn_h1m2_scoreexit",
        "mode": "skip_up_gap",
        "up_cut": 0.03,
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


def _load_market(keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    if not keys:
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        for stock_code, trade_date in keys:
            row = con.execute(
                "select open, close, pre_close from STOCK_DAILY_DATA where stock_code=? and trade_date=?",
                (stock_code, trade_date),
            ).fetchone()
            if row is None:
                continue
            out[(stock_code, trade_date)] = {
                "open": float(row[0]) if row[0] is not None else math.nan,
                "close": float(row[1]) if row[1] is not None else math.nan,
                "pre_close": float(row[2]) if row[2] is not None else math.nan,
            }
    finally:
        con.close()
    return out


def _enrich_with_gap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        stock_code = str(row.get("stock_code", ""))
        keys.add((stock_code, str(row.get("signal_date", ""))))
        keys.add((stock_code, str(row.get("buy_date", ""))))
    market = _load_market(keys)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        stock_code = str(row.get("stock_code", ""))
        signal_m = market.get((stock_code, str(row.get("signal_date", ""))))
        buy_m = market.get((stock_code, str(row.get("buy_date", ""))))
        gap = None
        if signal_m and buy_m:
            signal_close = signal_m["close"]
            buy_open = buy_m["open"]
            if math.isfinite(signal_close) and signal_close > 0 and math.isfinite(buy_open):
                gap = buy_open / signal_close - 1.0
                item["buy_open_gap"] = f"{gap:.8f}"
        if gap is None:
            item["buy_open_gap"] = ""
        enriched.append(item)
    return enriched


def _scaled_target(row: dict[str, Any], case: dict[str, Any], target_cap: float) -> float:
    target = _float(row.get("target_pct")) or 0.0
    gap = _float(row.get("buy_open_gap"))
    if gap is None:
        return min(target, target_cap)
    if gap >= float(case.get("up_cut", 99.0)):
        target *= float(case.get("up_scale", 1.0))
    elif gap <= float(case.get("deep_down_cut", -99.0)):
        target *= float(case.get("deep_down_scale", 1.0))
    elif float(case.get("mid_down_low", -99.0)) <= gap <= float(case.get("mid_down_high", 99.0)):
        target *= float(case.get("mid_down_scale", 1.0))
    return max(0.0, min(target, target_cap))


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = BASES[str(case["base"])]
    base_rows = list(csv.DictReader(Path(base["path"]).open("r", encoding="utf-8-sig", newline="")))
    rows = _enrich_with_gap(base_rows)
    out_rows: list[dict[str, Any]] = []
    gap_counts = Counter()
    skipped_up = 0
    missing_gap = 0
    for row in rows:
        item = dict(row)
        gap = _float(item.get("buy_open_gap"))
        if gap is None:
            missing_gap += 1
        elif gap >= 0.03:
            gap_counts["up3"] += 1
        elif gap >= 0.02:
            gap_counts["up2"] += 1
        elif gap <= -0.06:
            gap_counts["deep_down"] += 1
        elif -0.03 <= gap <= 0:
            gap_counts["mid_down"] += 1
        else:
            gap_counts["other"] += 1

        if case["mode"] == "skip_up_gap" and gap is not None and gap >= float(case["up_cut"]):
            skipped_up += 1
            continue
        if case["mode"] == "scale_by_gap":
            item["target_pct"] = f"{_scaled_target(item, case, float(base['target_cap'])):.5f}"

        item["holding_days"] = str(int(base["holding_days"]))
        item["max_holding_days"] = str(int(base["max_holding_days"]))
        item["strategy_variant"] = str(case["name"])
        item["filter_name"] = str(case["name"])
        item["dynamic_hold_name"] = str(case["name"])
        out_rows.append(item)

    out_rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, out_rows)
    day_counts = Counter(row["signal_date"] for row in out_rows)
    below_target_days = sum(1 for count in day_counts.values() if count < int(base["topn"]))
    return output, {
        "signal_rows": len(out_rows),
        "signal_days": len(day_counts),
        "below_target_days": below_target_days,
        "skipped_up": skipped_up,
        "missing_gap": missing_gap,
        "up3_rows": gap_counts["up3"],
        "up2_rows": gap_counts["up2"],
        "deep_down_rows": gap_counts["deep_down"],
        "mid_down_rows": gap_counts["mid_down"],
        "other_gap_rows": gap_counts["other"],
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
                "GM_OPEN_DAILY_SCORE_EXIT": str(int(base["open_daily_score_exit"])),
                "GM_MAX_DAILY_SELLS": str(int(base["max_daily_sells"])),
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
                "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
                "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.98",
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.99",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
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
            str(int(base["topn"])),
            "--holding-days",
            str(int(base["holding_days"])),
            "--max-holding-days",
            str(int(base["max_holding_days"])),
            "--target-position-pct",
            str(float(base["target_cap"])),
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
        "log_file": str(log_file),
        "score_db": str(SCORE_DB),
        "score_table": SCORE_TABLE,
        "note": "research-only formal next-open gap follow-up; production unchanged",
    }


def main() -> None:
    for base in BASES.values():
        if not Path(base["path"]).exists():
            raise FileNotFoundError(base["path"])
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

