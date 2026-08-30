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

OUT_SIGNAL_DIR = REPORT_DIR / "fragility_refill_optimization_signals"
OUT_LOG_DIR = REPORT_DIR / "fragility_refill_optimization_logs"
OUT_CSV = REPORT_DIR / "fragility_refill_optimization_20260630.csv"
OUT_JSON = REPORT_DIR / "fragility_refill_optimization_20260630.json"

FRAGILE_TOP1 = {"SZSE.301396"}
FRAGILE_TOP3 = {"SZSE.301396", "SZSE.301013", "SHSE.603823"}


BASES: dict[str, dict[str, Any]] = {
    "top5_s1000": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top5_s1000_cap99_h2m3_ddoff.csv",
        "target_position_pct": 0.99,
    },
    "top5_s500": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top5_s500_cap80_h2m3_ddoff.csv",
        "target_position_pct": 0.80,
    },
}


CASES: list[dict[str, Any]] = [
    {
        "name": "refill_s1000_rm301396_top3_h2m3",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "refill_s1000_rm301396_top5_h2m3",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "refill_s1000_rmtop3_top3_h2m3",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP3),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "refill_s1000_rm301396_top3_fast_h1m2",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
    },
    {
        "name": "refill_s1000_rm301396_top3_risk_h2m3",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "risk_mode": "basic",
    },
    {
        "name": "refill_s1000_rm301396_top5_risk_h2m3",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "risk_mode": "basic",
    },
    {
        "name": "refill_s1000_rm301396_top5_risk_fast_h1m2",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
        "risk_mode": "basic",
    },
    {
        "name": "refill_s1000_rm301396_top5_strict_h2m3",
        "base": "top5_s1000",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "risk_mode": "strict",
    },
    {
        "name": "refill_s500_rm301396_top3_h2m3",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "refill_s500_rm301396_top5_h2m3",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "refill_s500_rmtop3_top3_h2m3",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP3),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "refill_s500_rm301396_top3_fast_h1m2",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
    },
    {
        "name": "refill_s500_rm301396_top3_risk_h2m3",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "risk_mode": "basic",
    },
    {
        "name": "refill_s500_rm301396_top5_risk_h2m3",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "risk_mode": "basic",
    },
    {
        "name": "refill_s500_rm301396_top5_risk_fast_h1m2",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
        "risk_mode": "basic",
    },
    {
        "name": "refill_s500_rm301396_top3_strict_h2m3",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "risk_mode": "strict",
    },
    {
        "name": "refill_s500_rm301396_top5_strict_h2m3",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "risk_mode": "strict",
    },
    {
        "name": "refill_s500_rm301396_top5_strict_fast_h1m2",
        "base": "top5_s500",
        "remove_symbols": sorted(FRAGILE_TOP1),
        "select_topn": 5,
        "max_positions": 5,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
        "risk_mode": "strict",
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


def _market_day_cache(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    keys = {(str(row.get("stock_code") or ""), str(row.get("signal_date") or "")) for row in rows}
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        out: dict[tuple[str, str], dict[str, float]] = {}
        for stock_code, trade_date in sorted(key for key in keys if key[0] and key[1]):
            row = con.execute(
                """
                select amount, total_mv, turnover_rate, atr_qfq, pct_chg
                from STOCK_DAILY_DATA
                where stock_code=? and trade_date=?
                """,
                (stock_code, trade_date),
            ).fetchone()
            if not row:
                continue
            amount, total_mv, turnover, atr, pct = row
            item: dict[str, float] = {}
            for key, value in [
                ("amount", amount),
                ("total_mv", total_mv),
                ("turnover_rate", turnover),
                ("atr_qfq", atr),
                ("pct_chg", pct),
            ]:
                if value is not None:
                    item[key] = float(value)
            out[(stock_code, trade_date)] = item
        return out
    finally:
        con.close()


def _risk_scale(row: dict[str, Any], market: dict[tuple[str, str], dict[str, float]], mode: str) -> tuple[float, str]:
    if mode == "none":
        return 1.0, "base"
    stock_code = str(row.get("stock_code") or "")
    signal_date = str(row.get("signal_date") or "")
    m = market.get((stock_code, signal_date), {})
    amount = _float(row.get("amount")) or m.get("amount")
    total_mv = _float(row.get("total_mv")) or m.get("total_mv")
    turnover = _float(row.get("turnover_rate")) or m.get("turnover_rate")
    atr = _float(row.get("atr_qfq")) if row.get("atr_qfq") not in (None, "") else m.get("atr_qfq")
    pct = _float(row.get("pct_chg")) if row.get("pct_chg") not in (None, "") else m.get("pct_chg")
    two_day = _float(row.get("two_day_ret"))
    scale = 1.0
    tags: list[str] = []

    def apply(cond: bool, factor: float, tag: str) -> None:
        nonlocal scale
        if cond:
            scale *= factor
            tags.append(tag)

    if mode == "strict":
        apply(amount is not None and amount < 300000, 0.55, "amount_lt30w")
        apply(total_mv is not None and total_mv < 500000, 0.60, "mv_lt50w")
        apply(turnover is not None and turnover >= 15.0, 0.65, "turnover_ge15")
        apply(atr is None, 0.60, "atr_missing")
        apply(pct is not None and pct <= -4.0, 0.45, "pct_drop4")
        apply(two_day is not None and two_day <= -0.06, 0.50, "twoday_drop6")
    else:
        apply(amount is not None and amount < 300000, 0.70, "amount_lt30w")
        apply(total_mv is not None and total_mv < 500000, 0.75, "mv_lt50w")
        apply(turnover is not None and turnover >= 15.0, 0.80, "turnover_ge15")
        apply(atr is None, 0.80, "atr_missing")
        apply(pct is not None and pct <= -4.0, 0.60, "pct_drop4")
        apply(two_day is not None and two_day <= -0.06, 0.65, "twoday_drop6")
    return max(0.0, min(scale, 1.0)), "|".join(tags) if tags else "base"


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = BASES[str(case["base"])]
    rows = list(csv.DictReader(base["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    market = _market_day_cache(rows)
    remove_symbols = set(case.get("remove_symbols") or [])
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    removed = 0
    for row in rows:
        if str(row.get("symbol") or "") in remove_symbols:
            removed += 1
            continue
        by_date[str(row.get("signal_date") or "")].append(row)

    output_rows: list[dict[str, Any]] = []
    risk_counter: Counter[str] = Counter()
    for signal_date, items in by_date.items():
        selected = sorted(items, key=lambda row: (int(float(row.get("rank") or 999999)), str(row.get("stock_code") or "")))[
            : int(case["select_topn"])
        ]
        for idx, row in enumerate(selected, start=1):
            item = dict(row)
            item["original_rank"] = str(row.get("rank") or "")
            item["rank"] = str(idx)
            base_target = _float(row.get("target_pct")) or float(base["target_position_pct"])
            scale, tag = _risk_scale(row, market, str(case.get("risk_mode", "none")))
            risk_counter[tag] += 1
            item["target_pct"] = f"{min(float(base['target_position_pct']), max(0.0, base_target * scale)):.5f}"
            item["holding_days"] = str(int(case["holding_days"]))
            item["max_holding_days"] = str(int(case["max_holding_days"]))
            item["score_exit_entry_ratio"] = f"{float(case['score_exit']):.5f}"
            item["score_continue_entry_ratio"] = f"{float(case['score_continue']):.5f}"
            item["min_holding_days_before_score_exit"] = "1"
            item["strategy_variant"] = str(case["name"])
            item["filter_name"] = str(case["name"])
            item["dynamic_hold_name"] = str(case["name"])
            item["fragility_refill_removed_symbols"] = ",".join(sorted(remove_symbols))
            item["risk_scale"] = f"{scale:.6f}"
            item["risk_tags"] = tag
            output_rows.append(item)

    output_rows.sort(key=lambda row: (row["signal_date"], int(float(row["rank"])), row["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, output_rows)
    counts = Counter(row["signal_date"] for row in output_rows)
    target = int(case["select_topn"])
    return output, {
        "signal_rows": len(output_rows),
        "signal_days": len(counts),
        "removed_rows": removed,
        "target_names_per_day": target,
        "days_below_target": sum(1 for value in counts.values() if value < target),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "risk_tag_top": dict(risk_counter.most_common(8)),
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    base = BASES[str(case["base"])]
    signal_file, meta = _make_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
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
                "GM_EQUITY_DD_RISK_MODE": "0",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
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
            str(int(case["max_positions"])),
            "--holding-days",
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["max_holding_days"])),
            "--target-position-pct",
            str(float(base["target_position_pct"])),
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
    annual = indicator.get("pnl_ratio_annual") if indicator else None
    sharpe = indicator.get("sharp_ratio") if indicator else None
    max_drawdown = indicator.get("max_drawdown") if indicator else None
    return {
        **case,
        **meta,
        "returncode": returncode,
        "annual": annual,
        "annual_return": annual,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "target_hit_500_sharpe4_mdd40": bool(
            annual is not None and sharpe is not None and max_drawdown is not None and annual >= 5.0 and sharpe >= 4.0 and max_drawdown <= 0.40
        ),
        "signal_file": str(signal_file),
        "score_db": str(SCORE_DB),
        "score_table": SCORE_TABLE,
        "log_file": str(log_file),
        "note": "research-only fragility refill optimization; production unchanged",
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
