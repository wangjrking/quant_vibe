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

OUT_SIGNAL_DIR = REPORT_DIR / "open_gap_frequency_signals"
OUT_LOG_DIR = REPORT_DIR / "open_gap_frequency_logs"
OUT_CSV = REPORT_DIR / "open_gap_frequency_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "open_gap_frequency_probe_20260630.json"


BASES: list[dict[str, Any]] = [
    {
        "base": "dyn_mild_09_12_15",
        "signal_file": REPORT_DIR / "dynamic_target_signals" / "dyn_mild_09_12_15.csv",
        "score_db": REPORT_DIR / "scores" / "grid_scores.duckdb",
        "score_table": "score_w72_23_05_amt150_mv30_latest_l4_full",
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "min_score_exit_days": 1,
        "max_daily_sells": 0,
        "target_position_pct": 0.15,
        "sell_slippage_mult": 0.0,
    },
    {
        "base": "hp_new70_turn80_top5_s170_cap27_h2m3",
        "signal_file": REPORT_DIR / "soft_newstock_highpos_signals" / "hp_new70_turn80_top5_s170_cap27_h2m3.csv",
        "score_db": REPORT_DIR / "scores" / "soft_newstock_highpos_scores.duckdb",
        "score_table": "score_hp_new70_turn80_top5_s170_cap27_h2m3",
        "max_positions": 5,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "min_score_exit_days": 1,
        "max_daily_sells": 1,
        "target_position_pct": 0.27,
        "sell_slippage_mult": 0.25,
    },
]


OPEN_GAP_CASES: list[dict[str, Any]] = [
    {"name": "gap_base", "mode": "base"},
    {"name": "gap_skip_up_gt_2p", "mode": "skip_up", "up": 0.02},
    {"name": "gap_skip_up_gt_3p", "mode": "skip_up", "up": 0.03},
    {"name": "gap_skip_up_gt_5p", "mode": "skip_up", "up": 0.05},
    {"name": "gap_scale_up_gt_2p_x50", "mode": "scale_up", "up": 0.02, "scale": 0.5},
    {"name": "gap_scale_up_gt_3p_x60", "mode": "scale_up", "up": 0.03, "scale": 0.6},
    {"name": "gap_abs_le_6p", "mode": "skip_abs", "abs": 0.06},
    {"name": "gap_rerank_penalty_1p5", "mode": "rerank_penalty", "penalty": 1.5},
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
                buy_date = str(row.get("buy_date") or "").strip()
                stock_code = str(row.get("stock_code") or "").strip()
                if buy_date and stock_code:
                    pairs.add((stock_code, buy_date))
    if not pairs:
        return {}

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
        row["holding_days"] = str(base["holding_days"])
        row["max_holding_days"] = str(base["max_holding_days"])
        row["score_exit_entry_ratio"] = f"{float(base['score_exit']):.5f}"
        row["score_continue_entry_ratio"] = f"{float(base['score_continue']):.5f}"
        row["min_holding_days_before_score_exit"] = str(base["min_score_exit_days"])
    return rows


def _signal_rows(
    base: dict[str, Any],
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    gaps: dict[tuple[str, str], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(case["mode"])
    out: list[dict[str, Any]] = []
    skipped = 0
    scaled = 0
    missing_gap = 0

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("signal_date") or "")].append(row)

    for signal_date, items in grouped.items():
        day_rows: list[dict[str, Any]] = []
        for row in items:
            buy_date = str(row.get("buy_date") or "").strip()
            stock_code = str(row.get("stock_code") or "").strip()
            gap = gaps.get((stock_code, buy_date)) if buy_date else None
            keep = True
            target = _float(row.get("target_pct")) or float(base["target_position_pct"])
            if gap is None and buy_date:
                missing_gap += 1
            if mode == "base":
                pass
            elif mode == "skip_up":
                if gap is not None and gap > float(case["up"]):
                    keep = False
            elif mode == "scale_up":
                if gap is not None and gap > float(case["up"]):
                    target *= float(case["scale"])
                    scaled += 1
            elif mode == "skip_abs":
                if gap is not None and abs(gap) > float(case["abs"]):
                    keep = False
            elif mode == "rerank_penalty":
                pass
            else:
                raise ValueError(f"unknown mode: {mode}")
            if not keep:
                skipped += 1
                continue
            item = dict(row)
            item["target_pct"] = f"{target:.5f}"
            item["strategy_variant"] = f"{base['base']}__{case['name']}"
            item["filter_name"] = str(case["name"])
            item["buy_open_gap"] = "" if gap is None else f"{gap:.8f}"
            day_rows.append(item)

        if mode == "rerank_penalty":
            def score(item: dict[str, Any]) -> float:
                entry = _float(item.get("entry_score")) or _float(item.get("pred_prob")) or 0.0
                gap = _float(item.get("buy_open_gap"))
                return entry - float(case["penalty"]) * max(gap or 0.0, 0.0)

            day_rows.sort(key=lambda item: (-score(item), int(float(item.get("rank") or 9999))))
            day_rows = day_rows[: int(base["max_positions"])]
            for rank, item in enumerate(day_rows, start=1):
                item["rank"] = str(rank)
        out.extend(day_rows)

    out.sort(key=lambda row: (str(row.get("signal_date") or ""), int(float(row.get("rank") or 9999)), str(row.get("stock_code") or "")))
    counts = Counter(row["signal_date"] for row in out)
    return out, {
        "signal_rows": len(out),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < int(base["max_positions"])),
        "empty_days_vs_base": 986 - len(counts),
        "skipped_rows": skipped,
        "scaled_rows": scaled,
        "missing_gap_rows": missing_gap,
    }


def _run(base: dict[str, Any], case: dict[str, Any], rows: list[dict[str, Any]], gaps: dict[tuple[str, str], float]) -> dict[str, Any]:
    case_key = f"{base['base']}__{case['name']}"
    signal_rows, signal_meta = _signal_rows(base, case, rows, gaps)
    signal_file = OUT_SIGNAL_DIR / f"{case_key}.csv"
    log_file = OUT_LOG_DIR / f"{case_key}.log"
    _write_rows(signal_file, signal_rows)

    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
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
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(int(base["min_score_exit_days"])),
                "GM_SCORE_EXIT_ENTRY_RATIO": str(float(base["score_exit"])),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(base["score_continue"])),
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
            str(int(base["holding_days"])),
            "--max-holding-days",
            str(int(base["max_holding_days"])),
            "--target-position-pct",
            str(float(base["target_position_pct"])),
            "--score-db",
            str(base["score_db"]),
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
        "base": base["base"],
        "case": case["name"],
        "mode": case["mode"],
        **{k: v for k, v in case.items() if k not in {"name", "mode"}},
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
        "intraday_risk": 0,
        "note": "research-only buy-open-gap frequency probe; no production parameter change",
    }


def main() -> None:
    for base in BASES:
        if not base["signal_file"].exists():
            raise FileNotFoundError(base["signal_file"])
    gaps = _load_open_gaps([base["signal_file"] for base in BASES])
    results: list[dict[str, Any]] = []
    for base in BASES:
        rows = _base_rows(base)
        for case in OPEN_GAP_CASES:
            row = _run(base, case, rows, gaps)
            results.append(row)
            _write_rows(OUT_CSV, results)
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
