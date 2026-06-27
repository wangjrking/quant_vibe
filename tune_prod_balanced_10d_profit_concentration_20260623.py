from __future__ import annotations

import argparse
import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
PROD_DIR = MAIN / "strategy_library" / "production" / "prod_measure_balanced_10d_v20260623"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260623"
    / "prod_balanced_10d_profit_concentration_tune"
)
SOURCE_SIGNAL = PROD_DIR / "signals" / "historical_repro_current_best.csv"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
SCORE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_day_pool_replace_20260621"
    / "weak_day_pool_scores.db"
)
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-18 15:30:00"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune current balanced 10D production strategy for profit-first concentrated portfolios."
    )
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--limit", type=int, default=0, help="Optional max variants to run after sorting.")
    parser.add_argument("--min-buy-days", type=int, default=350)
    parser.add_argument("--min-signal-count", type=int, default=350)
    parser.add_argument("--force", action="store_true", help="Rerun even when a parseable log already exists.")
    return parser.parse_args()


def _variant(
    name: str,
    max_positions: int,
    target_pct: float,
    holding_days: int,
    max_holding_days: int,
    min_score_hold: int,
    score_exit_ratio: float,
    max_daily_sells: int,
    score_continue_ratio: float = 1.02,
) -> dict:
    return {
        "name": name,
        "max_positions": max_positions,
        "top_n": max_positions,
        "target_pct": target_pct,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "min_score_hold": min_score_hold,
        "score_exit_ratio": score_exit_ratio,
        "score_continue_ratio": score_continue_ratio,
        "max_daily_sells": max_daily_sells,
    }


def _targets_for(max_positions: int, mode: str) -> list[float]:
    if mode == "smoke":
        return {1: [0.99], 2: [0.55], 3: [0.42], 4: [0.36], 5: [0.32], 7: [0.42]}[max_positions]
    if mode == "quick":
        return {
            1: [0.99],
            2: [0.48, 0.55, 0.65],
            3: [0.38, 0.42, 0.50, 0.60],
            4: [0.32, 0.36, 0.42, 0.50],
            5: [0.28, 0.32, 0.42],
            7: [0.24, 0.32, 0.42],
        }[max_positions]
    return {
        1: [0.80, 0.90, 0.99],
        2: [0.42, 0.48, 0.55, 0.65],
        3: [0.34, 0.38, 0.42, 0.50, 0.60],
        4: [0.28, 0.32, 0.36, 0.42, 0.50],
        5: [0.24, 0.28, 0.32, 0.42, 0.50],
        7: [0.18, 0.24, 0.32, 0.42],
    }[max_positions]


def _build_variants(mode: str) -> list[dict]:
    if mode == "smoke":
        max_positions_grid = [1, 3, 5, 7]
        hold_grid = [(4, 4), (6, 6)]
        min_score_hold_grid = [1, 4]
        exit_grid = [1.00]
        sell_grid = [1]
    elif mode == "quick":
        max_positions_grid = [1, 2, 3, 4, 5, 7]
        hold_grid = [(4, 4), (5, 5), (6, 6)]
        min_score_hold_grid = [1, 2, 4]
        exit_grid = [0.98, 1.00, 1.02]
        sell_grid = [1]
    else:
        max_positions_grid = [1, 2, 3, 4, 5, 7]
        hold_grid = [(4, 4), (5, 5), (6, 6)]
        min_score_hold_grid = [1, 2, 3, 4]
        exit_grid = [0.98, 1.00, 1.02]
        sell_grid = [1, 2]

    variants: list[dict] = []
    for max_positions in max_positions_grid:
        for target_pct in _targets_for(max_positions, mode):
            for holding_days, max_holding_days in hold_grid:
                for min_score_hold in min_score_hold_grid:
                    for score_exit_ratio in exit_grid:
                        for max_daily_sells in sell_grid:
                            name = (
                                f"top{max_positions}_t{str(target_pct).replace('.', 'p')}"
                                f"_h{holding_days}_mh{max_holding_days}"
                                f"_score{str(score_exit_ratio).replace('.', 'p')}"
                                f"_min{min_score_hold}_sell{max_daily_sells}"
                            )
                            variants.append(
                                _variant(
                                    name=name,
                                    max_positions=max_positions,
                                    target_pct=target_pct,
                                    holding_days=holding_days,
                                    max_holding_days=max_holding_days,
                                    min_score_hold=min_score_hold,
                                    score_exit_ratio=score_exit_ratio,
                                    max_daily_sells=max_daily_sells,
                                )
                            )
    return variants


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normal_code(value: str) -> str:
    code = (value or "").strip()
    if "." in code:
        return code
    if len(code) >= 6:
        prefix = code[:6]
        return prefix + (".SH" if prefix.startswith("6") else ".SZ")
    return code


def _is_st_text(*values: str) -> bool:
    text = " ".join(value or "" for value in values).upper()
    return "ST" in text or "退" in text or "RISK" in text


def _load_st_flags(rows: list[dict]) -> dict[tuple[str, str], dict]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        code = _normal_code(row.get("stock_code") or row.get("symbol") or "")
        for field in ("signal_date", "buy_date"):
            date = str(row.get(field) or "").strip()
            if code and date:
                keys.add((code, date))
    if not keys:
        return {}
    conn = sqlite3.connect(MARKET_DB)
    cur = conn.cursor()
    flags: dict[tuple[str, str], dict] = {}
    for code, date in sorted(keys):
        cur.execute(
            "select ST_TYPE, ST_TYPE_name, name from STOCK_DAILY_DATA where stock_code=? and trade_date=? limit 1",
            (code, date),
        )
        rec = cur.fetchone()
        if rec is None:
            flags[(code, date)] = {"missing": True, "is_st": False}
            continue
        st_type, st_name, name = [(value or "") for value in rec]
        flags[(code, date)] = {
            "missing": False,
            "is_st": _is_st_text(st_type, st_name, name),
            "ST_TYPE": st_type,
            "ST_TYPE_name": st_name,
            "name": name,
        }
    conn.close()
    return flags


def _row_is_st(row: dict, st_flags: dict[tuple[str, str], dict]) -> bool:
    code = _normal_code(row.get("stock_code") or row.get("symbol") or "")
    checks = []
    for field in ("signal_date", "buy_date"):
        date = str(row.get(field) or "").strip()
        if code and date:
            checks.append(st_flags.get((code, date), {}).get("is_st", False))
    return any(checks)


def _sort_key(row: dict) -> tuple:
    rank = _to_float(row.get("rank"), 999999.0)
    pred = _to_float(row.get("pred_prob"), float("-inf"))
    return (rank, -pred)


def _write_signal(variant: dict, source_rows: list[dict], st_flags: dict[tuple[str, str], dict], path: Path) -> dict:
    kept_by_day: dict[str, list[dict]] = defaultdict(list)
    st_removed = 0
    for row in source_rows:
        if _row_is_st(row, st_flags):
            st_removed += 1
            continue
        buy_date = str(row.get("buy_date") or "").strip()
        if buy_date:
            kept_by_day[buy_date].append(dict(row))

    out_rows: list[dict] = []
    for buy_date in sorted(kept_by_day):
        day_rows = sorted(kept_by_day[buy_date], key=_sort_key)[: int(variant["top_n"])]
        for rank, row in enumerate(day_rows, start=1):
            row["rank"] = str(rank)
            row["target_pct"] = f"{float(variant['target_pct']):.5f}"
            row["holding_days"] = str(int(variant["holding_days"]))
            row["max_holding_days"] = str(int(variant["max_holding_days"]))
            row["signal_score_exit_entry_ratio"] = str(float(variant["score_exit_ratio"]))
            row["signal_min_holding_days_before_score_exit"] = str(int(variant["min_score_hold"]))
            row["signal_score_continue_entry_ratio"] = str(float(variant["score_continue_ratio"]))
            out_rows.append(row)

    fieldnames = list(source_rows[0].keys())
    for extra in (
        "max_holding_days",
        "signal_score_exit_entry_ratio",
        "signal_min_holding_days_before_score_exit",
        "signal_score_continue_entry_ratio",
    ):
        if extra not in fieldnames:
            fieldnames.append(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in out_rows])
    day_target = defaultdict(float)
    for row in out_rows:
        day_target[row.get("buy_date") or ""] += _to_float(row.get("target_pct"), 0.0) or 0.0
    return {
        "signal_count": len(out_rows),
        "buy_days": len([day for day in day_target if day]),
        "st_removed_rows": st_removed,
        "avg_day_target_sum": sum(day_target.values()) / len(day_target) if day_target else None,
        "max_day_target_sum": max(day_target.values()) if day_target else None,
        "min_day_target_sum": min(day_target.values()) if day_target else None,
    }


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


def _run_backtest(variant: dict, signal_file: Path, log_file: Path, force: bool) -> int:
    if not force and log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": str(float(variant["score_exit_ratio"])),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(int(variant["min_score_hold"])),
            "GM_MAX_DAILY_SELLS": str(int(variant["max_daily_sells"])),
            "GM_STOP_LOSS_PCT": "0.08",
            "GM_TAKE_PROFIT_PCT": "none",
            "GM_LIGHT_STOP_LOSS_PCT": "none",
            "GM_SCORE_EXIT_RANK": "none",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "1",
            "GM_CASH_BUFFER": "0.99",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(variant["score_continue_ratio"])),
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.09",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.17",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.82",
            "GM_EQUITY_DD_HARD_SCALE": "0.58",
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
        str(float(variant["target_pct"])),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        str(float(variant["score_exit_ratio"])),
        "--min-holding-days-before-score-exit",
        str(int(variant["min_score_hold"])),
        "--score-continue-entry-ratio",
        str(float(variant["score_continue_ratio"])),
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


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def _passes_coverage(row: dict, min_buy_days: int, min_signal_count: int) -> bool:
    return int(row.get("buy_days") or 0) >= min_buy_days and int(row.get("signal_count") or 0) >= min_signal_count


def _objective(row: dict, min_buy_days: int, min_signal_count: int) -> tuple:
    valid = (
        int(row.get("returncode") or 1) == 0
        and int(row.get("st_violation_count") or 0) == 0
        and _passes_coverage(row, min_buy_days, min_signal_count)
        and _metric(row, "avg_invested_pct") >= 0.80
        and _metric(row, "sharpe") >= 3.0
    )
    return (1 if valid else 0, _metric(row, "annual"), _metric(row, "sharpe"), -_metric(row, "max_drawdown"))


def _write_report(rows: list[dict], min_buy_days: int, min_signal_count: int) -> None:
    if not rows:
        return
    filtered = [
        row
        for row in rows
        if int(row.get("returncode") or 1) == 0
        and int(row.get("st_violation_count") or 0) == 0
        and _passes_coverage(row, min_buy_days, min_signal_count)
        and _metric(row, "avg_invested_pct") >= 0.80
        and _metric(row, "sharpe") >= 3.0
    ]
    hits_500 = [row for row in filtered if _metric(row, "annual") >= 5.0]
    best = max(rows, key=lambda row: _objective(row, min_buy_days, min_signal_count))
    best_filtered = max(filtered, key=lambda row: _metric(row, "annual")) if filtered else None
    lines = [
        "# prod balanced 10D profit concentration tune",
        "",
        f"- variants: {len(rows)}",
        "- industry/style-exposure filters: not used",
        f"- coverage gate: buy_days>={min_buy_days}, signal_count>={min_signal_count}",
        f"- filtered candidates sharpe>=3 avg_invested>=80 st=0: {len(filtered)}",
        f"- annual>=5.0 hits: {len(hits_500)}",
        f"- best objective: {best.get('name')} annual={best.get('annual')} sharpe={best.get('sharpe')} max_dd={best.get('max_drawdown')} avg_inv={best.get('avg_invested_pct')}",
    ]
    if best_filtered:
        lines.append(
            f"- best filtered annual: {best_filtered.get('name')} annual={best_filtered.get('annual')} sharpe={best_filtered.get('sharpe')} max_dd={best_filtered.get('max_drawdown')} avg_inv={best_filtered.get('avg_invested_pct')}"
        )
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if not SOURCE_SIGNAL.exists():
        raise SystemExit(f"source signal not found: {SOURCE_SIGNAL}")
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"strategy main.py not found: {STRATEGY_DIR}")

    source_rows = _load_rows(SOURCE_SIGNAL)
    st_flags = _load_st_flags(source_rows)
    variants = _build_variants(args.mode)
    if args.limit and args.limit > 0:
        variants = variants[: args.limit]

    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, variant in enumerate(variants, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        signal_stats = _write_signal(variant, source_rows, st_flags, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file, args.force)
        indicator = _extract_indicator(log_file) or {}
        row = {
            **variant,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "st_violation_count": 0,
            **signal_stats,
            **_exposure_stats(log_file),
        }
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(variants)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"maxdd={row.get('max_drawdown')} avg_inv={row.get('avg_invested_pct')}"
        )

    _write_rows(
        REPORT_DIR / "summary_by_objective.csv",
        sorted(results, key=lambda row: _objective(row, args.min_buy_days, args.min_signal_count), reverse=True),
    )
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_filtered_candidates.csv",
        [
            row
            for row in sorted(results, key=lambda item: _metric(item, "annual"), reverse=True)
            if int(row.get("returncode") or 1) == 0
            and int(row.get("st_violation_count") or 0) == 0
            and _passes_coverage(row, args.min_buy_days, args.min_signal_count)
            and _metric(row, "avg_invested_pct") >= 0.80
            and _metric(row, "sharpe") >= 3.0
        ],
    )
    _write_report(results, args.min_buy_days, args.min_signal_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
