from __future__ import annotations

import argparse
import ast
import csv
import datetime as datetime_module
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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260623" / "prod_balanced_10d_wide_refill_tune"
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
    / "fusion_5d10d.db"
)
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
START_DATE = "20240604"
END_SIGNAL_DATE = "20260617"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-ST wide-pool refill candidates for current balanced 10D strategy.")
    parser.add_argument("--mode", choices=["smoke", "quick"], default="smoke")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-buy-days", type=int, default=350)
    parser.add_argument("--min-signal-count", type=int, default=350)
    return parser.parse_args()


def _variant(
    name: str,
    top_n: int,
    target_pct: float,
    hold: int,
    min_score_hold: int,
    score_exit: float,
    w10: float,
    max_total_mv: float = 200000.0,
) -> dict:
    return {
        "name": name,
        "top_n": top_n,
        "max_positions": top_n,
        "target_pct": target_pct,
        "holding_days": hold,
        "max_holding_days": hold,
        "min_score_hold": min_score_hold,
        "score_exit_ratio": score_exit,
        "w10": w10,
        "w5": 1.0 - w10,
        "max_total_mv": max_total_mv,
    }


def _build_variants(mode: str) -> list[dict]:
    base = []
    if mode == "smoke":
        specs = [
            (3, 0.42),
            (4, 0.36),
            (5, 0.32),
            (7, 0.24),
        ]
        holds = [4, 6]
        exits = [1.0]
        mins = [1, 3]
        weights = [0.9]
    else:
        specs = [
            (2, 0.55),
            (3, 0.42),
            (3, 0.50),
            (4, 0.36),
            (4, 0.42),
            (5, 0.32),
            (5, 0.42),
            (7, 0.24),
            (7, 0.32),
        ]
        holds = [4, 5, 6]
        exits = [0.98, 1.0, 1.02]
        mins = [1, 2, 3]
        weights = [1.0, 0.9, 0.8]
    for top_n, target in specs:
        for hold in holds:
            for min_score_hold in mins:
                for score_exit in exits:
                    for w10 in weights:
                        base.append(
                            _variant(
                                f"wide_top{top_n}_t{str(target).replace('.', 'p')}_h{hold}_min{min_score_hold}_score{str(score_exit).replace('.', 'p')}_w10{str(w10).replace('.', 'p')}",
                                top_n,
                                target,
                                hold,
                                min_score_hold,
                                score_exit,
                                w10,
                            )
                        )
    return base


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return value


def _symbol(code: str) -> str:
    code = str(code)
    if code.endswith(".SZ"):
        return "SZSE." + code[:6]
    if code.endswith(".SH"):
        return "SHSE." + code[:6]
    return code


def _is_st_text(*values: str) -> bool:
    text = " ".join(value or "" for value in values).upper()
    return "ST" in text or "退" in text or "RISK" in text


def _load_calendar() -> list[str]:
    conn = sqlite3.connect(FUSION_DB)
    dates = [
        row[0]
        for row in conn.execute(
            "select distinct trade_date from fusion_rank_base where trade_date>=? order by trade_date", (START_DATE,)
        )
    ]
    conn.close()
    return dates


def _load_st_set(dates: list[str]) -> set[tuple[str, str]]:
    conn = sqlite3.connect(MARKET_DB)
    cur = conn.cursor()
    st: set[tuple[str, str]] = set()
    for start in range(0, len(dates), 100):
        chunk = dates[start : start + 100]
        placeholders = ",".join("?" for _ in chunk)
        for code, date, st_type, st_name, name in cur.execute(
            f"""
            select stock_code, trade_date, ST_TYPE, ST_TYPE_name, name
            from STOCK_DAILY_DATA
            where trade_date in ({placeholders})
            """,
            chunk,
        ):
            if _is_st_text(st_type or "", st_name or "", name or ""):
                st.add((str(code), str(date)))
    conn.close()
    return st


def _load_candidates(variant: dict, dates: list[str], st_set: set[tuple[str, str]]) -> list[dict]:
    next_date = {date: dates[index + 1] for index, date in enumerate(dates[:-1])}
    valid_dates = [date for date in dates if START_DATE <= date <= END_SIGNAL_DATE and date in next_date]
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    by_date: dict[str, list[dict]] = defaultdict(list)
    w10 = float(variant["w10"])
    w5 = float(variant["w5"])
    for start in range(0, len(valid_dates), 40):
        chunk = valid_dates[start : start + 40]
        placeholders = ",".join("?" for _ in chunk)
        query = f"""
            select trade_date, stock_code, pred_10d, pred_5d, name, close, amount,
                   turnover_rate, total_mv, atr_qfq, limit_times, rank_5d, rank_10d
            from fusion_rank_base
            where trade_date in ({placeholders})
        """
        for row in conn.execute(query, chunk):
            code = str(row["stock_code"])
            signal_date = str(row["trade_date"])
            buy_date = next_date[signal_date]
            if code.endswith(".BJ") or code.startswith("8") or code.startswith("4"):
                continue
            if (code, signal_date) in st_set or (code, buy_date) in st_set:
                continue
            if str(row["limit_times"] or "").strip() not in {"", "0", "0.0", "None", "none"}:
                continue
            amount = _to_float(row["amount"], 0.0)
            turnover = _to_float(row["turnover_rate"], 0.0)
            total_mv = _to_float(row["total_mv"], 0.0)
            if amount < 10000.0 or turnover < 0.3 or total_mv <= 0 or total_mv > float(variant["max_total_mv"]):
                continue
            rank_10d = _to_float(row["rank_10d"], 0.0)
            rank_5d = _to_float(row["rank_5d"], 0.0)
            pred_10d = _to_float(row["pred_10d"], 0.0)
            pred_5d = _to_float(row["pred_5d"], 0.0)
            close = _to_float(row["close"], 0.0)
            atr = _to_float(row["atr_qfq"], None)
            score = w10 * rank_10d + w5 * rank_5d
            by_date[signal_date].append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "symbol": _symbol(code),
                    "stock_code": code,
                    "name": row["name"],
                    "rank": "",
                    "pred_prob": f"{score:.12f}",
                    "atr_ratio": f"{(atr / close):.8f}" if atr is not None and close else "",
                    "holding_days": str(int(variant["holding_days"])),
                    "target_pct": f"{float(variant['target_pct']):.5f}",
                    "trade_date": signal_date,
                    "pred_5d": pred_5d,
                    "pred_10d": pred_10d,
                    "total_mv": total_mv,
                    "amount": amount,
                    "turnover_rate": turnover,
                    "atr_qfq": atr,
                    "close": close,
                    "pred_gap": abs(pred_10d - pred_5d),
                    "pool_role": "wide_refill",
                    "score": score,
                    "rank_10d": rank_10d,
                    "rank_5d": rank_5d,
                    "max_holding_days": str(int(variant["max_holding_days"])),
                    "signal_score_exit_entry_ratio": str(float(variant["score_exit_ratio"])),
                    "signal_min_holding_days_before_score_exit": str(int(variant["min_score_hold"])),
                    "signal_score_continue_entry_ratio": "1.02",
                }
            )
    conn.close()
    selected: list[dict] = []
    for signal_date, day_rows in sorted(by_date.items()):
        day_rows.sort(key=lambda item: (-float(item["score"]), -float(item["pred_10d"]), item["stock_code"]))
        for rank, item in enumerate(day_rows[: int(variant["top_n"])], start=1):
            item["rank"] = str(rank)
            item.pop("score", None)
            selected.append(item)
    return selected


def _write_signal(rows: list[dict], path: Path) -> dict:
    fields = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "atr_ratio",
        "holding_days",
        "target_pct",
        "trade_date",
        "pred_5d",
        "pred_10d",
        "total_mv",
        "amount",
        "turnover_rate",
        "atr_qfq",
        "close",
        "pred_gap",
        "pool_role",
        "rank_10d",
        "rank_5d",
        "max_holding_days",
        "signal_score_exit_entry_ratio",
        "signal_min_holding_days_before_score_exit",
        "signal_score_continue_entry_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    day_target = defaultdict(float)
    for row in rows:
        day_target[row["buy_date"]] += _to_float(row["target_pct"], 0.0) or 0.0
    return {
        "signal_count": len(rows),
        "buy_days": len(day_target),
        "avg_day_target_sum": sum(day_target.values()) / len(day_target) if day_target else None,
        "min_day_target_sum": min(day_target.values()) if day_target else None,
        "max_day_target_sum": max(day_target.values()) if day_target else None,
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
            "GM_MAX_DAILY_SELLS": "1",
            "GM_STOP_LOSS_PCT": "0.08",
            "GM_TAKE_PROFIT_PCT": "none",
            "GM_LIGHT_STOP_LOSS_PCT": "none",
            "GM_SCORE_EXIT_RANK": "none",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_SYNC_POSITIONS": "1",
            "GM_CASH_BUFFER": "0.99",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.09",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.17",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.82",
            "GM_EQUITY_DD_HARD_SCALE": "0.58",
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
        "1.02",
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
        proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def _passes(row: dict, min_buy_days: int, min_signal_count: int) -> bool:
    return (
        int(row.get("returncode") or 1) == 0
        and int(row.get("st_violation_count") or 0) == 0
        and int(row.get("buy_days") or 0) >= min_buy_days
        and int(row.get("signal_count") or 0) >= min_signal_count
        and _metric(row, "avg_invested_pct") >= 0.80
        and _metric(row, "sharpe") >= 3.0
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    dates = _load_calendar()
    st_set = _load_st_set(dates)
    variants = _build_variants(args.mode)
    if args.limit > 0:
        variants = variants[: args.limit]
    rows_out: list[dict] = []
    for index, variant in enumerate(variants, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        signal_rows = _load_candidates(variant, dates, st_set)
        stats = _write_signal(signal_rows, signal_file)
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
            **stats,
            **_exposure_stats(log_file),
        }
        rows_out.append(row)
        _write_rows(REPORT_DIR / "summary.csv", rows_out)
        print(
            f"[{index}/{len(variants)}] {variant['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} maxdd={row.get('max_drawdown')} avg_inv={row.get('avg_invested_pct')}"
        )
    ranked = sorted(
        rows_out,
        key=lambda row: (
            1 if _passes(row, args.min_buy_days, args.min_signal_count) else 0,
            _metric(row, "annual"),
            _metric(row, "sharpe"),
        ),
        reverse=True,
    )
    _write_rows(REPORT_DIR / "summary_by_objective.csv", ranked)
    _write_rows(REPORT_DIR / "summary_filtered_candidates.csv", [r for r in ranked if _passes(r, args.min_buy_days, args.min_signal_count)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
