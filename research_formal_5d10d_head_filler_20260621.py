from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path

from gm_signal_module import load_market_rows_by_trade_date, to_gm_symbol


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
    / "fusion_5d10d.db"
)
SCORE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_day_pool_replace_20260621"
    / "weak_day_pool_scores.db"
)
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_head_filler_20260621"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"
START_DATE = "20240604"
END_DATE = "20260618"


VARIANTS = [
    {
        "name": "hf_rank90_top10_21191815100806040302",
        "table": "rank_10d90_5d10",
        "top_k": 10,
        "weights": {1: 0.21, 2: 0.19, 3: 0.18, 4: 0.15, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.04, 9: 0.03, 10: 0.02},
        "min_pred_10d": 0.0,
        "max_total_mv": 200000.0,
        "min_amount": 10000.0,
        "min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 10,
    },
    {
        "name": "hf_rank90_top12_201817141008060504030201",
        "table": "rank_10d90_5d10",
        "top_k": 12,
        "weights": {1: 0.20, 2: 0.18, 3: 0.17, 4: 0.14, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.05, 9: 0.04, 10: 0.03, 11: 0.02, 12: 0.01},
        "min_pred_10d": 0.0,
        "max_total_mv": 250000.0,
        "min_amount": 10000.0,
        "min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 12,
    },
    {
        "name": "hf_rank80_top10_21191815100806040302",
        "table": "rank_10d80_5d20",
        "top_k": 10,
        "weights": {1: 0.21, 2: 0.19, 3: 0.18, 4: 0.15, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.04, 9: 0.03, 10: 0.02},
        "min_pred_10d": 0.0,
        "max_total_mv": 200000.0,
        "min_amount": 10000.0,
        "min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 10,
    },
    {
        "name": "hf_rank60_top12_201817141008060504030201",
        "table": "rank_10d60_5d40",
        "top_k": 12,
        "weights": {1: 0.20, 2: 0.18, 3: 0.17, 4: 0.14, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.05, 9: 0.04, 10: 0.03, 11: 0.02, 12: 0.01},
        "min_pred_10d": 0.0,
        "max_total_mv": 250000.0,
        "min_amount": 10000.0,
        "min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 12,
    },
    {
        "name": "hf_rank90_top10_liq_mid",
        "table": "rank_10d90_5d10",
        "top_k": 10,
        "weights": {1: 0.21, 2: 0.19, 3: 0.18, 4: 0.15, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.04, 9: 0.03, 10: 0.02},
        "min_pred_10d": 0.0,
        "max_total_mv": 200000.0,
        "min_amount": 20000.0,
        "min_turnover_rate": 0.5,
        "holding_days": 7,
        "max_positions": 10,
    },
    {
        "name": "hf_rank90_top10_avg205",
        "table": "rank_10d90_5d10",
        "top_k": 10,
        "weights": {1: 0.21, 2: 0.19, 3: 0.18, 4: 0.15, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.04, 9: 0.03, 10: 0.02},
        "post_filter_avg_pred_min": 2.05,
        "min_pred_10d": 0.0,
        "max_total_mv": 200000.0,
        "min_amount": 10000.0,
        "min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 10,
    },
    {
        "name": "hf_rank90_top10_h9",
        "table": "rank_10d90_5d10",
        "top_k": 10,
        "weights": {1: 0.18, 2: 0.17, 3: 0.16, 4: 0.13, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.05, 9: 0.04, 10: 0.03},
        "min_pred_10d": 0.0,
        "max_total_mv": 200000.0,
        "min_amount": 10000.0,
        "min_turnover_rate": 0.3,
        "holding_days": 9,
        "max_positions": 10,
    },
    {
        "name": "hf_rank90_top10_exit099",
        "table": "rank_10d90_5d10",
        "top_k": 10,
        "weights": {1: 0.21, 2: 0.19, 3: 0.18, 4: 0.15, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.04, 9: 0.03, 10: 0.02},
        "min_pred_10d": 0.0,
        "max_total_mv": 200000.0,
        "min_amount": 10000.0,
        "min_turnover_rate": 0.3,
        "holding_days": 7,
        "max_positions": 10,
        "extra_env": {"GM_SCORE_EXIT_ENTRY_RATIO": "0.99"},
    },
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _is_unbuyable_next_day(row: dict | None) -> bool:
    if not row:
        return True
    pre_close = _to_float(row.get("pre_close"))
    open_price = _to_float(row.get("open"))
    if open_price is None or open_price <= 0:
        return True
    if row.get("limit_times") not in (None, "", "0", "0.0"):
        try:
            if float(row.get("limit_times")) > 0:
                return True
        except Exception:
            return True
    if pre_close and pre_close > 0 and open_price / pre_close - 1.0 >= 0.195:
        return True
    return False


def _load_rows(table: str) -> list[dict]:
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT trade_date, stock_code, pred_prob, pred_5d, pred_10d, rank_5d, rank_10d,
                   name, pre_close, open, close, amount, turnover_rate, total_mv, atr_qfq, limit_times
            FROM {_quote_ident(table)}
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date, pred_prob ASC, stock_code
            """,
            (START_DATE, END_DATE),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _passes(row: dict, variant: dict) -> bool:
    code = str(row.get("stock_code") or "")
    name = str(row.get("name") or "")
    if code.endswith(".BJ") or code.startswith("BJSE."):
        return False
    if name.startswith("ST") or name.startswith("*ST") or "退市" in name or name.startswith("退"):
        return False
    if row.get("limit_times") not in (None, "", "0", "0.0"):
        try:
            if float(row.get("limit_times")) > 0:
                return False
        except Exception:
            return False
    checks = [
        ("min_amount", "amount", lambda a, b: a >= b),
        ("min_turnover_rate", "turnover_rate", lambda a, b: a >= b),
        ("max_total_mv", "total_mv", lambda a, b: a <= b),
        ("min_pred_10d", "pred_10d", lambda a, b: a >= b),
        ("min_pred_5d", "pred_5d", lambda a, b: a >= b),
    ]
    close = _to_float(row.get("close"))
    atr = _to_float(row.get("atr_qfq"))
    atr_ratio = atr / close if atr is not None and close and close > 0 else None
    for key, col, pred in checks:
        if key not in variant:
            continue
        actual = _to_float(row.get(col))
        if actual is None or not pred(actual, float(variant[key])):
            return False
    if "max_atr_ratio" in variant and variant["max_atr_ratio"] is not None:
        if atr_ratio is None or atr_ratio > float(variant["max_atr_ratio"]):
            return False
    return True


def _write_signal(variant: dict, signal_file: Path) -> None:
    rows = _load_rows(variant["table"])
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if _passes(row, variant):
            grouped.setdefault(str(row["trade_date"]), []).append(row)
    trade_dates = sorted(grouped)
    next_date = {date: trade_dates[idx + 1] for idx, date in enumerate(trade_dates[:-1])}
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    output = []
    for signal_date in trade_dates[:-1]:
        buy_date = next_date[signal_date]
        selected = grouped[signal_date][: int(variant["top_k"])]
        if "post_filter_avg_pred_min" in variant:
            vals = [_to_float(row.get("pred_10d"), 0.0) or 0.0 for row in selected]
            if not vals or sum(vals) / len(vals) < float(variant["post_filter_avg_pred_min"]):
                continue
        rank = 0
        for row in selected:
            if _is_unbuyable_next_day((market_rows.get(buy_date) or {}).get(str(row.get("stock_code") or ""))):
                continue
            rank += 1
            if rank > int(variant["top_k"]):
                break
            target_pct = float(variant["weights"].get(rank, 0.0))
            if target_pct <= 0:
                continue
            output.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "symbol": to_gm_symbol(row.get("stock_code")),
                    "stock_code": row.get("stock_code"),
                    "name": row.get("name"),
                    "rank": rank,
                    "pred_prob": row.get("pred_prob"),
                    "pred_10d": row.get("pred_10d"),
                    "pred_5d": row.get("pred_5d"),
                    "holding_days": variant["holding_days"],
                    "target_pct": f"{target_pct:.5f}",
                    "score_exit_entry_ratio": str((variant.get("extra_env") or {}).get("GM_SCORE_EXIT_ENTRY_RATIO", "1.0")),
                    "min_holding_days_before_score_exit": "3",
                }
            )
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "signal_date",
            "buy_date",
            "symbol",
            "stock_code",
            "name",
            "rank",
            "pred_prob",
            "pred_10d",
            "pred_5d",
            "holding_days",
            "target_pct",
            "score_exit_entry_ratio",
            "min_holding_days_before_score_exit",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


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
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
        "min_signal_target_pct": min(targets) if targets else None,
    }


def _run_backtest(variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    if _signal_stats(signal_file)["signal_count"] <= 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("SKIPPED: no signals generated\n", encoding="utf-8")
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
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
        }
    )
    env.update({str(k): str(v) for k, v in (variant.get("extra_env") or {}).items()})
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
        str(int(variant.get("max_holding_days", variant["holding_days"]))),
        "--target-position-pct",
        f"{max(float(x) for x in variant['weights'].values()):.5f}",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
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
    return (float(row.get("annual") or -999), float(row.get("sharpe") or -999), float(row.get("avg_invested_pct") or -999))


def _sort_by_sharpe(row: dict) -> tuple[float, float, float]:
    return (float(row.get("sharpe") or -999), float(row.get("annual") or -999), float(row.get("avg_invested_pct") or -999))


def main() -> int:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "table": variant["table"],
            "top_k": variant["top_k"],
            "weights": variant["weights"],
            "holding_days": variant["holding_days"],
            "max_positions": variant["max_positions"],
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')} signals={row.get('signal_count')}"
        )
    _write_rows(REPORT_DIR / "summary.csv", results)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
    target = [
        row
        for row in results
        if row.get("annual") is not None
        and float(row["annual"]) >= 3.0
        and row.get("sharpe") is not None
        and float(row["sharpe"]) >= 4.0
        and row.get("avg_invested_pct") is not None
        and float(row["avg_invested_pct"]) >= 0.8
    ]
    _write_rows(REPORT_DIR / "summary_target_hits.csv", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
