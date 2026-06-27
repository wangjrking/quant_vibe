from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path

from gm_signal_module import load_market_rows_by_trade_date, to_gm_symbol
from research_list_age_current_best_probe_20260620 import (
    END_DATE,
    JUEJIN_PYTHON,
    MAIN,
    MARKET_DB,
    PRED_DB,
    START_DATE,
    STRATEGY_DIR,
    TABLE_10D,
    TABLE_5D,
    _load_rows,
    _rank5d_by_day,
)


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260623"
    / "prod_balanced_10d_stclean_refill_tune"
)
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


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4",
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


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _is_missing(value) -> bool:
    if value in (None, "", "None", "NONE", "nan", "NaN"):
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _is_st_like(row: dict | None) -> bool:
    if not row:
        return False
    name = str(row.get("name") or "")
    st_type = row.get("st_type")
    st_name = str(row.get("st_type_name") or row.get("ST_TYPE_name") or "")
    if name.startswith(("ST", "*ST")) or "退" in name:
        return True
    if "风险警示" in st_name or "退市" in st_name:
        return True
    if _is_missing(st_type):
        return False
    text = str(st_type).strip().upper()
    if text in {"0", "0.0", "FALSE", "NONE", "NAN"}:
        return False
    return True


def _is_unbuyable_next_day(row: dict | None) -> bool:
    if not row:
        return False
    if _is_st_like(row):
        return True
    limit_times = row.get("limit_times")
    if not _is_missing(limit_times) and (_to_float(limit_times, 1.0) or 0.0) > 0.0:
        return True
    pre_close = _to_float(row.get("pre_close"))
    open_price = _to_float(row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return False
    code = str(row.get("stock_code") or "")
    pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _load_features() -> dict[tuple[str, str], dict]:
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    features: dict[tuple[str, str], dict] = {}
    try:
        for row in conn.execute(
            """
            SELECT trade_date, stock_code, pred_5d, pred_10d, rank_5d, rank_10d,
                   amount, turnover_rate, total_mv, atr_qfq, close
            FROM fusion_rank_base
            WHERE trade_date >= ? AND trade_date <= ?
            """,
            (START_DATE, END_DATE),
        ):
            pred_5d = _to_float(row["pred_5d"])
            pred_10d = _to_float(row["pred_10d"])
            close = _to_float(row["close"])
            atr = _to_float(row["atr_qfq"])
            features[(str(row["trade_date"]), str(row["stock_code"]))] = {
                "pred_5d": pred_5d,
                "pred_10d": pred_10d,
                "rank_5d": _to_float(row["rank_5d"]),
                "rank_10d": _to_float(row["rank_10d"]),
                "amount": _to_float(row["amount"]),
                "turnover_rate": _to_float(row["turnover_rate"]),
                "total_mv": _to_float(row["total_mv"]),
                "atr_ratio": atr / close if atr is not None and close and close > 0 else None,
                "pred_gap": abs(pred_10d - pred_5d) if pred_10d is not None and pred_5d is not None else None,
            }
    finally:
        conn.close()
    return features


def _date_map() -> tuple[list[str], dict[str, str]]:
    conn = sqlite3.connect(PRED_DB)
    try:
        dates = [
            str(row[0])
            for row in conn.execute(
                f'SELECT DISTINCT trade_date FROM "{TABLE_10D}" WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date',
                (START_DATE, END_DATE),
            )
        ]
    finally:
        conn.close()
    return dates, {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}


def _load_candidate_rows(max_pred_prob: float, confirm_rank_min: float) -> list[dict]:
    market = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    _, next_date = _date_map()
    rank5d = _rank5d_by_day()
    rows = _load_rows()
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        date = str(row.get("trade_date") or "")
        code = str(row.get("stock_code") or "")
        pred = _to_float(row.get("pred_prob"))
        close_rate = _to_float(row.get("close_rate"))
        total_mv = _to_float(row.get("total_mv"))
        if not date or not code or pred is None:
            continue
        if pred > max_pred_prob:
            continue
        if _to_float(rank5d.get((date, code))) is None or float(rank5d[(date, code)]) < confirm_rank_min:
            continue
        if close_rate is None or close_rate < 0.965 or close_rate > 1.085:
            continue
        if total_mv is None or total_mv > 200000.0:
            continue
        if code.endswith(".BJ"):
            continue
        signal_market = market.get(date, {}).get(code)
        buy_market = market.get(next_date.get(date, ""), {}).get(code)
        if _is_st_like(signal_market) or _is_st_like(buy_market):
            continue
        if _is_unbuyable_next_day(buy_market):
            continue
        enriched = dict(row)
        if signal_market:
            for field in ("name", "amount", "turnover_rate", "total_mv", "limit_times"):
                if signal_market.get(field) not in (None, ""):
                    enriched[field] = signal_market[field]
        enriched["rank5d"] = rank5d.get((date, code))
        by_date.setdefault(date, []).append(enriched)
    selected: list[dict] = []
    for date, day_rows in sorted(by_date.items()):
        buy_date = next_date.get(date)
        if not buy_date:
            continue
        day_rows.sort(key=lambda item: (-float(item.get("pred_prob") or 0.0), str(item.get("stock_code") or "")))
        for rank, row in enumerate(day_rows[:5], start=1):
            selected.append(
                {
                    "signal_date": date,
                    "buy_date": buy_date,
                    "symbol": to_gm_symbol(str(row["stock_code"])),
                    "stock_code": row["stock_code"],
                    "name": row.get("name"),
                    "rank": rank,
                    "pred_prob": row.get("pred_prob"),
                    "atr_ratio": "",
                    "holding_days": 6,
                    "target_pct": 0.20,
                }
            )
    return selected


def _assign_targets(rows: list[dict], cfg: dict) -> list[dict]:
    features = _load_features()
    out_rows: list[dict] = []
    for row in rows:
        out = dict(row)
        feat = features.get((str(row["signal_date"]), str(row["stock_code"])), {})
        out.update(feat)
        gap = _to_float(out.get("pred_gap"), 999.0)
        amount = _to_float(out.get("amount"))
        total_mv = _to_float(out.get("total_mv"))
        rank = int(float(out.get("rank") or 999))
        role = "strong" if gap < float(cfg["strong_gap"]) else "weak"
        target = float(cfg["strong"]) if role == "strong" else float(cfg["weak"])
        if role == "weak" and rank in {3, 5}:
            target = min(target, float(cfg["weak_rank35"]))
        if gap >= 0.06 and gap < 0.09:
            target = min(target, float(cfg["gap0609"]))
        if total_mv is not None and 50000 <= total_mv < 100000:
            target = min(target, float(cfg["mv50_100"]))
        if amount is not None and amount < 10000:
            target = max(target, float(cfg["low_amount"]))
        target = min(target, float(cfg["cap"]))
        out["pool_role"] = role
        out["target_pct"] = f"{target:.5f}"
        out["holding_days"] = str(int(cfg["holding_days"]))
        out["max_holding_days"] = str(int(cfg["max_holding_days"]))
        out_rows.append(out)
    return out_rows


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
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
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


def _st_audit(rows: list[dict]) -> dict:
    market = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    hits = 0
    for row in rows:
        code = str(row.get("stock_code") or "")
        if _is_st_like(market.get(str(row.get("signal_date")), {}).get(code)):
            hits += 1
        if _is_st_like(market.get(str(row.get("buy_date")), {}).get(code)):
            hits += 1
    return {"st_check_hits": hits, "st_checked_pairs": len(rows) * 2}


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(cfg.get("env", {}))
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
        str(int(cfg["max_positions"])),
        "--holding-days",
        str(int(cfg["holding_days"])),
        "--max-holding-days",
        str(int(cfg["max_holding_days"])),
        "--target-position-pct",
        str(float(cfg["cap"])),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        "2024-06-05 09:00:00",
        "--backtest-end",
        "2026-06-18 15:30:00",
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


def _configs() -> list[dict]:
    configs = []
    for max_pred_prob in (0.50, 0.55):
        for confirm_rank_min in (0.50, 0.55):
            for max_positions in (5, 6, 7):
                configs.append(
                    {
                        "name": f"stclean_cap{str(max_pred_prob).replace('.', 'p')}_c{str(confirm_rank_min).replace('.', 'p')}_mp{max_positions}",
                        "max_pred_prob": max_pred_prob,
                        "confirm_rank_min": confirm_rank_min,
                        "strong_gap": 0.02,
                        "strong": 0.32,
                        "weak": 0.206,
                        "weak_rank35": 0.14,
                        "gap0609": 0.20,
                        "mv50_100": 0.20,
                        "low_amount": 0.42,
                        "cap": 0.42,
                        "holding_days": 6,
                        "max_holding_days": 6,
                        "max_positions": max_positions,
                        "env": {},
                    }
                )
    configs.append(
        {
            "name": "stclean_cap0p50_c0p50_mp6_score_mh2",
            "max_pred_prob": 0.50,
            "confirm_rank_min": 0.50,
            "strong_gap": 0.02,
            "strong": 0.32,
            "weak": 0.206,
            "weak_rank35": 0.14,
            "gap0609": 0.20,
            "mv50_100": 0.20,
            "low_amount": 0.42,
            "cap": 0.42,
            "holding_days": 6,
            "max_holding_days": 6,
            "max_positions": 6,
            "env": {"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"},
        }
    )
    return configs


def main() -> int:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    cache: dict[tuple[float, float], list[dict]] = {}
    for cfg in _configs():
        key = (float(cfg["max_pred_prob"]), float(cfg["confirm_rank_min"]))
        if key not in cache:
            cache[key] = _load_candidate_rows(*key)
        rows = _assign_targets(cache[key], cfg)
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        _write_rows(signal_file, rows)
        returncode = _run_backtest(cfg, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        day_counts: dict[str, int] = {}
        for row in rows:
            day_counts[str(row.get("signal_date"))] = day_counts.get(str(row.get("signal_date")), 0) + 1
        result = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "signal_count": len(rows),
            "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
            "days_ge3": sum(1 for value in day_counts.values() if value >= 3),
            "avg_per_day": len(rows) / len(day_counts) if day_counts else None,
            "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        result.update(_st_audit(rows))
        result.update(_exposure_stats(log_file))
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_to_float(row.get("annual"), -999) / 5.0, _to_float(row.get("sharpe"), -999) / 3.0, _to_float(row.get("avg_invested_pct"), -999) / 0.80), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _to_float(row.get("annual"), -999) >= 5.0 and _to_float(row.get("sharpe"), -999) >= 3.0 and _to_float(row.get("avg_invested_pct"), -999) >= 0.80 and int(row.get("st_check_hits") or 0) == 0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
