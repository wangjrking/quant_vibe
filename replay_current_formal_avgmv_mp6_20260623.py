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

from gm_signal_module import to_gm_symbol


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_avgmv_replay"
)
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "latest_formal_5d10d_stclean_refill"
    / "fusion_5d10d_latest_20260622.db"
)
SCORE_DB = REPORT_DIR / "current_formal_avgmv_scores.db"
SCORE_TABLE = "current_formal_10d_score"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
END_DATE = "20260622"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-23 15:30:00"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
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


CFG = {
    "name": "current_formal_avgmv_mp6_scale1p25_cap60_mh2_exact",
    "max_pred_prob": 0.50,
    "confirm_rank_min": 0.55,
    "strong_gap": 0.02,
    "strong": 0.32,
    "weak": 0.206,
    "weak_rank35": 0.14,
    "gap0609": 0.20,
    "mv50_100": 0.20,
    "low_amount": 0.42,
    "base_cap": 0.42,
    "target_scale": 1.25,
    "cap": 0.60,
    "holding_days": 6,
    "max_holding_days": 6,
    "max_positions": 6,
}

CFGS = [
    dict(CFG),
    {
        **CFG,
        "name": "current_formal_avgmv_mp6_scale1p25_cap60_mh2_p10cap055_diagnostic",
        "max_pred_prob": 0.55,
        "diagnostic_note": "Only relaxes old max_pred_prob from 0.50 to 0.55 because exact current formal 5D/10D intersection is empty.",
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
    st_type = row.get("st_type") if "st_type" in row else row.get("ST_TYPE")
    st_name = str(row.get("st_type_name") or row.get("ST_TYPE_name") or "")
    if name.startswith(("ST", "*ST")) or "退" in name:
        return True
    if "风险" in st_name or "退市" in st_name:
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


def _market_rows() -> dict[str, dict[str, dict]]:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open, close, amount,
                   turnover_rate, total_mv, limit_times, ST_TYPE AS st_type,
                   ST_TYPE_name AS st_type_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= ? AND trade_date <= ?
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        out.setdefault(str(row["trade_date"]), {})[str(row["stock_code"])] = dict(row)
    return out


def _date_map() -> dict[str, str]:
    conn = sqlite3.connect(FUSION_DB)
    try:
        dates = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT trade_date FROM fusion_rank_base WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                (START_DATE, END_DATE),
            )
        ]
    finally:
        conn.close()
    return {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}


def build_score_table() -> None:
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SCORE_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(FUSION_DB),))
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS {SCORE_TABLE};
            CREATE TABLE {SCORE_TABLE} AS
            SELECT trade_date, stock_code, pred_10d AS pred_prob
            FROM fusion.fusion_rank_base
            WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}';
            CREATE INDEX idx_{SCORE_TABLE}_trade_stock ON {SCORE_TABLE}(trade_date, stock_code);
            CREATE INDEX idx_{SCORE_TABLE}_trade_pred ON {SCORE_TABLE}(trade_date, pred_prob DESC);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _load_candidate_rows() -> list[dict]:
    market = _market_rows()
    next_date = _date_map()
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    try:
        base_rows = conn.execute(
            """
            SELECT trade_date, stock_code, pred_10d AS pred_prob, pred_5d,
                   rank_5d, rank_10d, close_rate, amount, turnover_rate,
                   total_mv, atr_qfq, close, name, limit_times
            FROM fusion_rank_base
            WHERE trade_date >= ? AND trade_date <= ?
              AND pred_10d IS NOT NULL
              AND pred_5d IS NOT NULL
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()
    by_date: dict[str, list[dict]] = {}
    for raw in base_rows:
        row = dict(raw)
        date = str(row.get("trade_date") or "")
        code = str(row.get("stock_code") or "")
        pred = _to_float(row.get("pred_prob"))
        rank5d = _to_float(row.get("rank_5d"))
        close_rate = _to_float(row.get("close_rate"))
        total_mv = _to_float(row.get("total_mv"))
        if not date or not code or pred is None:
            continue
        if pred > float(CFG["max_pred_prob"]):
            continue
        if rank5d is None or rank5d < float(CFG["confirm_rank_min"]):
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
        if signal_market:
            for field in ("name", "amount", "turnover_rate", "total_mv", "limit_times"):
                if signal_market.get(field) not in (None, ""):
                    row[field] = signal_market[field]
        by_date.setdefault(date, []).append(row)
    selected: list[dict] = []
    for date, day_rows in sorted(by_date.items()):
        buy_date = next_date.get(date)
        if not buy_date:
            continue
        day_rows.sort(key=lambda item: (-float(item.get("pred_prob") or 0.0), str(item.get("stock_code") or "")))
        for rank, row in enumerate(day_rows[:5], start=1):
            close = _to_float(row.get("close"))
            atr = _to_float(row.get("atr_qfq"))
            pred_10d = _to_float(row.get("pred_prob"))
            pred_5d = _to_float(row.get("pred_5d"))
            selected.append(
                {
                    "signal_date": date,
                    "buy_date": buy_date,
                    "symbol": to_gm_symbol(str(row["stock_code"])),
                    "stock_code": row["stock_code"],
                    "name": row.get("name"),
                    "rank": rank,
                    "pred_prob": row.get("pred_prob"),
                    "pred_5d": row.get("pred_5d"),
                    "pred_10d": row.get("pred_prob"),
                    "rank_5d": row.get("rank_5d"),
                    "rank_10d": row.get("rank_10d"),
                    "amount": row.get("amount"),
                    "turnover_rate": row.get("turnover_rate"),
                    "total_mv": row.get("total_mv"),
                    "atr_qfq": row.get("atr_qfq"),
                    "close": row.get("close"),
                    "close_rate": row.get("close_rate"),
                    "atr_ratio": atr / close if atr is not None and close and close > 0 else "",
                    "pred_gap": abs(pred_10d - pred_5d) if pred_10d is not None and pred_5d is not None else "",
                    "holding_days": int(CFG["holding_days"]),
                    "max_holding_days": int(CFG["max_holding_days"]),
                    "target_pct": 0.20,
                }
            )
    return selected


def _assign_base_targets(rows: list[dict]) -> list[dict]:
    out_rows: list[dict] = []
    for row in rows:
        out = dict(row)
        gap = _to_float(out.get("pred_gap"), 999.0)
        amount = _to_float(out.get("amount"))
        total_mv = _to_float(out.get("total_mv"))
        rank = int(float(out.get("rank") or 999))
        role = "strong" if gap < float(CFG["strong_gap"]) else "weak"
        target = float(CFG["strong"]) if role == "strong" else float(CFG["weak"])
        if role == "weak" and rank in {3, 5}:
            target = min(target, float(CFG["weak_rank35"]))
        if 0.06 <= gap < 0.09:
            target = min(target, float(CFG["gap0609"]))
        if total_mv is not None and 50000 <= total_mv < 100000:
            target = min(target, float(CFG["mv50_100"]))
        if amount is not None and amount < 10000:
            target = max(target, float(CFG["low_amount"]))
        target = min(target, float(CFG["base_cap"]))
        out["pool_role"] = role
        out["base_target_pct"] = f"{target:.5f}"
        out["target_pct"] = f"{target:.5f}"
        out_rows.append(out)
    return out_rows


def _day_features(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("signal_date")), []).append(row)
    out: dict[str, dict] = {}
    for day, items in grouped.items():
        mvs = [_to_float(row.get("total_mv")) for row in items]
        mvs = [value for value in mvs if value is not None]
        out[day] = {"avg_mv": sum(mvs) / len(mvs) if mvs else None}
    return out


def _daystate_scale(features: dict) -> float:
    avg_mv = _to_float(features.get("avg_mv"))
    if avg_mv is None:
        return 1.0
    if avg_mv < 132800.0:
        return 0.75
    if avg_mv < 171600.0:
        return 1.08
    return 1.0


def _apply_avgmv_overlay(rows: list[dict]) -> list[dict]:
    features = _day_features(rows)
    out_rows = []
    for row in rows:
        out = dict(row)
        feat = features.get(str(row.get("signal_date")), {})
        day_scale = _daystate_scale(feat)
        raw = _to_float(out.get("target_pct"), 0.0) or 0.0
        target = min(raw * float(CFG["target_scale"]) * day_scale, float(CFG["cap"]))
        out["raw_target_pct"] = f"{raw:.5f}"
        out["target_pct"] = f"{target:.5f}"
        out["day_avg_mv"] = "" if feat.get("avg_mv") is None else f"{float(feat['avg_mv']):.6f}"
        out["day_state_scale"] = f"{day_scale:.6f}"
        out_rows.append(out)
    return out_rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


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


def _st_audit(signal_file: Path) -> dict:
    conn = sqlite3.connect(MARKET_DB)
    try:
        signal_hits = 0
        buy_hits = 0
        limit_hits = 0
        checks = 0
        for row in csv.DictReader(signal_file.open(encoding="utf-8-sig")):
            code = str(row.get("stock_code") or "")
            for key in ("signal_date", "buy_date"):
                day = str(row.get(key) or "")
                checks += 1
                rec = conn.execute(
                    """
                    SELECT name, ST_TYPE AS st_type, ST_TYPE_name AS st_type_name, limit_times
                    FROM STOCK_DAILY_DATA
                    WHERE trade_date=? AND stock_code=?
                    """,
                    (day, code),
                ).fetchone()
                if not rec:
                    continue
                market_row = {
                    "name": rec[0],
                    "st_type": rec[1],
                    "st_type_name": rec[2],
                    "limit_times": rec[3],
                }
                if _is_st_like(market_row):
                    if key == "signal_date":
                        signal_hits += 1
                    else:
                        buy_hits += 1
                if key == "buy_date" and not _is_missing(rec[3]) and (_to_float(rec[3], 0.0) or 0.0) > 0:
                    limit_hits += 1
        return {
            "st_checked_pairs": checks,
            "signal_st_or_risk_warning": signal_hits,
            "buy_st_or_risk_warning": buy_hits,
            "buy_limit_times_gt0": limit_hits,
        }
    finally:
        conn.close()


def _signal_stats(rows: list[dict]) -> dict:
    day_sums: dict[str, float] = {}
    for row in rows:
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
    }


def _run_backtest(signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(BASE_ENV)
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
        str(int(CFG["max_positions"])),
        "--holding-days",
        str(int(CFG["holding_days"])),
        "--max-holding-days",
        str(int(CFG["max_holding_days"])),
        "--target-position-pct",
        str(float(CFG["cap"])),
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


def _run_one() -> dict:
    base_rows = _assign_base_targets(_load_candidate_rows())
    rows = _apply_avgmv_overlay(base_rows)
    signal_file = REPORT_DIR / "signals" / f"{CFG['name']}.csv"
    log_file = REPORT_DIR / "logs" / f"{CFG['name']}.log"
    _write_rows(signal_file, rows)
    if rows:
        returncode = _run_backtest(signal_file, log_file)
    else:
        returncode = 2
    indicator = _extract_indicator(log_file) or {}
    summary = {
        "name": CFG["name"],
        "returncode": returncode,
        "status": "no_signal" if not rows else "backtested",
        "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
        "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
        "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "config_json": json.dumps(CFG, ensure_ascii=False, sort_keys=True),
        "env_json": json.dumps(BASE_ENV, ensure_ascii=False, sort_keys=True),
        "fusion_db": str(FUSION_DB),
        "score_db": str(SCORE_DB),
        "score_table": SCORE_TABLE,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        **_signal_stats(rows),
        **_st_audit(signal_file),
        **_exposure_stats(log_file),
    }
    return summary


def main() -> int:
    if not FUSION_DB.exists():
        raise SystemExit(f"fusion db not found: {FUSION_DB}")
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"juejin strategy main.py not found: {STRATEGY_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    build_score_table()
    summaries = []
    global CFG
    for cfg in CFGS:
        CFG = dict(cfg)
        summary = _run_one()
        summaries.append(summary)
        _write_rows(REPORT_DIR / "summary.csv", summaries)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0 if any(row.get("status") == "backtested" for row in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
