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


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623" / "formal_l4_bucket_scale_stclean"
SOURCE_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_bucket_target_20260622"
SCORE_DB = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_weak_day_pool_replace_20260621" / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
TARGET_ANNUAL = 25.672466012925824


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
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.85",
    "GM_EQUITY_DD_HARD_SCALE": "0.60",
}


SOURCES = [
    "good_lowamount040_s032_w022_bad008",
    "good_lowamount034_s030_w020_bad014",
    "good_lowamount034_s030_w018_bad014",
]


def _variant(source: str, scale: float, risk_mode: int) -> dict:
    suffix = str(scale).replace(".", "p")
    return {
        "name": f"{source}_scale{suffix}_risk{risk_mode}",
        "source": source,
        "source_file": SOURCE_DIR / "signals" / f"{source}.csv",
        "scale": scale,
        "risk_mode": risk_mode,
        "max_positions": 6,
        "holding_days": 6,
        "max_holding_days": 6,
        "target_cap": 0.60,
    }


VARIANTS = [
    _variant(source, scale, risk_mode)
    for source in SOURCES
    for scale in [1.10, 1.25, 1.40]
    for risk_mode in [1, 0]
]


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


def _is_bad_st(name, st_type, st_type_name) -> bool:
    name_text = str(name or "")
    st_text = str(st_type or "").strip()
    st_name_text = str(st_type_name or "")
    if name_text.startswith("ST") or name_text.startswith("*ST"):
        return True
    if "风险警示" in st_name_text or "退市风险" in st_name_text:
        return True
    return st_text not in {"", "0", "0.0", "None", "NONE", "nan", "NaN"}


def _is_limit_up(limit_times) -> bool:
    try:
        return limit_times not in (None, "", "None") and float(limit_times) > 0.0
    except Exception:
        return bool(limit_times)


def _load_market_gate(rows: list[dict]) -> dict[tuple[str, str], tuple]:
    keys = {
        (str(row.get("stock_code") or ""), str(date or ""))
        for row in rows
        for date in [row.get("signal_date"), row.get("buy_date")]
        if row.get("stock_code") and date
    }
    if not keys:
        return {}
    conn = sqlite3.connect(MARKET_DB)
    try:
        out = {}
        codes = sorted({code for code, _ in keys})
        dates = sorted({date for _, date in keys})
        for code_start in range(0, len(codes), 300):
            code_chunk = codes[code_start : code_start + 300]
            code_ph = ",".join("?" for _ in code_chunk)
            date_ph = ",".join("?" for _ in dates)
            query = f"""
                SELECT stock_code, trade_date, name, ST_TYPE, ST_TYPE_name, limit_times
                FROM STOCK_DAILY_DATA
                WHERE stock_code IN ({code_ph})
                  AND trade_date IN ({date_ph})
            """
            for row in conn.execute(query, [*code_chunk, *dates]):
                out[(str(row[0]), str(row[1]))] = row[2:]
        return out
    finally:
        conn.close()


def _passes_market_gate(row: dict, market_gate: dict[tuple[str, str], tuple]) -> tuple[bool, str | None]:
    code = str(row.get("stock_code") or "")
    signal_date = str(row.get("signal_date") or "")
    buy_date = str(row.get("buy_date") or "")
    if code.endswith(".BJ") or code.startswith("8") or code.startswith("4"):
        return False, "bj"
    for date, phase in [(signal_date, "signal"), (buy_date, "buy")]:
        market_row = market_gate.get((code, date))
        if market_row is None:
            return False, f"{phase}_missing_market"
        name, st_type, st_type_name, limit_times = market_row
        if _is_bad_st(name, st_type, st_type_name):
            return False, f"{phase}_st"
        if phase == "signal" and _is_limit_up(limit_times):
            return False, "signal_limit_up"
        if phase == "buy" and _is_limit_up(limit_times):
            return False, "buy_limit_up"
    return True, None


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows = _load_rows(cfg["source_file"])
    market_gate = _load_market_gate(rows)
    fieldnames = list(rows[0].keys())
    day_target: dict[str, float] = {}
    output_rows = []
    drop_counts: dict[str, int] = {}
    for row in rows:
        ok, reason = _passes_market_gate(row, market_gate)
        if not ok:
            drop_counts[str(reason)] = drop_counts.get(str(reason), 0) + 1
            continue
        old_target = _to_float(row.get("target_pct"), 0.0) or 0.0
        new_target = min(float(cfg["target_cap"]), old_target * float(cfg["scale"]))
        row["target_pct"] = f"{new_target:.5f}"
        row["holding_days"] = str(int(cfg["holding_days"]))
        day = str(row.get("buy_date") or "")
        day_target[day] = day_target.get(day, 0.0) + new_target
        output_rows.append(row)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in output_rows])
    return {
        "signal_count": len(output_rows),
        "buy_days": len(day_target),
        "avg_day_target_sum": sum(day_target.values()) / len(day_target) if day_target else None,
        "min_day_target_sum": min(day_target.values()) if day_target else None,
        "max_day_target_sum": max(day_target.values()) if day_target else None,
        "dropped_json": json.dumps(drop_counts, ensure_ascii=False, sort_keys=True),
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


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["GM_EQUITY_DD_RISK_MODE"] = str(int(cfg["risk_mode"]))
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
        str(int(cfg["max_positions"])),
        "--holding-days",
        str(int(cfg["holding_days"])),
        "--max-holding-days",
        str(int(cfg["max_holding_days"])),
        "--target-position-pct",
        str(float(cfg["target_cap"])),
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
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


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
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    results = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_stats = _write_signal(cfg, signal_file)
        returncode = _run_backtest(cfg, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            **cfg,
            "source_file": str(cfg["source_file"]),
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            **signal_stats,
            **_exposure_stats(log_file),
        }
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}",
            flush=True,
        )
    ranked = sorted(results, key=lambda row: (_metric(row, "annual"), _metric(row, "sharpe")), reverse=True)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", ranked)
    _write_rows(
        REPORT_DIR / "summary_target_over_2567.csv",
        [
            row
            for row in ranked
            if _metric(row, "annual") > TARGET_ANNUAL
            and int(row.get("signal_count") or 0) >= 40
            and int(row.get("buy_days") or 0) >= 30
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
