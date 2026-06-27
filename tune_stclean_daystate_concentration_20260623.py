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
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260623"
    / "stclean_daystate_concentration_tune"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260623"
    / "prod_balanced_10d_stclean_refill_tune"
    / "signals"
    / "stclean_cap0p5_c0p55_mp6.csv"
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
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")


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


def _variant(
    name: str,
    *,
    target_scale: float,
    cap: float,
    max_positions: int,
    min_score_hold: int,
    daystate: bool = True,
) -> dict:
    env = dict(BASE_ENV)
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(min_score_hold)
    return {
        "name": name,
        "target_scale": target_scale,
        "cap": cap,
        "max_positions": max_positions,
        "holding_days": 6,
        "max_holding_days": 6,
        "daystate": daystate,
        "env": env,
    }


VARIANTS = [
    _variant("base_recheck_mp6", target_scale=1.0, cap=0.42, max_positions=6, min_score_hold=4, daystate=False),
    _variant("avgmv_mp6_scale1p0_cap42", target_scale=1.0, cap=0.42, max_positions=6, min_score_hold=4),
    _variant("avgmv_mp5_scale1p15_cap50_mh3", target_scale=1.15, cap=0.50, max_positions=5, min_score_hold=3),
    _variant("avgmv_mp6_scale1p25_cap60_mh2", target_scale=1.25, cap=0.60, max_positions=6, min_score_hold=2),
    _variant("avgmv_mp5_scale1p40_cap60_mh2", target_scale=1.40, cap=0.60, max_positions=5, min_score_hold=2),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


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


def _daystate_scale(features: dict, enabled: bool) -> float:
    if not enabled:
        return 1.0
    avg_mv = _to_float(features.get("avg_mv"))
    if avg_mv is None:
        return 1.0
    if avg_mv < 132800.0:
        return 0.75
    if avg_mv < 171600.0:
        return 1.08
    return 1.0


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _load_rows(SOURCE_SIGNAL)
    for field in ["day_avg_mv", "day_state_scale", "raw_target_pct"]:
        if field not in fields:
            fields.append(field)
    features = _day_features(rows)
    day_sums: dict[str, float] = {}
    for row in rows:
        feat = features.get(str(row.get("signal_date")), {})
        day_scale = _daystate_scale(feat, bool(cfg["daystate"]))
        raw = _to_float(row.get("target_pct"), 0.0) or 0.0
        target = min(raw * float(cfg["target_scale"]) * day_scale, float(cfg["cap"]))
        row["raw_target_pct"] = f"{raw:.5f}"
        row["target_pct"] = f"{target:.5f}"
        row["holding_days"] = str(int(cfg["holding_days"]))
        row["max_holding_days"] = str(int(cfg["max_holding_days"]))
        row["day_avg_mv"] = "" if feat.get("avg_mv") is None else f"{float(feat['avg_mv']):.6f}"
        row["day_state_scale"] = f"{day_scale:.6f}"
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + target
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
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
        hits = 0
        checks = 0
        for row in csv.DictReader(signal_file.open(encoding="utf-8-sig")):
            code = str(row.get("stock_code") or "")
            for day in [row.get("signal_date"), row.get("buy_date")]:
                checks += 1
                rec = conn.execute(
                    "SELECT name, ST_TYPE, ST_TYPE_name FROM STOCK_DAILY_DATA WHERE trade_date=? AND stock_code=?",
                    (day, code),
                ).fetchone()
                if not rec:
                    continue
                name, st_type, st_name = [str(value or "") for value in rec]
                if (
                    name.startswith(("ST", "*ST"))
                    or "退" in name
                    or st_type.upper() not in ("", "0", "0.0", "NONE", "NAN")
                    or "风险警示" in st_name
                    or "退市" in st_name
                ):
                    hits += 1
        return {"st_check_hits": hits, "st_checked_pairs": checks}
    finally:
        conn.close()


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
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


def main() -> int:
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_stats = _write_signal(cfg, signal_file)
        returncode = _run_backtest(cfg, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            **signal_stats,
            **_st_audit(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} st={row.get('st_check_hits')} avg={row.get('avg_invested_pct')}",
            flush=True,
        )
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _to_float(row.get("annual"), -999) >= 5.0
            and _to_float(row.get("sharpe"), -999) >= 3.0
            and int(row.get("st_check_hits") or 0) == 0
            and int(row.get("signal_count") or 0) >= 350
            and int(row.get("buy_days") or 0) >= 250
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
