from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_bucket_target_20260622"
SOURCE_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap_strong_weak_fill_20260622" / "signals" / "gap02_s0p32_w0p18_h6_mh6.csv"
SCORE_DB = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_weak_day_pool_replace_20260621" / "weak_day_pool_scores.db"
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
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.85",
    "GM_EQUITY_DD_HARD_SCALE": "0.60",
}


def _variant(name: str, **kwargs) -> dict:
    cfg = {
        "name": name,
        "strong": 0.30,
        "weak": 0.18,
        "gap0609": None,
        "weak_rank35": None,
        "mv50_100": None,
        "bad_combo": None,
        "low_turnover": None,
        "low_amount": None,
        "gap_lt03": None,
        "holding_days": 6,
        "max_positions": 6,
        "cap": 0.30,
        "env": dict(BASE_ENV),
    }
    cfg.update(kwargs)
    return cfg


VARIANTS = [
    _variant("base_s030_w018"),
    _variant("gap0609_to010", gap0609=0.10),
    _variant("gap0609_to006", gap0609=0.06),
    _variant("weak_rank35_to010", weak_rank35=0.10),
    _variant("weak_rank35_to006", weak_rank35=0.06),
    _variant("mv50_100_to010", mv50_100=0.10),
    _variant("mv50_100_to006", mv50_100=0.06),
    _variant("combo_bad_to010", bad_combo=0.10),
    _variant("combo_bad_to006", bad_combo=0.06),
    _variant("combo_bad_to010_gap0609_to006", bad_combo=0.10, gap0609=0.06),
    _variant("weak_rank35_to006_strong032", strong=0.32, weak_rank35=0.06, cap=0.32),
    _variant("weak_rank35_to006_weak020", weak=0.20, weak_rank35=0.06),
    _variant("weak_rank35_to006_s032_w020", strong=0.32, weak=0.20, weak_rank35=0.06, cap=0.32),
    _variant("weak_rank35_to008_s032_w020", strong=0.32, weak=0.20, weak_rank35=0.08, cap=0.32),
    _variant("good_lowturnover032_bad008", strong=0.30, weak=0.18, weak_rank35=0.08, low_turnover=0.32, cap=0.32),
    _variant("good_lowamount032_bad008", strong=0.30, weak=0.18, weak_rank35=0.08, low_amount=0.32, cap=0.32),
    _variant("good_gaplt03_032_bad008", strong=0.30, weak=0.18, weak_rank35=0.08, gap_lt03=0.32, cap=0.32),
    _variant("good_all032_bad008", strong=0.30, weak=0.18, weak_rank35=0.08, low_turnover=0.32, low_amount=0.32, gap_lt03=0.32, cap=0.32),
    _variant("good_all034_bad008", strong=0.32, weak=0.20, weak_rank35=0.08, low_turnover=0.34, low_amount=0.34, gap_lt03=0.34, cap=0.34),
    _variant("good_all034_bad006", strong=0.32, weak=0.20, weak_rank35=0.06, low_turnover=0.34, low_amount=0.34, gap_lt03=0.34, cap=0.34),
    _variant("good_lowamount034_bad008", strong=0.30, weak=0.18, weak_rank35=0.08, low_amount=0.34, cap=0.34),
    _variant("good_lowamount036_bad008", strong=0.30, weak=0.18, weak_rank35=0.08, low_amount=0.36, cap=0.36),
    _variant("good_lowamount034_bad010", strong=0.30, weak=0.18, weak_rank35=0.10, low_amount=0.34, cap=0.34),
    _variant("good_lowamount034_s032_w020_bad008", strong=0.32, weak=0.20, weak_rank35=0.08, low_amount=0.34, cap=0.34),
    _variant("good_lowamount036_s032_w020_bad008", strong=0.32, weak=0.20, weak_rank35=0.08, low_amount=0.36, cap=0.36),
    _variant("good_lowamount034_s030_w020_bad008", strong=0.30, weak=0.20, weak_rank35=0.08, low_amount=0.34, cap=0.34),
    _variant("good_lowamount034_s030_w022_bad008", strong=0.30, weak=0.22, weak_rank35=0.08, low_amount=0.34, cap=0.34),
    _variant("good_lowamount034_s030_w024_bad008", strong=0.30, weak=0.24, weak_rank35=0.08, low_amount=0.34, cap=0.34),
    _variant("good_lowamount036_s030_w022_bad008", strong=0.30, weak=0.22, weak_rank35=0.08, low_amount=0.36, cap=0.36),
    _variant("good_lowamount040_s030_w022_bad008", strong=0.30, weak=0.22, weak_rank35=0.08, low_amount=0.40, cap=0.40),
    _variant("good_lowamount040_s032_w022_bad008", strong=0.32, weak=0.22, weak_rank35=0.08, low_amount=0.40, cap=0.40),
    _variant("good_lowamount040_s032_w024_bad008", strong=0.32, weak=0.24, weak_rank35=0.08, low_amount=0.40, cap=0.40),
    _variant("good_lowamount034_s030_w020_bad012", strong=0.30, weak=0.20, weak_rank35=0.12, low_amount=0.34, cap=0.34),
    _variant("good_lowamount034_s030_w020_bad014", strong=0.30, weak=0.20, weak_rank35=0.14, low_amount=0.34, cap=0.34),
    _variant("good_lowamount040_s030_w022_bad012", strong=0.30, weak=0.22, weak_rank35=0.12, low_amount=0.40, cap=0.40),
    _variant("good_lowamount040_s030_w022_bad014", strong=0.30, weak=0.22, weak_rank35=0.14, low_amount=0.40, cap=0.40),
    _variant("good_lowamount040_s032_w024_bad012", strong=0.32, weak=0.24, weak_rank35=0.12, low_amount=0.40, cap=0.40),
    _variant("good_lowamount040_s032_w024_bad014", strong=0.32, weak=0.24, weak_rank35=0.14, low_amount=0.40, cap=0.40),
    _variant("good_lowamount034_s030_w020_bad013", strong=0.30, weak=0.20, weak_rank35=0.13, low_amount=0.34, cap=0.34),
    _variant("good_lowamount034_s030_w020_bad015", strong=0.30, weak=0.20, weak_rank35=0.15, low_amount=0.34, cap=0.34),
    _variant("good_lowamount034_s030_w020_bad016", strong=0.30, weak=0.20, weak_rank35=0.16, low_amount=0.34, cap=0.34),
    _variant("good_lowamount032_s030_w020_bad014", strong=0.30, weak=0.20, weak_rank35=0.14, low_amount=0.32, cap=0.32),
    _variant("good_lowamount036_s030_w020_bad014", strong=0.30, weak=0.20, weak_rank35=0.14, low_amount=0.36, cap=0.36),
    _variant("good_lowamount034_s030_w018_bad014", strong=0.30, weak=0.18, weak_rank35=0.14, low_amount=0.34, cap=0.34),
    _variant("good_lowamount034_s032_w020_bad014", strong=0.32, weak=0.20, weak_rank35=0.14, low_amount=0.34, cap=0.34),
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


def _is_gap0609(row: dict) -> bool:
    gap = _to_float(row.get("pred_gap"), 999.0)
    return 0.06 <= gap < 0.09


def _is_weak_rank35(row: dict) -> bool:
    role = str(row.get("pool_role") or "")
    rank = int(float(row.get("rank") or 0))
    return role == "weak" and rank in {3, 5}


def _is_mv50_100(row: dict) -> bool:
    mv = _to_float(row.get("total_mv"), 999999999.0)
    return 50000.0 <= mv < 100000.0


def _is_low_turnover(row: dict) -> bool:
    return _to_float(row.get("turnover_rate"), 999.0) < 0.5


def _is_low_amount(row: dict) -> bool:
    return _to_float(row.get("amount"), 999999999.0) < 10000.0


def _is_gap_lt03(row: dict) -> bool:
    return _to_float(row.get("pred_gap"), 999.0) < 0.03


def _write_signal(cfg: dict, signal_file: Path) -> None:
    rows = _load_rows(SOURCE_SIGNAL)
    fieldnames = list(rows[0].keys())
    for row in rows:
        role = str(row.get("pool_role") or "")
        target = float(cfg["strong"] if role == "strong" else cfg["weak"])
        if cfg.get("gap0609") is not None and _is_gap0609(row):
            target = min(target, float(cfg["gap0609"]))
        if cfg.get("weak_rank35") is not None and _is_weak_rank35(row):
            target = min(target, float(cfg["weak_rank35"]))
        if cfg.get("mv50_100") is not None and _is_mv50_100(row):
            target = min(target, float(cfg["mv50_100"]))
        if cfg.get("bad_combo") is not None and (_is_gap0609(row) or _is_weak_rank35(row) or _is_mv50_100(row)):
            target = min(target, float(cfg["bad_combo"]))
        if cfg.get("low_turnover") is not None and _is_low_turnover(row):
            target = max(target, float(cfg["low_turnover"]))
        if cfg.get("low_amount") is not None and _is_low_amount(row):
            target = max(target, float(cfg["low_amount"]))
        if cfg.get("gap_lt03") is not None and _is_gap_lt03(row):
            target = max(target, float(cfg["gap_lt03"]))
        row["target_pct"] = f"{target:.5f}"
        row["holding_days"] = str(int(cfg["holding_days"]))
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


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
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    cmd = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", str(int(cfg["max_positions"])),
        "--holding-days", str(int(cfg["holding_days"])),
        "--max-holding-days", str(int(cfg["holding_days"])),
        "--target-position-pct", str(float(cfg["cap"])),
        "--score-db", str(SCORE_DB),
        "--score-table", SCORE_TABLE,
        "--market-db", str(MARKET_DB),
        "--backtest-start", "2024-06-05 09:00:00",
        "--backtest-end", "2026-06-18 15:30:00",
        "--backtest-adjust", "none",
        "--backtest-initial-cash", "600000",
        "--backtest-slippage-ratio", "0.0015",
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


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        _write_signal(cfg, signal_file)
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
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
