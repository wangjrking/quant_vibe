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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_regime_switch_20260622"
DEFENSIVE_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_core_breadth_target_scale2_20260622" / "signals" / "up0p7_g1p35_b0p7_cap0p32.csv"
AGGRESSIVE_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_bucket_target_20260622" / "signals" / "good_lowamount036_s030_w020_bad014.csv"
AGGRESSIVE_ALT_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_bucket_target_20260622" / "signals" / "good_lowamount034_s030_w020_bad008.csv"
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


def _variant(
    name: str,
    threshold: float,
    mode: str,
    aggressive_alt: bool = False,
    defensive_scale: float = 1.0,
    aggressive_scale: float = 1.0,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "threshold": threshold,
        "mode": mode,
        "aggressive_alt": aggressive_alt,
        "defensive_scale": defensive_scale,
        "aggressive_scale": aggressive_scale,
        "env": env,
    }


VARIANTS = []
for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    tag = str(threshold).replace(".", "p")
    VARIANTS.append(_variant(f"aggr_high_t{tag}", threshold, "aggressive_when_high"))
    VARIANTS.append(_variant(f"aggr_low_t{tag}", threshold, "aggressive_when_low"))
for threshold in [0.30, 0.40, 0.50]:
    tag = str(threshold).replace(".", "p")
    VARIANTS.append(_variant(f"alt_aggr_high_t{tag}", threshold, "aggressive_when_high", aggressive_alt=True))
    VARIANTS.append(_variant(f"alt_aggr_low_t{tag}", threshold, "aggressive_when_low", aggressive_alt=True))
VARIANTS.extend(
    [
        _variant("aggr_high_t0p40_def120", 0.40, "aggressive_when_high", defensive_scale=1.20),
        _variant("aggr_high_t0p50_def120", 0.50, "aggressive_when_high", defensive_scale=1.20),
        _variant("aggr_low_t0p40_def120", 0.40, "aggressive_when_low", defensive_scale=1.20),
        _variant("aggr_low_t0p50_def120", 0.50, "aggressive_when_low", defensive_scale=1.20),
        _variant("aggr_high_t0p40_strict", 0.40, "aggressive_when_high", extra_env={"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.16", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
        _variant("aggr_low_t0p40_strict", 0.40, "aggressive_when_low", extra_env={"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.16", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
    ]
)


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_market_up_ratio() -> dict[str, float]:
    conn = sqlite3.connect(MARKET_DB)
    try:
        rows = conn.execute(
            """
            SELECT trade_date,
                   AVG(CASE WHEN close > pre_close THEN 1.0 ELSE 0.0 END) AS up_ratio
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= '20240604' AND trade_date <= '20260618'
              AND close IS NOT NULL AND pre_close IS NOT NULL AND pre_close > 0
              AND stock_code NOT LIKE '%.BJ'
            GROUP BY trade_date
            """
        ).fetchall()
    finally:
        conn.close()
    return {str(row[0]): float(row[1]) for row in rows if row[1] is not None}


def _target(row: dict) -> float:
    return _to_float(row.get("target_pct"), 0.0) or 0.0


def _rows_by_day(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        out.setdefault((str(row["signal_date"]), str(row["buy_date"])), []).append(row)
    return out


def _scale_rows(rows: list[dict], scale: float, source_role: str) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["target_pct"] = f"{_target(item) * float(scale):.5f}"
        item["source_role"] = source_role
        out.append(item)
    return out


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    defensive_by_day = _rows_by_day(_read_csv(DEFENSIVE_SIGNAL))
    aggressive_path = AGGRESSIVE_ALT_SIGNAL if cfg["aggressive_alt"] else AGGRESSIVE_SIGNAL
    aggressive_by_day = _rows_by_day(_read_csv(aggressive_path))
    up_ratio = _load_market_up_ratio()
    output = []
    role_counts = {"aggressive": 0, "defensive": 0}
    all_days = sorted(set(defensive_by_day) | set(aggressive_by_day))
    for day_key in all_days:
        signal_date, _buy_date = day_key
        ratio = up_ratio.get(signal_date)
        if ratio is None:
            use_aggressive = True
        elif cfg["mode"] == "aggressive_when_high":
            use_aggressive = ratio >= float(cfg["threshold"])
        else:
            use_aggressive = ratio <= float(cfg["threshold"])
        if use_aggressive:
            rows = _scale_rows(aggressive_by_day.get(day_key) or defensive_by_day.get(day_key, []), float(cfg["aggressive_scale"]), "aggressive")
            role_counts["aggressive"] += 1
        else:
            rows = _scale_rows(defensive_by_day.get(day_key) or aggressive_by_day.get(day_key, []), float(cfg["defensive_scale"]), "defensive")
            role_counts["defensive"] += 1
        rows.sort(key=lambda item: int(float(item.get("rank") or 9999)))
        for index, row in enumerate(rows, start=1):
            row["rank"] = str(index)
            output.append(row)
    fields = []
    for row in output:
        for key in row:
            if key not in fields:
                fields.append(key)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    day_sums = {}
    for row in output:
        day_sums.setdefault(row["buy_date"], 0.0)
        day_sums[row["buy_date"]] += _target(row)
    return {
        "signal_count": len(output),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
        "aggressive_days": role_counts["aggressive"],
        "defensive_days": role_counts["defensive"],
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
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    cmd = [
        str(JUEJIN_PYTHON), str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", "8",
        "--holding-days", "7",
        "--max-holding-days", "10",
        "--target-position-pct", "0.42",
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
    fields = []
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


def _objective(row: dict) -> float:
    return min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80)


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
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
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
