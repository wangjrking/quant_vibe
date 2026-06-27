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

import research_formal_10d_bucket_cap_grid_20260622 as base_grid


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_best_core_breadth_switch_20260622"
)
BEST_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
)
CORE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_core_breadth_target_scale2_20260622"
    / "signals"
    / "up0p7_g1p35_b0p7_cap0p32.csv"
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


BASE_ENV = dict(base_grid.BASE_ENV)
BASE_ENV.update(base_grid.EQDD_PROFILES["mid"])


def _variant(
    name: str,
    *,
    threshold: float,
    best_when: str,
    best_scale: float = 1.0,
    core_scale: float = 1.0,
    max_positions: int = 8,
    target_position_pct: float = 0.42,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "threshold": threshold,
        "best_when": best_when,
        "best_scale": best_scale,
        "core_scale": core_scale,
        "max_positions": max_positions,
        "target_position_pct": target_position_pct,
        "env": env,
    }


VARIANTS = [
    _variant("best_high_t20", threshold=0.20, best_when="high"),
    _variant("best_high_t30", threshold=0.30, best_when="high"),
    _variant("best_high_t40", threshold=0.40, best_when="high"),
    _variant("best_high_t50", threshold=0.50, best_when="high"),
    _variant("best_high_t60", threshold=0.60, best_when="high"),
    _variant("best_low_t30", threshold=0.30, best_when="low"),
    _variant("best_low_t40", threshold=0.40, best_when="low"),
    _variant("best_low_t50", threshold=0.50, best_when="low"),
    _variant("best_high_t40_core120", threshold=0.40, best_when="high", core_scale=1.20),
    _variant("best_high_t50_core120", threshold=0.50, best_when="high", core_scale=1.20),
    _variant("best_high_t40_softbest", threshold=0.40, best_when="high", best_scale=0.90, core_scale=1.25, target_position_pct=0.38),
    _variant("best_high_t50_softbest", threshold=0.50, best_when="high", best_scale=0.90, core_scale=1.25, target_position_pct=0.38),
]


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


def _load_up_ratio() -> dict[str, float]:
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


def _day_key(row: dict) -> tuple[str, str]:
    return str(row["signal_date"]), str(row["buy_date"])


def _target(row: dict) -> float:
    return _to_float(row.get("target_pct"), 0.0) or 0.0


def _rank(row: dict) -> int:
    return int(float(row.get("rank") or 999999))


def _rows_by_day(path: Path) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for row in _read_csv(path):
        out.setdefault(_day_key(row), []).append(dict(row))
    for rows in out.values():
        rows.sort(key=_rank)
    return out


def _scale(rows: list[dict], scale: float, role: str) -> list[dict]:
    out = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["rank"] = str(index)
        item["source_role"] = role
        item["target_pct"] = f"{_target(item) * float(scale):.5f}"
        out.append(item)
    return out


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    best_by_day = _rows_by_day(BEST_SIGNAL)
    core_by_day = _rows_by_day(CORE_SIGNAL)
    up_ratio = _load_up_ratio()
    output: list[dict] = []
    role_days = {"best": 0, "core": 0}

    for day in sorted(set(best_by_day) | set(core_by_day)):
        signal_date, _buy_date = day
        ratio = up_ratio.get(signal_date)
        use_best = True
        if ratio is not None:
            if cfg["best_when"] == "high":
                use_best = ratio >= float(cfg["threshold"])
            else:
                use_best = ratio <= float(cfg["threshold"])
        if use_best:
            rows = _scale(best_by_day.get(day) or core_by_day.get(day, []), float(cfg["best_scale"]), "best10d")
            role_days["best"] += 1
        else:
            rows = _scale(core_by_day.get(day) or best_by_day.get(day, []), float(cfg["core_scale"]), "core5d10d")
            role_days["core"] += 1
        output.extend(rows)

    fields: list[str] = []
    for row in output:
        for key in row:
            if key not in fields:
                fields.append(key)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in output])

    day_sums: dict[str, float] = {}
    role_counts = {"best10d": 0, "core5d10d": 0}
    for row in output:
        day_sums[str(row.get("buy_date"))] = day_sums.get(str(row.get("buy_date")), 0.0) + _target(row)
        role = str(row.get("source_role") or "")
        role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "signal_count": len(output),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
        "role_days_json": json.dumps(role_days, ensure_ascii=False, sort_keys=True),
        "role_counts_json": json.dumps(role_counts, ensure_ascii=False, sort_keys=True),
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
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(cfg["max_positions"]),
        "--holding-days",
        "6",
        "--max-holding-days",
        "6",
        "--target-position-pct",
        str(cfg["target_position_pct"]),
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
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
