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
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_core_plus_best10d_filler_20260622"
)
CORE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_core_breadth_target_scale2_20260622"
    / "signals"
    / "up0p7_g1p35_b0p7_cap0p32.csv"
)
FILLER_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
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
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
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
    *,
    day_target_cap: float,
    filler_scale: float,
    filler_single_cap: float,
    filler_rank_limit: int,
    core_scale: float = 1.0,
    max_positions: int = 8,
    min_filler_gap: float | None = None,
    max_filler_turnover: float | None = None,
) -> dict:
    return {
        "name": name,
        "day_target_cap": day_target_cap,
        "filler_scale": filler_scale,
        "filler_single_cap": filler_single_cap,
        "filler_rank_limit": filler_rank_limit,
        "core_scale": core_scale,
        "max_positions": max_positions,
        "min_filler_gap": min_filler_gap,
        "max_filler_turnover": max_filler_turnover,
        "env": dict(BASE_ENV),
    }


VARIANTS = [
    _variant("cap80_fill15_rank2", day_target_cap=0.80, filler_scale=0.15, filler_single_cap=0.08, filler_rank_limit=2),
    _variant("cap80_fill20_rank2", day_target_cap=0.80, filler_scale=0.20, filler_single_cap=0.10, filler_rank_limit=2),
    _variant("cap80_fill25_rank2", day_target_cap=0.80, filler_scale=0.25, filler_single_cap=0.12, filler_rank_limit=2),
    _variant("cap84_fill20_rank2", day_target_cap=0.84, filler_scale=0.20, filler_single_cap=0.10, filler_rank_limit=2),
    _variant("cap84_fill25_rank2", day_target_cap=0.84, filler_scale=0.25, filler_single_cap=0.12, filler_rank_limit=2),
    _variant("cap84_fill30_rank2", day_target_cap=0.84, filler_scale=0.30, filler_single_cap=0.14, filler_rank_limit=2),
    _variant("cap80_fill20_rank3", day_target_cap=0.80, filler_scale=0.20, filler_single_cap=0.10, filler_rank_limit=3),
    _variant("cap84_fill25_rank3", day_target_cap=0.84, filler_scale=0.25, filler_single_cap=0.12, filler_rank_limit=3),
    _variant("cap88_fill25_rank3", day_target_cap=0.88, filler_scale=0.25, filler_single_cap=0.12, filler_rank_limit=3),
    _variant("cap84_core110_fill20_rank2", day_target_cap=0.84, filler_scale=0.20, filler_single_cap=0.10, filler_rank_limit=2, core_scale=1.10),
    _variant("cap84_gap06_fill25_rank3", day_target_cap=0.84, filler_scale=0.25, filler_single_cap=0.12, filler_rank_limit=3, min_filler_gap=0.06),
    _variant("cap84_turn4_fill25_rank3", day_target_cap=0.84, filler_scale=0.25, filler_single_cap=0.12, filler_rank_limit=3, max_filler_turnover=4.0),
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


def _target(row: dict) -> float:
    return _to_float(row.get("target_pct"), 0.0) or 0.0


def _rank(row: dict) -> int:
    return int(float(row.get("rank") or 999999))


def _day_key(row: dict) -> tuple[str, str]:
    return str(row["signal_date"]), str(row["buy_date"])


def _passes_filler(row: dict, cfg: dict) -> bool:
    if _rank(row) > int(cfg["filler_rank_limit"]):
        return False
    min_gap = cfg.get("min_filler_gap")
    if min_gap is not None and (_to_float(row.get("pred_gap"), -999.0) or -999.0) < float(min_gap):
        return False
    max_turnover = cfg.get("max_filler_turnover")
    if max_turnover is not None and (_to_float(row.get("turnover_rate"), 999.0) or 999.0) > float(max_turnover):
        return False
    return True


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    core_by_day: dict[tuple[str, str], list[dict]] = {}
    for row in _read_csv(CORE_SIGNAL):
        item = dict(row)
        item["source_role"] = "core"
        item["target_pct"] = f"{min(_target(item) * float(cfg['core_scale']), 0.32):.5f}"
        core_by_day.setdefault(_day_key(item), []).append(item)

    filler_by_day: dict[tuple[str, str], list[dict]] = {}
    for row in _read_csv(FILLER_SIGNAL):
        if not _passes_filler(row, cfg):
            continue
        item = dict(row)
        item["source_role"] = "best10d_filler"
        item["target_pct"] = f"{min(_target(item) * float(cfg['filler_scale']), float(cfg['filler_single_cap'])):.5f}"
        item["holding_days"] = "6"
        filler_by_day.setdefault(_day_key(item), []).append(item)

    output: list[dict] = []
    for day_key in sorted(set(core_by_day) | set(filler_by_day)):
        selected = sorted(core_by_day.get(day_key, []), key=_rank)
        used_codes = {str(row.get("stock_code")) for row in selected}
        target_sum = sum(_target(row) for row in selected)
        next_rank = len(selected) + 1
        for row in sorted(filler_by_day.get(day_key, []), key=_rank):
            if str(row.get("stock_code")) in used_codes:
                continue
            remaining = float(cfg["day_target_cap"]) - target_sum
            if remaining <= 0:
                break
            item = dict(row)
            item["target_pct"] = f"{min(_target(item), remaining):.5f}"
            item["rank"] = str(next_rank)
            selected.append(item)
            used_codes.add(str(item.get("stock_code")))
            target_sum += _target(item)
            next_rank += 1
            if len(selected) >= int(cfg["max_positions"]):
                break
        output.extend(sorted(selected, key=_rank))

    fields: list[str] = []
    for row in output:
        for key in row:
            if key not in fields:
                fields.append(key)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    day_sums: dict[str, float] = {}
    source_counts: dict[str, int] = {}
    for row in output:
        day_sums.setdefault(str(row.get("buy_date")), 0.0)
        day_sums[str(row.get("buy_date"))] += _target(row)
        source = str(row.get("source_role") or "")
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "signal_count": len(output),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
        "source_counts_json": json.dumps(source_counts, ensure_ascii=False, sort_keys=True),
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
        "7",
        "--max-holding-days",
        "10",
        "--target-position-pct",
        "0.28",
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
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}", flush=True)

    by_objective = sorted(results, key=_objective, reverse=True)
    _write_rows(REPORT_DIR / "summary_by_objective.csv", by_objective)
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: (_metric(row, "sharpe"), _metric(row, "annual")), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0
            and _metric(row, "sharpe") >= 4.0
            and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
