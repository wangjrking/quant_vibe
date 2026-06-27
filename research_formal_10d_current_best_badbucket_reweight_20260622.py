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

import research_formal_10d_bucket_cap_grid_20260622 as base_grid


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_badbucket_reweight_20260622"
)
SOURCE_SIGNAL = (
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


def _variant(
    name: str,
    rules: list[dict],
    *,
    cap: float = 0.52,
    normalize: str = "same_sum",
    max_day_sum_mult: float = 1.04,
    eqdd: str = "mid",
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(base_grid.BASE_ENV)
    env.update(base_grid.EQDD_PROFILES[eqdd])
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "rules": rules,
        "cap": cap,
        "normalize": normalize,
        "max_day_sum_mult": max_day_sum_mult,
        "eqdd": eqdd,
        "env": env,
    }


BAD_PRED10 = {"field": "pred_10d", "op": "between", "low": 0.08, "high": 0.12}
BAD_GAP = {"field": "pred_gap", "op": "between", "low": 0.05, "high": 0.07}
BAD_MV = {"field": "total_mv", "op": "between", "low": 80000.0, "high": 120000.0}
GOOD_LOW_AMOUNT = {"field": "amount", "op": "lt", "threshold": 10000.0}
GOOD_LOW_TURN = {"field": "turnover_rate", "op": "lt", "threshold": 1.0}
GOOD_HIGH_PRED10 = {"field": "pred_10d", "op": "ge", "threshold": 0.16}
GOOD_SMALL_MV = {"field": "total_mv", "op": "lt", "threshold": 80000.0}
GOOD_LARGE_MV = {"field": "total_mv", "op": "ge", "threshold": 180000.0}


def _rule(template: dict, scale: float) -> dict:
    out = dict(template)
    out["scale"] = scale
    return out


VARIANTS = [
    _variant("bad_pred10_down70_same", [_rule(BAD_PRED10, 0.70)]),
    _variant("bad_pred10_down55_same", [_rule(BAD_PRED10, 0.55)]),
    _variant("bad_gap_down70_same", [_rule(BAD_GAP, 0.70)]),
    _variant("bad_gap_down55_same", [_rule(BAD_GAP, 0.55)]),
    _variant("bad_mv_down75_same", [_rule(BAD_MV, 0.75)]),
    _variant("bad_mv_down60_same", [_rule(BAD_MV, 0.60)]),
    _variant("bad_combo_soft_same", [
        _rule(BAD_PRED10, 0.75),
        _rule(BAD_GAP, 0.75),
        _rule(BAD_MV, 0.82),
    ]),
    _variant("bad_combo_mid_same", [
        _rule(BAD_PRED10, 0.62),
        _rule(BAD_GAP, 0.65),
        _rule(BAD_MV, 0.72),
    ]),
    _variant("bad_combo_hard_same", [
        _rule(BAD_PRED10, 0.50),
        _rule(BAD_GAP, 0.55),
        _rule(BAD_MV, 0.62),
    ]),
    _variant("bad_good_soft_same", [
        _rule(BAD_PRED10, 0.68),
        _rule(BAD_GAP, 0.70),
        _rule(BAD_MV, 0.78),
        _rule(GOOD_LOW_AMOUNT, 1.12),
        _rule(GOOD_LOW_TURN, 1.10),
        _rule(GOOD_HIGH_PRED10, 1.15),
        _rule(GOOD_SMALL_MV, 1.08),
        _rule(GOOD_LARGE_MV, 1.04),
    ]),
    _variant("bad_good_mid_same", [
        _rule(BAD_PRED10, 0.55),
        _rule(BAD_GAP, 0.60),
        _rule(BAD_MV, 0.70),
        _rule(GOOD_LOW_AMOUNT, 1.18),
        _rule(GOOD_LOW_TURN, 1.15),
        _rule(GOOD_HIGH_PRED10, 1.25),
        _rule(GOOD_SMALL_MV, 1.12),
        _rule(GOOD_LARGE_MV, 1.06),
    ], cap=0.55),
    _variant("bad_good_mid_grow", [
        _rule(BAD_PRED10, 0.55),
        _rule(BAD_GAP, 0.60),
        _rule(BAD_MV, 0.70),
        _rule(GOOD_LOW_AMOUNT, 1.18),
        _rule(GOOD_LOW_TURN, 1.15),
        _rule(GOOD_HIGH_PRED10, 1.25),
        _rule(GOOD_SMALL_MV, 1.12),
        _rule(GOOD_LARGE_MV, 1.06),
    ], cap=0.55, normalize="grow", max_day_sum_mult=1.06),
    _variant("pred10_down50_cap48_same", [_rule(BAD_PRED10, 0.50)], cap=0.48),
    _variant("pred10_down50_cap55_same", [_rule(BAD_PRED10, 0.50)], cap=0.55),
    _variant("pred10_down60_cap48_same", [_rule(BAD_PRED10, 0.60)], cap=0.48),
    _variant("pred10_down60_cap55_same", [_rule(BAD_PRED10, 0.60)], cap=0.55),
    _variant("pred10_down55_base_same", [_rule(BAD_PRED10, 0.55)], eqdd="base"),
    _variant("pred10_down55_strict_same", [_rule(BAD_PRED10, 0.55)], eqdd="strict"),
    _variant("mv_down70_cap48_same", [_rule(BAD_MV, 0.70)], cap=0.48),
    _variant("mv_down80_cap48_same", [_rule(BAD_MV, 0.80)], cap=0.48),
    _variant("mv_down70_cap55_same", [_rule(BAD_MV, 0.70)], cap=0.55),
    _variant("pred10_down55_mv_down90_same", [
        _rule(BAD_PRED10, 0.55),
        _rule(BAD_MV, 0.90),
    ], cap=0.52),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _matches(value: float | None, rule: dict) -> bool:
    if value is None:
        return False
    op = str(rule["op"])
    if op == "lt":
        return value < float(rule["threshold"])
    if op == "le":
        return value <= float(rule["threshold"])
    if op == "ge":
        return value >= float(rule["threshold"])
    if op == "gt":
        return value > float(rule["threshold"])
    if op == "between":
        return float(rule["low"]) <= value < float(rule["high"])
    raise ValueError(f"unsupported op: {op}")


def _scale(row: dict, rules: list[dict]) -> float:
    out = 1.0
    for rule in rules:
        if _matches(_to_float(row.get(rule["field"])), rule):
            out *= float(rule["scale"])
    return out


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in ["badbucket_scale", "badbucket_source_target"]:
        if field not in fields:
            fields.append(field)

    original_by_day: dict[str, float] = {}
    scaled_by_day: dict[str, float] = {}
    for row in rows:
        day = str(row.get("signal_date"))
        base_target = _to_float(row.get("target_pct"), 0.0) or 0.0
        scale = _scale(row, cfg["rules"])
        target = min(base_target * scale, float(cfg["cap"]))
        row["badbucket_source_target"] = f"{base_target:.5f}"
        row["badbucket_scale"] = f"{scale:.6f}"
        row["target_pct"] = f"{target:.5f}"
        original_by_day[day] = original_by_day.get(day, 0.0) + base_target
        scaled_by_day[day] = scaled_by_day.get(day, 0.0) + target

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("signal_date")), []).append(row)
    for day, day_rows in grouped.items():
        current_sum = scaled_by_day.get(day, 0.0)
        if current_sum <= 0:
            continue
        original_sum = original_by_day.get(day, current_sum)
        if cfg["normalize"] == "same_sum":
            target_sum = original_sum
        elif cfg["normalize"] == "grow":
            target_sum = min(current_sum, original_sum * float(cfg["max_day_sum_mult"]))
        else:
            target_sum = current_sum
        multiplier = target_sum / current_sum
        if abs(multiplier - 1.0) < 1e-12:
            continue
        for row in day_rows:
            target = _to_float(row.get("target_pct"), 0.0) or 0.0
            row["target_pct"] = f"{min(target * multiplier, float(cfg['cap'])):.5f}"

    day_sums: dict[str, float] = {}
    for row in rows:
        day = str(row.get("signal_date"))
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
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
        "7",
        "--holding-days",
        "6",
        "--max-holding-days",
        "6",
        "--target-position-pct",
        "0.42",
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
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
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
