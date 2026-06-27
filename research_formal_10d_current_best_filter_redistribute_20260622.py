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
    / "formal_10d_current_best_filter_redistribute_20260622"
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


BASE_ENV = dict(base_grid.BASE_ENV)
BASE_ENV.update(base_grid.EQDD_PROFILES["mid"])


def _variant(
    name: str,
    filters: list[dict],
    *,
    cap: float = 0.48,
    max_positions: int = 7,
    holding_days: int = 6,
    min_day_sum: float = 0.75,
) -> dict:
    return {
        "name": name,
        "filters": filters,
        "cap": cap,
        "max_positions": max_positions,
        "holding_days": holding_days,
        "min_day_sum": min_day_sum,
        "env": dict(BASE_ENV),
    }


VARIANTS = [
    _variant("repro", [], cap=0.42, min_day_sum=0.0),
    _variant("drop_turn_ge8", [{"field": "turnover_rate", "op": "ge", "threshold": 8.0}]),
    _variant("drop_turn_ge7", [{"field": "turnover_rate", "op": "ge", "threshold": 7.0}]),
    _variant("drop_gap_ge008", [{"field": "pred_gap", "op": "ge", "threshold": 0.08}]),
    _variant("drop_gap_ge007", [{"field": "pred_gap", "op": "ge", "threshold": 0.07}]),
    _variant("drop_mv_ge180k", [{"field": "total_mv", "op": "ge", "threshold": 180000.0}]),
    _variant("drop_mv_ge160k", [{"field": "total_mv", "op": "ge", "threshold": 160000.0}]),
    _variant("drop_amount_ge60k", [{"field": "amount", "op": "ge", "threshold": 60000.0}]),
    _variant("drop_amount_ge50k", [{"field": "amount", "op": "ge", "threshold": 50000.0}]),
    _variant("drop_weak_rank35", [{"field": "weak_rank35", "op": "eq", "threshold": 1.0}]),
    _variant("drop_bad_gap_rank", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.07},
        {"field": "weak_rank35", "op": "eq", "threshold": 1.0},
    ]),
    _variant("drop_bad_mv_turn", [
        {"field": "total_mv", "op": "ge", "threshold": 180000.0},
        {"field": "turnover_rate", "op": "ge", "threshold": 7.0},
    ]),
    _variant("drop_bad_combo_soft", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.08},
        {"field": "total_mv", "op": "ge", "threshold": 180000.0},
        {"field": "turnover_rate", "op": "ge", "threshold": 8.0},
    ]),
    _variant("drop_bad_combo_mid", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.07},
        {"field": "total_mv", "op": "ge", "threshold": 160000.0},
        {"field": "turnover_rate", "op": "ge", "threshold": 7.0},
    ]),
    _variant("drop_turn_ge8_cap52", [{"field": "turnover_rate", "op": "ge", "threshold": 8.0}], cap=0.52),
    _variant("drop_gap_ge008_cap52", [{"field": "pred_gap", "op": "ge", "threshold": 0.08}], cap=0.52),
    _variant("drop_mv_ge180k_cap52", [{"field": "total_mv", "op": "ge", "threshold": 180000.0}], cap=0.52),
    _variant("drop_bad_combo_soft_cap52", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.08},
        {"field": "total_mv", "op": "ge", "threshold": 180000.0},
        {"field": "turnover_rate", "op": "ge", "threshold": 8.0},
    ], cap=0.52),
    _variant("drop_turn_ge8_pos6", [{"field": "turnover_rate", "op": "ge", "threshold": 8.0}], max_positions=6, cap=0.52),
    _variant("drop_bad_combo_soft_pos6", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.08},
        {"field": "total_mv", "op": "ge", "threshold": 180000.0},
        {"field": "turnover_rate", "op": "ge", "threshold": 8.0},
    ], max_positions=6, cap=0.52),
    _variant("drop_turn_ge8_h5", [{"field": "turnover_rate", "op": "ge", "threshold": 8.0}], holding_days=5, cap=0.52),
    _variant("drop_bad_combo_soft_h5", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.08},
        {"field": "total_mv", "op": "ge", "threshold": 180000.0},
        {"field": "turnover_rate", "op": "ge", "threshold": 8.0},
    ], holding_days=5, cap=0.52),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _derived_value(row: dict, field: str) -> float | None:
    if field == "weak_rank35":
        role = str(row.get("pool_role") or "")
        rank = int(_to_float(row.get("rank"), 0.0) or 0)
        return 1.0 if role == "weak" and rank in {3, 5} else 0.0
    return _to_float(row.get(field))


def _matches(row: dict, rule: dict) -> bool:
    value = _derived_value(row, str(rule["field"]))
    if value is None:
        return False
    op = str(rule["op"])
    threshold = float(rule["threshold"])
    if op == "lt":
        return value < threshold
    if op == "le":
        return value <= threshold
    if op == "ge":
        return value >= threshold
    if op == "gt":
        return value > threshold
    if op == "eq":
        return abs(value - threshold) < 1e-12
    raise ValueError(f"unsupported op: {op}")


def _drop(row: dict, filters: list[dict]) -> bool:
    return any(_matches(row, rule) for rule in filters)


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in ["filter_redistribute_source_target", "filter_redistribute_dropped"]:
        if field not in fields:
            fields.append(field)

    original_by_day: dict[str, float] = {}
    kept_by_day: dict[str, list[dict]] = {}
    dropped = 0
    for row in rows:
        day = str(row.get("signal_date"))
        base_target = _to_float(row.get("target_pct"), 0.0) or 0.0
        original_by_day[day] = original_by_day.get(day, 0.0) + base_target
        row["filter_redistribute_source_target"] = f"{base_target:.5f}"
        row["filter_redistribute_dropped"] = "0"
        row["holding_days"] = str(int(cfg["holding_days"]))
        if _drop(row, cfg["filters"]):
            row["filter_redistribute_dropped"] = "1"
            dropped += 1
            continue
        kept_by_day.setdefault(day, []).append(row)

    output_rows: list[dict] = []
    day_sums: dict[str, float] = {}
    skipped_days = 0
    for day, day_rows in kept_by_day.items():
        current_sum = sum(_to_float(row.get("target_pct"), 0.0) or 0.0 for row in day_rows)
        target_sum = original_by_day.get(day, current_sum)
        if target_sum < float(cfg["min_day_sum"]):
            skipped_days += 1
            continue
        if current_sum <= 0:
            skipped_days += 1
            continue
        multiplier = target_sum / current_sum
        for row in day_rows:
            target = _to_float(row.get("target_pct"), 0.0) or 0.0
            row["target_pct"] = f"{min(target * multiplier, float(cfg['cap'])):.5f}"
            output_rows.append(row)
            day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)

    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in output_rows])
    return {
        "signal_count": len(output_rows),
        "buy_days": len({row.get("buy_date") for row in output_rows if row.get("buy_date")}),
        "dropped_rows": dropped,
        "skipped_days": skipped_days,
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
        str(cfg["max_positions"]),
        "--holding-days",
        str(cfg["holding_days"]),
        "--max-holding-days",
        str(cfg["holding_days"]),
        "--target-position-pct",
        str(cfg["cap"]),
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
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_avg80_by_sharpe.csv",
        [
            row
            for row in sorted(results, key=lambda item: _metric(item, "sharpe"), reverse=True)
            if _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
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
