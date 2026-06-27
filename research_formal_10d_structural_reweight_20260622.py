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

import research_formal_10d_intraday_replace_refine_20260622 as current_best


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_structural_reweight_20260622"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_exit_grid_20260622"
    / "signals"
    / "repro_current_best.csv"
)
SCORE_DB = current_best.SCORE_DB
SCORE_TABLE = current_best.SCORE_TABLE
MARKET_DB = current_best.MARKET_DB
STRATEGY_DIR = current_best.STRATEGY_DIR
JUEJIN_PYTHON = current_best.JUEJIN_PYTHON
BASE_ENV = dict(current_best.BASE_ENV)


def _variant(name: str, rules: list[dict], *, cap: float = 0.42, extra_env: dict[str, str] | None = None) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {"name": name, "rules": rules, "cap": cap, "env": env}


VARIANTS = [
    _variant("repro", []),
    _variant("board20_down90", [{"kind": "board20", "scale": 0.90}]),
    _variant("board20_down80", [{"kind": "board20", "scale": 0.80}]),
    _variant("board20_down70", [{"kind": "board20", "scale": 0.70}]),
    _variant("star_down80", [{"kind": "prefix", "prefixes": ["688"], "scale": 0.80}]),
    _variant("cyb_down80", [{"kind": "prefix", "prefixes": ["300", "301"], "scale": 0.80}]),
    _variant("lowprice3_down80", [{"kind": "close_lt", "threshold": 3.0, "scale": 0.80}]),
    _variant("lowprice5_down88", [{"kind": "close_lt", "threshold": 5.0, "scale": 0.88}]),
    _variant("lowprice3_down70", [{"kind": "close_lt", "threshold": 3.0, "scale": 0.70}]),
    _variant("board20_lowprice3_combo", [{"kind": "board20", "scale": 0.88}, {"kind": "close_lt", "threshold": 3.0, "scale": 0.82}]),
    _variant("board20_lowprice5_combo", [{"kind": "board20", "scale": 0.90}, {"kind": "close_lt", "threshold": 5.0, "scale": 0.90}]),
    _variant("main10_boost_board20_down", [{"kind": "board20", "scale": 0.82}, {"kind": "main10", "scale": 1.08}]),
    _variant("lowmv8e4_down70", [{"kind": "mv_lt", "threshold": 80000.0, "scale": 0.70}]),
    _variant("highturn6_down85", [{"kind": "turn_ge", "threshold": 6.0, "scale": 0.85}]),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _code(row: dict) -> str:
    return str(row.get("stock_code") or "").split(".")[0]


def _match(row: dict, rule: dict) -> bool:
    code = _code(row)
    kind = str(rule["kind"])
    if kind == "board20":
        return code.startswith(("300", "301", "688"))
    if kind == "main10":
        return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))
    if kind == "prefix":
        return any(code.startswith(str(prefix)) for prefix in rule["prefixes"])
    if kind == "close_lt":
        value = _to_float(row.get("close"))
        return value is not None and value < float(rule["threshold"])
    if kind == "mv_lt":
        value = _to_float(row.get("total_mv"))
        return value is not None and value < float(rule["threshold"])
    if kind == "turn_ge":
        value = _to_float(row.get("turnover_rate"))
        return value is not None and value >= float(rule["threshold"])
    raise ValueError(f"unsupported rule kind: {kind}")


def _scale(row: dict, rules: list[dict]) -> float:
    out = 1.0
    for rule in rules:
        if _match(row, rule):
            out *= float(rule["scale"])
    return out


def _read_rows() -> tuple[list[dict], list[str]]:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _read_rows()
    for field in ["structural_scale", "structural_base_target_pct"]:
        if field not in fields:
            fields.append(field)
    original_day_sum: dict[str, float] = {}
    scaled_day_sum: dict[str, float] = {}
    for row in rows:
        day = str(row.get("buy_date") or "")
        base = _to_float(row.get("target_pct"), 0.0) or 0.0
        scale = _scale(row, cfg["rules"])
        target = min(base * scale, float(cfg["cap"]))
        row["structural_scale"] = f"{scale:.6f}"
        row["structural_base_target_pct"] = f"{base:.5f}"
        row["target_pct"] = f"{target:.5f}"
        original_day_sum[day] = original_day_sum.get(day, 0.0) + base
        scaled_day_sum[day] = scaled_day_sum.get(day, 0.0) + target

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("buy_date") or ""), []).append(row)
    for day, day_rows in grouped.items():
        current = scaled_day_sum.get(day, 0.0)
        target_sum = original_day_sum.get(day, current)
        if current <= 0:
            continue
        multiplier = target_sum / current
        for row in day_rows:
            value = _to_float(row.get("target_pct"), 0.0) or 0.0
            row["target_pct"] = f"{min(value * multiplier, float(cfg['cap'])):.5f}"

    day_sums: dict[str, float] = {}
    for row in rows:
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
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
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}", flush=True)
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
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
