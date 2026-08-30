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

import research_formal_signal_state_rule_scan_20260622 as market_scan
import research_formal_10d_bucket_risk_reweight_20260622 as risk_base


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_market_state_scale_20260622"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
)
SCORE_DB = risk_base.SCORE_DB
SCORE_TABLE = risk_base.SCORE_TABLE
MARKET_DB = risk_base.MARKET_DB
STRATEGY_DIR = risk_base.STRATEGY_DIR
JUEJIN_PYTHON = risk_base.JUEJIN_PYTHON
BASE_ENV = dict(risk_base.BASE_ENV)


def _variant(name: str, rules: list[dict], *, cap: float = 0.42, extra_env: dict[str, str] | None = None) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {"name": name, "rules": rules, "cap": cap, "env": env}


VARIANTS = [
    _variant("repro", []),
    _variant("up_low_down94_up_high_boost103", [
        {"field": "mkt_up_ratio", "op": "lt", "threshold": 0.45, "scale": 0.94},
        {"field": "mkt_up_ratio", "op": "ge", "threshold": 0.62, "scale": 1.03},
    ]),
    _variant("up_low_down90_up_high_boost105", [
        {"field": "mkt_up_ratio", "op": "lt", "threshold": 0.45, "scale": 0.90},
        {"field": "mkt_up_ratio", "op": "ge", "threshold": 0.62, "scale": 1.05},
    ]),
    _variant("avgchg_neg_down92_pos_boost103", [
        {"field": "mkt_avg_pct_chg", "op": "lt", "threshold": -0.30, "scale": 0.92},
        {"field": "mkt_avg_pct_chg", "op": "ge", "threshold": 0.50, "scale": 1.03},
    ]),
    _variant("idxcc_neg_down92_pos_boost103", [
        {"field": "mkt_index_cc", "op": "lt", "threshold": -0.01, "scale": 0.92},
        {"field": "mkt_index_cc", "op": "ge", "threshold": 0.01, "scale": 1.03},
    ]),
    _variant("idxintra_neg_down94_pos_boost102", [
        {"field": "mkt_index_intraday", "op": "lt", "threshold": -0.006, "scale": 0.94},
        {"field": "mkt_index_intraday", "op": "ge", "threshold": 0.006, "scale": 1.02},
    ]),
    _variant("market_combo_soft", [
        {"field": "mkt_up_ratio", "op": "lt", "threshold": 0.45, "scale": 0.96},
        {"field": "mkt_avg_pct_chg", "op": "lt", "threshold": -0.30, "scale": 0.96},
        {"field": "mkt_index_cc", "op": "lt", "threshold": -0.01, "scale": 0.96},
        {"field": "mkt_up_ratio", "op": "ge", "threshold": 0.62, "scale": 1.03},
    ]),
    _variant("market_combo_mid", [
        {"field": "mkt_up_ratio", "op": "lt", "threshold": 0.45, "scale": 0.93},
        {"field": "mkt_avg_pct_chg", "op": "lt", "threshold": -0.30, "scale": 0.94},
        {"field": "mkt_index_cc", "op": "lt", "threshold": -0.01, "scale": 0.94},
        {"field": "mkt_up_ratio", "op": "ge", "threshold": 0.62, "scale": 1.04},
        {"field": "mkt_avg_pct_chg", "op": "ge", "threshold": 0.50, "scale": 1.02},
    ]),
    _variant("market_combo_hard", [
        {"field": "mkt_up_ratio", "op": "lt", "threshold": 0.45, "scale": 0.90},
        {"field": "mkt_avg_pct_chg", "op": "lt", "threshold": -0.30, "scale": 0.92},
        {"field": "mkt_index_cc", "op": "lt", "threshold": -0.01, "scale": 0.92},
        {"field": "mkt_up_ratio", "op": "ge", "threshold": 0.62, "scale": 1.05},
        {"field": "mkt_avg_pct_chg", "op": "ge", "threshold": 0.50, "scale": 1.03},
    ]),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _read_rows() -> tuple[list[dict], list[str]]:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _matches(value: float | None, rule: dict) -> bool:
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
    raise ValueError(f"unsupported op: {op}")


def _scale(feature_row: dict, rules: list[dict]) -> float:
    scale = 1.0
    for rule in rules:
        if _matches(_to_float(feature_row.get(rule["field"])), rule):
            scale *= float(rule["scale"])
    return scale


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _read_rows()
    dates = {str(row.get("signal_date")) for row in rows if row.get("signal_date")}
    market = market_scan._market_features(dates)
    for field in [
        "market_state_scale",
        "mkt_up_ratio",
        "mkt_avg_pct_chg",
        "mkt_index_cc",
        "mkt_index_intraday",
    ]:
        if field not in fields:
            fields.append(field)
    day_sums: dict[str, float] = {}
    scales: dict[str, float] = {}
    for row in rows:
        day = str(row.get("signal_date"))
        feature_row = market.get(day, {})
        scale = _scale(feature_row, cfg["rules"])
        target = _to_float(row.get("target_pct"), 0.0) or 0.0
        row["target_pct"] = f"{min(target * scale, float(cfg['cap'])):.5f}"
        row["market_state_scale"] = f"{scale:.6f}"
        for field in ("mkt_up_ratio", "mkt_avg_pct_chg", "mkt_index_cc", "mkt_index_intraday"):
            value = feature_row.get(field)
            row[field] = "" if value is None else value
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
        scales[day] = scale
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
        "avg_market_state_scale": sum(scales.values()) / len(scales) if scales else None,
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
        "--max-positions", "7",
        "--holding-days", "6",
        "--max-holding-days", "6",
        "--target-position-pct", str(cfg["cap"]),
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


def _objective(row: dict) -> float:
    return min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80)


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
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
