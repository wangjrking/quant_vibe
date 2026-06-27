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

import research_formal_10d_intraday_replace_refine_20260622 as intraday_grid


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_selective_intraday_stop_20260622"
)
SOURCE_SIGNAL = intraday_grid.SOURCE_SIGNAL
SCORE_DB = intraday_grid.SCORE_DB
SCORE_TABLE = intraday_grid.SCORE_TABLE
MARKET_DB = intraday_grid.MARKET_DB
STRATEGY_DIR = intraday_grid.STRATEGY_DIR
JUEJIN_PYTHON = intraday_grid.JUEJIN_PYTHON

BASE_ENV = dict(intraday_grid.BASE_ENV)
BASE_ENV.update(
    {
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_INTRADAY_RISK_SELL_FRACTION": "1.0",
        "GM_INTRADAY_REPLACE_BUY": "1",
        "GM_INTRADAY_REPLACE_MAX_BUYS": "1",
        "GM_STOP_LOSS_PCT": "0.99",
        "GM_TAKE_PROFIT_PCT": "none",
    }
)


def _variant(name: str, selector: dict, stop_loss: float = 0.05, replace_buys: int = 1, extra_env: dict[str, str] | None = None) -> dict:
    env = dict(BASE_ENV)
    env["GM_INTRADAY_REPLACE_MAX_BUYS"] = str(replace_buys)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "selector": selector,
        "stop_loss": stop_loss,
        "env": env,
        "max_positions": 7,
        "holding_days": 6,
        "cap": 0.42,
    }


VARIANTS = [
    _variant("repro_no_selective_stop", {"kind": "none"}, stop_loss=0.99),
    _variant("risk_sl05_replace1", {"kind": "field_eq", "field": "risk_role", "value": "risk"}, 0.05, 1),
    _variant("risk_sl05_replace2", {"kind": "field_eq", "field": "risk_role", "value": "risk"}, 0.05, 2),
    _variant("risk_sl045_replace1", {"kind": "field_eq", "field": "risk_role", "value": "risk"}, 0.045, 1),
    _variant("weak_sl05_replace1", {"kind": "field_eq", "field": "pool_role", "value": "weak"}, 0.05, 1),
    _variant("weak_sl05_replace2", {"kind": "field_eq", "field": "pool_role", "value": "weak"}, 0.05, 2),
    _variant("risk_weak_sl05_replace1", {"kind": "any", "rules": [{"kind": "field_eq", "field": "risk_role", "value": "risk"}, {"kind": "field_eq", "field": "pool_role", "value": "weak"}]}, 0.05, 1),
    _variant("high_gap_sl05_replace1", {"kind": "ge", "field": "pred_gap", "threshold": 0.05323}, 0.05, 1),
    _variant("very_high_gap_sl05_replace1", {"kind": "ge", "field": "pred_gap", "threshold": 0.07261}, 0.05, 1),
    _variant("low_5d_sl05_replace1", {"kind": "lt", "field": "pred_5d", "threshold": 0.00247}, 0.05, 1),
    _variant("low_5d_risk_sl05_replace1", {"kind": "all", "rules": [{"kind": "lt", "field": "pred_5d", "threshold": 0.00247}, {"kind": "field_eq", "field": "risk_role", "value": "risk"}]}, 0.05, 1),
    _variant("risk_sl05_replace1_softdd", {"kind": "field_eq", "field": "risk_role", "value": "risk"}, 0.05, 1, {"GM_EQUITY_DD_SOFT_TRIGGER": "0.11", "GM_EQUITY_DD_HARD_TRIGGER": "0.20", "GM_EQUITY_DD_SOFT_SCALE": "0.90", "GM_EQUITY_DD_HARD_SCALE": "0.70"}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _matches(row: dict, rule: dict) -> bool:
    kind = str(rule.get("kind"))
    if kind == "none":
        return False
    if kind == "field_eq":
        return str(row.get(str(rule["field"])) or "") == str(rule["value"])
    if kind == "ge":
        value = _to_float(row.get(str(rule["field"])))
        return value is not None and value >= float(rule["threshold"])
    if kind == "lt":
        value = _to_float(row.get(str(rule["field"])))
        return value is not None and value < float(rule["threshold"])
    if kind == "any":
        return any(_matches(row, item) for item in rule.get("rules", []))
    if kind == "all":
        return all(_matches(row, item) for item in rule.get("rules", []))
    raise ValueError(f"unsupported selector kind: {kind}")


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    with SOURCE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in ["signal_stop_loss_pct", "selective_stop_flag"]:
        if field not in fields:
            fields.append(field)
    selected = 0
    for row in rows:
        if _matches(row, cfg["selector"]):
            row["signal_stop_loss_pct"] = f"{float(cfg['stop_loss']):.6f}"
            row["selective_stop_flag"] = "1"
            selected += 1
        else:
            row["signal_stop_loss_pct"] = "0.990000"
            row["selective_stop_flag"] = "0"
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"signal_count": len(rows), "selective_stop_count": selected, "selective_stop_ratio": selected / len(rows) if rows else None}


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
