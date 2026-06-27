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
    / "formal_10d_bucket_risk_reweight_v2_20260622"
)
SIGNAL_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_bucket_cap_fine_20260622" / "signals"
SOURCES = {
    "mid": SIGNAL_DIR / "mid_s32_w206_la42_bad14_cap42_pos7.csv",
    "base20627": SIGNAL_DIR / "base_s32_w20627_la42_bad14_cap42_pos7.csv",
    "base2063": SIGNAL_DIR / "base_s32_w2063_la42_bad14_cap42_pos7.csv",
    "base2062": SIGNAL_DIR / "base_s32_w2062_la42_bad14_cap42_pos7.csv",
}
SCORE_DB = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_weak_day_pool_replace_20260621" / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")


BASE_ENV = dict(base_grid.BASE_ENV)
BASE_ENV.update(base_grid.EQDD_PROFILES["mid"])


def _variant(
    name: str,
    rules: list[dict],
    *,
    source: str = "mid",
    normalize: str = "same_day_sum",
    cap: float = 0.42,
    floor_day_sum: float | None = None,
    holding_days: int = 6,
    max_positions: int = 7,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "source": source,
        "rules": rules,
        "normalize": normalize,
        "cap": cap,
        "floor_day_sum": floor_day_sum,
        "holding_days": holding_days,
        "max_positions": max_positions,
        "env": env,
    }


VARIANTS = [
    _variant("repro", []),
    _variant("turn6_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}]),
    _variant("turn6_down83", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.83}]),
    _variant("turn6_down84", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.84}]),
    _variant("turn6_down86", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.86}]),
    _variant("turn6_down87", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.87}]),
    _variant("turn59_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 5.9, "scale": 0.85}]),
    _variant("turn61_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 6.1, "scale": 0.85}]),
    _variant("turn6_down85_h5_pos7", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], holding_days=5, max_positions=7),
    _variant("turn6_down85_h7_pos7", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], holding_days=7, max_positions=7),
    _variant("turn6_down85_h6_pos6", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], holding_days=6, max_positions=6),
    _variant("turn6_down85_h6_pos8", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], holding_days=6, max_positions=8),
    _variant("turn6_down85_eqdd_base", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], extra_env=base_grid.EQDD_PROFILES["base"]),
    _variant("turn6_down85_eqdd_strict", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], extra_env=base_grid.EQDD_PROFILES["strict"]),
    _variant("turn6_down85_no_dd", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], extra_env={"GM_EQUITY_DD_RISK_MODE": "0"}),
    _variant("turn6_down85_no_score_exit", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], extra_env={"GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("turn6_down85_score99", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.99", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"}),
    _variant("turn6_down85_score98", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"}),
    _variant("base20627_turn6_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], source="base20627"),
    _variant("base2063_turn6_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], source="base2063"),
    _variant("base2062_turn6_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], source="base2062"),
    _variant("base20627_turn6_down85_gap06_down95", [
        {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85},
        {"field": "pred_gap", "op": "ge", "threshold": 0.06, "scale": 0.95},
    ], source="base20627"),
    _variant("turn6_down80", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.80}]),
    _variant("turn6_down90", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.90}]),
    _variant("turn55_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 5.5, "scale": 0.85}]),
    _variant("turn65_down85", [{"field": "turnover_rate", "op": "ge", "threshold": 6.5, "scale": 0.85}]),
    _variant("turn6_down85_turn8_down80", [
        {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85},
        {"field": "turnover_rate", "op": "ge", "threshold": 8.0, "scale": 0.94},
    ]),
    _variant("turn6_down85_lowturn_boost", [
        {"field": "turnover_rate", "op": "lt", "threshold": 1.5, "scale": 1.04},
        {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85},
    ]),
    _variant("turn6_down85_gap06_down95", [
        {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85},
        {"field": "pred_gap", "op": "ge", "threshold": 0.06, "scale": 0.95},
    ]),
    _variant("turn6_down85_mv180_down95", [
        {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85},
        {"field": "total_mv", "op": "ge", "threshold": 180000.0, "scale": 0.95},
    ]),
    _variant("turn6_down85_cap40", [{"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.85}], cap=0.40, floor_day_sum=0.95),
    _variant("turn5_down82", [{"field": "turnover_rate", "op": "ge", "threshold": 5.0, "scale": 0.82}]),
    _variant("turn4_down90_turn7_down75", [
        {"field": "turnover_rate", "op": "ge", "threshold": 4.0, "scale": 0.90},
        {"field": "turnover_rate", "op": "ge", "threshold": 7.0, "scale": 0.83},
    ]),
    _variant("mv180_down85", [{"field": "total_mv", "op": "ge", "threshold": 180000.0, "scale": 0.85}]),
    _variant("mv160_down88_mv190_down80", [
        {"field": "total_mv", "op": "ge", "threshold": 160000.0, "scale": 0.88},
        {"field": "total_mv", "op": "ge", "threshold": 190000.0, "scale": 0.91},
    ]),
    _variant("gap06_down88", [{"field": "pred_gap", "op": "ge", "threshold": 0.06, "scale": 0.88}]),
    _variant("gap05_down90_turn6_down86", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.05, "scale": 0.90},
        {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 0.86},
    ]),
    _variant("risk_combo_soft", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.05, "scale": 0.93},
        {"field": "turnover_rate", "op": "ge", "threshold": 5.0, "scale": 0.90},
        {"field": "total_mv", "op": "ge", "threshold": 180000.0, "scale": 0.90},
    ]),
    _variant("risk_combo_mid", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.05, "scale": 0.90},
        {"field": "turnover_rate", "op": "ge", "threshold": 5.0, "scale": 0.86},
        {"field": "total_mv", "op": "ge", "threshold": 170000.0, "scale": 0.88},
    ]),
    _variant("risk_combo_mid_cap38", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.05, "scale": 0.90},
        {"field": "turnover_rate", "op": "ge", "threshold": 5.0, "scale": 0.86},
        {"field": "total_mv", "op": "ge", "threshold": 170000.0, "scale": 0.88},
    ], cap=0.38, floor_day_sum=0.95),
    _variant("risk_combo_hard", [
        {"field": "pred_gap", "op": "ge", "threshold": 0.045, "scale": 0.86},
        {"field": "turnover_rate", "op": "ge", "threshold": 4.5, "scale": 0.82},
        {"field": "total_mv", "op": "ge", "threshold": 160000.0, "scale": 0.84},
    ], floor_day_sum=0.95),
    _variant("low_turn_boost", [
        {"field": "turnover_rate", "op": "lt", "threshold": 1.5, "scale": 1.08},
        {"field": "turnover_rate", "op": "ge", "threshold": 5.0, "scale": 0.88},
    ]),
    _variant("low_gap_boost", [
        {"field": "pred_gap", "op": "lt", "threshold": 0.02, "scale": 1.06},
        {"field": "pred_gap", "op": "ge", "threshold": 0.06, "scale": 0.86},
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


def _scale(row: dict, rules: list[dict]) -> float:
    out = 1.0
    for rule in rules:
        if _matches(_to_float(row.get(rule["field"])), rule):
            out *= float(rule["scale"])
    return out


def _read_source(source: str) -> tuple[list[dict], list[str]]:
    path = SOURCES[str(source)]
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    rows, fields = _read_source(str(cfg["source"]))
    if "reweight_scale" not in fields:
        fields.append("reweight_scale")
    if "base_target_pct" not in fields:
        fields.append("base_target_pct")
    day_original: dict[str, float] = {}
    day_scaled: dict[str, float] = {}
    for row in rows:
        day = str(row.get("signal_date"))
        base_target = _to_float(row.get("target_pct"), 0.0) or 0.0
        scale = _scale(row, cfg["rules"])
        target = min(base_target * scale, float(cfg["cap"]))
        row["base_target_pct"] = f"{base_target:.5f}"
        row["reweight_scale"] = f"{scale:.6f}"
        row["target_pct"] = f"{target:.5f}"
        day_original[day] = day_original.get(day, 0.0) + base_target
        day_scaled[day] = day_scaled.get(day, 0.0) + target

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("signal_date")), []).append(row)
    for day, day_rows in grouped.items():
        current_sum = day_scaled.get(day, 0.0)
        target_sum = current_sum
        if cfg["normalize"] == "same_day_sum":
            target_sum = day_original.get(day, current_sum)
        if cfg.get("floor_day_sum") is not None and 0 < target_sum < float(cfg["floor_day_sum"]):
            target_sum = float(cfg["floor_day_sum"])
        if current_sum <= 0 or abs(target_sum - current_sum) < 1e-12:
            continue
        multiplier = target_sum / current_sum
        for row in day_rows:
            target = _to_float(row.get("target_pct"), 0.0) or 0.0
            row["target_pct"] = f"{min(target * multiplier, float(cfg['cap'])):.5f}"

    day_sums: dict[str, float] = {}
    for row in rows:
        day_sums[str(row.get("signal_date"))] = day_sums.get(str(row.get("signal_date")), 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
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
        str(JUEJIN_PYTHON), str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", str(cfg["max_positions"]),
        "--holding-days", str(cfg["holding_days"]),
        "--max-holding-days", str(cfg["holding_days"]),
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
