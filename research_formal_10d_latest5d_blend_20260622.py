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
    / "formal_10d_latest5d_blend_20260622"
)
BEST10_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
)
LATEST_FUSION_SIGNALS = {
    "prod": DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_latest_manifest_pool_grid_20260622"
    / "signals"
    / "prod_rules_latest.csv",
    "rank5_10d80": DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_latest_manifest_pool_grid_20260622"
    / "signals"
    / "rank5_10d80_5d20_h6.csv",
}
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
    fusion_name: str,
    best10_scale: float,
    fusion_scale: float,
    day_cap: float,
    max_positions: int,
    max_single: float,
    holding_days: int = 6,
    max_holding_days: int = 6,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "fusion_name": fusion_name,
        "best10_scale": best10_scale,
        "fusion_scale": fusion_scale,
        "day_cap": day_cap,
        "max_positions": max_positions,
        "max_single": max_single,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "env": env,
    }


VARIANTS = [
    _variant("best80_prod20_mp8_cap90", fusion_name="prod", best10_scale=0.80, fusion_scale=0.20, day_cap=0.90, max_positions=8, max_single=0.30),
    _variant("best70_prod30_mp9_cap92", fusion_name="prod", best10_scale=0.70, fusion_scale=0.30, day_cap=0.92, max_positions=9, max_single=0.28),
    _variant("best60_prod40_mp10_cap95", fusion_name="prod", best10_scale=0.60, fusion_scale=0.40, day_cap=0.95, max_positions=10, max_single=0.25),
    _variant("best85_rank15_mp8_cap90", fusion_name="rank5_10d80", best10_scale=0.85, fusion_scale=0.15, day_cap=0.90, max_positions=8, max_single=0.32),
    _variant("best75_rank25_mp9_cap92", fusion_name="rank5_10d80", best10_scale=0.75, fusion_scale=0.25, day_cap=0.92, max_positions=9, max_single=0.28),
    _variant("best65_rank35_mp10_cap95", fusion_name="rank5_10d80", best10_scale=0.65, fusion_scale=0.35, day_cap=0.95, max_positions=10, max_single=0.25),
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


def _day_key(row: dict) -> tuple[str, str]:
    return str(row["signal_date"]), str(row["buy_date"])


def _rank(row: dict) -> int:
    return int(float(row.get("rank") or 999999))


def _target(row: dict) -> float:
    return _to_float(row.get("target_pct"), 0.0) or 0.0


def _normal_code(row: dict) -> str:
    return str(row.get("stock_code") or row.get("symbol") or "")


def _copy_scaled(row: dict, source_role: str, scale: float, cap: float) -> dict:
    item = dict(row)
    item["source_role"] = source_role
    item["target_pct"] = f"{min(_target(row) * scale, cap):.5f}"
    return item


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    by_day: dict[tuple[str, str], dict[str, dict]] = {}

    for row in _read_csv(BEST10_SIGNAL):
        day = _day_key(row)
        code = _normal_code(row)
        item = _copy_scaled(row, "best10d", float(cfg["best10_scale"]), float(cfg["max_single"]))
        by_day.setdefault(day, {})[code] = item

    fusion_path = LATEST_FUSION_SIGNALS[str(cfg["fusion_name"])]
    for row in _read_csv(fusion_path):
        day = _day_key(row)
        code = _normal_code(row)
        item = _copy_scaled(row, f"latest5d10d_{cfg['fusion_name']}", float(cfg["fusion_scale"]), float(cfg["max_single"]))
        existing = by_day.setdefault(day, {}).get(code)
        if existing is None:
            by_day[day][code] = item
            continue
        merged_target = min(_target(existing) + _target(item), float(cfg["max_single"]))
        existing["target_pct"] = f"{merged_target:.5f}"
        existing["source_role"] = "blend_overlap"

    output: list[dict] = []
    source_counts: dict[str, int] = {}
    for day in sorted(by_day):
        rows = list(by_day[day].values())
        rows.sort(key=lambda row: (0 if str(row.get("source_role")) in {"best10d", "blend_overlap"} else 1, _rank(row)))
        selected: list[dict] = []
        target_sum = 0.0
        for row in rows:
            if len(selected) >= int(cfg["max_positions"]):
                break
            remaining = float(cfg["day_cap"]) - target_sum
            if remaining <= 0:
                break
            item = dict(row)
            item["target_pct"] = f"{min(_target(item), remaining):.5f}"
            item["rank"] = str(len(selected) + 1)
            item["holding_days"] = str(int(cfg["holding_days"]))
            selected.append(item)
            target_sum += _target(item)
        for row in selected:
            source = str(row.get("source_role") or "")
            source_counts[source] = source_counts.get(source, 0) + 1
        output.extend(selected)

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
    for row in output:
        day_sums[str(row.get("buy_date"))] = day_sums.get(str(row.get("buy_date")), 0.0) + _target(row)
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
        str(cfg["holding_days"]),
        "--max-holding-days",
        str(cfg["max_holding_days"]),
        "--target-position-pct",
        str(cfg["max_single"]),
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
