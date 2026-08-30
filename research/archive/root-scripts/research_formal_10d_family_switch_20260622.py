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

import research_formal_10d_bucket_risk_reweight_20260622 as risk_base
import research_formal_signal_state_rule_scan_20260622 as market_scan


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_family_switch_20260622"
)
BASE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
)
AGGR_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_gap02_target_shape_20260622"
    / "signals"
    / "s030_w018.csv"
)
AGGR_SIGNAL_2 = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_gap02_target_shape_20260622"
    / "signals"
    / "s034_w014.csv"
)
DEF_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_current_best_badbucket_reweight_20260622"
    / "signals"
    / "mv_down80_cap48_same.csv"
)
SCORE_DB = risk_base.SCORE_DB
SCORE_TABLE = risk_base.SCORE_TABLE
MARKET_DB = risk_base.MARKET_DB
STRATEGY_DIR = risk_base.STRATEGY_DIR
JUEJIN_PYTHON = risk_base.JUEJIN_PYTHON


def _variant(
    name: str,
    *,
    feature: str,
    threshold: float,
    op: str,
    aggressive: str = "s030_w018",
    defensive_on_bad: bool = False,
    max_positions: int = 8,
    cap_scale: float = 1.0,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(risk_base.BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "feature": feature,
        "threshold": threshold,
        "op": op,
        "aggressive": aggressive,
        "defensive_on_bad": defensive_on_bad,
        "max_positions": max_positions,
        "cap_scale": cap_scale,
        "env": env,
    }


VARIANTS = [
    _variant("up_ge55_aggr30", feature="mkt_up_ratio", threshold=0.55, op="ge"),
    _variant("up_ge60_aggr30", feature="mkt_up_ratio", threshold=0.60, op="ge"),
    _variant("up_ge65_aggr30", feature="mkt_up_ratio", threshold=0.65, op="ge"),
    _variant("up_ge70_aggr30", feature="mkt_up_ratio", threshold=0.70, op="ge"),
    _variant("avg_ge03_aggr30", feature="mkt_avg_pct_chg", threshold=0.30, op="ge"),
    _variant("avg_ge05_aggr30", feature="mkt_avg_pct_chg", threshold=0.50, op="ge"),
    _variant("avg_ge08_aggr30", feature="mkt_avg_pct_chg", threshold=0.80, op="ge"),
    _variant("idxcc_ge0_aggr30", feature="mkt_index_cc", threshold=0.0, op="ge"),
    _variant("idxcc_ge005_aggr30", feature="mkt_index_cc", threshold=0.005, op="ge"),
    _variant("idxcc_ge01_aggr30", feature="mkt_index_cc", threshold=0.01, op="ge"),
    _variant("up_ge60_aggr34", feature="mkt_up_ratio", threshold=0.60, op="ge", aggressive="s034_w014"),
    _variant("avg_ge05_aggr34", feature="mkt_avg_pct_chg", threshold=0.50, op="ge", aggressive="s034_w014"),
    _variant("idxcc_ge005_aggr34", feature="mkt_index_cc", threshold=0.005, op="ge", aggressive="s034_w014"),
    _variant("up_ge60_aggr30_defbad", feature="mkt_up_ratio", threshold=0.60, op="ge", defensive_on_bad=True),
    _variant("avg_ge05_aggr30_defbad", feature="mkt_avg_pct_chg", threshold=0.50, op="ge", defensive_on_bad=True),
    _variant("idxcc_ge005_aggr30_defbad", feature="mkt_index_cc", threshold=0.005, op="ge", defensive_on_bad=True),
    _variant("up_ge60_aggr30_nodd", feature="mkt_up_ratio", threshold=0.60, op="ge", extra_env={"GM_EQUITY_DD_RISK_MODE": "0"}),
    _variant("avg_ge05_aggr30_nodd", feature="mkt_avg_pct_chg", threshold=0.50, op="ge", extra_env={"GM_EQUITY_DD_RISK_MODE": "0"}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _by_day(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        out.setdefault((str(row.get("signal_date")), str(row.get("buy_date"))), []).append(row)
    return out


def _match(value: float | None, op: str, threshold: float) -> bool:
    if value is None:
        return False
    if op == "ge":
        return value >= threshold
    if op == "gt":
        return value > threshold
    if op == "le":
        return value <= threshold
    if op == "lt":
        return value < threshold
    raise ValueError(f"unsupported op: {op}")


def _write_signal(cfg: dict, signal_file: Path) -> dict:
    base_by_day = _by_day(_load(BASE_SIGNAL))
    defensive_by_day = _by_day(_load(DEF_SIGNAL))
    aggressive_path = AGGR_SIGNAL_2 if cfg["aggressive"] == "s034_w014" else AGGR_SIGNAL
    aggr_by_day = _by_day(_load(aggressive_path))
    all_days = sorted(set(base_by_day) | set(aggr_by_day) | set(defensive_by_day))
    dates = {date for date, _buy_date in all_days}
    market = market_scan._market_features(dates)

    output: list[dict] = []
    family_counts = {"aggressive": 0, "base": 0, "defensive": 0}
    for day_key in all_days:
        signal_date, _buy_date = day_key
        feature_row = market.get(signal_date, {})
        value = _to_float(feature_row.get(str(cfg["feature"])))
        use_aggressive = _match(value, str(cfg["op"]), float(cfg["threshold"]))
        if use_aggressive:
            rows = aggr_by_day.get(day_key) or base_by_day.get(day_key) or []
            family = "aggressive"
        elif cfg["defensive_on_bad"]:
            rows = defensive_by_day.get(day_key) or base_by_day.get(day_key) or []
            family = "defensive"
        else:
            rows = base_by_day.get(day_key) or []
            family = "base"
        family_counts[family] += 1
        for index, row in enumerate(sorted(rows, key=lambda item: int(float(item.get("rank") or 999999))), start=1):
            item = dict(row)
            item["rank"] = str(index)
            item["source_family"] = family
            item["switch_feature"] = str(cfg["feature"])
            item["switch_threshold"] = str(cfg["threshold"])
            item["switch_value"] = "" if value is None else f"{value:.8f}"
            item["target_pct"] = f"{(_to_float(item.get('target_pct'), 0.0) or 0.0) * float(cfg['cap_scale']):.5f}"
            output.append(item)

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
        day_sums.setdefault(str(row.get("buy_date")), 0.0)
        day_sums[str(row.get("buy_date"))] += _to_float(row.get("target_pct"), 0.0) or 0.0
    return {
        "signal_count": len(output),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
        "family_counts_json": json.dumps(family_counts, ensure_ascii=False, sort_keys=True),
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
