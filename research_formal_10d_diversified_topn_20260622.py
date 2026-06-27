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

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig

import research_formal_10d_intraday_replace_refine_20260622 as current_best


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_diversified_topn_20260622"
)
SCORE_DB = current_best.SCORE_DB
SCORE_TABLE = current_best.SCORE_TABLE
MARKET_DB = current_best.MARKET_DB
STRATEGY_DIR = current_best.STRATEGY_DIR
JUEJIN_PYTHON = current_best.JUEJIN_PYTHON
START_DATE = "20240604"
END_DATE = "20260618"

BASE_ENV = dict(current_best.BASE_ENV)


def _variant(
    name: str,
    *,
    top_k: int,
    max_positions: int,
    target_total_pct: float,
    target_position_pct: float,
    holding_days: int = 6,
    weight_mode: str = "equal",
    atr_required: bool = False,
    min_amount: float | None = 10000.0,
    min_turnover: float | None = 0.3,
    max_mv: float | None = 200000.0,
    close_rate_min: float | None = None,
    close_rate_max: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "top_k": top_k,
        "max_positions": max_positions,
        "target_total_pct": target_total_pct,
        "target_position_pct": target_position_pct,
        "holding_days": holding_days,
        "weight_mode": weight_mode,
        "atr_required": atr_required,
        "min_amount": min_amount,
        "min_turnover": min_turnover,
        "max_mv": max_mv,
        "close_rate_min": close_rate_min,
        "close_rate_max": close_rate_max,
        "env": env,
    }


VARIANTS = [
    _variant("top5_repro_shape", top_k=5, max_positions=7, target_total_pct=1.15, target_position_pct=0.42),
    _variant("top7_eq_t98_p14", top_k=7, max_positions=9, target_total_pct=0.98, target_position_pct=0.14),
    _variant("top7_eq_t112_p16", top_k=7, max_positions=9, target_total_pct=1.12, target_position_pct=0.16),
    _variant("top7_rank_t112_p20", top_k=7, max_positions=9, target_total_pct=1.12, target_position_pct=0.20, weight_mode="rank"),
    _variant("top9_eq_t99_p11", top_k=9, max_positions=11, target_total_pct=0.99, target_position_pct=0.11),
    _variant("top9_eq_t117_p13", top_k=9, max_positions=11, target_total_pct=1.17, target_position_pct=0.13),
    _variant("top9_rank_t117_p18", top_k=9, max_positions=11, target_total_pct=1.17, target_position_pct=0.18, weight_mode="rank"),
    _variant("top12_eq_t96_p08", top_k=12, max_positions=14, target_total_pct=0.96, target_position_pct=0.08),
    _variant("top12_eq_t120_p10", top_k=12, max_positions=14, target_total_pct=1.20, target_position_pct=0.10),
    _variant("top9_eq_t117_h5", top_k=9, max_positions=11, target_total_pct=1.17, target_position_pct=0.13, holding_days=5),
    _variant("top9_eq_t117_atr", top_k=9, max_positions=11, target_total_pct=1.17, target_position_pct=0.13, atr_required=True),
    _variant("top7_eq_t112_atr", top_k=7, max_positions=9, target_total_pct=1.12, target_position_pct=0.16, atr_required=True),
    _variant("top9_eq_close98_106", top_k=9, max_positions=11, target_total_pct=1.17, target_position_pct=0.13, close_rate_min=0.98, close_rate_max=1.06),
    _variant("top7_eq_close98_106", top_k=7, max_positions=9, target_total_pct=1.12, target_position_pct=0.16, close_rate_min=0.98, close_rate_max=1.06),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_score_rows() -> list[dict]:
    conn = sqlite3.connect(SCORE_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_prob
                FROM "{SCORE_TABLE}"
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date, pred_prob DESC
                """,
                (START_DATE, END_DATE),
            )
        ]
    finally:
        conn.close()
    return rows


def _passes_extra_filters(row: dict, cfg: dict) -> bool:
    close_rate_min = cfg.get("close_rate_min")
    close_rate_max = cfg.get("close_rate_max")
    if close_rate_min is None and close_rate_max is None:
        return True
    close = _to_float(row.get("close"))
    pre_close = _to_float(row.get("pre_close"))
    if close is None or pre_close is None or pre_close <= 0:
        return False
    rate = close / pre_close
    if close_rate_min is not None and rate < float(close_rate_min):
        return False
    if close_rate_max is not None and rate > float(close_rate_max):
        return False
    return True


def _write_signal(score_rows: list[dict], market_rows: dict[str, dict[str, dict]], cfg: dict, signal_file: Path) -> dict:
    max_atr_ratio = 0.10 if cfg["atr_required"] else None
    merged_rows: list[dict] = []
    for row in score_rows:
        day = str(row.get("trade_date"))
        market = market_rows.get(day, {}).get(str(row.get("stock_code")), {})
        merged = dict(market)
        merged.update(row)
        if not _passes_extra_filters(merged, cfg):
            continue
        merged_rows.append(merged)

    signals = build_gm_signal_rows(
        merged_rows,
        config=SelectionConfig(
            top_k=int(cfg["top_k"]),
            pred_col="pred_prob",
            min_pred_prob=None,
            min_pred_quantile=None,
            max_atr_ratio=max_atr_ratio,
            min_amount=cfg["min_amount"],
            min_turnover_rate=cfg["min_turnover"],
            max_total_mv=cfg["max_mv"],
            max_per_industry=999999,
            exclude_bj=True,
            exclude_st=True,
            exclude_delisting=True,
            exclude_current_limit=True,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=int(cfg["holding_days"]),
        max_positions=int(cfg["max_positions"]),
        weight_mode=str(cfg["weight_mode"]),
        target_total_pct=float(cfg["target_total_pct"]),
    )
    for signal in signals:
        target = _to_float(signal.get("target_pct"), float(cfg["target_position_pct"])) or 0.0
        signal["target_pct"] = min(target, float(cfg["target_position_pct"]))
        signal["diversify_variant"] = cfg["name"]

    signal_file.parent.mkdir(parents=True, exist_ok=True)
    write_gm_signals_csv(signals, signal_file)
    day_sums: dict[str, float] = {}
    for signal in signals:
        day_sums[str(signal.get("buy_date"))] = day_sums.get(str(signal.get("buy_date")), 0.0) + (_to_float(signal.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": len(signals),
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
        str(cfg["max_positions"]),
        "--holding-days",
        str(cfg["holding_days"]),
        "--max-holding-days",
        str(cfg["holding_days"]),
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
    score_rows = _load_score_rows()
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_stats = _write_signal(score_rows, market_rows, cfg, signal_file)
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
