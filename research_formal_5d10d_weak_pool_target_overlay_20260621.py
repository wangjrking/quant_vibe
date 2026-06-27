from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import math
import os
import re
import sqlite3
import statistics
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_day_pool_replace_20260621"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_pool_target_overlay_20260621"
)
BASE_SIGNAL = SOURCE_DIR / "signals" / "weak_f60_liq.csv"
SCORE_DB = SOURCE_DIR / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"
BASE_TARGET = 0.10925


VARIANTS = [
    {"name": "baseline", "rules": []},
    {"name": "weak_primary_scale095", "rules": [{"type": "weak_primary", "scale": 0.95, "primary_count_max": 1}]},
    {"name": "weak_primary_scale090", "rules": [{"type": "weak_primary", "scale": 0.90, "primary_count_max": 1}]},
    {"name": "weak_fallback_only_scale095", "rules": [{"type": "weak_fallback_only", "scale": 0.95, "primary_count_max": 1}]},
    {"name": "weak_fallback_only_scale090", "rules": [{"type": "weak_fallback_only", "scale": 0.90, "primary_count_max": 1}]},
    {"name": "weak_avgpred_scale095", "rules": [{"type": "weak_avg_pred", "scale": 0.95, "avg_pred_lt": 2.25}]},
    {"name": "weak_avgpred_scale090", "rules": [{"type": "weak_avg_pred", "scale": 0.90, "avg_pred_lt": 2.25}]},
    {
        "name": "weak095_strong101",
        "rules": [
            {"type": "weak_primary", "scale": 0.95, "primary_count_max": 1},
            {"type": "strong_primary", "scale": 1.01, "primary_count_min": 2},
        ],
        "max_target_pct": BASE_TARGET * 1.01,
    },
    {
        "name": "weak090_strong101",
        "rules": [
            {"type": "weak_primary", "scale": 0.90, "primary_count_max": 1},
            {"type": "strong_primary", "scale": 1.01, "primary_count_min": 2},
        ],
        "max_target_pct": BASE_TARGET * 1.01,
    },
    {
        "name": "weak095_strong102",
        "rules": [
            {"type": "weak_primary", "scale": 0.95, "primary_count_max": 1},
            {"type": "strong_primary", "scale": 1.02, "primary_count_min": 2},
        ],
        "max_target_pct": BASE_TARGET * 1.02,
    },
    {
        "name": "idx_or_weak_scale090_strong101",
        "rules": [
            {"type": "idx_or_weak", "scale": 0.90, "idx_ret_5_lt": -0.03, "primary_count_max": 1, "avg_pred_lt": 2.25},
            {"type": "strong_primary", "scale": 1.01, "primary_count_min": 2},
        ],
        "max_target_pct": BASE_TARGET * 1.01,
    },
    {"name": "weak_combined_scale090", "rules": [{"type": "weak_combined", "scale": 0.90, "primary_count_max": 1, "avg_pred_lt": 2.25}]},
    {"name": "weak_combined_scale085", "rules": [{"type": "weak_combined", "scale": 0.85, "primary_count_max": 1, "avg_pred_lt": 2.25}]},
    {"name": "idx5_neg3_scale075", "rules": [{"type": "idx_ret", "field": "idx_ret_5", "lt": -0.03, "scale": 0.75}]},
    {"name": "idx5_neg3_scale085", "rules": [{"type": "idx_ret", "field": "idx_ret_5", "lt": -0.03, "scale": 0.85}]},
    {"name": "idx5_neg2_scale090", "rules": [{"type": "idx_ret", "field": "idx_ret_5", "lt": -0.02, "scale": 0.90}]},
    {"name": "idx10_neg5_scale075", "rules": [{"type": "idx_ret", "field": "idx_ret_10", "lt": -0.05, "scale": 0.75}]},
    {"name": "idx20_dd10_scale075", "rules": [{"type": "idx_drawdown", "field": "idx_dd_20", "lt": -0.10, "scale": 0.75}]},
    {
        "name": "idx_or_weak_scale090",
        "rules": [
            {"type": "idx_or_weak", "scale": 0.90, "idx_ret_5_lt": -0.03, "primary_count_max": 1, "avg_pred_lt": 2.25}
        ],
    },
    {
        "name": "idx_or_weak_scale085",
        "rules": [
            {"type": "idx_or_weak", "scale": 0.85, "idx_ret_5_lt": -0.03, "primary_count_max": 1, "avg_pred_lt": 2.25}
        ],
    },
    {
        "name": "bad_vol_scale075",
        "rules": [
            {"type": "bad_vol", "scale": 0.75, "idx_vol_10_gt": 0.018, "idx_ret_5_lt": -0.01}
        ],
    },
    {
        "name": "quality_gate_scale080",
        "rules": [
            {"type": "quality_gate", "scale": 0.80, "avg_pred_lt": 2.30, "primary_count_max": 1, "idx_ret_5_lt": -0.01}
        ],
    },
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_index_features() -> dict[str, dict]:
    conn = sqlite3.connect(MARKET_DB)
    rows = conn.execute(
        """
        SELECT trade_date,
               MAX(index_2000_open) AS open_price,
               MAX(index_2000_close) AS close_price
        FROM STOCK_DAILY_DATA
        WHERE trade_date >= '20240501' AND trade_date <= '20260630'
          AND index_2000_close IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
        """
    ).fetchall()
    conn.close()
    features: dict[str, dict] = {}
    closes = []
    dates = []
    prev_close = None
    for trade_date, open_price, close_price in rows:
        trade_date = str(trade_date)
        close_price = _to_float(close_price)
        open_price = _to_float(open_price)
        if close_price is None:
            continue
        dates.append(trade_date)
        closes.append(close_price)
        idx = len(closes) - 1
        rets = []
        for lookback in (5, 10, 20):
            if idx >= lookback and closes[idx - lookback]:
                ret = close_price / closes[idx - lookback] - 1.0
            else:
                ret = None
            features.setdefault(trade_date, {})[f"idx_ret_{lookback}"] = ret
        if idx >= 20:
            high_20 = max(closes[idx - 20 : idx + 1])
            features[trade_date]["idx_dd_20"] = close_price / high_20 - 1.0 if high_20 else None
        else:
            features[trade_date]["idx_dd_20"] = None
        if idx >= 10:
            daily_rets = [closes[i] / closes[i - 1] - 1.0 for i in range(idx - 9, idx + 1) if closes[i - 1]]
            features[trade_date]["idx_vol_10"] = statistics.pstdev(daily_rets) if len(daily_rets) > 1 else None
        else:
            features[trade_date]["idx_vol_10"] = None
        features[trade_date]["idx_ret_1"] = close_price / prev_close - 1.0 if prev_close else None
        features[trade_date]["idx_intraday"] = close_price / open_price - 1.0 if open_price else None
        prev_close = close_price
    return features


def _load_base_rows() -> list[dict]:
    with BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _signal_features(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("signal_date") or ""), []).append(row)
    features = {}
    for signal_date, items in grouped.items():
        preds = [_to_float(row.get("pred_prob"), 0.0) or 0.0 for row in items]
        primary_count = sum(1 for value in preds if value >= 2.0)
        features[signal_date] = {
            "avg_pred": sum(preds) / len(preds) if preds else None,
            "min_pred": min(preds) if preds else None,
            "primary_count": primary_count,
            "primary_ratio": primary_count / len(preds) if preds else None,
        }
    return features


def _rule_scale(row: dict, signal_feat: dict, index_feat: dict, rule: dict) -> float:
    rule_type = rule["type"]
    if rule_type == "weak_primary":
        return float(rule["scale"]) if int(signal_feat.get("primary_count") or 0) <= int(rule["primary_count_max"]) else 1.0
    if rule_type == "strong_primary":
        return float(rule["scale"]) if int(signal_feat.get("primary_count") or 0) >= int(rule["primary_count_min"]) else 1.0
    if rule_type == "weak_avg_pred":
        return float(rule["scale"]) if (_to_float(signal_feat.get("avg_pred"), 999) or 999) < float(rule["avg_pred_lt"]) else 1.0
    if rule_type == "weak_combined":
        weak = int(signal_feat.get("primary_count") or 0) <= int(rule["primary_count_max"]) or (
            (_to_float(signal_feat.get("avg_pred"), 999) or 999) < float(rule["avg_pred_lt"])
        )
        return float(rule["scale"]) if weak else 1.0
    if rule_type == "weak_fallback_only":
        pred = _to_float(row.get("pred_prob"), 0.0) or 0.0
        weak = int(signal_feat.get("primary_count") or 0) <= int(rule["primary_count_max"]) and pred < 2.0
        return float(rule["scale"]) if weak else 1.0
    if rule_type == "idx_ret":
        value = _to_float(index_feat.get(rule["field"]))
        return float(rule["scale"]) if value is not None and value < float(rule["lt"]) else 1.0
    if rule_type == "idx_drawdown":
        value = _to_float(index_feat.get(rule["field"]))
        return float(rule["scale"]) if value is not None and value < float(rule["lt"]) else 1.0
    if rule_type == "idx_or_weak":
        idx_value = _to_float(index_feat.get("idx_ret_5"))
        weak = int(signal_feat.get("primary_count") or 0) <= int(rule["primary_count_max"]) or (
            (_to_float(signal_feat.get("avg_pred"), 999) or 999) < float(rule["avg_pred_lt"])
        )
        bad_index = idx_value is not None and idx_value < float(rule["idx_ret_5_lt"])
        return float(rule["scale"]) if (weak or bad_index) else 1.0
    if rule_type == "bad_vol":
        vol = _to_float(index_feat.get("idx_vol_10"))
        ret = _to_float(index_feat.get("idx_ret_5"))
        return (
            float(rule["scale"])
            if vol is not None and ret is not None and vol > float(rule["idx_vol_10_gt"]) and ret < float(rule["idx_ret_5_lt"])
            else 1.0
        )
    if rule_type == "quality_gate":
        avg_pred = _to_float(signal_feat.get("avg_pred"))
        primary_count = int(signal_feat.get("primary_count") or 0)
        idx_ret = _to_float(index_feat.get("idx_ret_5"))
        weak = (avg_pred is not None and avg_pred < float(rule["avg_pred_lt"])) or primary_count <= int(rule["primary_count_max"])
        bad_index = idx_ret is not None and idx_ret < float(rule["idx_ret_5_lt"])
        return float(rule["scale"]) if weak and bad_index else 1.0
    raise ValueError(rule_type)


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    rows = _load_base_rows()
    signal_features = _signal_features(rows)
    index_features = _load_index_features()
    fieldnames = list(rows[0].keys()) if rows else []
    if "regime_scale" not in fieldnames:
        fieldnames.append("regime_scale")
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            signal_date = str(row.get("signal_date") or "")
            scale = 1.0
            for rule in variant["rules"]:
                scale *= _rule_scale(row, signal_features.get(signal_date, {}), index_features.get(signal_date, {}), rule)
            row = dict(row)
            row["target_pct"] = f"{BASE_TARGET * scale:.5f}"
            row["regime_scale"] = f"{scale:.5f}"
            row["holding_days"] = "7"
            row["score_exit_entry_ratio"] = "1.0"
            row["min_holding_days_before_score_exit"] = "3"
            writer.writerow(row)


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


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    scales = [_to_float(row.get("regime_scale")) for row in rows]
    scales = [value for value in scales if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_regime_scale": sum(scales) / len(scales) if scales else 1.0,
        "scaled_signal_count": sum(1 for value in scales if value < 0.999),
    }


def _run_backtest(variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
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
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
        }
    )
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        "10",
        "--holding-days",
        "7",
        "--max-holding-days",
        "7",
        "--target-position-pct",
        f"{float(variant.get('max_target_pct', BASE_TARGET)):.5f}",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        BACKTEST_START,
        "--backtest-end",
        BACKTEST_END,
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _sort_by_annual(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("annual") or -999.0),
        float(row.get("sharpe") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _sort_by_sharpe(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_variant_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            "name": variant["name"],
            "rules": repr(variant["rules"]),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted(results, key=_sort_by_sharpe, reverse=True)
        if float(row.get("annual") or -999.0) >= 2.0
        and float(row.get("sharpe") or -999.0) >= 3.0
        and float(row.get("avg_invested_pct") or -999.0) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
