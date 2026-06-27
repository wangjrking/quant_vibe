from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_feature_filter_refine_20260622"
)
BASE_SIGNAL = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_best_v20260621"
    / "signals"
    / "backtest_signals_best_noncal_avgpred205_rw2121201919.csv"
)
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
    / "fusion_5d10d.db"
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
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


RANK_TARGETS = {
    "rw2121201919": {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19},
    "rw2322211816": {1: 0.23, 2: 0.22, 3: 0.21, 4: 0.18, 5: 0.16},
    "rw2523211917": {1: 0.25, 2: 0.23, 3: 0.21, 4: 0.19, 5: 0.17},
    "top3_343332": {1: 0.34, 2: 0.33, 3: 0.32},
    "top2_6040": {1: 0.60, 2: 0.40},
    "top1_098": {1: 0.98},
}


def _variant(name: str, filters: dict, ranks: str = "rw2121201919", max_positions: int = 5) -> dict:
    return {
        "name": name,
        "filters": filters,
        "rank_target_pct": RANK_TARGETS[ranks],
        "rank_targets_name": ranks,
        "rank_max": max(RANK_TARGETS[ranks]),
        "max_positions": max_positions,
        "holding_days": 7,
        "max_holding_days": 10,
    }


VARIANTS = [
    _variant("gap_lt002_rw2121201919", {"max_pred_gap": 0.02}),
    _variant("gap_lt005_rw2121201919", {"max_pred_gap": 0.05}),
    _variant("gap_lt010_rw2121201919", {"max_pred_gap": 0.10}),
    _variant("gap_lt002_rw2523211917", {"max_pred_gap": 0.02}, "rw2523211917"),
    _variant("gap_lt005_rw2523211917", {"max_pred_gap": 0.05}, "rw2523211917"),
    _variant("gap_lt002_top3", {"max_pred_gap": 0.02}, "top3_343332", 3),
    _variant("gap_lt005_top3", {"max_pred_gap": 0.05}, "top3_343332", 3),
    _variant("gap_lt002_top2", {"max_pred_gap": 0.02}, "top2_6040", 2),
    _variant("gap_lt005_top2", {"max_pred_gap": 0.05}, "top2_6040", 2),
    _variant("rank5_lt02_top1", {"max_rank_5d": 0.20}, "top1_098", 1),
    _variant("rank5_lt05_rw2121201919", {"max_rank_5d": 0.50}),
    _variant("turn_ge6_rw2121201919", {"min_turnover": 6.0}),
    _variant("turn_lt1_rw2121201919", {"max_turnover": 1.0}),
    _variant("gap_lt002_or_turn_ge6_rw2121201919", {"max_pred_gap": 0.02, "or_min_turnover": 6.0}),
    _variant("gap_lt002_or_rank5_lt02_rw2121201919", {"max_pred_gap": 0.02, "or_max_rank_5d": 0.20}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_base_rows() -> list[dict]:
    with BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_fusion_features(keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    features = {}
    try:
        dates = sorted({date for date, _stock in keys})
        for start in range(0, len(dates), 100):
            chunk = dates[start : start + 100]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_5d, pred_10d, rank_5d, rank_10d,
                       amount, turnover_rate, total_mv, atr_qfq, close
                FROM fusion_rank_base
                WHERE trade_date IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                key = (str(row["trade_date"]), str(row["stock_code"]))
                if key not in keys:
                    continue
                pred_5d = _to_float(row["pred_5d"])
                pred_10d = _to_float(row["pred_10d"])
                close = _to_float(row["close"])
                atr = _to_float(row["atr_qfq"])
                features[key] = {
                    "pred_5d": pred_5d,
                    "pred_10d": pred_10d,
                    "rank_5d": _to_float(row["rank_5d"]),
                    "rank_10d": _to_float(row["rank_10d"]),
                    "amount": _to_float(row["amount"]),
                    "turnover_rate": _to_float(row["turnover_rate"]),
                    "total_mv": _to_float(row["total_mv"]),
                    "atr_ratio": atr / close if atr is not None and close and close > 0 else None,
                    "pred_gap": abs(pred_10d - pred_5d) if pred_10d is not None and pred_5d is not None else None,
                }
    finally:
        conn.close()
    return features


def _passes_one_filter(features: dict, filters: dict) -> bool:
    checks = [
        ("max_pred_gap", "pred_gap", lambda actual, threshold: actual <= threshold),
        ("min_pred_gap", "pred_gap", lambda actual, threshold: actual >= threshold),
        ("max_rank_5d", "rank_5d", lambda actual, threshold: actual <= threshold),
        ("min_rank_5d", "rank_5d", lambda actual, threshold: actual >= threshold),
        ("max_turnover", "turnover_rate", lambda actual, threshold: actual <= threshold),
        ("min_turnover", "turnover_rate", lambda actual, threshold: actual >= threshold),
    ]
    for filter_key, feature_key, predicate in checks:
        if filter_key not in filters:
            continue
        actual = features.get(feature_key)
        if actual is None or not predicate(float(actual), float(filters[filter_key])):
            return False
    return True


def _passes(row: dict, variant: dict, features: dict) -> bool:
    rank = int(float(row.get("rank") or 999999))
    if rank > int(variant["rank_max"]):
        return False
    filters = variant["filters"]
    base_filters = {key: value for key, value in filters.items() if not key.startswith("or_")}
    if _passes_one_filter(features, base_filters):
        return True
    or_filters = {key[3:]: value for key, value in filters.items() if key.startswith("or_")}
    return bool(or_filters) and _passes_one_filter(features, or_filters)


def _write_variant_signal(variant: dict, signal_file: Path, base_rows: list[dict], fusion_features: dict) -> None:
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(base_rows[0].keys()) + ["pred_5d", "pred_10d", "rank_5d", "rank_10d", "turnover_rate", "pred_gap"]
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            features = fusion_features.get((str(row.get("signal_date")), str(row.get("stock_code"))), {})
            if not _passes(row, variant, features):
                continue
            rank = int(float(row.get("rank") or 999999))
            out = dict(row)
            out["target_pct"] = f"{float(variant['rank_target_pct'].get(rank, 0.0)):.5f}"
            for key in ["pred_5d", "pred_10d", "rank_5d", "rank_10d", "turnover_rate", "pred_gap"]:
                out[key] = features.get(key)
            writer.writerow(out)


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
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
        "min_signal_target_pct": min(targets) if targets else None,
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
            "GM_CASH_BUFFER": "0.99",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.90",
            "GM_EQUITY_DD_HARD_SCALE": "0.70",
        }
    )
    max_target = max(float(value) for value in variant["rank_target_pct"].values())
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
        str(int(variant["max_positions"])),
        "--holding-days",
        "7",
        "--max-holding-days",
        "10",
        "--target-position-pct",
        f"{max_target:.5f}",
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


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    base_rows = _load_base_rows()
    keys = {(str(row.get("signal_date")), str(row.get("stock_code"))) for row in base_rows}
    fusion_features = _load_fusion_features(keys)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_variant_signal(variant, signal_file, base_rows, fusion_features)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file)
        row = {
            "name": variant["name"],
            "filters": variant["filters"],
            "rank_targets_name": variant["rank_targets_name"],
            "rank_max": variant["rank_max"],
            "max_positions": variant["max_positions"],
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        if indicator:
            row.update(
                {
                    "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
                    "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
                    "max_drawdown": indicator.get("max_drawdown"),
                    "win_ratio": indicator.get("win_ratio"),
                    "open_count": indicator.get("open_count"),
                    "close_count": indicator.get("close_count"),
                }
            )
        row.update(_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')} signals={row.get('signal_count')}"
        )
    _write_rows(REPORT_DIR / "summary.csv", results)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
    target_hits = [
        row
        for row in results
        if (row.get("annual") is not None and float(row["annual"]) >= 3.0)
        and (row.get("sharpe") is not None and float(row["sharpe"]) >= 4.0)
        and (row.get("avg_invested_pct") is not None and float(row["avg_invested_pct"]) >= 0.80)
    ]
    _write_rows(REPORT_DIR / "summary_target_hits.csv", sorted(target_hits, key=_sort_by_sharpe, reverse=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
