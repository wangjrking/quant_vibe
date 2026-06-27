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
    / "formal_5d10d_pool_quality_refine_20260622"
)
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
    / "fusion_5d10d.db"
)
SCORE_DB = SOURCE_DIR / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


BASE_SIGNALS = {
    "baseline": SOURCE_DIR / "signals" / "baseline.csv",
    "weak_f60_liq": SOURCE_DIR / "signals" / "weak_f60_liq.csv",
    "weak_f50_liq": SOURCE_DIR / "signals" / "weak_f50_liq.csv",
    "weak_f70": SOURCE_DIR / "signals" / "weak_f70.csv",
    "weak_fallback150": SOURCE_DIR / "signals" / "weak_fallback150.csv",
    "weak_primary80": SOURCE_DIR / "signals" / "weak_primary80.csv",
    "weak_pmv100": SOURCE_DIR / "signals" / "weak_pmv100.csv",
    "weak_avgpred_f60_liq": SOURCE_DIR / "signals" / "weak_avgpred_f60_liq.csv",
}


RANK_TARGETS = {
    "rw2121201919": {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19},
    "rw2221201918": {1: 0.22, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.18},
    "rw2322211816": {1: 0.23, 2: 0.22, 3: 0.21, 4: 0.18, 5: 0.16},
    "rw2422201816": {1: 0.24, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.16},
}


def _variant(
    name: str,
    base_name: str,
    avg_pred_min: float,
    rank_targets_name: str = "rw2121201919",
    rank_max: int = 5,
    primary_count_min: int = 2,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "base_name": base_name,
        "base_signal": BASE_SIGNALS[base_name],
        "post_filter_avg_pred_min": avg_pred_min,
        "rank_targets_name": rank_targets_name,
        "rank_target_pct": RANK_TARGETS[rank_targets_name],
        "rank_max": rank_max,
        "primary_count_min": primary_count_min,
        "max_positions": 5,
        "holding_days": 7,
        "max_holding_days": 10,
        "extra_env": extra_env or {},
    }


VARIANTS = [
    _variant("repro_weak_f60_liq_avg205", "weak_f60_liq", 2.05),
    _variant("baseline_avg205", "baseline", 2.05),
    _variant("weak_f50_liq_avg205", "weak_f50_liq", 2.05),
    _variant("weak_f70_avg205", "weak_f70", 2.05),
    _variant("weak_fallback150_avg205", "weak_fallback150", 2.05),
    _variant("weak_primary80_avg205", "weak_primary80", 2.05),
    _variant("weak_pmv100_avg205", "weak_pmv100", 2.05),
    _variant("weak_avgpred_f60_liq_avg205", "weak_avgpred_f60_liq", 2.05),
    _variant("weak_f50_liq_avg210", "weak_f50_liq", 2.10),
    _variant("weak_f70_avg210", "weak_f70", 2.10),
    _variant("weak_fallback150_avg210", "weak_fallback150", 2.10),
    _variant("weak_primary80_avg210", "weak_primary80", 2.10),
    _variant("weak_pmv100_avg210", "weak_pmv100", 2.10),
    _variant("weak_f50_liq_avg200", "weak_f50_liq", 2.00),
    _variant("weak_f70_avg200", "weak_f70", 2.00),
    _variant("weak_fallback150_avg200", "weak_fallback150", 2.00),
    _variant("weak_primary80_avg200", "weak_primary80", 2.00),
    _variant("weak_pmv100_avg200", "weak_pmv100", 2.00),
    _variant("weak_f50_liq_avg205_rw2221201918", "weak_f50_liq", 2.05, "rw2221201918"),
    _variant("weak_f70_avg205_rw2221201918", "weak_f70", 2.05, "rw2221201918"),
    _variant("weak_primary80_avg205_rw2221201918", "weak_primary80", 2.05, "rw2221201918"),
    _variant("weak_pmv100_avg205_rw2221201918", "weak_pmv100", 2.05, "rw2221201918"),
    _variant("weak_f50_liq_avg205_rw2322211816", "weak_f50_liq", 2.05, "rw2322211816"),
    _variant("weak_f70_avg205_rw2322211816", "weak_f70", 2.05, "rw2322211816"),
    _variant("weak_primary80_avg205_rw2322211816", "weak_primary80", 2.05, "rw2322211816"),
    _variant("weak_pmv100_avg205_rw2322211816", "weak_pmv100", 2.05, "rw2322211816"),
    _variant("weak_f50_liq_avg205_rw2422201816", "weak_f50_liq", 2.05, "rw2422201816"),
    _variant("weak_f70_avg205_rw2422201816", "weak_f70", 2.05, "rw2422201816"),
    _variant("weak_primary80_avg205_rw2422201816", "weak_primary80", 2.05, "rw2422201816"),
    _variant("weak_pmv100_avg205_rw2422201816", "weak_pmv100", 2.05, "rw2422201816"),
    _variant("weak_f60_liq_avg205_defer_mh20", "weak_f60_liq", 2.05, extra_env={"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "20"}),
    _variant("weak_f60_liq_avg205_resize_held", "weak_f60_liq", 2.05, extra_env={"GM_RESIZE_HELD_ON_SIGNAL": "1", "GM_RESIZE_HELD_TARGET_MULT": "1.05", "GM_RESIZE_HELD_MIN_DELTA_PCT": "0.02"}),
    _variant("weak_f60_liq_avg205_eqdd_strict", "weak_f60_liq", 2.05, extra_env={"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
    _variant("weak_f60_liq_avg205_eqdd_mid", "weak_f60_liq", 2.05, extra_env={"GM_EQUITY_DD_SOFT_TRIGGER": "0.10", "GM_EQUITY_DD_HARD_TRIGGER": "0.18", "GM_EQUITY_DD_SOFT_SCALE": "0.85", "GM_EQUITY_DD_HARD_SCALE": "0.60"}),
    _variant("weak_f60_liq_avg205_eqdd_loose", "weak_f60_liq", 2.05, extra_env={"GM_EQUITY_DD_SOFT_TRIGGER": "0.14", "GM_EQUITY_DD_HARD_TRIGGER": "0.24", "GM_EQUITY_DD_SOFT_SCALE": "0.92", "GM_EQUITY_DD_HARD_SCALE": "0.75"}),
    _variant("weak_f60_liq_avg205_takeprofit12", "weak_f60_liq", 2.05, extra_env={"GM_TAKE_PROFIT_PCT": "0.12"}),
    _variant("weak_f60_liq_avg205_takeprofit18", "weak_f60_liq", 2.05, extra_env={"GM_TAKE_PROFIT_PCT": "0.18"}),
    _variant("weak_f60_liq_avg205_takeprofit25", "weak_f60_liq", 2.05, extra_env={"GM_TAKE_PROFIT_PCT": "0.25"}),
    _variant("weak_f60_liq_avg205_lightstop04_mh2", "weak_f60_liq", 2.05, extra_env={"GM_LIGHT_STOP_LOSS_PCT": "0.04", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "2"}),
    _variant("weak_f60_liq_avg205_lightstop06_mh2", "weak_f60_liq", 2.05, extra_env={"GM_LIGHT_STOP_LOSS_PCT": "0.06", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "2"}),
    _variant("weak_f60_liq_avg205_score_drop085", "weak_f60_liq", 2.05, extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.85"}),
    _variant("weak_f60_liq_avg205_score_drop090", "weak_f60_liq", 2.05, extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90"}),
    _variant("weak_f60_liq_avg205_exit100_mh4", "weak_f60_liq", 2.05, extra_env={"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4"}),
    _variant("weak_f60_liq_avg205_exit100_mh5", "weak_f60_liq", 2.05, extra_env={"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "5"}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_fusion_features() -> dict[tuple[str, str], dict]:
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT trade_date, stock_code, pred_10d, pred_5d, amount, turnover_rate,
               total_mv, atr_qfq, close, rank_5d, rank_10d
        FROM fusion_rank_base
        """
    ).fetchall()
    conn.close()
    features = {}
    for row in rows:
        pred_5d = _to_float(row["pred_5d"])
        pred_10d = _to_float(row["pred_10d"])
        features[(str(row["trade_date"]), str(row["stock_code"]))] = {
            "pred_5d": pred_5d,
            "pred_10d": pred_10d,
            "amount": _to_float(row["amount"]),
            "turnover_rate": _to_float(row["turnover_rate"]),
            "total_mv": _to_float(row["total_mv"]),
            "rank_5d": _to_float(row["rank_5d"]),
            "rank_10d": _to_float(row["rank_10d"]),
        }
    return features


def _load_base_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _signal_features(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("signal_date") or ""), []).append(row)
    features = {}
    for signal_date, items in grouped.items():
        preds = [_to_float(row.get("pred_prob"), 0.0) or 0.0 for row in items]
        features[signal_date] = {
            "count": len(items),
            "primary_count": sum(1 for value in preds if value >= 2.0),
            "avg_pred": sum(preds) / len(preds) if preds else None,
        }
    return features


def _passes_first_stage(row: dict, variant: dict, base_features: dict, fusion_features: dict) -> bool:
    rank = int(float(row.get("rank") or 999999))
    if rank > int(variant["rank_max"]):
        return False
    day_features = base_features.get(str(row.get("signal_date") or ""), {})
    if int(day_features.get("primary_count") or 0) < int(variant["primary_count_min"]):
        return False
    key = (str(row.get("signal_date") or ""), str(row.get("stock_code") or ""))
    stock_features = fusion_features.get(key)
    if not stock_features:
        return False
    pred_10d = stock_features.get("pred_10d")
    if pred_10d is None or float(pred_10d) < 0.0:
        return False
    return True


def _write_variant_signal(variant: dict, signal_file: Path, fusion_features: dict) -> None:
    rows = _load_base_rows(variant["base_signal"])
    base_features = _signal_features(rows)
    first_stage = [
        row
        for row in rows
        if _passes_first_stage(row, variant, base_features, fusion_features)
    ]
    post_features = _signal_features(first_stage)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    extra_fields = ["pred_10d", "pred_5d"]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in first_stage:
            signal_date = str(row.get("signal_date") or "")
            avg_pred = _to_float(post_features.get(signal_date, {}).get("avg_pred"))
            if avg_pred is None or avg_pred < float(variant["post_filter_avg_pred_min"]):
                continue
            output = dict(row)
            rank = int(float(output.get("rank") or 0))
            output["target_pct"] = f"{float(variant['rank_target_pct'].get(rank, 0.0)):.5f}"
            output["holding_days"] = str(int(variant["holding_days"]))
            output["score_exit_entry_ratio"] = "1.0"
            output["min_holding_days_before_score_exit"] = "3"
            features = fusion_features.get((signal_date, str(output.get("stock_code") or "")), {})
            output["pred_10d"] = features.get("pred_10d")
            output["pred_5d"] = features.get("pred_5d")
            writer.writerow(output)


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
    env.update({str(key): str(value) for key, value in variant.get("extra_env", {}).items()})
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
        str(int(variant["holding_days"])),
        "--max-holding-days",
        str(int(variant["max_holding_days"])),
        "--target-position-pct",
        f"{max(float(value) for value in variant['rank_target_pct'].values()):.5f}",
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


def _write_empty_target_hits(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "name,annual,sharpe,avg_invested_pct,ge80_ratio,signal_count,buy_days\n",
        encoding="utf-8-sig",
    )


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
    fusion_features = _load_fusion_features()
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_variant_signal(variant, signal_file, fusion_features)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file)
        row = {
            "name": variant["name"],
            "base_name": variant["base_name"],
            "post_filter_avg_pred_min": variant["post_filter_avg_pred_min"],
            "rank_targets_name": variant["rank_targets_name"],
            "rank_max": variant["rank_max"],
            "primary_count_min": variant["primary_count_min"],
            "max_positions": variant["max_positions"],
            "holding_days": variant["holding_days"],
            "max_holding_days": variant["max_holding_days"],
            "extra_env": variant.get("extra_env"),
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
    if target_hits:
        _write_rows(REPORT_DIR / "summary_target_hits.csv", sorted(target_hits, key=_sort_by_sharpe, reverse=True))
    else:
        _write_empty_target_hits(REPORT_DIR / "summary_target_hits.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
