from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import duckdb
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"
BASE_SIGNAL = REPORT_DIR / "dynamic_target_signals" / "dyn_mild_09_12_15.csv"
OUT_SIGNAL_DIR = REPORT_DIR / "buy_frequency_signals"
OUT_LOG_DIR = REPORT_DIR / "buy_frequency_logs"
OUT_CSV = REPORT_DIR / "buy_frequency_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "buy_frequency_probe_20260630.json"


CASES: list[dict[str, Any]] = [
    {"name": "buy_base_top5_dyn", "mode": "base", "max_positions": 5, "cap": 0.15},
    {"name": "buy_top4_t18", "mode": "topn_fixed", "topn": 4, "target": 0.18, "max_positions": 4, "cap": 0.18},
    {"name": "buy_top3_t22", "mode": "topn_fixed", "topn": 3, "target": 0.22, "max_positions": 3, "cap": 0.22},
    {"name": "buy_top3_t25", "mode": "topn_fixed", "topn": 3, "target": 0.25, "max_positions": 3, "cap": 0.25},
    {"name": "buy_score_scale_min95", "mode": "day_min_score_scale", "threshold": 0.95, "scale": 0.4, "max_positions": 5, "cap": 0.15},
    {"name": "buy_score_skip_min95", "mode": "day_min_score_skip", "threshold": 0.95, "max_positions": 5, "cap": 0.15},
    {"name": "buy_agree_scale_avg90", "mode": "day_agreement_scale", "threshold": 0.90, "scale": 0.4, "max_positions": 5, "cap": 0.15},
    {"name": "buy_agree_skip_avg90", "mode": "day_agreement_skip", "threshold": 0.90, "max_positions": 5, "cap": 0.15},
    {"name": "buy_ma20_scale", "mode": "market_ma_scale", "ma": "ma20", "scale": 0.4, "max_positions": 5, "cap": 0.15},
    {"name": "buy_ma20_skip", "mode": "market_ma_skip", "ma": "ma20", "max_positions": 5, "cap": 0.15},
    {"name": "buy_ma60_scale", "mode": "market_ma_scale", "ma": "ma60", "scale": 0.4, "max_positions": 5, "cap": 0.15},
    {"name": "buy_row_confirm_3d5d90", "mode": "row_confirm", "threshold": 0.90, "max_positions": 5, "cap": 0.15},
    {"name": "buy_nochase_2d5", "mode": "row_two_day_cap", "threshold": 0.05, "max_positions": 5, "cap": 0.15},
    {"name": "buy_nochase_2d3", "mode": "row_two_day_cap", "threshold": 0.03, "max_positions": 5, "cap": 0.15},
    {
        "name": "buy_hybrid_ma20_score",
        "mode": "hybrid_market_score",
        "score_threshold": 0.95,
        "market_ma": "ma20",
        "weak_scale": 0.4,
        "max_positions": 5,
        "cap": 0.15,
    },
]


SELL_RULE = {
    "holding_days": 1,
    "max_holding_days": 2,
    "score_exit": 0.99,
    "score_continue": 0.995,
    "min_score_exit_days": 1,
    "max_daily_sells": 1,
    "day_drop_ratio": 0.995,
}


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
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


def _load_market_regime() -> dict[str, dict[str, float]]:
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        rows = con.execute(
            """
            WITH idx AS (
              SELECT trade_date, max(index_2000_close) AS close_value
              FROM STOCK_DAILY_DATA
              WHERE trade_date BETWEEN '20220606' AND '20260629'
              GROUP BY trade_date
            ),
            ma AS (
              SELECT
                trade_date,
                close_value,
                avg(close_value) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
                avg(close_value) OVER (ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60
              FROM idx
            )
            SELECT trade_date, close_value, ma20, ma60 FROM ma
            """
        ).fetchall()
    finally:
        con.close()
    return {str(date): {"close": float(close), "ma20": float(ma20), "ma60": float(ma60)} for date, close, ma20, ma60 in rows}


def _base_rows() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="")))
    for row in rows:
        row["holding_days"] = str(SELL_RULE["holding_days"])
        row["max_holding_days"] = str(SELL_RULE["max_holding_days"])
        row["score_exit_entry_ratio"] = f"{SELL_RULE['score_exit']:.5f}"
        row["score_continue_entry_ratio"] = f"{SELL_RULE['score_continue']:.5f}"
        row["min_holding_days_before_score_exit"] = str(SELL_RULE["min_score_exit_days"])
    return rows


def _day_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["signal_date"]].append(row)
    stats: dict[str, dict[str, float]] = {}
    for date, items in grouped.items():
        scores = [_float(row.get("pred_prob")) or 0.0 for row in items]
        r3 = [_float(row.get("rank_3d")) or 0.0 for row in items]
        r5 = [_float(row.get("rank_5d")) or 0.0 for row in items]
        r10 = [_float(row.get("rank_10d")) or 0.0 for row in items]
        stats[date] = {
            "min_score": min(scores),
            "avg_score": sum(scores) / len(scores),
            "avg_r3": sum(r3) / len(r3),
            "avg_r5": sum(r5) / len(r5),
            "avg_r10": sum(r10) / len(r10),
        }
    return stats


def _with_target(row: dict[str, Any], target: float, case_name: str) -> dict[str, Any]:
    out = dict(row)
    out["target_pct"] = f"{target:.5f}"
    out["strategy_variant"] = case_name
    out["filter_name"] = case_name
    out["dynamic_hold_name"] = "daily_h1m2_e099_c0995"
    return out


def _transform(case: dict[str, Any], rows: list[dict[str, Any]], day_stats: dict[str, dict[str, float]], market: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        date = row["signal_date"]
        rank = int(float(row["rank"]))
        base_target = _float(row.get("target_pct")) or 0.0
        mode = case["mode"]
        keep = True
        target = base_target
        if mode == "base":
            pass
        elif mode == "topn_fixed":
            keep = rank <= int(case["topn"])
            target = float(case["target"])
        elif mode == "day_min_score_scale":
            if day_stats[date]["min_score"] < float(case["threshold"]):
                target *= float(case["scale"])
        elif mode == "day_min_score_skip":
            keep = day_stats[date]["min_score"] >= float(case["threshold"])
        elif mode == "day_agreement_scale":
            if min(day_stats[date]["avg_r3"], day_stats[date]["avg_r5"]) < float(case["threshold"]):
                target *= float(case["scale"])
        elif mode == "day_agreement_skip":
            keep = min(day_stats[date]["avg_r3"], day_stats[date]["avg_r5"]) >= float(case["threshold"])
        elif mode == "market_ma_scale":
            regime = market.get(date)
            if regime and regime["close"] < regime[str(case["ma"])]:
                target *= float(case["scale"])
        elif mode == "market_ma_skip":
            regime = market.get(date)
            keep = bool(regime and regime["close"] >= regime[str(case["ma"])])
        elif mode == "row_confirm":
            keep = (_float(row.get("rank_3d")) or 0.0) >= float(case["threshold"]) and (_float(row.get("rank_5d")) or 0.0) >= float(case["threshold"])
        elif mode == "row_two_day_cap":
            two_day = _float(row.get("two_day_ret"))
            keep = two_day is None or two_day < float(case["threshold"])
        elif mode == "hybrid_market_score":
            regime = market.get(date)
            weak_market = bool(regime and regime["close"] < regime[str(case["market_ma"])])
            weak_score = day_stats[date]["min_score"] < float(case["score_threshold"])
            if weak_market or weak_score:
                target *= float(case["weak_scale"])
        else:
            raise ValueError(f"unknown mode: {mode}")
        if keep and target > 0:
            out.append(_with_target(row, target, str(case["name"])))
    out.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    return out


def _make_signal(case: dict[str, Any], rows: list[dict[str, Any]], day_stats: dict[str, dict[str, float]], market: dict[str, dict[str, float]]) -> tuple[Path, dict[str, Any]]:
    OUT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    signal_rows = _transform(case, rows, day_stats, market)
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, signal_rows)
    counts = Counter(row["signal_date"] for row in signal_rows)
    target = int(case.get("max_positions", 5))
    return output, {
        "signal_rows": len(signal_rows),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < target),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "empty_days_vs_base": 986 - len(counts),
    }


def _run(case: dict[str, Any], rows: list[dict[str, Any]], day_stats: dict[str, dict[str, float]], market: dict[str, dict[str, float]]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case, rows, day_stats, market)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(SELL_RULE["max_daily_sells"]),
                "GM_LIGHT_STOP_LOSS_PCT": "none",
                "GM_LOG_EXPOSURE": "1",
                "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
                "GM_SYNC_POSITIONS": "1",
                "GM_CASH_BUFFER": "0.99",
                "GM_VERBOSE_TRADES": "1",
                "GM_FORCE_SELL_MARKET_ORDER": "0",
                "GM_FORCE_BUY_MARKET_ORDER": "0",
                "GM_INTRADAY_RISK_MODE": "0",
                "GM_INTRADAY_REPLACE_BUY": "0",
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
                "GM_EQUITY_DD_SOFT_SCALE": "0.80",
                "GM_EQUITY_DD_HARD_SCALE": "0.60",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(SELL_RULE["min_score_exit_days"]),
                "GM_SCORE_EXIT_ENTRY_RATIO": str(SELL_RULE["score_exit"]),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(SELL_RULE["score_continue"]),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": str(SELL_RULE["day_drop_ratio"]),
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
            }
        )
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
            str(int(case.get("max_positions", 5))),
            "--holding-days",
            str(SELL_RULE["holding_days"]),
            "--max-holding-days",
            str(SELL_RULE["max_holding_days"]),
            "--target-position-pct",
            str(float(case.get("cap", 0.15))),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            "2022-06-07 09:00:00",
            "--backtest-end",
            "2026-06-29 15:30:00",
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        **case,
        **signal_meta,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "intraday_risk": 0,
        "sell_rule": "daily_h1m2_e099_c0995_maxsell1",
        "note": "research-only buy-frequency probe; no production parameter change",
    }


def main() -> None:
    if not BASE_SIGNAL.exists():
        raise FileNotFoundError(BASE_SIGNAL)
    rows = _base_rows()
    day_stats = _day_stats(rows)
    market = _load_market_regime()
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case, rows, day_stats, market)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

