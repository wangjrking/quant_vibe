from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
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
BASE_SIGNAL = REPORT_DIR / "open_gap_frequency_signals" / "hp_new70_turn80_top5_s170_cap27_h2m3__gap_rerank_penalty_1p5.csv"
OUT_SIGNAL_DIR = REPORT_DIR / "concentration_risk_filter_signals"
OUT_LOG_DIR = REPORT_DIR / "concentration_risk_filter_logs"
OUT_CSV = REPORT_DIR / "concentration_risk_filter_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "concentration_risk_filter_probe_20260630.json"


CASES: list[dict[str, Any]] = [
    {
        "name": "top1_80_skip_hot_turn_new_refill",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "reject_hot_turn": True,
        "reject_new_or_no_atr": True,
        "refill": True,
    },
    {
        "name": "top1_80_skip_hot_turn_refill",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "reject_hot_turn": True,
        "refill": True,
    },
    {
        "name": "top1_80_skip_new_refill",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "reject_new_or_no_atr": True,
        "refill": True,
    },
    {
        "name": "top1_80_scale_hot_new",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "scale_hot_turn": 0.40,
        "scale_new_or_no_atr": 0.35,
        "refill": False,
    },
    {
        "name": "top1_80_scale_hot_new90",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "scale_hot_turn": 0.90,
        "scale_new_or_no_atr": 0.90,
        "refill": False,
    },
    {
        "name": "top1_80_scale_hot_new75",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "scale_hot_turn": 0.75,
        "scale_new_or_no_atr": 0.75,
        "refill": False,
    },
    {
        "name": "top1_80_scale_hot_new60",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "scale_hot_turn": 0.60,
        "scale_new_or_no_atr": 0.60,
        "refill": False,
    },
    {
        "name": "top1_80_scale_hot_new50",
        "topn": 1,
        "fixed_target": 0.80,
        "target_cap": 0.80,
        "scale_hot_turn": 0.50,
        "scale_new_or_no_atr": 0.50,
        "refill": False,
    },
    {
        "name": "top2_50_skip_hot_turn_new_refill",
        "topn": 2,
        "fixed_target": 0.50,
        "target_cap": 0.50,
        "reject_hot_turn": True,
        "reject_new_or_no_atr": True,
        "refill": True,
    },
    {
        "name": "top2_50_skip_hot_turn_refill",
        "topn": 2,
        "fixed_target": 0.50,
        "target_cap": 0.50,
        "reject_hot_turn": True,
        "refill": True,
    },
    {
        "name": "top2_50_skip_new_refill",
        "topn": 2,
        "fixed_target": 0.50,
        "target_cap": 0.50,
        "reject_new_or_no_atr": True,
        "refill": True,
    },
    {
        "name": "top2_50_scale_hot_new",
        "topn": 2,
        "fixed_target": 0.50,
        "target_cap": 0.50,
        "scale_hot_turn": 0.50,
        "scale_new_or_no_atr": 0.45,
        "refill": False,
    },
    {
        "name": "top2_50_scale_hot_new75",
        "topn": 2,
        "fixed_target": 0.50,
        "target_cap": 0.50,
        "scale_hot_turn": 0.75,
        "scale_new_or_no_atr": 0.75,
        "refill": False,
    },
    {
        "name": "top3_34_skip_hot_turn_new_refill",
        "topn": 3,
        "fixed_target": 0.34,
        "target_cap": 0.34,
        "reject_hot_turn": True,
        "reject_new_or_no_atr": True,
        "refill": True,
    },
    {
        "name": "top3_34_scale_hot_new",
        "topn": 3,
        "fixed_target": 0.34,
        "target_cap": 0.34,
        "scale_hot_turn": 0.60,
        "scale_new_or_no_atr": 0.55,
        "refill": False,
    },
    {
        "name": "top3_34_scale_hot_new75",
        "topn": 3,
        "fixed_target": 0.34,
        "target_cap": 0.34,
        "scale_hot_turn": 0.75,
        "scale_new_or_no_atr": 0.75,
        "refill": False,
    },
]


SELL_RULE = {
    "holding_days": 2,
    "max_holding_days": 3,
    "score_exit": 0.98,
    "score_continue": 0.99,
    "min_score_exit_days": 1,
    "day_drop_ratio": 0.995,
    "max_daily_sells": 1,
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


def _risk_flags(row: dict[str, Any]) -> dict[str, bool]:
    pct_chg = _float(row.get("pct_chg"))
    turnover = _float(row.get("turnover_rate"))
    list_age = _float(row.get("list_age_days"))
    atr = _float(row.get("atr_qfq"))
    two_day_ret = _float(row.get("two_day_ret"))
    hot_turn = pct_chg is not None and turnover is not None and pct_chg >= 15.0 and turnover >= 15.0
    two_day_hot = two_day_ret is not None and turnover is not None and two_day_ret >= 0.20 and turnover >= 10.0
    new_or_no_atr = (list_age is not None and list_age < 60.0) or atr is None
    return {
        "hot_turn": hot_turn or two_day_hot,
        "new_or_no_atr": new_or_no_atr,
    }


def _reject_reason(row: dict[str, Any], case: dict[str, Any]) -> str | None:
    flags = _risk_flags(row)
    if case.get("reject_hot_turn") and flags["hot_turn"]:
        return "hot_turn"
    if case.get("reject_new_or_no_atr") and flags["new_or_no_atr"]:
        return "new_or_no_atr"
    return None


def _target_for_case(row: dict[str, Any], case: dict[str, Any]) -> tuple[float, str]:
    target = float(case["fixed_target"])
    flags = _risk_flags(row)
    applied: list[str] = []
    hot_scale = _float(case.get("scale_hot_turn"))
    new_scale = _float(case.get("scale_new_or_no_atr"))
    if hot_scale is not None and flags["hot_turn"]:
        target *= hot_scale
        applied.append("hot_turn_scale")
    if new_scale is not None and flags["new_or_no_atr"]:
        target *= new_scale
        applied.append("new_or_no_atr_scale")
    return max(0.0, min(target, float(case["target_cap"]))), "|".join(applied)


def _make_signal(case: dict[str, Any], by_date: dict[str, list[dict[str, Any]]]) -> tuple[Path, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rejected = Counter()
    target = int(case["topn"])
    for signal_date, candidates in sorted(by_date.items()):
        picked = 0
        for row in sorted(candidates, key=lambda item: (int(float(item["rank"])), item["stock_code"])):
            rank = int(float(row["rank"]))
            if not case.get("refill") and rank > target:
                continue
            reason = _reject_reason(row, case)
            if reason:
                rejected[reason] += 1
                continue
            target_pct, scale_reason = _target_for_case(row, case)
            item = dict(row)
            picked += 1
            item["rank"] = str(picked)
            item["target_pct"] = f"{target_pct:.5f}"
            item["holding_days"] = str(SELL_RULE["holding_days"])
            item["max_holding_days"] = str(SELL_RULE["max_holding_days"])
            item["score_exit_entry_ratio"] = f"{SELL_RULE['score_exit']:.5f}"
            item["score_continue_entry_ratio"] = f"{SELL_RULE['score_continue']:.5f}"
            item["min_holding_days_before_score_exit"] = str(SELL_RULE["min_score_exit_days"])
            item["strategy_variant"] = str(case["name"])
            item["dynamic_hold_name"] = "h2m3_e098_c099_daydrop995_ms1"
            item["filter_name"] = str(case["name"])
            item["risk_filter_scale_reason"] = scale_reason
            rows.append(item)
            if picked >= target:
                break
    rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, rows)
    counts = Counter(row["signal_date"] for row in rows)
    return output, {
        "signal_rows": len(rows),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < target),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "rejected_hot_turn": rejected["hot_turn"],
        "rejected_new_or_no_atr": rejected["new_or_no_atr"],
    }


def _run(case: dict[str, Any], by_date: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case, by_date)
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
                "GM_EQUITY_DD_RISK_MODE": "0",
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
                "GM_RESIZE_HELD_ON_SIGNAL": "0",
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
                "GM_INDEX_RISK_EXIT_MODE": "0",
                "GM_BREADTH_RISK_EXIT_MODE": "0",
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
            str(int(case["topn"])),
            "--holding-days",
            str(SELL_RULE["holding_days"]),
            "--max-holding-days",
            str(SELL_RULE["max_holding_days"]),
            "--target-position-pct",
            str(float(case["target_cap"])),
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
        "note": "research-only signal-date risk filter/refill probe; no production parameter change",
    }


def main() -> None:
    if not BASE_SIGNAL.exists():
        raise FileNotFoundError(BASE_SIGNAL)
    base_rows = list(csv.DictReader(BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="")))
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        by_date[row["signal_date"]].append(row)

    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case, by_date)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

