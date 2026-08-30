from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

import tune_strict_sync_liquidity_refill_20260624 as liq


SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = (
    SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "dynamic_h2_m3_weight_micro_20260625"
)

BASE_FILTER = {
    "amount_min": 100000,
    "total_mv_min": 200000,
    "total_mv_max": None,
    "turnover_min": None,
    "require_atr": False,
}

TARGET_PCT = 0.89
HOLDING_DAYS = 2
MAX_HOLDING_DAYS = 3
EXIT_RATIO = 0.97
MIN_HOLD = 1
CONTINUE_RATIO = 0.975
INTRADAY_STOP_LOSS = 0.06
TAKE_PROFIT = 0.07
DD_SOFT = 0.07
DD_HARD = 0.12
DD_RECOVER = 0.03
DD_SOFT_SCALE = 0.70
DD_HARD_SCALE = 0.50

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("recent120_20251224", "2025-12-24 09:00:00"),
    ("recent60_20260324", "2026-03-24 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

WEIGHT_CASES = [
    {"weight_name": "w78_12_10", "w10d": 0.78, "w5d": 0.12, "w3d": 0.10},
    {"weight_name": "w79_11_10", "w10d": 0.79, "w5d": 0.11, "w3d": 0.10},
    {"weight_name": "w77_13_10", "w10d": 0.77, "w5d": 0.13, "w3d": 0.10},
    {"weight_name": "w78_11_11", "w10d": 0.78, "w5d": 0.11, "w3d": 0.11},
    {"weight_name": "w78_13_09", "w10d": 0.78, "w5d": 0.13, "w3d": 0.09},
    {"weight_name": "w79_12_09", "w10d": 0.79, "w5d": 0.12, "w3d": 0.09},
    {"weight_name": "w77_12_11", "w10d": 0.77, "w5d": 0.12, "w3d": 0.11},
    {"weight_name": "w785_115_10", "w10d": 0.785, "w5d": 0.115, "w3d": 0.10},
    {"weight_name": "w775_125_10", "w10d": 0.775, "w5d": 0.125, "w3d": 0.10},
]


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
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _case_name(case: dict) -> str:
    return f"micro_{case['weight_name']}_pos89_h2m3_c975_tp070"


def _is_bj_code(stock_code: str) -> bool:
    code = str(stock_code or "")
    return code.endswith(".BJ") or code.startswith(("8", "4"))


def _entry_score(frame: pd.DataFrame, case: dict) -> pd.Series:
    return (
        float(case["w10d"]) * frame["rank_10d"].astype(float)
        + float(case["w5d"]) * frame["rank_5d"].astype(float)
        + float(case["w3d"]) * frame["rank_3d"].astype(float)
    )


def _build_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], case: dict) -> tuple[Path, list[dict]]:
    frame = base.loc[liq._filter_mask(base, BASE_FILTER)].copy()
    frame = frame.loc[~frame["stock_code"].astype(str).map(_is_bj_code)].copy()
    frame["entry_score"] = _entry_score(frame, case)
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    rows: list[dict] = []
    upper = max(float(base["rank_10d"].max()), 1.0)
    for signal_date, day in frame.groupby("trade_date", sort=True):
        buy_date = next_date.get(str(signal_date))
        if not buy_date:
            continue
        chosen = None
        for item in day.to_dict("records"):
            stock_code = str(item["stock_code"])
            buy_market = market.get(buy_date, {}).get(stock_code)
            if _is_bj_code(stock_code):
                continue
            if liq.top1._is_st_like(buy_market) or liq.top1._is_limit_buy(buy_market):
                continue
            chosen = item
            break
        if chosen is None:
            continue
        stock_code = str(chosen["stock_code"])
        rows.append(
            {
                "signal_date": str(signal_date),
                "buy_date": buy_date,
                "symbol": liq.top1.to_gm_symbol(stock_code),
                "stock_code": stock_code,
                "name": chosen.get("name"),
                "rank": 1,
                "pred_prob": liq.top1._tailpow(float(chosen["rank_10d"]), 2.0, upper),
                "entry_score": chosen["entry_score"],
                "pred_3d": chosen.get("pred_3d"),
                "pred_5d": chosen.get("pred_5d"),
                "pred_10d": chosen.get("pred_10d"),
                "rank_3d": chosen.get("rank_3d"),
                "rank_5d": chosen.get("rank_5d"),
                "rank_10d": chosen.get("rank_10d"),
                "amount": chosen.get("amount"),
                "turnover_rate": chosen.get("turnover_rate"),
                "total_mv": chosen.get("total_mv"),
                "atr_qfq": chosen.get("atr_qfq"),
                "target_pct": f"{TARGET_PCT:.5f}",
                "holding_days": HOLDING_DAYS,
                "max_holding_days": MAX_HOLDING_DAYS,
                "score_exit_entry_ratio": f"{EXIT_RATIO:.5f}",
                "score_continue_entry_ratio": f"{CONTINUE_RATIO:.5f}",
                "min_holding_days_before_score_exit": MIN_HOLD,
                "signal_stop_loss_pct": f"{INTRADAY_STOP_LOSS:.5f}",
                "signal_take_profit_pct": f"{TAKE_PROFIT:.5f}",
                "filter_name": "liq_amt10w_mv20w_no_bj",
                "entry_weight_name": case["weight_name"],
                "weight_10d": f"{float(case['w10d']):.5f}",
                "weight_5d": f"{float(case['w5d']):.5f}",
                "weight_3d": f"{float(case['w3d']):.5f}",
                "exit_score_basis": "10d_core_score_table",
                "execution_variant": "force_sell_mkt_intraday_sl_tp_dd",
                "strategy_variant": _case_name(case),
            }
        )
    path = REPORT_DIR / "signals" / f"{_case_name(case)}.csv"
    _write_rows(path, rows)
    return path, rows


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


def _run_case(case: dict, signal_file: Path, start_name: str, start: str) -> dict:
    name = _case_name(case)
    log_file = REPORT_DIR / "logs" / f"{name}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(EXIT_RATIO)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(MIN_HOLD)
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(CONTINUE_RATIO)
        env["GM_INTRADAY_RISK_MODE"] = "1"
        env["GM_INTRADAY_REPLACE_BUY"] = "0"
        env["GM_INTRADAY_RISK_TIMES"] = "10:00:00,11:00:00,14:30:00"
        env["GM_EQUITY_DD_RISK_MODE"] = "1"
        env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(DD_SOFT)
        env["GM_EQUITY_DD_HARD_TRIGGER"] = str(DD_HARD)
        env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(DD_RECOVER)
        env["GM_EQUITY_DD_SOFT_SCALE"] = str(DD_SOFT_SCALE)
        env["GM_EQUITY_DD_HARD_SCALE"] = str(DD_HARD_SCALE)
        env["GM_EQUITY_DD_RESIZE_EXISTING"] = "0"
        command = [
            str(liq.JUEJIN_PYTHON),
            str(liq.MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(liq.STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            str(HOLDING_DAYS),
            "--max-holding-days",
            str(MAX_HOLDING_DAYS),
            "--target-position-pct",
            str(TARGET_PCT),
            "--score-db",
            str(liq.SCORE_DB),
            "--score-table",
            liq.SCORE_TABLE,
            "--market-db",
            str(liq.MARKET_DB),
            "--backtest-start",
            start,
            "--backtest-end",
            liq.BACKTEST_END,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0.0015",
            "--stop-loss-pct",
            str(INTRADAY_STOP_LOSS),
            "--take-profit-pct",
            str(TAKE_PROFIT),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(liq.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "name": name,
        "weight_name": case["weight_name"],
        "start_name": start_name,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "returncode": returncode,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        **_exposure_stats(log_file),
        **case,
    }


def _summarize(detail: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for case in WEIGHT_CASES:
        name = _case_name(case)
        rows = [row for row in detail if row["name"] == name]
        by_start = {row["start_name"]: row for row in rows}
        late = [
            _f(by_start[key]["annual"])
            for key in ["late_20250701", "late_20251009", "late_20260105"]
            if key in by_start
        ]
        late = [value for value in late if value == value]
        full = by_start.get("full_20240605", {})
        recent120 = by_start.get("recent120_20251224", {})
        recent60 = by_start.get("recent60_20260324", {})
        out.append(
            {
                "name": name,
                **case,
                "full_annual": full.get("annual"),
                "full_sharpe": full.get("sharpe"),
                "full_max_drawdown": full.get("max_drawdown"),
                "recent120_annual": recent120.get("annual"),
                "recent60_annual": recent60.get("annual"),
                "late_min_annual": min(late) if late else None,
                "late_median_annual": float(pd.Series(late).median()) if late else None,
                "avg_invested_pct": full.get("avg_invested_pct"),
                "open_count": full.get("open_count"),
                "admissible_research_screen": (
                    _f(full.get("max_drawdown")) <= 0.40
                    and _f(recent60.get("annual")) >= 0.40
                    and (min(late) if late else float("-inf")) >= 1.50
                ),
                **signal_stats.get(name, {}),
            }
        )
    return out


def _admission_key(row: dict) -> tuple:
    return (
        bool(row["admissible_research_screen"]),
        _f(row["full_annual"]),
        _f(row["full_sharpe"]),
        -_f(row["full_max_drawdown"]),
        _f(row["recent60_annual"]),
        _f(row["late_min_annual"]),
    )


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = liq._load_base()
    market = liq._market_rows()
    next_date = liq._date_map(base)
    detail: list[dict] = []
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    total = len(WEIGHT_CASES) * len(STARTS)
    done = 0
    for case in WEIGHT_CASES:
        signal_file, rows = _build_signal(base, market, next_date, case)
        signal_files.append(signal_file)
        signal_stats[_case_name(case)] = {
            "signal_rows": len(rows),
            "signal_days": len({row["signal_date"] for row in rows}),
            "buy_days": len({row["buy_date"] for row in rows}),
            "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
            "latest_buy_date": max((row["buy_date"] for row in rows), default=None),
        }
        for start_name, start in STARTS:
            done += 1
            print(f"[{done}/{total}] {_case_name(case)} {start_name}", flush=True)
            detail.append(_run_case(case, signal_file, start_name, start))
    summary = _summarize(detail, signal_stats)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_admission.csv", sorted(summary, key=_admission_key, reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
