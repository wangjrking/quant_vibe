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


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_filter_narrow_grid_20260625"
BASE_CACHE = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "latest_formal_base_no_delist_v2.parquet"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_dynamic_h2m3_pos8975_scale7555_v20260625" / "code_snapshot"
SCORE_DB = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "diversification_tune_20260624" / "scores_diversification.db"
SCORE_TABLE = "score_div_top1_w90_5d10_h5_e099"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_FORCE_SELL_MARKET_ORDER": "1",
    "GM_INTRADAY_RISK_MODE": "1",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.115",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.75",
    "GM_EQUITY_DD_HARD_SCALE": "0.55",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.975",
}

WEIGHT = {"weight_name": "w78_5d12_3d10", "w10d": 0.78, "w5d": 0.12, "w3d": 0.10, "w1d": 0.00}
RULE = {"target_pct": 0.8975, "holding_days": 2, "max_holding_days": 3, "stop_loss": 0.06, "take_profit": 0.07}

FILTERS = [
    {"name": "base_amt10w_mv20w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "amt8w_mv20w", "amount_min": 80000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "amt12w_mv20w", "amount_min": 120000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "amt10w_mv25w", "amount_min": 100000, "total_mv_min": 250000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "amt10w_mv20w_cap200w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": 2000000, "turnover_min": None, "require_atr": False},
    {"name": "turn1_amt10w_mv20w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": 1.0, "require_atr": False},
]

SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
    ("late20250701", "2025-07-01 09:00:00", "2026-06-23 15:30:00"),
    ("late20260105", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
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


def _load_base() -> pd.DataFrame:
    frame = pd.read_parquet(BASE_CACHE)
    for column in ["rank_1d", "rank_3d", "rank_5d", "rank_10d", "amount", "turnover_rate", "total_mv", "atr_qfq"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _entry_score(frame: pd.DataFrame) -> pd.Series:
    return (
        float(WEIGHT["w10d"]) * frame["rank_10d"].astype(float)
        + float(WEIGHT["w5d"]) * frame["rank_5d"].astype(float)
        + float(WEIGHT["w3d"]) * frame["rank_3d"].astype(float)
        + float(WEIGHT["w1d"]) * frame["rank_1d"].astype(float)
    )


def _case_name(cfg: dict) -> str:
    return f"prod_filter_{cfg['name']}"


def _build_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], cfg: dict) -> tuple[Path, list[dict], dict]:
    frame = base.loc[liq._filter_mask(base, cfg)].copy()
    frame["entry_score"] = _entry_score(frame)
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    upper = max(float(base["rank_10d"].max()), 1.0)
    rows: list[dict] = []
    for signal_date, day in frame.groupby("trade_date", sort=True):
        buy_date = next_date.get(str(signal_date))
        if not buy_date:
            continue
        chosen = None
        for item in day.to_dict("records"):
            stock_code = str(item["stock_code"])
            buy_market = market.get(buy_date, {}).get(stock_code)
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
                "pred_1d": chosen.get("pred_1d"),
                "pred_3d": chosen.get("pred_3d"),
                "pred_5d": chosen.get("pred_5d"),
                "pred_10d": chosen.get("pred_10d"),
                "rank_1d": chosen.get("rank_1d"),
                "rank_3d": chosen.get("rank_3d"),
                "rank_5d": chosen.get("rank_5d"),
                "rank_10d": chosen.get("rank_10d"),
                "amount": chosen.get("amount"),
                "turnover_rate": chosen.get("turnover_rate"),
                "total_mv": chosen.get("total_mv"),
                "atr_qfq": chosen.get("atr_qfq"),
                "target_pct": f"{RULE['target_pct']:.5f}",
                "holding_days": RULE["holding_days"],
                "max_holding_days": RULE["max_holding_days"],
                "score_exit_entry_ratio": "0.97000",
                "min_holding_days_before_score_exit": 1,
                "score_continue_entry_ratio": "0.97500",
                "filter_name": cfg["name"],
                "entry_weight_name": WEIGHT["weight_name"],
                "dynamic_hold_name": "h2_m3_c975",
                "exit_score_basis": "10d_core_score_table",
                "execution_variant": "force_sell_mkt",
                "signal_stop_loss_pct": f"{RULE['stop_loss']:.5f}",
                "signal_take_profit_pct": f"{RULE['take_profit']:.5f}",
                "strategy_variant": _case_name(cfg),
            }
        )
    path = REPORT_DIR / "signals" / f"{_case_name(cfg)}.csv"
    _write_rows(path, rows)
    daily = frame.groupby("trade_date").size()
    stats = {
        "filtered_rows": int(len(frame)),
        "signal_rows": int(len(rows)),
        "signal_days": len({row["signal_date"] for row in rows}),
        "min_daily_candidates": int(daily.min()) if len(daily) else 0,
        "median_daily_candidates": float(daily.median()) if len(daily) else 0.0,
    }
    return path, rows, stats


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
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    values: list[float] = []
    actives: list[int] = []
    if not log_file.exists():
        return {"avg_invested_pct": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        actives.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "max_active_positions": max(actives) if actives else None,
        "exposure_points": len(values),
    }


def _run_case(cfg: dict, signal_file: Path, slice_name: str, start: str, end: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{slice_name}_{_case_name(cfg)}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
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
        "1",
        "--holding-days",
        str(RULE["holding_days"]),
        "--max-holding-days",
        str(RULE["max_holding_days"]),
        "--target-position-pct",
        str(RULE["target_pct"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        start,
        "--backtest-end",
        end,
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
        "--stop-loss-pct",
        str(RULE["stop_loss"]),
        "--take-profit-pct",
        str(RULE["take_profit"]),
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    exposure = _exposure_stats(log_file)
    return {
        "tag": f"{slice_name}_{_case_name(cfg)}",
        "filter_name": cfg["name"],
        "slice": slice_name,
        "start": start,
        "end": end,
        "returncode": proc.returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "avg_invested_pct": exposure["avg_invested_pct"],
        "amount_min": cfg.get("amount_min"),
        "total_mv_min": cfg.get("total_mv_min"),
        "total_mv_max": cfg.get("total_mv_max"),
        "turnover_min": cfg.get("turnover_min"),
        "require_atr": cfg.get("require_atr"),
        "log_file": str(log_file),
        "signal_file": str(signal_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_base()
    market = liq._market_rows()
    next_date = liq._date_map(base)
    rows: list[dict] = []
    summary: list[dict] = []
    signal_files: list[Path] = []
    for cfg in FILTERS:
        signal_file, signal_rows, stats = _build_signal(base, market, next_date, cfg)
        signal_files.append(signal_file)
        case_rows = []
        for slice_name, start, end in SLICES:
            row = _run_case(cfg, signal_file, slice_name, start, end)
            rows.append(row)
            case_rows.append(row)
        by_slice = {row["slice"]: row for row in case_rows}
        summary.append(
            {
                "filter_name": cfg["name"],
                "amount_min": cfg.get("amount_min"),
                "total_mv_min": cfg.get("total_mv_min"),
                "total_mv_max": cfg.get("total_mv_max"),
                "turnover_min": cfg.get("turnover_min"),
                "require_atr": cfg.get("require_atr"),
                "full_annual": by_slice["full"]["annual"],
                "full_sharpe": by_slice["full"]["sharpe"],
                "full_max_drawdown": by_slice["full"]["max_drawdown"],
                "recent120_annual": by_slice["recent120"]["annual"],
                "recent60_annual": by_slice["recent60"]["annual"],
                "ytd_annual": by_slice["ytd2026"]["annual"],
                "late20250701_annual": by_slice["late20250701"]["annual"],
                "late20260105_annual": by_slice["late20260105"]["annual"],
                **stats,
            }
        )
    _write_rows(REPORT_DIR / "detail.csv", rows)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "filters": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
