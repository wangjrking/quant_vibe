from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

import research_prod_filter_amount_interpolate_20260625 as filt


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_amount_target_micro_20260625"

AMOUNT_MINS = [90000, 92500, 95000, 97500, 100000]
TARGET_PCTS = [0.895, 0.8975]
SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
    ("late20250701", "2025-07-01 09:00:00", "2026-06-23 15:30:00"),
    ("late20251009", "2025-10-09 09:00:00", "2026-06-23 15:30:00"),
    ("late20260105", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
]

BASE_ENV = dict(filt.BASE_ENV)
BASE_ENV["GM_EQUITY_DD_SOFT_SCALE"] = "0.74"
BASE_ENV["GM_EQUITY_DD_HARD_SCALE"] = "0.54"


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


def _case_name(amount_min: int, target_pct: float) -> str:
    pct = str(target_pct).replace(".", "p")
    return f"amt{amount_min}_tpct_{pct}"


def _build_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], amount_min: int, target_pct: float) -> tuple[Path, dict]:
    cfg = {
        "name": f"amt{amount_min}",
        "amount_min": amount_min,
        "total_mv_min": 200000,
        "total_mv_max": None,
        "turnover_min": None,
        "require_atr": False,
    }
    frame = base.loc[filt.liq._filter_mask(base, cfg)].copy()
    frame["entry_score"] = filt._entry_score(frame)
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    upper = max(float(base["rank_10d"].max()), 1.0)
    rows = []
    for signal_date, day in frame.groupby("trade_date", sort=True):
        buy_date = next_date.get(str(signal_date))
        if not buy_date:
            continue
        chosen = None
        for item in day.to_dict("records"):
            stock_code = str(item["stock_code"])
            buy_market = market.get(buy_date, {}).get(stock_code)
            if filt.liq.top1._is_st_like(buy_market) or filt.liq.top1._is_limit_buy(buy_market):
                continue
            chosen = item
            break
        if chosen is None:
            continue
        rows.append(
            {
                "signal_date": str(signal_date),
                "buy_date": buy_date,
                "symbol": filt.liq.top1.to_gm_symbol(str(chosen["stock_code"])),
                "stock_code": str(chosen["stock_code"]),
                "name": chosen.get("name"),
                "rank": 1,
                "pred_prob": filt.liq.top1._tailpow(float(chosen["rank_10d"]), 2.0, upper),
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
                "target_pct": f"{target_pct:.5f}",
                "holding_days": 2,
                "max_holding_days": 3,
                "score_exit_entry_ratio": "0.97000",
                "min_holding_days_before_score_exit": 1,
                "score_continue_entry_ratio": "0.97500",
                "filter_name": cfg["name"],
                "entry_weight_name": filt.WEIGHT["weight_name"],
                "dynamic_hold_name": "h2_m3_c975",
                "signal_stop_loss_pct": "0.06000",
                "signal_take_profit_pct": "0.07000",
                "strategy_variant": _case_name(amount_min, target_pct),
            }
        )
    path = REPORT_DIR / "signals" / f"{_case_name(amount_min, target_pct)}.csv"
    _write_rows(path, rows)
    daily = frame.groupby("trade_date").size()
    return path, {
        "filtered_rows": int(len(frame)),
        "signal_rows": int(len(rows)),
        "min_daily_candidates": int(daily.min()) if len(daily) else 0,
        "median_daily_candidates": float(daily.median()) if len(daily) else 0.0,
    }


def _run(signal_file: Path, case_name: str, target_pct: float, slice_name: str, start: str, end: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{case_name}_{slice_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        command = [
            str(filt.JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir", str(filt.STRATEGY_DIR),
            "--signal-file", str(signal_file),
            "--log-file", str(log_file),
            "--max-positions", "1",
            "--holding-days", "2",
            "--max-holding-days", "3",
            "--target-position-pct", str(target_pct),
            "--score-db", str(filt.SCORE_DB),
            "--score-table", filt.SCORE_TABLE,
            "--market-db", str(filt.MARKET_DB),
            "--backtest-start", start,
            "--backtest-end", end,
            "--backtest-adjust", "none",
            "--backtest-initial-cash", "600000",
            "--backtest-slippage-ratio", "0.0015",
            "--stop-loss-pct", "0.06",
            "--take-profit-pct", "0.07",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        indicator = _extract_indicator(log_file)
    return {
        "case_name": case_name,
        "slice": slice_name,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "log_file": str(log_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = filt._load_base()
    market = filt.liq._market_rows()
    next_date = filt.liq._date_map(base)

    detail = []
    summary = []
    for amount_min in AMOUNT_MINS:
        for target_pct in TARGET_PCTS:
            case_name = _case_name(amount_min, target_pct)
            signal_file, stats = _build_signal(base, market, next_date, amount_min, target_pct)
            case_rows = []
            for slice_name, start, end in SLICES:
                row = _run(signal_file, case_name, target_pct, slice_name, start, end)
                detail.append(row)
                case_rows.append(row)
            by_slice = {row["slice"]: row for row in case_rows}
            summary.append(
                {
                    "case_name": case_name,
                    "amount_min": amount_min,
                    "target_pct": target_pct,
                    "full_annual": by_slice["full"]["annual"],
                    "full_sharpe": by_slice["full"]["sharpe"],
                    "full_max_drawdown": by_slice["full"]["max_drawdown"],
                    "recent120_annual": by_slice["recent120"]["annual"],
                    "recent60_annual": by_slice["recent60"]["annual"],
                    "late20250701_annual": by_slice["late20250701"]["annual"],
                    "late20251009_annual": by_slice["late20251009"]["annual"],
                    "late20260105_annual": by_slice["late20260105"]["annual"],
                    **stats,
                }
            )
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
