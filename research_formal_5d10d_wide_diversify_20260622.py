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

from gm_signal_module import load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig, select_candidates


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_wide_diversify_20260622"
MANIFEST_5D = MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json"
MANIFEST_10D = MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
START_DATE = "20240604"
END_DATE = "20260618"


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"manifest is not approved_for_l5: {path}")
    return manifest


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / str(value)).resolve()


M5 = _load_manifest(MANIFEST_5D)
M10 = _load_manifest(MANIFEST_10D)
PRED_DB = _resolve(MANIFEST_10D, M10["db_path"])
MARKET_DB = _resolve(MANIFEST_10D, M10["market_db_path"])
TABLE_5D = str(M5["table"])
TABLE_10D = str(M10["table"])


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def _to_float(value, default=None):
    if value in (None, "", "None"):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_ranked_rows() -> list[dict]:
    conn = _connect_ro(PRED_DB)
    try:
        rows = conn.execute(
            f"""
            SELECT d10.trade_date, d10.stock_code,
                   d10.pred_prob AS pred_10d, d5.pred_prob AS pred_5d,
                   d10.close, d10.pre_close, d10.industry_encode,
                   d10.atr_qfq, d10.close_rate, d10.amount, d10.turnover_rate,
                   d10.total_mv
            FROM "{TABLE_10D}" d10
            JOIN "{TABLE_5D}" d5
              ON d10.trade_date = d5.trade_date AND d10.stock_code = d5.stock_code
            WHERE d10.trade_date >= ? AND d10.trade_date <= ?
              AND d10.pred_prob IS NOT NULL AND d5.pred_prob IS NOT NULL
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        item = dict(row)
        grouped.setdefault(str(item["trade_date"]), []).append(item)

    out: list[dict] = []
    for trade_date, day_rows in grouped.items():
        day_rows.sort(key=lambda row: _to_float(row.get("pred_10d"), -999.0), reverse=True)
        denom10 = max(len(day_rows) - 1, 1)
        rank10 = {str(row["stock_code"]): 1.0 - idx / denom10 for idx, row in enumerate(day_rows)}
        day_rows.sort(key=lambda row: _to_float(row.get("pred_5d"), -999.0), reverse=True)
        denom5 = max(len(day_rows) - 1, 1)
        rank5 = {str(row["stock_code"]): 1.0 - idx / denom5 for idx, row in enumerate(day_rows)}
        for row in day_rows:
            stock = str(row["stock_code"])
            pred10 = _to_float(row.get("pred_10d"), 0.0)
            pred5 = _to_float(row.get("pred_5d"), 0.0)
            close = _to_float(row.get("close"))
            atr = _to_float(row.get("atr_qfq"))
            row["rank_10d_pct"] = rank10[stock]
            row["rank_5d_pct"] = rank5[stock]
            row["pred_gap"] = abs(pred10 - pred5)
            row["atr_ratio"] = atr / close if atr is not None and close and close > 0 else None
            out.append(row)
    return out


def _score(row: dict, mode: str) -> float:
    r10 = _to_float(row.get("rank_10d_pct"), 0.0)
    r5 = _to_float(row.get("rank_5d_pct"), 0.0)
    gap = _to_float(row.get("pred_gap"), 0.0)
    amount = _to_float(row.get("amount"), 0.0)
    turnover = _to_float(row.get("turnover_rate"), 0.0)
    mv = _to_float(row.get("total_mv"), 999999999.0)
    atr = _to_float(row.get("atr_ratio"), 0.2)
    if mode == "blend_gap":
        return 0.78 * r10 + 0.22 * r5 - 0.65 * gap
    if mode == "blend_liq":
        return 0.74 * r10 + 0.20 * r5 - 0.40 * gap + 0.015 * math.log1p(max(amount, 0.0)) + 0.01 * min(turnover, 8.0)
    if mode == "blend_lowvol":
        return 0.76 * r10 + 0.20 * r5 - 0.45 * gap - 0.60 * atr - 0.00000015 * mv
    return 0.90 * r10 + 0.10 * r5


def _passes(row: dict, cfg: dict) -> bool:
    if str(row.get("stock_code") or "").upper().endswith(".BJ"):
        return False
    if cfg.get("min_amount") is not None and _to_float(row.get("amount"), -1.0) < float(cfg["min_amount"]):
        return False
    if cfg.get("min_turnover") is not None and _to_float(row.get("turnover_rate"), -1.0) < float(cfg["min_turnover"]):
        return False
    if cfg.get("max_mv") is not None and _to_float(row.get("total_mv"), 999999999.0) > float(cfg["max_mv"]):
        return False
    if cfg.get("max_gap") is not None and _to_float(row.get("pred_gap"), 999.0) > float(cfg["max_gap"]):
        return False
    if cfg.get("min_rank5") is not None and _to_float(row.get("rank_5d_pct"), 0.0) < float(cfg["min_rank5"]):
        return False
    if cfg.get("max_atr") is not None and (_to_float(row.get("atr_ratio")) is None or _to_float(row.get("atr_ratio")) > float(cfg["max_atr"])):
        return False
    return True


def _next_dates(trade_dates: list[str]) -> dict[str, str]:
    return {date: trade_dates[idx + 1] for idx, date in enumerate(trade_dates[:-1])}


def _gm_symbol(stock_code: str) -> str:
    code = str(stock_code).upper()
    if code.endswith(".SH"):
        return "SHSE." + code[:6]
    if code.endswith(".SZ"):
        return "SZSE." + code[:6]
    if code.endswith(".BJ"):
        return "BJSE." + code[:6]
    return code


def _is_open_limit_up(next_row: dict | None) -> bool:
    if not next_row:
        return False
    limit_times = _to_float(next_row.get("limit_times"), 0.0)
    if limit_times and limit_times > 0:
        return True
    pre_close = _to_float(next_row.get("pre_close"))
    open_price = _to_float(next_row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0:
        return False
    code = str(next_row.get("stock_code") or "")
    name = str(next_row.get("name") or "")
    pct = 0.05 if name.startswith(("ST", "*ST")) else (0.20 if code.startswith(("300", "301", "688")) else 0.10)
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _write_signal(rows: list[dict], market: dict[str, dict[str, dict]], cfg: dict, path: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if _passes(row, cfg):
            item = dict(row)
            item["pred_score"] = _score(item, str(cfg["score_mode"]))
            grouped.setdefault(str(item["trade_date"]), []).append(item)
    trade_dates = sorted(grouped)
    next_by = _next_dates(trade_dates)
    signals: list[dict] = []
    for date in trade_dates[:-1]:
        candidates = sorted(grouped[date], key=lambda row: (-_to_float(row.get("pred_score"), -999.0), row.get("atr_ratio") or 999.0))[: int(cfg["top_k"]) * 4]
        next_rows = market.get(next_by[date], {})
        clean = [row for row in candidates if not _is_open_limit_up(next_rows.get(str(row.get("stock_code"))))]
        clean = clean[: int(cfg["top_k"])]
        if not clean:
            continue
        total_weight = sum(max(_to_float(row.get("pred_score"), 0.0), 0.0) + 0.01 for row in clean)
        cap = float(cfg["target_total"]) / max(int(cfg["max_positions"]), 1)
        for idx, row in enumerate(clean, start=1):
            weight = max(_to_float(row.get("pred_score"), 0.0), 0.0) + 0.01
            target = min(cap, float(cfg["target_total"]) * weight / max(total_weight, 1e-9))
            signals.append(
                {
                    "signal_date": date,
                    "buy_date": next_by[date],
                    "symbol": _gm_symbol(str(row["stock_code"])),
                    "stock_code": row["stock_code"],
                    "name": row.get("name"),
                    "rank": idx,
                    "pred_prob": row.get("pred_score"),
                    "pred_10d": row.get("pred_10d"),
                    "pred_5d": row.get("pred_5d"),
                    "rank_10d_pct": row.get("rank_10d_pct"),
                    "rank_5d_pct": row.get("rank_5d_pct"),
                    "pred_gap": row.get("pred_gap"),
                    "atr_ratio": row.get("atr_ratio"),
                    "amount": row.get("amount"),
                    "turnover_rate": row.get("turnover_rate"),
                    "total_mv": row.get("total_mv"),
                    "holding_days": int(cfg["holding_days"]),
                    "target_pct": target,
                }
            )
    write_gm_signals_csv(signals, path)


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
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4",
            "GM_MAX_DAILY_SELLS": "1",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_SYNC_POSITIONS": "1",
            "GM_CASH_BUFFER": "0.99",
            "GM_VERBOSE_TRADES": "0",
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.85",
            "GM_EQUITY_DD_HARD_SCALE": "0.60",
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
        str(int(cfg["max_positions"])),
        "--holding-days",
        str(int(cfg["holding_days"])),
        "--max-holding-days",
        str(int(cfg["max_holding_days"])),
        "--target-position-pct",
        str(float(cfg["target_total"]) / max(int(cfg["max_positions"]), 1)),
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
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
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _load_signal_stats(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


def _write_csv(path: Path, rows: list[dict]) -> None:
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


VARIANTS = [
    {"name": "top6_gap_t98", "top_k": 6, "max_positions": 8, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_gap", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 300000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": 0.12},
    {"name": "top8_gap_t98", "top_k": 8, "max_positions": 10, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_gap", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 300000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": 0.12},
    {"name": "top8_liq_t98", "top_k": 8, "max_positions": 10, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_liq", "min_amount": 20000.0, "min_turnover": 0.5, "max_mv": 250000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": 0.12},
    {"name": "top8_lowvol_t98", "top_k": 8, "max_positions": 10, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_lowvol", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 250000.0, "max_gap": 0.08, "min_rank5": 0.50, "max_atr": 0.10},
    {"name": "top10_gap_t98", "top_k": 10, "max_positions": 12, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_gap", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 300000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": 0.12},
    {"name": "top10_liq_t98", "top_k": 10, "max_positions": 12, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_liq", "min_amount": 20000.0, "min_turnover": 0.5, "max_mv": 250000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": 0.12},
    {"name": "top6_gap_noatr_t98", "top_k": 6, "max_positions": 8, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_gap", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 300000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": None},
    {"name": "top8_gap_noatr_t98", "top_k": 8, "max_positions": 10, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_gap", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 300000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": None},
    {"name": "top8_liq_noatr_t98", "top_k": 8, "max_positions": 10, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_liq", "min_amount": 20000.0, "min_turnover": 0.5, "max_mv": 250000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": None},
    {"name": "top8_lowvol_noatr_t98", "top_k": 8, "max_positions": 10, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_lowvol", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 250000.0, "max_gap": 0.08, "min_rank5": 0.50, "max_atr": None},
    {"name": "top10_gap_noatr_t98", "top_k": 10, "max_positions": 12, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_gap", "min_amount": 10000.0, "min_turnover": 0.3, "max_mv": 300000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": None},
    {"name": "top10_liq_noatr_t98", "top_k": 10, "max_positions": 12, "holding_days": 6, "max_holding_days": 6, "target_total": 0.98, "score_mode": "blend_liq", "min_amount": 20000.0, "min_turnover": 0.5, "max_mv": 250000.0, "max_gap": 0.10, "min_rank5": 0.45, "max_atr": None},
]


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    rows = _load_ranked_rows()
    market = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        _write_signal(rows, market, cfg, signal_file)
        returncode = _run_backtest(cfg, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            **cfg,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(_load_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_csv(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_csv(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
    _write_csv(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
