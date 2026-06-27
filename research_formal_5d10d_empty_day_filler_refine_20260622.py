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


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_empty_day_filler_refine_20260622"
)
CORE_SIGNAL = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_gap_v20260622"
    / "signals"
    / "backtest_signals_gap_t0p03_g1p08_b0p88.csv"
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


FILTERS = {
    "strict_small_consensus": {
        "max_total_mv": 200000.0,
        "min_amount": 50000.0,
        "min_turnover": 0.5,
        "min_rank_10d": 0.95,
        "min_rank_5d": 0.80,
        "min_pred_10d": 0.0,
        "min_pred_5d": 0.0,
        "max_gap": None,
    },
    "gap_small": {
        "max_total_mv": 200000.0,
        "min_amount": 30000.0,
        "min_turnover": 0.3,
        "min_rank_10d": 0.90,
        "min_rank_5d": 0.70,
        "min_pred_10d": 0.0,
        "min_pred_5d": None,
        "max_gap": 0.10,
    },
    "liquid_top10d": {
        "max_total_mv": 300000.0,
        "min_amount": 100000.0,
        "min_turnover": 0.5,
        "min_rank_10d": 0.97,
        "min_rank_5d": None,
        "min_pred_10d": 0.0,
        "min_pred_5d": None,
        "max_gap": None,
    },
    "loose_small": {
        "max_total_mv": 200000.0,
        "min_total_mv": None,
        "min_amount": 20000.0,
        "min_turnover": 0.3,
        "max_turnover": None,
        "min_rank_10d": 0.85,
        "min_rank_5d": None,
        "min_pred_10d": 0.0,
        "min_pred_5d": None,
        "max_gap": None,
    },
    "large_liquid_top10d": {
        "max_total_mv": None,
        "min_total_mv": 1000000.0,
        "min_amount": 500000.0,
        "min_turnover": 0.1,
        "max_turnover": 3.0,
        "min_rank_10d": 0.97,
        "min_rank_5d": None,
        "min_pred_10d": 0.0,
        "min_pred_5d": None,
        "max_gap": None,
    },
    "large_liquid_consensus": {
        "max_total_mv": None,
        "min_total_mv": 1000000.0,
        "min_amount": 500000.0,
        "min_turnover": 0.1,
        "max_turnover": 3.0,
        "min_rank_10d": 0.95,
        "min_rank_5d": 0.80,
        "min_pred_10d": 0.0,
        "min_pred_5d": 0.0,
        "max_gap": None,
    },
    "mid_liquid_lowturn": {
        "max_total_mv": 3000000.0,
        "min_total_mv": 300000.0,
        "min_amount": 200000.0,
        "min_turnover": 0.1,
        "max_turnover": 2.0,
        "min_rank_10d": 0.95,
        "min_rank_5d": None,
        "min_pred_10d": 0.0,
        "min_pred_5d": None,
        "max_gap": None,
    },
}


def _variant(name: str, filter_name: str, fill_mode: str, filler_target: float, max_fillers: int = 5, extra_env=None) -> dict:
    return {
        "name": name,
        "filter_name": filter_name,
        "filters": FILTERS[filter_name],
        "fill_mode": fill_mode,
        "filler_target": filler_target,
        "max_fillers": max_fillers,
        "max_positions": 5,
        "holding_days": 7,
        "max_holding_days": 10,
        "extra_env": extra_env or {},
    }


VARIANTS = []
for filter_name in FILTERS:
    for target in [0.10, 0.15, 0.19]:
        VARIANTS.append(_variant(f"{filter_name}_missing_t{str(target).replace('.', 'p')}", filter_name, "missing_core_day", target))
for filter_name in ["strict_small_consensus", "gap_small", "liquid_top10d"]:
    VARIANTS.append(_variant(f"{filter_name}_lowcount_t015", filter_name, "core_count_lt3", 0.15))
    VARIANTS.append(_variant(f"{filter_name}_missing_t015_defer8", filter_name, "missing_core_day", 0.15, extra_env={"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8", "GM_DEFER_EXIT_MIN_POSITION_RETURN": "0.06"}))
for filter_name in ["large_liquid_top10d", "large_liquid_consensus", "mid_liquid_lowturn"]:
    for target in [0.10, 0.15, 0.19]:
        VARIANTS.append(_variant(f"{filter_name}_missing_t{str(target).replace('.', 'p')}", filter_name, "missing_core_day", target))
for filter_name in ["large_liquid_top10d", "large_liquid_consensus", "strict_small_consensus"]:
    for streak in [3, 5, 8]:
        for target in [0.10, 0.15, 0.19]:
            VARIANTS.append(_variant(f"{filter_name}_streak{streak}_t{str(target).replace('.', 'p')}", filter_name, f"missing_streak_ge{streak}", target))


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _symbol(stock_code: str) -> str:
    code, suffix = stock_code.split(".", 1)
    return f"{'SHSE' if suffix == 'SH' else 'SZSE'}.{code}"


def _trade_dates() -> list[str]:
    conn = sqlite3.connect(MARKET_DB)
    try:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                ("20240604", "20260630"),
            )
            if row[0]
        ]
    finally:
        conn.close()


def _next_trade_date_map() -> dict[str, str]:
    dates = _trade_dates()
    return {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}


def _core_counts(rows: list[dict]) -> dict[str, int]:
    counts = {}
    for row in rows:
        counts[str(row.get("signal_date") or "")] = counts.get(str(row.get("signal_date") or ""), 0) + 1
    return counts


def _all_fusion_dates() -> list[str]:
    conn = sqlite3.connect(FUSION_DB)
    try:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT trade_date FROM fusion_rank_base WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                ("20240604", "20260618"),
            )
            if row[0]
        ]
    finally:
        conn.close()


def _where_clause(filters: dict) -> tuple[str, list[object]]:
    clauses = [
        "trade_date = ?",
        "stock_code NOT LIKE '%.BJ'",
        "COALESCE(name, '') NOT LIKE 'ST%'",
        "COALESCE(name, '') NOT LIKE '*ST%'",
        "COALESCE(name, '') NOT LIKE ?",
        "COALESCE(name, '') NOT LIKE ?",
        "(limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) = 0)",
        "amount IS NOT NULL AND amount >= ?",
        "turnover_rate IS NOT NULL AND turnover_rate >= ?",
        "rank_10d IS NOT NULL AND rank_10d >= ?",
    ]
    params: list[object] = ["%退市%", "退%", filters["min_amount"], filters["min_turnover"], filters["min_rank_10d"]]
    if filters.get("max_total_mv") is not None:
        clauses.append("total_mv IS NOT NULL AND total_mv <= ?")
        params.append(filters["max_total_mv"])
    if filters.get("min_total_mv") is not None:
        clauses.append("total_mv IS NOT NULL AND total_mv >= ?")
        params.append(filters["min_total_mv"])
    if filters.get("max_turnover") is not None:
        clauses.append("turnover_rate IS NOT NULL AND turnover_rate <= ?")
        params.append(filters["max_turnover"])
    if filters.get("min_rank_5d") is not None:
        clauses.append("rank_5d IS NOT NULL AND rank_5d >= ?")
        params.append(filters["min_rank_5d"])
    if filters.get("min_pred_10d") is not None:
        clauses.append("pred_10d IS NOT NULL AND pred_10d > ?")
        params.append(filters["min_pred_10d"])
    if filters.get("min_pred_5d") is not None:
        clauses.append("pred_5d IS NOT NULL AND pred_5d > ?")
        params.append(filters["min_pred_5d"])
    if filters.get("max_gap") is not None:
        clauses.append("pred_10d IS NOT NULL AND pred_5d IS NOT NULL AND ABS(pred_10d - pred_5d) <= ?")
        params.append(filters["max_gap"])
    return " AND ".join(clauses), params


def _load_fillers_for_dates(dates: list[str], filters: dict, max_fillers: int) -> dict[str, list[dict]]:
    where, params_tail = _where_clause(filters)
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    by_date: dict[str, list[dict]] = {}
    try:
        for date in dates:
            rows = conn.execute(
                f"""
                SELECT trade_date, stock_code, name, pred_5d, pred_10d, rank_5d, rank_10d,
                       amount, turnover_rate, total_mv, atr_qfq, close
                FROM fusion_rank_base
                WHERE {where}
                ORDER BY ((rank_10d * 0.8) + (rank_5d * 0.2)) DESC, stock_code
                LIMIT ?
                """,
                [date, *params_tail, int(max_fillers)],
            ).fetchall()
            by_date[date] = [dict(row) for row in rows]
    finally:
        conn.close()
    return by_date


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    core_rows = _load_rows(CORE_SIGNAL)
    counts = _core_counts(core_rows)
    next_trade = _next_trade_date_map()
    selected_dates = []
    missing_streak = 0
    for date in _all_fusion_dates():
        count = counts.get(date, 0)
        missing_streak = missing_streak + 1 if count == 0 else 0
        if variant["fill_mode"] == "missing_core_day" and count == 0:
            selected_dates.append(date)
        elif variant["fill_mode"] == "core_count_lt3" and count < 3:
            selected_dates.append(date)
        elif variant["fill_mode"].startswith("missing_streak_ge") and count == 0:
            threshold = int(variant["fill_mode"].replace("missing_streak_ge", ""))
            if missing_streak >= threshold:
                selected_dates.append(date)
    fillers_by_date = _load_fillers_for_dates(selected_dates, variant["filters"], int(variant["max_fillers"]))
    rows = [dict(row) for row in core_rows]
    seen = {(row.get("signal_date"), row.get("stock_code")) for row in rows}
    for date in selected_dates:
        buy_date = next_trade.get(date)
        if not buy_date:
            continue
        existing_count = counts.get(date, 0)
        rank = existing_count + 1
        for filler in fillers_by_date.get(date, []):
            key = (date, filler["stock_code"])
            if key in seen or rank > 5:
                continue
            close = _to_float(filler.get("close"))
            atr = _to_float(filler.get("atr_qfq"))
            pred_5d = _to_float(filler.get("pred_5d"))
            pred_10d = _to_float(filler.get("pred_10d"))
            row = {
                "signal_date": date,
                "buy_date": buy_date,
                "symbol": _symbol(str(filler["stock_code"])),
                "stock_code": filler["stock_code"],
                "name": filler.get("name"),
                "rank": str(rank),
                "pred_prob": f"{1.0 + (_to_float(filler.get('rank_10d'), 0.0) or 0.0) * 0.8 + (_to_float(filler.get('rank_5d'), 0.0) or 0.0) * 0.2:.12f}",
                "atr_ratio": "" if atr is None or not close else f"{atr / close:.10f}",
                "holding_days": "7",
                "target_pct": f"{float(variant['filler_target']):.5f}",
                "score_exit_entry_ratio": "1.0",
                "min_holding_days_before_score_exit": "3",
                "weak_mode": f"empty_day_filler_{variant['filter_name']}",
                "pred_5d": filler.get("pred_5d"),
                "pred_10d": filler.get("pred_10d"),
                "rank_5d": filler.get("rank_5d"),
                "rank_10d": filler.get("rank_10d"),
                "turnover_rate": filler.get("turnover_rate"),
                "amount": filler.get("amount"),
                "total_mv": filler.get("total_mv"),
                "pred_gap": "" if pred_5d is None or pred_10d is None else f"{abs(pred_10d - pred_5d):.10f}",
            }
            rows.append(row)
            seen.add(key)
            rank += 1
    rows.sort(key=lambda item: (str(item.get("signal_date")), int(float(item.get("rank") or 999999)), str(item.get("stock_code"))))
    fieldnames = list(_load_rows(CORE_SIGNAL)[0].keys())
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fieldnames} for row in rows])


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
    rows = _load_rows(signal_file)
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
        "0.28",
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


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def _write_report(rows: list[dict]) -> None:
    hits = [
        row
        for row in rows
        if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
    ]
    best_objective = max(
        rows,
        key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80),
    )
    best_annual = max(rows, key=lambda row: _metric(row, "annual"))
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    report = f"""# 5D10D 空窗日高质量补位调参结论

## 当前结论

本轮只使用 formal L4 的 5D 与 10D 融合预测资产，在当前 L5 核心信号缺失或低数量日期补入小市值、流动性、非 ST、非退市、非北交所、非当前涨停候选。不使用 1D、不使用行业限制、不使用月份或日期过滤。所有绩效以掘金回测日志为准。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均持仓 >= 0.80` 的候选数量：{len(hits)}
- 是否建议替换当前 L5：{"是" if hits else "否"}

## 最优候选

- 综合最接近目标：`{best_objective["name"]}`，年化 `{_metric(best_objective, "annual"):.6f}`，夏普 `{_metric(best_objective, "sharpe"):.6f}`，平均持仓 `{_metric(best_objective, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，平均持仓 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，平均持仓 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据路径

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 信号目录：`{REPORT_DIR / "signals"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "5D10D空窗日高质量补位调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_variant_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "filter_name": variant["filter_name"],
            "fill_mode": variant["fill_mode"],
            "filler_target": variant["filler_target"],
            "max_fillers": variant["max_fillers"],
            "extra_env_json": json.dumps(variant.get("extra_env", {}), ensure_ascii=False, sort_keys=True),
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
        row.update(_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')} signals={row.get('signal_count')}"
        )
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    _write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
