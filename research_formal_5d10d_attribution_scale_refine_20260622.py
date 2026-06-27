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
    / "formal_5d10d_attribution_scale_refine_20260622"
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


def _variant(name: str, rules: list[dict], *, normalize_day: bool, min_day_target: float | None = None) -> dict:
    return {
        "name": name,
        "rules": rules,
        "normalize_day": normalize_day,
        "min_day_target": min_day_target,
        "max_positions": 5,
        "holding_days": 7,
        "max_holding_days": 10,
        "extra_env": {},
    }


VARIANTS = [
    _variant("repro", [], normalize_day=False),
    _variant(
        "gap_good_bad_soft",
        [
            {"field": "pred_gap", "op": "lt", "threshold": 0.02, "scale": 1.12},
            {"field": "pred_gap", "op": "between", "lower": 0.05, "upper": 0.10, "scale": 0.88},
        ],
        normalize_day=True,
    ),
    _variant(
        "turn_good_bad_soft",
        [
            {"field": "turnover_rate", "op": "lt", "threshold": 1.0, "scale": 1.15},
            {"field": "turnover_rate", "op": "between", "lower": 3.0, "upper": 6.0, "scale": 0.85},
            {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 1.10},
        ],
        normalize_day=True,
    ),
    _variant(
        "gap_turn_combo_soft",
        [
            {"field": "pred_gap", "op": "lt", "threshold": 0.02, "scale": 1.12},
            {"field": "pred_gap", "op": "between", "lower": 0.05, "upper": 0.10, "scale": 0.90},
            {"field": "turnover_rate", "op": "between", "lower": 3.0, "upper": 6.0, "scale": 0.90},
            {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 1.08},
        ],
        normalize_day=True,
    ),
    _variant(
        "gap_turn_combo_hard",
        [
            {"field": "pred_gap", "op": "lt", "threshold": 0.02, "scale": 1.20},
            {"field": "pred_gap", "op": "between", "lower": 0.05, "upper": 0.10, "scale": 0.78},
            {"field": "turnover_rate", "op": "between", "lower": 3.0, "upper": 6.0, "scale": 0.78},
            {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 1.15},
        ],
        normalize_day=True,
    ),
    _variant(
        "low_count_floor95",
        [],
        normalize_day=True,
        min_day_target=0.95,
    ),
    _variant(
        "gap_turn_floor95",
        [
            {"field": "pred_gap", "op": "lt", "threshold": 0.02, "scale": 1.12},
            {"field": "pred_gap", "op": "between", "lower": 0.05, "upper": 0.10, "scale": 0.90},
            {"field": "turnover_rate", "op": "between", "lower": 3.0, "upper": 6.0, "scale": 0.90},
            {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 1.08},
        ],
        normalize_day=True,
        min_day_target=0.95,
    ),
    _variant(
        "gap_turn_floor98",
        [
            {"field": "pred_gap", "op": "lt", "threshold": 0.02, "scale": 1.10},
            {"field": "pred_gap", "op": "between", "lower": 0.05, "upper": 0.10, "scale": 0.92},
            {"field": "turnover_rate", "op": "between", "lower": 3.0, "upper": 6.0, "scale": 0.92},
            {"field": "turnover_rate", "op": "ge", "threshold": 6.0, "scale": 1.06},
        ],
        normalize_day=True,
        min_day_target=0.98,
    ),
    _variant(
        "rank_tail_boost",
        [
            {"field": "rank", "op": "eq", "threshold": 5.0, "scale": 1.10},
            {"field": "rank", "op": "eq", "threshold": 4.0, "scale": 0.92},
        ],
        normalize_day=True,
    ),
    _variant(
        "amount_mid_down",
        [
            {"field": "amount", "op": "between", "lower": 20000.0, "upper": 50000.0, "scale": 0.88},
            {"field": "amount", "op": "between", "lower": 50000.0, "upper": 100000.0, "scale": 1.08},
        ],
        normalize_day=True,
    ),
]


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


def _matches(value: float | None, rule: dict) -> bool:
    if value is None:
        return False
    op = rule["op"]
    if op == "lt":
        return value < float(rule["threshold"])
    if op == "le":
        return value <= float(rule["threshold"])
    if op == "ge":
        return value >= float(rule["threshold"])
    if op == "gt":
        return value > float(rule["threshold"])
    if op == "eq":
        return abs(value - float(rule["threshold"])) < 1e-9
    if op == "between":
        return value >= float(rule["lower"]) and value < float(rule["upper"])
    raise ValueError(f"unsupported op: {op}")


def _scaled_target(row: dict, features: dict, rules: list[dict], max_single_position_pct: float = 0.28) -> float:
    target = _to_float(row.get("target_pct"), 0.0) or 0.0
    scale = 1.0
    merged = {**features, "rank": _to_float(row.get("rank"))}
    for rule in rules:
        if _matches(_to_float(merged.get(rule["field"])), rule):
            scale *= float(rule["scale"])
    return max(0.0, min(target * scale, float(max_single_position_pct)))


def _write_variant_signal(variant: dict, signal_file: Path, base_rows: list[dict], fusion_features: dict) -> None:
    rows = []
    max_single_position_pct = float(variant.get("max_single_position_pct", 0.28))
    for row in base_rows:
        features = fusion_features.get((str(row.get("signal_date")), str(row.get("stock_code"))), {})
        out = dict(row)
        out["target_pct"] = f"{_scaled_target(row, features, variant['rules'], max_single_position_pct):.5f}"
        for key in ["pred_5d", "pred_10d", "rank_5d", "rank_10d", "turnover_rate", "amount", "total_mv", "pred_gap"]:
            out[key] = features.get(key)
        rows.append(out)

    if variant["normalize_day"]:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("signal_date")), []).append(row)
        for day_rows in grouped.values():
            original_sum = sum(_to_float(row.get("target_pct"), 0.0) or 0.0 for row in day_rows)
            target_sum = original_sum
            if variant["min_day_target"] is not None and 0 < original_sum < float(variant["min_day_target"]):
                target_sum = float(variant["min_day_target"])
            target_sum = min(target_sum, 0.99)
            if original_sum <= 0 or abs(target_sum - original_sum) < 1e-12:
                continue
            multiplier = target_sum / original_sum
            for row in day_rows:
                row["target_pct"] = f"{min((_to_float(row.get('target_pct'), 0.0) or 0.0) * multiplier, max_single_position_pct):.5f}"

    fieldnames = list(base_rows[0].keys())
    for key in ["pred_5d", "pred_10d", "rank_5d", "rank_10d", "turnover_rate", "amount", "total_mv", "pred_gap"]:
        if key not in fieldnames:
            fieldnames.append(key)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


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
    day_sums: dict[str, float] = {}
    for row in rows:
        day_sums[str(row.get("signal_date"))] = day_sums.get(str(row.get("signal_date")), 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
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
        str(float(variant.get("target_position_pct", variant.get("max_single_position_pct", 0.28)))),
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
    target_hits = [
        row
        for row in rows
        if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
    ]
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    best_annual = max(rows, key=lambda row: _metric(row, "annual"))
    report = f"""# 5D10D 归因调仓调参结论

## 结论

本轮基于当前 L5 信号的本地前向收益归因，只使用非行业、非日历字段进行目标仓位缩放：`pred_gap`、`turnover_rate`、`amount` 和 `rank`。候选仍通过掘金回测验证，未修改生产策略。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均持仓 >= 0.80` 的候选数量：{len(target_hits)}
- 是否建议替换当前 L5：{"是" if target_hits else "否"}

## 最优候选

- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，平均持仓 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，平均持仓 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 信号目录：`{REPORT_DIR / "signals"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "5D10D归因调仓调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    base_rows = _load_rows(BASE_SIGNAL)
    keys = {(str(row.get("signal_date")), str(row.get("stock_code"))) for row in base_rows}
    fusion_features = _load_fusion_features(keys)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_variant_signal(variant, signal_file, base_rows, fusion_features)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "rules_json": json.dumps(variant["rules"], ensure_ascii=False, sort_keys=True),
            "normalize_day": variant["normalize_day"],
            "min_day_target": variant["min_day_target"],
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
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    target_hits = [
        row
        for row in results
        if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_target_hits.csv", target_hits)
    _write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
