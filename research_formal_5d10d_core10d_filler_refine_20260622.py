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
    / "formal_5d10d_core10d_filler_refine_20260622"
)
CORE_SIGNAL = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_gap_v20260622"
    / "signals"
    / "backtest_signals_gap_t0p03_g1p08_b0p88.csv"
)
FILLER_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "core10d_original_signal_score_cut_20260621"
    / "signals"
    / "drop_gt0p5_all.csv"
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
BACKTEST_END = "2026-06-18 15:30:00"


def _variant(
    name: str,
    fill_mode: str,
    filler_target: float,
    max_fillers: int,
    max_positions: int = 5,
    filter_rule: dict | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "fill_mode": fill_mode,
        "filler_target": filler_target,
        "max_fillers": max_fillers,
        "max_positions": max_positions,
        "filter_rule": filter_rule or {},
        "holding_days": 7,
        "max_holding_days": 10,
        "extra_env": extra_env or {},
    }


VARIANTS = []
for mode in ["missing_core_day", "core_count_lt3", "core_count_lt5"]:
    for target in [0.05, 0.08, 0.10, 0.12]:
        for max_fillers in [1, 2, 3]:
            VARIANTS.append(_variant(f"{mode}_t{str(target).replace('.', 'p')}_n{max_fillers}", mode, target, max_fillers))

for target in [0.05, 0.08, 0.10]:
    VARIANTS.append(
        _variant(
            f"missing_core_day_t{str(target).replace('.', 'p')}_n2_defer8_pos",
            "missing_core_day",
            target,
            2,
            extra_env={
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1",
                "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8",
                "GM_DEFER_EXIT_MIN_POSITION_RETURN": "0.02",
            },
        )
    )

for mode in ["core_count_lt3", "core_count_lt5"]:
    for target in [0.08, 0.10, 0.12, 0.15]:
        for max_fillers in [1, 2, 3]:
            VARIANTS.append(
                _variant(
                    f"filtered_amt50_mv200_{mode}_t{str(target).replace('.', 'p')}_n{max_fillers}",
                    mode,
                    target,
                    max_fillers,
                    max_positions=6,
                    filter_rule={"amount_min": 50000.0, "total_mv_max": 200000.0},
                )
            )
            VARIANTS.append(
                _variant(
                    f"filtered_amt50_mv200_gap03_{mode}_t{str(target).replace('.', 'p')}_n{max_fillers}",
                    mode,
                    target,
                    max_fillers,
                    max_positions=6,
                    filter_rule={"amount_min": 50000.0, "total_mv_max": 200000.0, "pred_gap_max": 0.03},
                )
            )


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


def _stock_code(row: dict) -> str:
    stock_code = str(row.get("stock_code") or "")
    if stock_code:
        return stock_code
    symbol = str(row.get("symbol") or "")
    if symbol.startswith("SZSE."):
        return symbol[5:] + ".SZ"
    if symbol.startswith("SHSE."):
        return symbol[5:] + ".SH"
    return symbol


def _load_fusion_features(rows: list[dict]) -> dict[tuple[str, str], dict]:
    keys = {(str(row.get("signal_date") or ""), _stock_code(row)) for row in rows}
    dates = sorted({date for date, _stock in keys if date})
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    features = {}
    try:
        for start in range(0, len(dates), 100):
            chunk = dates[start : start + 100]
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_5d, pred_10d, amount,
                       turnover_rate, total_mv, atr_qfq, close
                FROM fusion_rank_base
                WHERE trade_date IN ({placeholders})
                """,
                chunk,
            ):
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
                    "amount": _to_float(row["amount"]),
                    "turnover_rate": _to_float(row["turnover_rate"]),
                    "total_mv": _to_float(row["total_mv"]),
                    "atr_ratio": atr / close if atr is not None and close and close > 0 else None,
                    "pred_gap": abs(pred_10d - pred_5d) if pred_10d is not None and pred_5d is not None else None,
                }
    finally:
        conn.close()
    return features


def _passes_filter(row: dict, features: dict, filter_rule: dict) -> bool:
    if not filter_rule:
        return True
    merged = dict(row)
    merged.update(features)
    checks = [
        ("amount_min", "amount", lambda value, limit: value >= limit),
        ("amount_max", "amount", lambda value, limit: value <= limit),
        ("total_mv_min", "total_mv", lambda value, limit: value >= limit),
        ("total_mv_max", "total_mv", lambda value, limit: value <= limit),
        ("turnover_rate_min", "turnover_rate", lambda value, limit: value >= limit),
        ("turnover_rate_max", "turnover_rate", lambda value, limit: value <= limit),
        ("pred_gap_max", "pred_gap", lambda value, limit: value <= limit),
        ("atr_ratio_max", "atr_ratio", lambda value, limit: value <= limit),
    ]
    for rule_key, field, predicate in checks:
        if rule_key not in filter_rule:
            continue
        value = _to_float(merged.get(field))
        if value is None or not predicate(value, float(filter_rule[rule_key])):
            return False
    return True


def _core_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        signal_date = str(row.get("signal_date") or "")
        counts[signal_date] = counts.get(signal_date, 0) + 1
    return counts


def _include_filler(fill_mode: str, core_count: int) -> bool:
    if fill_mode == "missing_core_day":
        return core_count == 0
    if fill_mode == "core_count_lt3":
        return core_count < 3
    if fill_mode == "core_count_lt5":
        return core_count < 5
    raise ValueError(f"unsupported fill_mode: {fill_mode}")


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    core_rows = _load_rows(CORE_SIGNAL)
    filler_rows = _load_rows(FILLER_SIGNAL)
    filler_features = _load_fusion_features(filler_rows)
    counts = _core_counts(core_rows)
    selected: list[dict] = []
    seen = set()
    for row in core_rows:
        key = (row.get("signal_date"), row.get("buy_date"), row.get("stock_code"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(row))

    added_by_day: dict[str, int] = {}
    for row in sorted(filler_rows, key=lambda item: (str(item.get("signal_date")), int(float(item.get("rank") or 999999)), str(item.get("stock_code")))):
        signal_date = str(row.get("signal_date") or "")
        if not _include_filler(str(variant["fill_mode"]), counts.get(signal_date, 0)):
            continue
        if added_by_day.get(signal_date, 0) >= int(variant["max_fillers"]):
            continue
        key = (row.get("signal_date"), row.get("buy_date"), row.get("stock_code"))
        if key in seen:
            continue
        feature_key = (signal_date, _stock_code(row))
        if not _passes_filter(row, filler_features.get(feature_key, {}), variant.get("filter_rule") or {}):
            continue
        out = dict(row)
        out.update({key: value for key, value in filler_features.get(feature_key, {}).items() if key not in out})
        out["target_pct"] = f"{float(variant['filler_target']):.5f}"
        out["holding_days"] = str(int(variant["holding_days"]))
        out["score_exit_entry_ratio"] = "1.0"
        out["min_holding_days_before_score_exit"] = "3"
        out["score_continue_entry_ratio"] = "1.02"
        out["filler_source"] = "core10d_drop_gt0p5"
        seen.add(key)
        added_by_day[signal_date] = added_by_day.get(signal_date, 0) + 1
        selected.append(out)

    selected.sort(key=lambda item: (str(item.get("signal_date")), int(float(item.get("rank") or 999999)), str(item.get("stock_code"))))
    fieldnames = list(core_rows[0].keys())
    for field in ["filler_source", "score_continue_entry_ratio"]:
        if field not in fieldnames:
            fieldnames.append(field)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in selected])


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
        "filler_count": sum(1 for row in rows if row.get("filler_source")),
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
    fieldnames: list[str] = []
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
        key=lambda row: min(
            _metric(row, "annual") / 3.0,
            _metric(row, "sharpe") / 4.0,
            _metric(row, "avg_invested_pct") / 0.80,
        ),
    )
    best_annual = max(rows, key=lambda row: _metric(row, "annual"))
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    report = f"""# 5D10D 核心策略叠加 10D 补位调参结论

## 当前结论

本轮使用当前 L5 的 5D+10D gap 策略作为核心，仅在核心无信号或信号较少时加入 formal 10D 高年化候选补位。未使用 1D、行业限制、月份排除或日期排除，回测截止日对齐到 `2026-06-18`。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均仓位 >= 0.80` 的候选数量：{len(hits)}
- 是否建议替换当前 L5：{"是" if hits else "否"}

## 最优候选

- 综合最接近目标：`{best_objective["name"]}`，年化 `{_metric(best_objective, "annual"):.6f}`，夏普 `{_metric(best_objective, "sharpe"):.6f}`，平均仓位 `{_metric(best_objective, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，平均仓位 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，平均仓位 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据路径

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 信号目录：`{REPORT_DIR / "signals"}`
- GM 日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "核心策略叠加10D补位调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    selected_variants = list(VARIANTS)
    name_filter = os.environ.get("NAME_FILTER", "").strip()
    if name_filter:
        selected_variants = [variant for variant in selected_variants if name_filter in str(variant["name"])]
    max_variants = int(os.environ.get("MAX_VARIANTS", str(len(selected_variants))))
    selected_variants = selected_variants[:max_variants]
    for index, variant in enumerate(selected_variants, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_variant_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "fill_mode": variant["fill_mode"],
            "filler_target": variant["filler_target"],
            "max_fillers": variant["max_fillers"],
            "max_positions": variant["max_positions"],
            "filter_rule": json.dumps(variant.get("filter_rule") or {}, ensure_ascii=False, sort_keys=True),
            "extra_env_json": json.dumps(variant["extra_env"], ensure_ascii=False, sort_keys=True),
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
            f"[{index}/{len(selected_variants)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
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
