from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_current_l5_execution_refine_20260622"
)
SIGNAL_FILE = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_gap_v20260622"
    / "signals"
    / "backtest_signals_gap_t0p03_g1p08_b0p88.csv"
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
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b003-10ffe0295517")
BACKTEST_STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-18 15:30:00"


def _variant(
    name: str,
    holding_days: int = 7,
    max_holding_days: int = 10,
    max_daily_sells: int = 1,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "max_daily_sells": max_daily_sells,
        "max_positions": 5,
        "extra_env": extra_env or {},
    }


VARIANTS = [
    _variant("current_repro"),
    _variant("no_score_exit", extra_env={"GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("score_exit_mh4", extra_env={"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4"}),
    _variant("score_exit_mh5", extra_env={"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "5"}),
    _variant("score_exit_ratio095", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.95"}),
    _variant("score_exit_ratio098", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.98"}),
    _variant("score_exit_ratio102", extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.02"}),
    _variant("max_sells0", max_daily_sells=0),
    _variant("max_sells2", max_daily_sells=2),
    _variant("max_sells3", max_daily_sells=3),
    _variant("hold5_max7", holding_days=5, max_holding_days=7),
    _variant("hold6_max8", holding_days=6, max_holding_days=8),
    _variant("hold8_max12", holding_days=8, max_holding_days=12),
    _variant("hold10_max15", holding_days=10, max_holding_days=15),
    _variant("continue100", extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}),
    _variant("continue101", extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"}),
    _variant("continue105", extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.05"}),
    _variant("sync1", extra_env={"GM_SYNC_POSITIONS": "1"}),
    _variant("sync1_no_score_exit", extra_env={"GM_SYNC_POSITIONS": "1", "GM_OPEN_DAILY_SCORE_EXIT": "0"}),
    _variant("sync1_max_sells0", max_daily_sells=0, extra_env={"GM_SYNC_POSITIONS": "1"}),
    _variant("sync1_max_sells2", max_daily_sells=2, extra_env={"GM_SYNC_POSITIONS": "1"}),
    _variant("no_eqdd", extra_env={"GM_EQUITY_DD_RISK_MODE": "0"}),
    _variant("eqdd_loose", extra_env={"GM_EQUITY_DD_SOFT_TRIGGER": "0.14", "GM_EQUITY_DD_HARD_TRIGGER": "0.24", "GM_EQUITY_DD_SOFT_SCALE": "0.92", "GM_EQUITY_DD_HARD_SCALE": "0.75"}),
    _variant("eqdd_strict", extra_env={"GM_EQUITY_DD_SOFT_TRIGGER": "0.08", "GM_EQUITY_DD_HARD_TRIGGER": "0.14", "GM_EQUITY_DD_SOFT_SCALE": "0.80", "GM_EQUITY_DD_HARD_SCALE": "0.55"}),
    _variant("defer8", extra_env={"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8"}),
    _variant("defer12", extra_env={"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "12"}),
    _variant("defer8_minret02", extra_env={"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "8", "GM_DEFER_EXIT_MIN_POSITION_RETURN": "0.02"}),
    _variant("resize_held", extra_env={"GM_RESIZE_HELD_ON_SIGNAL": "1", "GM_RESIZE_HELD_TARGET_MULT": "1.10", "GM_RESIZE_HELD_MIN_DELTA_PCT": "0.015"}),
    _variant("sync1_resize_held", extra_env={"GM_SYNC_POSITIONS": "1", "GM_RESIZE_HELD_ON_SIGNAL": "1", "GM_RESIZE_HELD_TARGET_MULT": "1.10", "GM_RESIZE_HELD_MIN_DELTA_PCT": "0.015"}),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


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


def _signal_stats(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _run_backtest(variant: dict, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
            "GM_MAX_DAILY_SELLS": str(int(variant["max_daily_sells"])),
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
        str(BACKTEST_STRATEGY_DIR),
        "--signal-file",
        str(SIGNAL_FILE),
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
    report = f"""# 当前 L5 执行参数调参结论

## 当前结论

本轮固定使用当前 L5 已发布信号文件，只改变掘金执行参数，回测截止日对齐到 `2026-06-18`。未修改选股信号，未使用 1D、行业限制、月份排除或日期排除。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均仓位 >= 0.80` 的候选数量：{len(hits)}
- 是否建议替换当前 L5：{"是" if hits else "否"}

## 最优候选

- 综合最接近目标：`{best_objective["name"]}`，年化 `{_metric(best_objective, "annual"):.6f}`，夏普 `{_metric(best_objective, "sharpe"):.6f}`，平均仓位 `{_metric(best_objective, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，平均仓位 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，平均仓位 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据路径

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- GM 日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "当前L5执行参数调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    signal_stats = _signal_stats(SIGNAL_FILE)
    for index, variant in enumerate(VARIANTS, start=1):
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        returncode = _run_backtest(variant, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "holding_days": variant["holding_days"],
            "max_holding_days": variant["max_holding_days"],
            "max_daily_sells": variant["max_daily_sells"],
            "max_positions": variant["max_positions"],
            "extra_env_json": json.dumps(variant["extra_env"], ensure_ascii=False, sort_keys=True),
            "returncode": returncode,
            "signal_file": str(SIGNAL_FILE),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
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
