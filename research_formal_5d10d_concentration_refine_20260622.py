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
    / "formal_5d10d_concentration_refine_20260622"
)
CORE_SIGNAL = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_best_v20260621"
    / "signals"
    / "backtest_signals_best_noncal_avgpred205_rw2121201919.csv"
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


def _variant(
    name: str,
    rank_targets: dict[int, float],
    max_positions: int,
    min_holding_days_before_score_exit: int = 3,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "rank_targets": rank_targets,
        "rank_max": max(rank_targets),
        "max_positions": max_positions,
        "holding_days": 7,
        "max_holding_days": 10,
        "min_holding_days_before_score_exit": min_holding_days_before_score_exit,
        "extra_env": extra_env or {},
    }


VARIANTS = [
    _variant("top2_5049_mh3", {1: 0.50, 2: 0.49}, 2, 3),
    _variant("top2_6040_mh3", {1: 0.60, 2: 0.40}, 2, 3),
    _variant("top2_5544_mh2", {1: 0.55, 2: 0.44}, 2, 2),
    _variant("top2_5544_defer20", {1: 0.55, 2: 0.44}, 2, 3, {"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "20"}),
    _variant("top3_343332_mh3", {1: 0.34, 2: 0.33, 3: 0.32}, 3, 3),
    _variant("top3_403326_mh3", {1: 0.40, 2: 0.33, 3: 0.26}, 3, 3),
    _variant("top3_453024_mh3", {1: 0.45, 2: 0.30, 3: 0.24}, 3, 3),
    _variant("top3_403326_mh2", {1: 0.40, 2: 0.33, 3: 0.26}, 3, 2),
    _variant("top3_403326_defer20", {1: 0.40, 2: 0.33, 3: 0.26}, 3, 3, {"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "20"}),
    _variant("top4_28252422_mh3", {1: 0.28, 2: 0.25, 3: 0.24, 4: 0.22}, 4, 3),
    _variant("top4_32282415_mh3", {1: 0.32, 2: 0.28, 3: 0.24, 4: 0.15}, 4, 3),
    _variant("top4_28252422_defer20", {1: 0.28, 2: 0.25, 3: 0.24, 4: 0.22}, 4, 3, {"GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1", "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "20"}),
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


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    rows = _load_rows(CORE_SIGNAL)
    output_rows = []
    for row in rows:
        rank = int(float(row.get("rank") or 999999))
        if rank > int(variant["rank_max"]):
            continue
        output = dict(row)
        output["target_pct"] = f"{float(variant['rank_targets'].get(rank, 0.0)):.5f}"
        output["holding_days"] = str(int(variant["holding_days"]))
        output["score_exit_entry_ratio"] = "1.0"
        output["min_holding_days_before_score_exit"] = str(int(variant["min_holding_days_before_score_exit"]))
        output_rows.append(output)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in output_rows])


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
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(int(variant["min_holding_days_before_score_exit"])),
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
        f"{max(variant['rank_targets'].values()):.5f}",
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
    proc = subprocess.run(command, cwd=str(MAIN), env=env, text=True, capture_output=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip())
    return proc.returncode


def _summarize(variant: dict, signal_file: Path, log_file: Path, returncode: int) -> dict:
    indicator = _extract_indicator(log_file) or {}
    if not isinstance(indicator, dict):
        indicator = {"raw_indicator": str(indicator)}
    row = {
        "name": variant["name"],
        "returncode": returncode,
        "rank_targets_json": json.dumps(variant["rank_targets"], ensure_ascii=False, sort_keys=True),
        "rank_max": variant["rank_max"],
        "max_positions": variant["max_positions"],
        "cum_return": indicator.get("pnl_ratio"),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "max_drawdown": indicator.get("max_drawdown"),
        "sharpe": indicator.get("sharpe_ratio", indicator.get("sharp_ratio")),
        "calmar": indicator.get("calmar_ratio"),
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "extra_env_json": json.dumps(variant.get("extra_env", {}), ensure_ascii=False, sort_keys=True),
    }
    row.update(_signal_stats(signal_file))
    row.update(_exposure_stats(log_file))
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: list[dict]) -> None:
    def metric(row: dict, key: str) -> float:
        return _to_float(row.get(key), float("-inf"))

    best_sharpe = max(rows, key=lambda row: metric(row, "sharpe"))
    best_annual = max(rows, key=lambda row: metric(row, "annual_return"))
    best_invested = max(rows, key=lambda row: metric(row, "avg_invested_pct"))
    target_hits = [
        row
        for row in rows
        if metric(row, "annual_return") >= 3.0 and metric(row, "sharpe") >= 4.0 and metric(row, "avg_invested_pct") >= 0.80
    ]
    report = f"""# 集中持仓调参结论

## 结论

本轮只基于 formal L4 的 5D+10D 当前核心信号，测试 Top2、Top3、Top4 集中持仓和单票仓位提高。没有加入行业过滤、月份过滤、日期过滤、外部数据或 1D 模型。

- 候选数量：{len(rows)}
- 达到 300% 年化、4 夏普、80% 平均持仓的候选数量：{len(target_hits)}
- 当前是否建议替换 L5：{"是" if target_hits else "否"}

## 最优候选

- 最高夏普：`{best_sharpe["name"]}`，年化 `{metric(best_sharpe, "annual_return"):.6f}`，夏普 `{metric(best_sharpe, "sharpe"):.6f}`，最大回撤 `{metric(best_sharpe, "max_drawdown"):.6f}`，平均持仓 `{metric(best_sharpe, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{metric(best_annual, "annual_return"):.6f}`，夏普 `{metric(best_annual, "sharpe"):.6f}`，最大回撤 `{metric(best_annual, "max_drawdown"):.6f}`，平均持仓 `{metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高平均持仓：`{best_invested["name"]}`，年化 `{metric(best_invested, "annual_return"):.6f}`，夏普 `{metric(best_invested, "sharpe"):.6f}`，最大回撤 `{metric(best_invested, "max_drawdown"):.6f}`，平均持仓 `{metric(best_invested, "avg_invested_pct"):.6f}`。

## 判断

如果集中持仓不能超过当前 L5 指标，说明当前模型分数在前几名之间的排序优势不足以支撑进一步集中化；集中化主要放大波动和回撤，而不是稳定提高夏普。

## 证据

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 候选信号目录：`{REPORT_DIR / "signals"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "集中持仓调参结论.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_variant_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        rows.append(_summarize(variant, signal_file, log_file, returncode))
        _write_csv(REPORT_DIR / "summary.csv", rows)
    _write_report(rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "variants": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
