from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[3]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[5]
REPORT_ROOT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260621"
REPORT_DIR = REPORT_ROOT / "core10d_risk_sharpe_refine_20260622"
SOURCES = {
    "orig": REPORT_ROOT / "core10d_original_signal_score_cut_20260621" / "signals" / "orig_copy.csv",
    "drop": REPORT_ROOT / "core10d_original_signal_score_cut_20260621" / "signals" / "drop_gt0p5_all.csv",
}
TABLE_10D = "stock_predict_data_model_agent_10d_tune_20260620_executable_10d_open_return_d4_l4_fs160_gate2_score_20240604_20260618"


def _variant(
    name: str,
    source: str,
    *,
    sync: int = 0,
    max_daily_sells: int = 1,
    open_score_exit: int = 0,
    score_exit_ratio: float | None = None,
    min_score_exit_hold: int = 2,
    target_scale: float = 1.0,
    holding_days: int = 5,
    max_holding_days: int = 5,
    cash_buffer: float = 0.995,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "source": source,
        "source_file": SOURCES[source],
        "sync": sync,
        "max_daily_sells": max_daily_sells,
        "open_score_exit": open_score_exit,
        "score_exit_ratio": score_exit_ratio,
        "min_score_exit_hold": min_score_exit_hold,
        "target_scale": target_scale,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "cash_buffer": cash_buffer,
        "max_positions": 5,
        "target_position_pct": 0.196 * target_scale,
        "extra_env": extra_env or {},
    }


VARIANTS = [
    _variant("orig_base", "orig"),
    _variant("orig_sync1", "orig", sync=1),
    _variant("orig_eqdd_mid", "orig", extra_env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.10", "GM_EQUITY_DD_HARD_TRIGGER": "0.18", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05", "GM_EQUITY_DD_SOFT_SCALE": "0.85", "GM_EQUITY_DD_HARD_SCALE": "0.60"}),
    _variant("orig_eqdd_loose", "orig", extra_env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.14", "GM_EQUITY_DD_HARD_TRIGGER": "0.24", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.06", "GM_EQUITY_DD_SOFT_SCALE": "0.92", "GM_EQUITY_DD_HARD_SCALE": "0.75"}),
    _variant("orig_eqdd_mid_resize", "orig", extra_env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_RESIZE_EXISTING": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.10", "GM_EQUITY_DD_HARD_TRIGGER": "0.18", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05", "GM_EQUITY_DD_SOFT_SCALE": "0.85", "GM_EQUITY_DD_HARD_SCALE": "0.60"}),
    _variant("orig_index_loose", "orig", extra_env={"GM_INDEX_RISK_EXIT_MODE": "1", "GM_INDEX_RISK_CC_THRESHOLD": "-0.04", "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.04", "GM_INDEX_RISK_BUY_SCALE": "0.85"}),
    _variant("orig_index_mid", "orig", extra_env={"GM_INDEX_RISK_EXIT_MODE": "1", "GM_INDEX_RISK_CC_THRESHOLD": "-0.03", "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.03", "GM_INDEX_RISK_BUY_SCALE": "0.80"}),
    _variant("orig_intraday_stop06", "orig", extra_env={"GM_INTRADAY_RISK_MODE": "1", "GM_STOP_LOSS_PCT": "0.06"}),
    _variant("orig_intraday_stop08", "orig", extra_env={"GM_INTRADAY_RISK_MODE": "1", "GM_STOP_LOSS_PCT": "0.08"}),
    _variant("orig_score_exit095_mh2", "orig", open_score_exit=1, score_exit_ratio=0.95, min_score_exit_hold=2),
    _variant("orig_score_exit100_mh2", "orig", open_score_exit=1, score_exit_ratio=1.00, min_score_exit_hold=2),
    _variant("orig_score_exit105_mh3", "orig", open_score_exit=1, score_exit_ratio=1.05, min_score_exit_hold=3),
    _variant("orig_daydrop090", "orig", open_score_exit=1, extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90"}),
    _variant("orig_daydrop080", "orig", open_score_exit=1, extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.80"}),
    _variant("orig_h4_mh4", "orig", holding_days=4, max_holding_days=4),
    _variant("orig_h6_mh6", "orig", holding_days=6, max_holding_days=6),
    _variant("orig_target095", "orig", target_scale=0.95),
    _variant("orig_target102", "orig", target_scale=1.02),
    _variant("drop_base", "drop"),
    _variant("drop_eqdd_loose", "drop", extra_env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.14", "GM_EQUITY_DD_HARD_TRIGGER": "0.24", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.06", "GM_EQUITY_DD_SOFT_SCALE": "0.92", "GM_EQUITY_DD_HARD_SCALE": "0.75"}),
    _variant("drop_index_loose", "drop", extra_env={"GM_INDEX_RISK_EXIT_MODE": "1", "GM_INDEX_RISK_CC_THRESHOLD": "-0.04", "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.04", "GM_INDEX_RISK_BUY_SCALE": "0.85"}),
    _variant("drop_score_exit095_mh2", "drop", open_score_exit=1, score_exit_ratio=0.95, min_score_exit_hold=2),
    _variant("drop_daydrop090", "drop", open_score_exit=1, extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.90"}),
    _variant("drop_target102", "drop", target_scale=1.02),
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
    rows = _load_rows(variant["source_file"])
    fieldnames = list(rows[0].keys())
    for field in ["score_exit_entry_ratio", "min_holding_days_before_score_exit"]:
        if field not in fieldnames:
            fieldnames.append(field)
    output_rows = []
    for row in rows:
        output = dict(row)
        base_target = _to_float(output.get("target_pct"), 0.196) or 0.196
        output["target_pct"] = f"{base_target * float(variant['target_scale']):.5f}"
        output["holding_days"] = str(int(variant["holding_days"]))
        if variant["score_exit_ratio"] is not None:
            output["score_exit_entry_ratio"] = f"{float(variant['score_exit_ratio']):.5f}"
        output["min_holding_days_before_score_exit"] = str(int(variant["min_score_exit_hold"]))
        output_rows.append(output)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
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
            "GM_OPEN_DAILY_SCORE_EXIT": str(int(variant["open_score_exit"])),
            "GM_MAX_DAILY_SELLS": str(int(variant["max_daily_sells"])),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": str(int(variant["sync"])),
            "GM_CASH_BUFFER": f"{float(variant['cash_buffer']):.5f}",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "none",
        }
    )
    if variant["score_exit_ratio"] is not None:
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(variant["score_exit_ratio"])
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(variant["min_score_exit_hold"]))
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
        f"{float(variant['target_position_pct']):.5f}",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        "2024-06-05 09:00:00",
        "--backtest-end",
        "2026-06-30 15:30:00",
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
        "source": variant["source"],
        "returncode": returncode,
        "sync": variant["sync"],
        "max_daily_sells": variant["max_daily_sells"],
        "open_score_exit": variant["open_score_exit"],
        "score_exit_ratio": variant["score_exit_ratio"],
        "min_score_exit_hold": variant["min_score_exit_hold"],
        "target_scale": variant["target_scale"],
        "holding_days": variant["holding_days"],
        "max_holding_days": variant["max_holding_days"],
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
    best_target = max(rows, key=lambda row: min(metric(row, "annual_return") / 3.0, metric(row, "sharpe") / 4.0, metric(row, "avg_invested_pct") / 0.8))
    target_hits = [
        row
        for row in rows
        if metric(row, "annual_return") >= 3.0 and metric(row, "sharpe") >= 4.0 and metric(row, "avg_invested_pct") >= 0.80
    ]
    report = f"""# 10D formal 高年化候选夏普调参结论

## 结论

本轮使用 `executable_10d_open_return_l4_formal_20260617.json` 对应的 10D formal L4 资产候选信号，目标是在已有高年化/高持仓基础上提升夏普。未使用 1D 模型、行业过滤、月份过滤、日期过滤或外部数据。本轮是 10D 单资产研究，不冒充当前 5D+10D L5 生产策略。

- 候选数量：{len(rows)}
- 达到 300% 年化、4 夏普、80% 平均持仓的候选数量：{len(target_hits)}
- 当前是否建议替换 L5：{"是" if target_hits else "否"}

## 最优候选

- 最高夏普：`{best_sharpe["name"]}`，年化 `{metric(best_sharpe, "annual_return"):.6f}`，夏普 `{metric(best_sharpe, "sharpe"):.6f}`，最大回撤 `{metric(best_sharpe, "max_drawdown"):.6f}`，平均持仓 `{metric(best_sharpe, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{metric(best_annual, "annual_return"):.6f}`，夏普 `{metric(best_annual, "sharpe"):.6f}`，最大回撤 `{metric(best_annual, "max_drawdown"):.6f}`，平均持仓 `{metric(best_annual, "avg_invested_pct"):.6f}`。
- 最接近三目标：`{best_target["name"]}`，年化 `{metric(best_target, "annual_return"):.6f}`，夏普 `{metric(best_target, "sharpe"):.6f}`，最大回撤 `{metric(best_target, "max_drawdown"):.6f}`，平均持仓 `{metric(best_target, "avg_invested_pct"):.6f}`。

## 判断

10D formal 单资产可以把年化和持仓率推到目标附近，但目前新增风控没有把夏普提升到 4。若继续推进，需要进一步寻找能降低波动且不牺牲收益的非日历、非行业横截面质量规则。

## 证据

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 候选信号目录：`{REPORT_DIR / "signals"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "10D_formal高年化候选夏普调参结论.md").write_text(report, encoding="utf-8")


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
