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

import research_formal_signal_state_rule_scan_20260622 as scan


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_market_dynamic_target_refine_20260622"
)
SOURCE_SIGNAL = scan.TEN_SIGNAL
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
    feature: str,
    threshold: float,
    strong_target: float,
    weak_target: float,
    holding_days: int = 5,
    max_holding_days: int = 6,
    sync: bool = True,
) -> dict:
    return {
        "name": name,
        "feature": feature,
        "threshold": threshold,
        "strong_target": strong_target,
        "weak_target": weak_target,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "max_positions": 5,
        "extra_env": {"GM_SYNC_POSITIONS": "1"} if sync else {},
    }


VARIANTS = []
FEATURE_THRESHOLDS = [
    ("mkt_up_ratio", 0.48256676557863504),
    ("mkt_up_ratio", 0.5555348142617137),
    ("mkt_up_ratio", 0.6403244058845718),
    ("mkt_avg_pct_chg", 0.5622473636329383),
    ("mkt_avg_pct_chg", 0.868236505167544),
    ("mkt_index_cc", 0.002871307432844583),
    ("mkt_index_cc", 0.005918612645613441),
]
for feature, threshold in FEATURE_THRESHOLDS:
    feature_tag = feature.replace("mkt_", "")
    threshold_tag = str(round(threshold, 6)).replace(".", "p").replace("-", "m")
    for strong_target in [0.23, 0.25, 0.28]:
        for weak_target in [0.14, 0.16, 0.18, 0.196, 0.21]:
            if weak_target > strong_target:
                continue
            for hold, max_hold in [(5, 5), (5, 6), (6, 6)]:
                VARIANTS.append(
                    _variant(
                        f"{feature_tag}_{threshold_tag}_s{str(strong_target).replace('.', 'p')}_w{str(weak_target).replace('.', 'p')}_h{hold}_mh{max_hold}",
                        feature,
                        threshold,
                        strong_target,
                        weak_target,
                        hold,
                        max_hold,
                        sync=True,
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


def _write_signal(variant: dict, path: Path, market: dict[str, dict]) -> None:
    rows = _load_rows(SOURCE_SIGNAL)
    fieldnames = list(rows[0].keys())
    for field in ["regime_strong", "regime_feature", "regime_threshold"]:
        if field not in fieldnames:
            fieldnames.append(field)
    output = []
    for row in rows:
        out = dict(row)
        signal_date = str(out.get("signal_date") or "")
        value = (market.get(signal_date) or {}).get(str(variant["feature"]))
        strong = value is not None and float(value) >= float(variant["threshold"])
        out["target_pct"] = f"{float(variant['strong_target'] if strong else variant['weak_target']):.5f}"
        out["holding_days"] = str(int(variant["holding_days"]))
        out["regime_strong"] = "1" if strong else "0"
        out["regime_feature"] = str(variant["feature"])
        out["regime_threshold"] = str(variant["threshold"])
        output.append(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in output])


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
    rows = _load_rows(path)
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "strong_signal_count": sum(1 for row in rows if row.get("regime_strong") == "1"),
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
        str(float(variant["strong_target"])),
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
    report = f"""# 10D 市场宽度动态仓位调参结论

## 当前结论

本轮固定使用 formal 10D `drop_gt0p5_all` 选股，只按信号日已知市场宽度/指数状态动态调整目标仓位。未使用 1D、行业限制、月份排除或具体日期排除。

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
    (REPORT_DIR / "10D市场宽度动态仓位调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    source_rows = _load_rows(SOURCE_SIGNAL)
    source_dates = {str(row.get("signal_date") or "") for row in source_rows if row.get("signal_date")}
    market = scan._market_features(source_dates)
    selected_variants = list(VARIANTS)
    feature_filter = os.environ.get("FEATURE_FILTER", "").strip()
    if feature_filter:
        allowed_features = {item.strip() for item in feature_filter.split(",") if item.strip()}
        selected_variants = [variant for variant in selected_variants if variant["feature"] in allowed_features]
    strong_filter = os.environ.get("STRONG_TARGET_FILTER", "").strip()
    if strong_filter:
        allowed_strong = {float(item.strip()) for item in strong_filter.split(",") if item.strip()}
        selected_variants = [variant for variant in selected_variants if float(variant["strong_target"]) in allowed_strong]
    weak_filter = os.environ.get("WEAK_TARGET_FILTER", "").strip()
    if weak_filter:
        allowed_weak = {float(item.strip()) for item in weak_filter.split(",") if item.strip()}
        selected_variants = [variant for variant in selected_variants if float(variant["weak_target"]) in allowed_weak]
    hold_filter = os.environ.get("HOLD_FILTER", "").strip()
    if hold_filter:
        allowed_holds = {
            tuple(int(part) for part in item.strip().split(":"))
            for item in hold_filter.split(",")
            if item.strip()
        }
        selected_variants = [
            variant
            for variant in selected_variants
            if (int(variant["holding_days"]), int(variant["max_holding_days"])) in allowed_holds
        ]
    variant_start = int(os.environ.get("VARIANT_START", "0"))
    variant_end_value = os.environ.get("VARIANT_END", "").strip()
    variant_end = int(variant_end_value) if variant_end_value else None
    selected_variants = selected_variants[variant_start:variant_end]
    max_variants = int(os.environ.get("MAX_VARIANTS", str(len(selected_variants))))
    selected_variants = selected_variants[:max_variants]
    for index, variant in enumerate(selected_variants, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_signal(variant, signal_file, market)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "feature": variant["feature"],
            "threshold": variant["threshold"],
            "strong_target": variant["strong_target"],
            "weak_target": variant["weak_target"],
            "holding_days": variant["holding_days"],
            "max_holding_days": variant["max_holding_days"],
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
