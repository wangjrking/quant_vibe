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
    / "formal_market_breadth_regime_refine_20260622"
)
CORE_SIGNAL = scan.CORE_SIGNAL
TEN_SIGNAL = scan.TEN_SIGNAL
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
    mode: str,
    core_scale: float,
    ten_target: float,
    ten_rank_max: int,
    max_positions: int,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "feature": feature,
        "threshold": threshold,
        "mode": mode,
        "core_scale": core_scale,
        "ten_target": ten_target,
        "ten_rank_max": ten_rank_max,
        "max_positions": max_positions,
        "holding_days": 7,
        "max_holding_days": 10,
        "extra_env": extra_env or {},
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
    for ten_target in [0.08, 0.10, 0.12]:
        for ten_rank in [1, 2]:
            VARIANTS.append(
                _variant(
                    f"core_all_sat_strong_{feature_tag}_{threshold_tag}_t{str(ten_target).replace('.', 'p')}_r{ten_rank}",
                    feature,
                    threshold,
                    "core_all_sat_strong",
                    1.0,
                    ten_target,
                    ten_rank,
                    5 + ten_rank,
                )
            )
    for core_scale in [0.65, 0.80, 1.00]:
        for ten_target in [0.21, 0.23]:
            VARIANTS.append(
                _variant(
                    f"switch_{feature_tag}_{threshold_tag}_c{str(core_scale).replace('.', 'p')}_t{str(ten_target).replace('.', 'p')}",
                    feature,
                    threshold,
                    "switch_strong10d_else_core",
                    core_scale,
                    ten_target,
                    5,
                    5,
                )
            )
    for core_scale in [0.65, 0.80]:
        VARIANTS.append(
            _variant(
                f"weak_scaled_core_sat_{feature_tag}_{threshold_tag}_c{str(core_scale).replace('.', 'p')}",
                feature,
                threshold,
                "weak_scaled_core_sat_strong",
                core_scale,
                0.10,
                2,
                7,
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


def _market_by_signal_date(core_rows: list[dict], ten_rows: list[dict]) -> dict[str, dict]:
    dates = {str(row.get("signal_date") or "") for row in core_rows + ten_rows if row.get("signal_date")}
    return scan._market_features(dates)


def _is_strong(date: str, market: dict[str, dict], variant: dict) -> bool:
    value = (market.get(date) or {}).get(str(variant["feature"]))
    return value is not None and float(value) >= float(variant["threshold"])


def _write_signal(variant: dict, path: Path) -> None:
    core_rows = _load_rows(CORE_SIGNAL)
    ten_rows = _load_rows(TEN_SIGNAL)
    market = _market_by_signal_date(core_rows, ten_rows)
    selected: list[dict] = []
    seen = set()

    if variant["mode"] in {"core_all_sat_strong", "weak_scaled_core_sat_strong"}:
        for row in core_rows:
            signal_date = str(row.get("signal_date") or "")
            strong = _is_strong(signal_date, market, variant)
            scale = 1.0 if variant["mode"] == "core_all_sat_strong" or strong else float(variant["core_scale"])
            out = dict(row)
            out["target_pct"] = f"{(_to_float(out.get('target_pct'), 0.0) or 0.0) * scale:.5f}"
            out["regime_strong"] = "1" if strong else "0"
            out["regime_sleeve"] = "core"
            key = (out.get("signal_date"), out.get("buy_date"), out.get("stock_code"))
            seen.add(key)
            selected.append(out)

    elif variant["mode"] == "switch_strong10d_else_core":
        for row in core_rows:
            signal_date = str(row.get("signal_date") or "")
            if _is_strong(signal_date, market, variant):
                continue
            out = dict(row)
            out["target_pct"] = f"{(_to_float(out.get('target_pct'), 0.0) or 0.0) * float(variant['core_scale']):.5f}"
            out["regime_strong"] = "0"
            out["regime_sleeve"] = "core_weak"
            key = (out.get("signal_date"), out.get("buy_date"), out.get("stock_code"))
            seen.add(key)
            selected.append(out)
    else:
        raise ValueError(f"unsupported mode: {variant['mode']}")

    added_by_day: dict[str, int] = {}
    for row in sorted(ten_rows, key=lambda item: (str(item.get("signal_date")), int(float(item.get("rank") or 999999)), str(item.get("stock_code")))):
        signal_date = str(row.get("signal_date") or "")
        if not _is_strong(signal_date, market, variant):
            continue
        rank = int(float(row.get("rank") or 999999))
        if rank > int(variant["ten_rank_max"]):
            continue
        if added_by_day.get(signal_date, 0) >= int(variant["ten_rank_max"]):
            continue
        key = (row.get("signal_date"), row.get("buy_date"), row.get("stock_code"))
        if key in seen:
            continue
        out = dict(row)
        out["target_pct"] = f"{float(variant['ten_target']):.5f}"
        out["holding_days"] = "5"
        out["score_exit_entry_ratio"] = "1.0"
        out["min_holding_days_before_score_exit"] = "3"
        out["score_continue_entry_ratio"] = "1.02"
        out["regime_strong"] = "1"
        out["regime_sleeve"] = "ten_strong"
        selected.append(out)
        seen.add(key)
        added_by_day[signal_date] = added_by_day.get(signal_date, 0) + 1

    selected.sort(key=lambda item: (str(item.get("signal_date")), int(float(item.get("rank") or 999999)), str(item.get("stock_code"))))
    fieldnames = list(core_rows[0].keys())
    for field in ["regime_strong", "regime_sleeve", "score_continue_entry_ratio"]:
        if field not in fieldnames:
            fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
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


def _signal_stats(path: Path) -> dict:
    rows = _load_rows(path)
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "strong_count": sum(1 for row in rows if row.get("regime_strong") == "1"),
        "ten_count": sum(1 for row in rows if row.get("regime_sleeve") == "ten_strong"),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
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
        "7",
        "--max-holding-days",
        "10",
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
    report = f"""# 市场宽度状态切换调参结论

## 当前结论

本轮只使用信号日已知的市场宽度/指数状态，将当前 L5 5D+10D gap 信号与 formal 10D 高年化信号做状态切换或强市场叠加。未使用 1D、行业限制、月份排除或具体日期排除。

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
    (REPORT_DIR / "市场宽度状态切换调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "feature": variant["feature"],
            "threshold": variant["threshold"],
            "mode": variant["mode"],
            "core_scale": variant["core_scale"],
            "ten_target": variant["ten_target"],
            "ten_rank_max": variant["ten_rank_max"],
            "max_positions": variant["max_positions"],
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
