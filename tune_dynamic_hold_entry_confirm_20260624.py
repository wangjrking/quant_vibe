from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

import tune_formal_horizon_entry_confirmation_20260624 as entry
import tune_strict_sync_liquidity_refill_20260624 as liq


REPORT_DIR = entry.SOURCE_DIR / "formal_horizon_entry_confirmation_20260624" / "dynamic_hold_20260624"
BASE_FILTER = {"amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False}
WEIGHT = {"weight_name": "w78_5d12_3d10", "w10d": 0.78, "w5d": 0.12, "w3d": 0.10, "w1d": 0.00}
TARGET_PCT = 0.56
EXIT_RATIO = 0.97
MIN_SCORE_EXIT_HOLD = 1

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

CASES = [
    {"name": "fixed_h3", "hold": 3, "max_hold": 3, "continue_ratio": None},
    {"name": "h2_m3_c098", "hold": 2, "max_hold": 3, "continue_ratio": 0.98},
    {"name": "h2_m3_c100", "hold": 2, "max_hold": 3, "continue_ratio": 1.00},
    {"name": "h2_m3_c102", "hold": 2, "max_hold": 3, "continue_ratio": 1.02},
    {"name": "h2_m4_c098", "hold": 2, "max_hold": 4, "continue_ratio": 0.98},
    {"name": "h2_m4_c100", "hold": 2, "max_hold": 4, "continue_ratio": 1.00},
    {"name": "h2_m4_c102", "hold": 2, "max_hold": 4, "continue_ratio": 1.02},
    {"name": "h3_m4_c098", "hold": 3, "max_hold": 4, "continue_ratio": 0.98},
    {"name": "h3_m4_c100", "hold": 3, "max_hold": 4, "continue_ratio": 1.00},
    {"name": "h3_m5_c100", "hold": 3, "max_hold": 5, "continue_ratio": 1.00},
]


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


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _case_name(case: dict) -> str:
    return f"dynamic_{case['name']}_w78_5d12_3d10_pos56_e097_mh1"


def _entry_score(frame: pd.DataFrame) -> pd.Series:
    return (
        float(WEIGHT["w10d"]) * frame["rank_10d"].astype(float)
        + float(WEIGHT["w5d"]) * frame["rank_5d"].astype(float)
        + float(WEIGHT["w3d"]) * frame["rank_3d"].astype(float)
        + float(WEIGHT["w1d"]) * frame["rank_1d"].astype(float)
    )


def _build_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], case: dict) -> tuple[Path, list[dict]]:
    frame = base.loc[liq._filter_mask(base, BASE_FILTER)].copy()
    frame["entry_score"] = _entry_score(frame)
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    rows: list[dict] = []
    upper = max(float(base["rank_10d"].max()), 1.0)
    for signal_date, day in frame.groupby("trade_date", sort=True):
        buy_date = next_date.get(str(signal_date))
        if not buy_date:
            continue
        chosen = None
        for item in day.to_dict("records"):
            stock_code = str(item["stock_code"])
            buy_market = market.get(buy_date, {}).get(stock_code)
            if liq.top1._is_st_like(buy_market) or liq.top1._is_limit_buy(buy_market):
                continue
            chosen = item
            break
        if chosen is None:
            continue
        stock_code = str(chosen["stock_code"])
        row = {
            "signal_date": str(signal_date),
            "buy_date": buy_date,
            "symbol": liq.top1.to_gm_symbol(stock_code),
            "stock_code": stock_code,
            "name": chosen.get("name"),
            "rank": 1,
            "pred_prob": liq.top1._tailpow(float(chosen["rank_10d"]), 2.0, upper),
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
            "atr_qfq": chosen.get("atr_qfq"),
            "target_pct": f"{TARGET_PCT:.5f}",
            "holding_days": int(case["hold"]),
            "max_holding_days": int(case["max_hold"]),
            "score_exit_entry_ratio": f"{EXIT_RATIO:.5f}",
            "min_holding_days_before_score_exit": MIN_SCORE_EXIT_HOLD,
            "filter_name": "liq_amt10w_mv20w",
            "entry_weight_name": WEIGHT["weight_name"],
            "dynamic_hold_name": case["name"],
            "score_continue_entry_ratio": "" if case.get("continue_ratio") is None else f"{float(case['continue_ratio']):.5f}",
            "exit_score_basis": "10d_core_score_table",
            "execution_variant": "force_sell_mkt",
        }
        rows.append(row)
    path = REPORT_DIR / "signals" / f"{_case_name(case)}.csv"
    _write_rows(path, rows)
    return path, rows


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


def _run(case: dict, signal_file: Path, start_name: str, start: str) -> dict:
    name = _case_name(case)
    log_file = REPORT_DIR / "logs" / f"{name}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(EXIT_RATIO)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(MIN_SCORE_EXIT_HOLD)
        if case.get("continue_ratio") is None:
            env.pop("GM_SCORE_CONTINUE_ENTRY_RATIO", None)
        else:
            env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(case["continue_ratio"])
        command = [
            str(liq.JUEJIN_PYTHON),
            str(liq.MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(liq.STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            str(int(case["hold"])),
            "--max-holding-days",
            str(int(case["max_hold"])),
            "--target-position-pct",
            str(TARGET_PCT),
            "--score-db",
            str(liq.SCORE_DB),
            "--score-table",
            liq.SCORE_TABLE,
            "--market-db",
            str(liq.MARKET_DB),
            "--backtest-start",
            start,
            "--backtest-end",
            liq.BACKTEST_END,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(liq.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "name": name,
        "hold": case["hold"],
        "max_hold": case["max_hold"],
        "continue_ratio": case.get("continue_ratio"),
        "start_name": start_name,
        "start": start,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        **_exposure_stats(log_file),
    }


def _summarize(rows: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out = []
    for name in sorted({row["name"] for row in rows}):
        items = [row for row in rows if row["name"] == name]
        full = next((row for row in items if row["start_name"] == "full_20240605"), None)
        late = [_f(row["annual"]) for row in items if row["start_name"].startswith("late_")]
        late = [value for value in late if value == value]
        first = items[0]
        stat = signal_stats.get(name, {})
        out.append(
            {
                "name": name,
                "hold": first["hold"],
                "max_hold": first["max_hold"],
                "continue_ratio": first.get("continue_ratio"),
                "full_annual": full.get("annual") if full else None,
                "full_sharpe": full.get("sharpe") if full else None,
                "full_max_drawdown": full.get("max_drawdown") if full else None,
                "late_min_annual": min(late) if late else None,
                "late_median_annual": float(pd.Series(late).median()) if late else None,
                "full_avg_invested_pct": full.get("avg_invested_pct") if full else None,
                "open_count": full.get("open_count") if full else None,
                "signal_rows": stat.get("signal_rows"),
                "signal_days": stat.get("signal_days"),
                "latest_signal_date": stat.get("latest_signal_date"),
            }
        )
    return out


def _write_report(summary: list[dict], audit: dict) -> None:
    ranked = sorted(summary, key=lambda row: (_f(row["late_min_annual"]), _f(row["full_annual"])), reverse=True)
    annual_ranked = sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True)
    lines = [
        "# 动态持有规则验证",
        "",
        "## 当前结论",
        "",
        "本实验固定 formal 10D/5D/3D 入场排序，测试最小检查日、最大持有日和分数延持阈值。规则只依赖信号日和持仓期间可见的 formal 分数，不使用日期、月份、行业过滤。",
        "",
        "## 按近期开仓最差年化排序",
        "",
        "| 候选 | 检查日 | 最大持有 | 延持阈值 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 | 中位年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        cont = "" if row.get("continue_ratio") in (None, "") else f"{float(row['continue_ratio']):.2f}"
        lines.append(
            f"| {row['name']} | {row['hold']} | {row['max_hold']} | {cont} | {_f(row['full_annual']):.2%} | "
            f"{_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | {_f(row['late_min_annual']):.2%} | {_f(row['late_median_annual']):.2%} |"
        )
    lines.extend([
        "",
        "## 按全周期年化排序",
        "",
        "| 候选 | 检查日 | 最大持有 | 延持阈值 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in annual_ranked:
        cont = "" if row.get("continue_ratio") in (None, "") else f"{float(row['continue_ratio']):.2f}"
        lines.append(
            f"| {row['name']} | {row['hold']} | {row['max_hold']} | {cont} | {_f(row['full_annual']):.2%} | "
            f"{_f(row['full_sharpe']):.2f} | {_f(row['full_max_drawdown']):.2%} | {_f(row['late_min_annual']):.2%} |"
        )
    lines.extend([
        "",
        "## 硬过滤审计",
        "",
        f"- 审计文件数：`{audit.get('audited_files')}`",
        f"- 失败文件数：`{audit.get('failed_files')}`",
        f"- 买入日行情缺失：`{audit.get('buy_join_missing')}`",
        f"- 北交所：`{audit.get('bj_rows')}`",
        f"- ST / 风险警示：`{audit.get('buy_st_rows')}`",
        f"- 退市：`{audit.get('buy_delist_rows')}`",
        f"- 开盘涨停：`{audit.get('open_limit_up_buy_rows')}`",
        "",
        "## 证据路径",
        "",
        f"- 明细：`{REPORT_DIR / 'detail.csv'}`",
        f"- 汇总：`{REPORT_DIR / 'summary.csv'}`",
        f"- 硬过滤审计：`{REPORT_DIR / 'hard_gate_audit.json'}`",
        f"- 信号文件：`{REPORT_DIR / 'signals'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
    ])
    (REPORT_DIR / "dynamic_hold_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = liq._load_base()
    market = liq._market_rows()
    next_date = liq._date_map(base)
    detail: list[dict] = []
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    for case in CASES:
        signal_file, rows = _build_signal(base, market, next_date, case)
        signal_files.append(signal_file)
        name = _case_name(case)
        signal_stats[name] = {
            "signal_rows": len(rows),
            "signal_days": len({row["signal_date"] for row in rows}),
            "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
        }
        for start_name, start in STARTS:
            row = _run(case, signal_file, start_name, start)
            detail.append(row)
            print(f"{name} {start_name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}", flush=True)
    summary = _summarize(detail, signal_stats)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_late_min.csv", sorted(summary, key=lambda row: _f(row["late_min_annual"]), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
