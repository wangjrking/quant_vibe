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

import pandas as pd

import tune_strict_sync_liquidity_refill_20260624 as liq


SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "force_sell_position_neighborhood_20260624"
BASE_FILTER = {"amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False}

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

TARGET_PCTS = [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.65, 0.70, 0.75]
HOLDING_DAYS = [3]
EXIT_RATIOS = [0.96]
MIN_HOLD = 1


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
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _case_name(case: dict) -> str:
    return (
        "force_sell_liq_amt10w_mv20w_w90_5d10"
        f"_pos{int(round(float(case['target_pct']) * 100)):02d}"
        f"_h{int(case['holding_days'])}"
        f"_e{int(round(float(case['exit_ratio']) * 100)):03d}"
        f"_mh{int(case['min_hold'])}"
    )


def _iter_cases() -> list[dict]:
    return [
        {"target_pct": target_pct, "holding_days": holding_days, "exit_ratio": exit_ratio, "min_hold": MIN_HOLD}
        for target_pct in TARGET_PCTS
        for holding_days in HOLDING_DAYS
        for exit_ratio in EXIT_RATIOS
    ]


def _entry_score(frame: pd.DataFrame) -> pd.Series:
    return 0.90 * frame["rank_10d"].astype(float) + 0.10 * frame["rank_5d"].astype(float)


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
        rows.append(
            {
                "signal_date": str(signal_date),
                "buy_date": buy_date,
                "symbol": liq.top1.to_gm_symbol(stock_code),
                "stock_code": stock_code,
                "name": chosen.get("name"),
                "rank": 1,
                "pred_prob": liq.top1._tailpow(float(chosen["rank_10d"]), 2.0, upper),
                "entry_score": chosen["entry_score"],
                "pred_5d": chosen.get("pred_5d"),
                "pred_10d": chosen.get("pred_10d"),
                "rank_5d": chosen.get("rank_5d"),
                "rank_10d": chosen.get("rank_10d"),
                "amount": chosen.get("amount"),
                "turnover_rate": chosen.get("turnover_rate"),
                "total_mv": chosen.get("total_mv"),
                "atr_qfq": chosen.get("atr_qfq"),
                "target_pct": f"{float(case['target_pct']):.5f}",
                "holding_days": int(case["holding_days"]),
                "max_holding_days": int(case["holding_days"]),
                "score_exit_entry_ratio": f"{float(case['exit_ratio']):.5f}",
                "min_holding_days_before_score_exit": int(case["min_hold"]),
                "filter_name": "liq_amt10w_mv20w",
                "execution_variant": "force_sell_mkt",
            }
        )
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


def _run_case(case: dict, signal_file: Path, start_name: str, start: str) -> dict:
    name = _case_name(case)
    log_file = REPORT_DIR / "logs" / f"{name}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(case["exit_ratio"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(case["min_hold"]))
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
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["holding_days"])),
            "--target-position-pct",
            str(float(case["target_pct"])),
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
        "target_pct": case["target_pct"],
        "holding_days": case["holding_days"],
        "exit_ratio": case["exit_ratio"],
        "min_hold": case["min_hold"],
        "start_name": start_name,
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


def _signal_stats(rows: list[dict]) -> dict:
    dates = [str(row.get("signal_date") or "") for row in rows if row.get("signal_date")]
    return {"signal_rows": len(rows), "signal_days": len(set(dates)), "latest_signal_date": max(dates) if dates else ""}


def _summarize(rows: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for name in sorted({row["name"] for row in rows}):
        items = [row for row in rows if row["name"] == name]
        full = next((row for row in items if row["start_name"] == "full_20240605"), None)
        if not full or full.get("annual") is None:
            continue
        late = [_f(row["annual"]) for row in items if row["start_name"].startswith("late_") and row.get("annual") is not None]
        late = [value for value in late if not math.isnan(value)]
        annual = _f(full.get("annual"))
        sharpe = _f(full.get("sharpe"))
        maxdd = _f(full.get("max_drawdown"))
        late_min = min(late) if late else None
        late_median = float(pd.Series(late).median()) if late else None
        score = annual + 0.8 * (late_min or 0.0) + 0.5 * (late_median or 0.0) + 0.35 * sharpe - 1.2 * maxdd
        out.append(
            {
                "name": name,
                "target_pct": full["target_pct"],
                "holding_days": full["holding_days"],
                "exit_ratio": full["exit_ratio"],
                "full_annual": annual,
                "full_sharpe": sharpe,
                "full_max_drawdown": maxdd,
                "late_min_annual": late_min,
                "late_median_annual": late_median,
                "full_avg_invested_pct": full.get("avg_invested_pct"),
                "open_count": full.get("open_count"),
                **signal_stats.get(name, {}),
                "score": score,
            }
        )
    return out


def _write_report(summary: list[dict], audit: dict) -> None:
    lines = [
        "# 强制卖出仓位邻域报告",
        "",
        "## 当前结论",
        "",
        "本轮固定宽过滤和 Top1 规则，仅测试 `GM_FORCE_SELL_MARKET_ORDER=1` 下仓位与持有期邻域。",
        "",
        "| 策略 | 仓位 | 持有 | 年化 | 夏普 | 最大回撤 | 近期开仓最差年化 | 近期开仓中位年化 | 开仓 | 综合分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary, key=lambda item: _f(item["score"]), reverse=True):
        lines.append(
            f"| {row['name']} | {float(row['target_pct']):.2f} | {row['holding_days']} | "
            f"{_f(row['full_annual']):.4f} | {_f(row['full_sharpe']):.4f} | {_f(row['full_max_drawdown']):.4f} | "
            f"{_f(row['late_min_annual']):.4f} | {_f(row['late_median_annual']):.4f} | {row['open_count']} | {_f(row['score']):.4f} |"
        )
    lines.extend([
        "",
        "## 硬过滤审计",
        "",
        f"- 审计文件数：`{audit['audited_files']}`",
        f"- 失败文件数：`{audit['failed_files']}`",
        f"- 买入日行情缺失：`{audit['buy_join_missing']}`",
        f"- 北交所：`{audit['bj_rows']}`",
        f"- 买入日 ST / 风险警示：`{audit['buy_st_rows']}`",
        f"- 买入日退市：`{audit['buy_delist_rows']}`",
        f"- 买入日开盘涨停：`{audit['open_limit_up_buy_rows']}`",
        "",
        "## 证据路径",
        "",
        f"- 明细：`{REPORT_DIR / 'cases.csv'}`",
        f"- 汇总：`{REPORT_DIR / 'summary.csv'}`",
        f"- 审计：`{REPORT_DIR / 'force_sell_position_hard_gate_audit.json'}`",
        f"- 日志：`{REPORT_DIR / 'logs'}`",
        f"- 信号：`{REPORT_DIR / 'signals'}`",
    ])
    (REPORT_DIR / "force_sell_position_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = liq._load_base()
    market = liq._market_rows()
    next_date = liq._date_map(base)
    rows: list[dict] = []
    cases_path = REPORT_DIR / "cases.csv"
    if cases_path.exists():
        with cases_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {(row["name"], row["start_name"]) for row in rows}
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    cases = _iter_cases()
    total = len(cases) * len(STARTS)
    done = 0
    for case in cases:
        signal_file, signal_rows = _build_signal(base, market, next_date, case)
        signal_files.append(signal_file)
        name = _case_name(case)
        signal_stats[name] = _signal_stats(signal_rows)
        for start_name, start in STARTS:
            done += 1
            if (name, start_name) in existing:
                print(f"[{done}/{total}] reuse {name} {start_name}", flush=True)
                continue
            row = _run_case(case, signal_file, start_name, start)
            rows.append(row)
            _write_rows(cases_path, rows)
            print(
                f"[{done}/{total}] {name} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']} open={row['open_count']}",
                flush=True,
            )
    summary = _summarize(rows, signal_stats)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_score.csv", sorted(summary, key=lambda row: _f(row["score"]), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "force_sell_position_hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
