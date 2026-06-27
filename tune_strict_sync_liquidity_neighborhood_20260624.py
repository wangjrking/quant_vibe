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
import tune_top1_soft_rank_latest_formal_20260623 as top1


REPORT_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"

FILTERS = [
    {"name": "liq_amt10w_mv20w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_amt10w_mv20w_mv200w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": 2000000, "turnover_min": None, "require_atr": False},
]
TARGET_PCTS = [0.75, 0.90, 0.99]
HOLDING_DAYS = [2, 3]
EXIT_RATIOS = [0.96, 0.97, 0.98]
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
        f"{case['filter_name']}_w90_5d10"
        f"_pos{int(round(float(case['target_pct']) * 100)):02d}"
        f"_h{int(case['holding_days'])}"
        f"_e{int(round(float(case['exit_ratio']) * 100)):03d}"
        f"_mh{int(case['min_hold'])}"
    )


def _iter_cases() -> list[dict]:
    cases: list[dict] = []
    for filt in FILTERS:
        for target_pct in TARGET_PCTS:
            for holding_days in HOLDING_DAYS:
                for exit_ratio in EXIT_RATIOS:
                    item = {
                        **filt,
                        "filter_name": filt["name"],
                        "target_pct": target_pct,
                        "holding_days": holding_days,
                        "exit_ratio": exit_ratio,
                        "min_hold": MIN_HOLD,
                        "gamma": 2.0,
                    }
                    item["name"] = _case_name(item)
                    cases.append(item)
    return cases


def _entry_score(frame: pd.DataFrame) -> pd.Series:
    return 0.90 * frame["rank_10d"].astype(float) + 0.10 * frame["rank_5d"].astype(float)


def _build_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], case: dict) -> tuple[Path, list[dict]]:
    frame = base.loc[liq._filter_mask(base, case)].copy()
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
            if top1._is_st_like(buy_market) or top1._is_limit_buy(buy_market):
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
                "symbol": top1.to_gm_symbol(stock_code),
                "stock_code": stock_code,
                "name": chosen.get("name"),
                "rank": 1,
                "pred_prob": top1._tailpow(float(chosen["rank_10d"]), float(case["gamma"]), upper),
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
                "filter_name": case["filter_name"],
            }
        )
    path = REPORT_DIR / "signals" / f"{case['name']}.csv"
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
    log_file = REPORT_DIR / "logs" / f"{case['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
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
        "name": case["name"],
        "filter_name": case["filter_name"],
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
    buy_dates = [str(row.get("buy_date") or "") for row in rows if row.get("buy_date")]
    return {
        "signal_rows": len(rows),
        "signal_days": len(set(dates)),
        "latest_signal_date": max(dates) if dates else "",
        "latest_buy_date": max(buy_dates) if buy_dates else "",
    }


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
                "filter_name": full["filter_name"],
                "target_pct": full["target_pct"],
                "holding_days": full["holding_days"],
                "exit_ratio": full["exit_ratio"],
                "min_hold": full["min_hold"],
                "full_annual": annual,
                "full_sharpe": sharpe,
                "full_max_drawdown": maxdd,
                "late_min_annual": late_min,
                "late_median_annual": late_median,
                "full_avg_invested_pct": full.get("avg_invested_pct"),
                "full_ge80_ratio": full.get("ge80_ratio"),
                "full_max_active_positions": full.get("max_active_positions"),
                "open_count": full.get("open_count"),
                **signal_stats.get(name, {}),
                "score": score,
            }
        )
    return out


def _write_report(summary: list[dict], audit: dict) -> None:
    lines = [
        "# 严格同步流动性邻域调参报告",
        "",
        "## 当前结论",
        "",
        "本轮围绕两个宽口径过滤做参数邻域：`amount>=100000 && total_mv>=200000`，以及在此基础上增加 `total_mv<=2000000` 的版本。收益仍是弱准入，强准入继续看参数邻域、持续性、低路径依赖和硬过滤审计。",
        "",
        "## 掘金回测汇总",
        "",
        "| 策略 | 年化 | 夏普 | 最大回撤 | 近期开仓最差年化 | 近期开仓中位年化 | 仓位 | 开仓 | 综合分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary, key=lambda item: _f(item["score"]), reverse=True)[:40]:
        lines.append(
            f"| {row['name']} | {_f(row['full_annual']):.4f} | {_f(row['full_sharpe']):.4f} | "
            f"{_f(row['full_max_drawdown']):.4f} | {_f(row['late_min_annual']):.4f} | "
            f"{_f(row['late_median_annual']):.4f} | {_f(row['full_avg_invested_pct']):.4f} | "
            f"{row['open_count']} | {_f(row['score']):.4f} |"
        )
    lines.extend([
        "",
        "## 硬过滤审计",
        "",
        f"- 审计文件数：`{audit['audited_files']}`",
        f"- 信号总行数：`{audit['total_signal_rows']}`",
        f"- 失败文件数：`{audit['failed_files']}`",
        f"- 买入日行情缺失：`{audit['buy_join_missing']}`",
        f"- 北交所：`{audit['bj_rows']}`",
        f"- 信号日 ST 名称：`{audit['signal_st_name_rows']}`",
        f"- 买入日 ST / 风险警示：`{audit['buy_st_rows']}`",
        f"- 信号日退市名称：`{audit['signal_delist_name_rows']}`",
        f"- 买入日退市名称：`{audit['buy_delist_rows']}`",
        f"- 买入日开盘涨停：`{audit['open_limit_up_buy_rows']}`",
        f"- 最新 signal_date：`{audit['latest_signal_date']}`",
        f"- 最新 buy_date：`{audit['latest_buy_date']}`",
        "",
        "## 证据路径",
        "",
        f"- 明细：`{REPORT_DIR / 'cases.csv'}`",
        f"- 汇总：`{REPORT_DIR / 'summary.csv'}`",
        f"- 年化排序：`{REPORT_DIR / 'summary_by_annual.csv'}`",
        f"- 综合分排序：`{REPORT_DIR / 'summary_by_score.csv'}`",
        f"- 硬过滤审计：`{REPORT_DIR / 'liquidity_neighborhood_hard_gate_audit.json'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        f"- 信号文件：`{REPORT_DIR / 'signals'}`",
    ])
    (REPORT_DIR / "liquidity_neighborhood_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not liq.STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"juejin strategy main.py not found: {liq.STRATEGY_DIR}")
    base = liq._load_base()
    market = liq._market_rows()
    next_date = liq._date_map(base)
    cases = _iter_cases()
    rows: list[dict] = []
    cases_path = REPORT_DIR / "cases.csv"
    if cases_path.exists():
        with cases_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {(row["name"], row["start_name"]) for row in rows}
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    total = len(cases) * len(liq.STARTS)
    done = 0
    for case in cases:
        signal_file, signal_rows = _build_signal(base, market, next_date, case)
        signal_files.append(signal_file)
        signal_stats[case["name"]] = _signal_stats(signal_rows)
        for start_name, start in liq.STARTS:
            done += 1
            if (case["name"], start_name) in existing:
                print(f"[{done}/{total}] reuse {case['name']} {start_name}", flush=True)
                continue
            row = _run_case(case, signal_file, start_name, start)
            rows.append(row)
            _write_rows(cases_path, rows)
            print(
                f"[{done}/{total}] {case['name']} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']} active={row['max_active_positions']}",
                flush=True,
            )
    summary = _summarize(rows, signal_stats)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_score.csv", sorted(summary, key=lambda row: _f(row["score"]), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "liquidity_neighborhood_hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
