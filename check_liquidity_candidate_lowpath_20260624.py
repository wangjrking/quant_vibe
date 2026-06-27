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
from datetime import datetime
from pathlib import Path

import pandas as pd

import tune_strict_sync_liquidity_neighborhood_20260624 as neigh
import tune_strict_sync_liquidity_refill_20260624 as liq


CASE_NAME = "liq_amt10w_mv20w_w90_5d10_pos75_h3_e096_mh1"
SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "low_path_check_20260624"
SIGNAL_FILE = SOURCE_DIR / "signals" / f"{CASE_NAME}.csv"
FULL_LOG = SOURCE_DIR / "logs" / f"{CASE_NAME}_full_20240605.log"

ANCHORS = ["20250701", "20251009", "20260105"]
NEARBY_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]


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


def _market_dates() -> list[str]:
    conn = sqlite3.connect(liq.MARKET_DB)
    try:
        dates = [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")]
    finally:
        conn.close()
    return dates


def _nearby_starts() -> list[dict]:
    dates = _market_dates()
    pos = {date: index for index, date in enumerate(dates)}
    rows: list[dict] = []
    for anchor in ANCHORS:
        idx = pos[anchor]
        for offset in NEARBY_OFFSETS:
            j = idx + offset
            if j < 0 or j >= len(dates):
                continue
            date = dates[j]
            rows.append(
                {
                    "anchor": anchor,
                    "offset": offset,
                    "start_date": date,
                    "start_time": f"{date[:4]}-{date[4:6]}-{date[6:]} 09:00:00",
                }
            )
    return rows


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


def _run_start(start: dict) -> dict:
    tag = f"{start['anchor']}_o{start['offset']:+d}_{start['start_date']}".replace("+", "p").replace("-", "m")
    log_file = REPORT_DIR / "logs" / f"{CASE_NAME}_{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = "0.96"
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = "1"
        command = [
            str(liq.JUEJIN_PYTHON),
            str(liq.MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(liq.STRATEGY_DIR),
            "--signal-file",
            str(SIGNAL_FILE),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            "3",
            "--max-holding-days",
            "3",
            "--target-position-pct",
            "0.75",
            "--score-db",
            str(liq.SCORE_DB),
            "--score-table",
            liq.SCORE_TABLE,
            "--market-db",
            str(liq.MARKET_DB),
            "--backtest-start",
            start["start_time"],
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
        **start,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
        **_exposure_stats(log_file),
    }


def _parse_nav(log_file: Path) -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"EXPOSURE (\d{8}) .* nav=([0-9.]+)")
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        rows.append({"date": match.group(1), "nav": float(match.group(2))})
    return rows


def _calendar_days(start: str, end: str) -> int:
    a = datetime.strptime(start, "%Y%m%d")
    b = datetime.strptime(end, "%Y%m%d")
    return max((b - a).days, 1)


def _continuous_slices(starts: list[dict]) -> list[dict]:
    nav_rows = _parse_nav(FULL_LOG)
    if not nav_rows:
        return []
    by_date = {row["date"]: row["nav"] for row in nav_rows}
    dates = [row["date"] for row in nav_rows]
    end_date = dates[-1]
    end_nav = by_date[end_date]
    out: list[dict] = []
    for start in starts:
        candidates = [date for date in dates if date >= start["start_date"]]
        if not candidates:
            continue
        actual_start = candidates[0]
        start_nav = by_date[actual_start]
        annual = (end_nav / start_nav) ** (365.0 / _calendar_days(actual_start, end_date)) - 1.0 if start_nav > 0 else None
        out.append(
            {
                "anchor": start["anchor"],
                "offset": start["offset"],
                "requested_start_date": start["start_date"],
                "actual_nav_start_date": actual_start,
                "end_date": end_date,
                "start_nav": start_nav,
                "end_nav": end_nav,
                "continuous_nav_slice_annual_approx": annual,
                "method": "full_run_exposure_nav_calendar_annualized_approx",
            }
        )
    return out


def _summarize_by_anchor(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for anchor in ANCHORS:
        items = [row for row in rows if row["anchor"] == anchor and row.get("annual") is not None]
        annuals = [float(row["annual"]) for row in items]
        sharpes = [float(row["sharpe"]) for row in items if row.get("sharpe") is not None]
        drawdowns = [float(row["max_drawdown"]) for row in items if row.get("max_drawdown") is not None]
        out.append(
            {
                "anchor": anchor,
                "starts": len(items),
                "annual_min": min(annuals) if annuals else None,
                "annual_median": float(pd.Series(annuals).median()) if annuals else None,
                "annual_max": max(annuals) if annuals else None,
                "sharpe_min": min(sharpes) if sharpes else None,
                "sharpe_median": float(pd.Series(sharpes).median()) if sharpes else None,
                "max_drawdown_max": max(drawdowns) if drawdowns else None,
            }
        )
    return out


def _write_report(rows: list[dict], anchor_summary: list[dict], continuous: list[dict]) -> None:
    lines = [
        "# 流动性候选低路径依赖检查",
        "",
        "## 当前结论",
        "",
        f"检查对象：`{CASE_NAME}`。该版本是宽过滤 `amount>=100000 && total_mv>=200000`、75% 目标仓位、持有 3 日、分数退出 0.96 的候选。",
        "",
        "本报告的空仓启动结果来自掘金回测；连续账户同区间结果来自全周期掘金日志中的 NAV 切片近似，只用于路径依赖对比，不替代正式掘金指标。",
        "",
        "## 近期相邻空仓启动",
        "",
        "| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 夏普最小 | 夏普中位 | 最大回撤最大 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in anchor_summary:
        lines.append(
            f"| {row['anchor']} | {row['starts']} | {float(row['annual_min']):.4f} | "
            f"{float(row['annual_median']):.4f} | {float(row['annual_max']):.4f} | "
            f"{float(row['sharpe_min']):.4f} | {float(row['sharpe_median']):.4f} | "
            f"{float(row['max_drawdown_max']):.4f} |"
        )
    lines.extend([
        "",
        "## 逐日起点明细",
        "",
        "| 锚点 | 偏移 | 起点 | 年化 | 夏普 | 最大回撤 | 开仓 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['anchor']} | {row['offset']} | {row['start_date']} | "
            f"{float(row['annual']):.4f} | {float(row['sharpe']):.4f} | "
            f"{float(row['max_drawdown']):.4f} | {row['open_count']} |"
        )
    if continuous:
        cdf = pd.DataFrame(continuous)
        lines.extend([
            "",
            "## 连续账户 NAV 切片近似",
            "",
            "| 锚点 | 近似年化最小 | 近似年化中位 | 近似年化最大 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for anchor in ANCHORS:
            items = cdf[cdf["anchor"] == anchor]["continuous_nav_slice_annual_approx"].dropna()
            if items.empty:
                continue
            lines.append(f"| {anchor} | {items.min():.4f} | {items.median():.4f} | {items.max():.4f} |")
    lines.extend([
        "",
        "## 证据路径",
        "",
        f"- 空仓启动明细：`{REPORT_DIR / 'nearby_cold_starts.csv'}`",
        f"- 锚点汇总：`{REPORT_DIR / 'nearby_cold_start_summary.csv'}`",
        f"- 连续账户 NAV 切片：`{REPORT_DIR / 'continuous_nav_slices.csv'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        f"- 信号文件：`{SIGNAL_FILE}`",
    ])
    (REPORT_DIR / "low_path_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(SIGNAL_FILE)
    if not FULL_LOG.exists():
        raise FileNotFoundError(FULL_LOG)
    starts = _nearby_starts()
    rows: list[dict] = []
    rows_path = REPORT_DIR / "nearby_cold_starts.csv"
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = {(row["anchor"], str(row["offset"]), row["start_date"]) for row in rows}
    total = len(starts)
    for index, start in enumerate(starts, start=1):
        key = (start["anchor"], str(start["offset"]), start["start_date"])
        if key in existing:
            print(f"[{index}/{total}] reuse {key}", flush=True)
            continue
        row = _run_start(start)
        rows.append(row)
        _write_rows(rows_path, rows)
        print(
            f"[{index}/{total}] {start['anchor']} {start['offset']} {start['start_date']} "
            f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
            flush=True,
        )
    anchor_summary = _summarize_by_anchor(rows)
    continuous = _continuous_slices(starts)
    _write_rows(REPORT_DIR / "nearby_cold_start_summary.csv", anchor_summary)
    _write_rows(REPORT_DIR / "continuous_nav_slices.csv", continuous)
    (REPORT_DIR / "low_path_summary.json").write_text(
        json.dumps({"nearby": rows, "summary": anchor_summary, "continuous_nav_slices": continuous}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(rows, anchor_summary, continuous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
