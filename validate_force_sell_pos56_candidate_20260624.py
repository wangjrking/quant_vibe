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

import tune_strict_sync_liquidity_refill_20260624 as liq


CASE_NAME = "force_sell_liq_amt10w_mv20w_w90_5d10_pos56_h3_e096_mh1"
SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
FORCE_DIR = SOURCE_DIR / "force_sell_position_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "force_sell_pos56_validation_20260624"
SIGNAL_FILE = FORCE_DIR / "signals" / f"{CASE_NAME}.csv"

BASE_ENV_EXTRA = {"GM_FORCE_SELL_MARKET_ORDER": "1"}
HOLDING_DAYS = 3
TARGET_PCT = 0.56
EXIT_RATIO = 0.96
MIN_HOLD = 1

ANCHORS = ["20250701", "20251009", "20260105"]
NEARBY_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
TIME_SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("slice_2024h2", "2024-06-05 09:00:00", "2024-12-31 15:30:00"),
    ("slice_2025h1", "2025-01-02 09:00:00", "2025-06-30 15:30:00"),
    ("slice_2025h2", "2025-07-01 09:00:00", "2025-12-31 15:30:00"),
    ("slice_2026ytd", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
    ("slice_recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("slice_recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
]
MAIN_STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
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


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


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


def _run(signal_file: Path, tag: str, start: str, end: str = liq.BACKTEST_END) -> dict:
    log_file = REPORT_DIR / "logs" / f"{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env.update(BASE_ENV_EXTRA)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(EXIT_RATIO)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(MIN_HOLD)
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
            str(HOLDING_DAYS),
            "--max-holding-days",
            str(HOLDING_DAYS),
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
            end,
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
        "tag": tag,
        "start": start,
        "end": end,
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


def _market_dates() -> list[str]:
    conn = sqlite3.connect(liq.MARKET_DB)
    try:
        return [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")]
    finally:
        conn.close()


def _nearby_starts() -> list[dict]:
    dates = _market_dates()
    pos = {date: index for index, date in enumerate(dates)}
    rows: list[dict] = []
    for anchor in ANCHORS:
        idx = pos[anchor]
        for offset in NEARBY_OFFSETS:
            date = dates[idx + offset]
            rows.append(
                {
                    "anchor": anchor,
                    "offset": offset,
                    "start_date": date,
                    "start_time": f"{date[:4]}-{date[4:6]}-{date[6:]} 09:00:00",
                }
            )
    return rows


def _summarize_nearby(rows: list[dict]) -> list[dict]:
    out = []
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


def _load_open_prices(codes: set[str], dates: set[str]) -> dict[tuple[str, str], float]:
    if not codes or not dates:
        return {}
    conn = sqlite3.connect(liq.MARKET_DB)
    try:
        code_q = ",".join("?" for _ in codes)
        date_q = ",".join("?" for _ in dates)
        rows = conn.execute(
            f"SELECT trade_date, stock_code, open FROM STOCK_DAILY_DATA WHERE stock_code IN ({code_q}) AND trade_date IN ({date_q})",
            [*codes, *dates],
        ).fetchall()
    finally:
        conn.close()
    return {(str(date), str(code)): float(open_price) for date, code, open_price in rows if open_price not in (None, "")}


def _proxy_enriched(rows: list[dict]) -> list[dict]:
    dates = _market_dates()
    pos = {date: index for index, date in enumerate(dates)}
    request_dates: set[str] = set()
    codes: set[str] = set()
    out = [dict(row) for row in rows]
    for row in out:
        buy_date = str(row.get("buy_date") or "")
        code = str(row.get("stock_code") or "")
        if buy_date not in pos:
            continue
        sell_date = dates[min(pos[buy_date] + HOLDING_DAYS, len(dates) - 1)]
        row["_proxy_sell_date"] = sell_date
        request_dates.add(buy_date)
        request_dates.add(sell_date)
        codes.add(code)
    prices = _load_open_prices(codes, request_dates)
    for row in out:
        buy_open = prices.get((str(row.get("buy_date") or ""), str(row.get("stock_code") or "")))
        sell_open = prices.get((str(row.get("_proxy_sell_date") or ""), str(row.get("stock_code") or "")))
        if not buy_open or not sell_open or buy_open <= 0:
            row["proxy_weighted_contribution"] = ""
            continue
        row["proxy_weighted_contribution"] = TARGET_PCT * (sell_open / buy_open - 1.0)
    return out


def _build_stress_variants(rows: list[dict]) -> tuple[list[dict], dict]:
    enriched = _proxy_enriched(rows)
    frame = pd.DataFrame(enriched)
    frame["proxy"] = pd.to_numeric(frame["proxy_weighted_contribution"], errors="coerce").fillna(0.0)
    frame["month"] = frame["signal_date"].astype(str).str[:6]
    pos_proxy = frame["proxy"].clip(lower=0.0)
    stock = frame.assign(pos_proxy=pos_proxy).groupby("stock_code")["pos_proxy"].sum().sort_values(ascending=False)
    day = frame.assign(pos_proxy=pos_proxy).groupby("signal_date")["pos_proxy"].sum().sort_values(ascending=False)
    month = frame.assign(pos_proxy=pos_proxy).groupby("month")["pos_proxy"].sum().sort_values(ascending=False)
    top_stock = str(stock.index[0])
    top_day = str(day.index[0])
    top_month = str(month.index[0])
    top_n = max(1, int(math.ceil(len(frame) * 0.01)))
    top_idx = set(frame.sort_values("proxy", ascending=False).head(top_n).index.tolist())
    variants = [
        {"name": "base", "reason": "原始信号", "rows": rows},
        {"name": "drop_top_stock", "reason": f"去最高代理股票 {top_stock}", "rows": [r for r in rows if str(r.get("stock_code") or "") != top_stock]},
        {"name": "drop_top_day", "reason": f"去最高代理信号日 {top_day}", "rows": [r for r in rows if str(r.get("signal_date") or "") != top_day]},
        {"name": "drop_top_month", "reason": f"去最高代理月份 {top_month}", "rows": [r for r in rows if str(r.get("signal_date") or "")[:6] != top_month]},
        {"name": "drop_top_1pct", "reason": f"去最高 1% 代理信号 {top_n} 条", "rows": [r for i, r in enumerate(rows) if i not in top_idx]},
    ]
    total = float(pos_proxy.sum())
    stats = {
        "top_stock": top_stock,
        "top_day": top_day,
        "top_month": top_month,
        "top_1pct_count": top_n,
        "total_positive_proxy": total,
        "top_stock_share": float(stock.iloc[0]) / total if total else None,
        "top_day_share": float(day.iloc[0]) / total if total else None,
        "top_month_share": float(month.iloc[0]) / total if total else None,
        "variant_reasons": {item["name"]: item["reason"] for item in variants},
    }
    _write_rows(REPORT_DIR / "proxy_signal_contribution.csv", [{k: v for k, v in row.items() if not k.startswith("_")} for row in enriched])
    return variants, stats


def _write_report(time_rows: list[dict], nearby_summary: list[dict], stress_summary: list[dict], proxy_stats: dict, audit: dict) -> None:
    lines = [
        "# 强制卖出 56% 仓位候选验证报告",
        "",
        "## 当前结论",
        "",
        f"验证对象：`{CASE_NAME}`。该候选在强制卖出仓位邻域中是当前更均衡的版本：年化超过 200%，回撤低于 75% 强制卖出版，近期空仓启动明显改善。",
        "",
        "该报告仍是研究候选验证，不代表已经发布生产。",
        "",
        "## 时间切片",
        "",
        "| 切片 | 年化 | 夏普 | 最大回撤 | 开仓 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in time_rows:
        lines.append(f"| {row['slice']} | {_f(row['annual']):.4f} | {_f(row['sharpe']):.4f} | {_f(row['max_drawdown']):.4f} | {row['open_count']} |")
    lines.extend([
        "",
        "## 低路径依赖",
        "",
        "| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in nearby_summary:
        lines.append(
            f"| {row['anchor']} | {row['starts']} | {_f(row['annual_min']):.4f} | {_f(row['annual_median']):.4f} | "
            f"{_f(row['annual_max']):.4f} | {_f(row['max_drawdown_max']):.4f} |"
        )
    lines.extend([
        "",
        "## 贡献集中度",
        "",
        f"- 最高代理股票：`{proxy_stats['top_stock']}`，占正代理贡献 `{_f(proxy_stats['top_stock_share']):.2%}`",
        f"- 最高代理日：`{proxy_stats['top_day']}`，占正代理贡献 `{_f(proxy_stats['top_day_share']):.2%}`",
        f"- 最高代理月份：`{proxy_stats['top_month']}`，占正代理贡献 `{_f(proxy_stats['top_month_share']):.2%}`",
        "",
        "| 压力变体 | 说明 | 年化 | 夏普 | 最大回撤 | 近期开仓最差年化 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in stress_summary:
        lines.append(
            f"| {row['variant']} | {row['reason']} | {_f(row['full_annual']):.4f} | {_f(row['full_sharpe']):.4f} | "
            f"{_f(row['full_max_drawdown']):.4f} | {_f(row['late_min_annual']):.4f} |"
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
        f"- 时间切片：`{REPORT_DIR / 'time_slices.csv'}`",
        f"- 低路径明细：`{REPORT_DIR / 'nearby_cold_starts.csv'}`",
        f"- 贡献压力：`{REPORT_DIR / 'stress_summary.csv'}`",
        f"- 审计：`{REPORT_DIR / 'hard_gate_audit.json'}`",
        f"- 日志：`{REPORT_DIR / 'logs'}`",
        f"- 信号变体：`{REPORT_DIR / 'signals'}`",
    ])
    (REPORT_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _stress_summary(rows: list[dict], stats: dict) -> list[dict]:
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        items = [row for row in rows if row["variant"] == variant]
        full = next((row for row in items if row["start_name"] == "full_20240605"), None)
        late = [_f(row["annual"]) for row in items if row["start_name"].startswith("late_") and row.get("annual") is not None]
        out.append(
            {
                "variant": variant,
                "reason": stats["variant_reasons"].get(variant),
                "full_annual": _f(full.get("annual")) if full else None,
                "full_sharpe": _f(full.get("sharpe")) if full else None,
                "full_max_drawdown": _f(full.get("max_drawdown")) if full else None,
                "late_min_annual": min(late) if late else None,
            }
        )
    return out


def main() -> int:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(SIGNAL_FILE)
    signal_rows = _read_rows(SIGNAL_FILE)

    time_rows = []
    for name, start, end in TIME_SLICES:
        row = _run(SIGNAL_FILE, f"time_{name}", start, end)
        row["slice"] = name
        time_rows.append(row)
        print(f"time {name} annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}", flush=True)
    _write_rows(REPORT_DIR / "time_slices.csv", time_rows)

    nearby_rows = []
    for item in _nearby_starts():
        tag = f"nearby_{item['anchor']}_{item['offset']}_{item['start_date']}".replace("-", "m")
        row = _run(SIGNAL_FILE, tag, item["start_time"], liq.BACKTEST_END)
        row.update(item)
        nearby_rows.append(row)
        print(f"nearby {item['anchor']} {item['offset']} annual={row['annual']} sharpe={row['sharpe']}", flush=True)
    nearby_summary = _summarize_nearby(nearby_rows)
    _write_rows(REPORT_DIR / "nearby_cold_starts.csv", nearby_rows)
    _write_rows(REPORT_DIR / "nearby_summary.csv", nearby_summary)

    variants, proxy_stats = _build_stress_variants(signal_rows)
    stress_rows = []
    signal_files = [SIGNAL_FILE]
    for variant in variants:
        path = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        _write_rows(path, variant["rows"])
        signal_files.append(path)
        for start_name, start in MAIN_STARTS:
            row = _run(path, f"stress_{variant['name']}_{start_name}", start, liq.BACKTEST_END)
            row["variant"] = variant["name"]
            row["start_name"] = start_name
            stress_rows.append(row)
            print(f"stress {variant['name']} {start_name} annual={row['annual']} sharpe={row['sharpe']}", flush=True)
    stress_summary = _stress_summary(stress_rows, proxy_stats)
    _write_rows(REPORT_DIR / "stress_cases.csv", stress_rows)
    _write_rows(REPORT_DIR / "stress_summary.csv", stress_summary)
    (REPORT_DIR / "proxy_summary.json").write_text(json.dumps(proxy_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "validation_summary.json").write_text(
        json.dumps(
            {"time_slices": time_rows, "nearby_summary": nearby_summary, "stress_summary": stress_summary, "proxy": proxy_stats, "audit": audit},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(time_rows, nearby_summary, stress_summary, proxy_stats, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
