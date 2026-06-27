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
from pathlib import Path

import pandas as pd

import tune_strict_sync_liquidity_refill_20260624 as liq


CASE_NAME = "liq_amt10w_mv20w_w90_5d10_pos75_h3_e096_mh1"
SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "concentration_stress_20260624"
BASE_SIGNAL_FILE = SOURCE_DIR / "signals" / f"{CASE_NAME}.csv"

STARTS = [
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


def _market_dates() -> list[str]:
    conn = sqlite3.connect(liq.MARKET_DB)
    try:
        return [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date")]
    finally:
        conn.close()


def _load_open_prices(codes: set[str], dates: set[str]) -> dict[tuple[str, str], float]:
    if not codes or not dates:
        return {}
    conn = sqlite3.connect(liq.MARKET_DB)
    try:
        code_q = ",".join("?" for _ in codes)
        date_q = ",".join("?" for _ in dates)
        sql = (
            f"SELECT trade_date, stock_code, open FROM STOCK_DAILY_DATA "
            f"WHERE stock_code IN ({code_q}) AND trade_date IN ({date_q})"
        )
        rows = conn.execute(sql, [*codes, *dates]).fetchall()
    finally:
        conn.close()
    return {(str(date), str(code)): float(open_price) for date, code, open_price in rows if open_price not in (None, "")}


def _attach_proxy_contribution(rows: list[dict]) -> list[dict]:
    market_dates = _market_dates()
    pos = {date: index for index, date in enumerate(market_dates)}
    requested_dates: set[str] = set()
    codes: set[str] = set()
    enriched = [dict(row) for row in rows]
    for row in enriched:
        buy_date = str(row.get("buy_date") or "")
        code = str(row.get("stock_code") or "")
        hold = int(float(row.get("holding_days") or 3))
        if buy_date not in pos or not code:
            continue
        sell_index = min(pos[buy_date] + hold, len(market_dates) - 1)
        sell_date = market_dates[sell_index]
        row["_proxy_sell_date"] = sell_date
        requested_dates.add(buy_date)
        requested_dates.add(sell_date)
        codes.add(code)
    prices = _load_open_prices(codes, requested_dates)
    for row in enriched:
        buy_date = str(row.get("buy_date") or "")
        sell_date = str(row.get("_proxy_sell_date") or "")
        code = str(row.get("stock_code") or "")
        buy_open = prices.get((buy_date, code))
        sell_open = prices.get((sell_date, code))
        target = _f(row.get("target_pct"))
        if not buy_open or not sell_open or buy_open <= 0 or math.isnan(target):
            row["proxy_open_to_holding_return"] = ""
            row["proxy_weighted_contribution"] = ""
            continue
        ret = sell_open / buy_open - 1.0
        row["proxy_open_to_holding_return"] = ret
        row["proxy_weighted_contribution"] = target * ret
    return enriched


def _positive_sum(frame: pd.DataFrame, group_col: str) -> pd.Series:
    data = frame.copy()
    data["positive_proxy"] = data["proxy_weighted_contribution_num"].clip(lower=0.0)
    return data.groupby(group_col)["positive_proxy"].sum().sort_values(ascending=False)


def _build_variants(enriched: list[dict]) -> tuple[list[dict], dict]:
    frame = pd.DataFrame(enriched)
    frame["proxy_weighted_contribution_num"] = pd.to_numeric(frame["proxy_weighted_contribution"], errors="coerce").fillna(0.0)
    frame["signal_month"] = frame["signal_date"].astype(str).str[:6]
    stock_contrib = _positive_sum(frame, "stock_code")
    day_contrib = _positive_sum(frame, "signal_date")
    month_contrib = _positive_sum(frame, "signal_month")
    top_stock = str(stock_contrib.index[0]) if len(stock_contrib) else ""
    top_day = str(day_contrib.index[0]) if len(day_contrib) else ""
    top_month = str(month_contrib.index[0]) if len(month_contrib) else ""
    top_n = max(1, int(math.ceil(len(frame) * 0.01)))
    top_idx = set(frame.sort_values("proxy_weighted_contribution_num", ascending=False).head(top_n).index.tolist())

    variants = [
        {"name": "base", "reason": "原始候选信号", "rows": enriched},
        {
            "name": "drop_top_proxy_stock",
            "reason": f"剔除代理正贡献最高股票 {top_stock}",
            "rows": [row for row in enriched if str(row.get("stock_code") or "") != top_stock],
        },
        {
            "name": "drop_top_proxy_day",
            "reason": f"剔除代理正贡献最高信号日 {top_day}",
            "rows": [row for row in enriched if str(row.get("signal_date") or "") != top_day],
        },
        {
            "name": "drop_top_proxy_month",
            "reason": f"剔除代理正贡献最高月份 {top_month}",
            "rows": [row for row in enriched if str(row.get("signal_date") or "")[:6] != top_month],
        },
        {
            "name": "drop_top_1pct_proxy_trades",
            "reason": f"剔除代理贡献最高 1% 信号，共 {top_n} 条",
            "rows": [row for index, row in enumerate(enriched) if index not in top_idx],
        },
    ]
    stats = {
        "method": "proxy_open_to_holding_days_open_not_juejin_trade_pnl",
        "top_stock": top_stock,
        "top_stock_positive_proxy": float(stock_contrib.iloc[0]) if len(stock_contrib) else None,
        "top_day": top_day,
        "top_day_positive_proxy": float(day_contrib.iloc[0]) if len(day_contrib) else None,
        "top_month": top_month,
        "top_month_positive_proxy": float(month_contrib.iloc[0]) if len(month_contrib) else None,
        "top_1pct_count": top_n,
        "total_positive_proxy": float(frame["proxy_weighted_contribution_num"].clip(lower=0.0).sum()),
        "variant_reasons": {item["name"]: item["reason"] for item in variants},
    }
    if stats["total_positive_proxy"]:
        stats["top_stock_positive_share"] = stats["top_stock_positive_proxy"] / stats["total_positive_proxy"]
        stats["top_day_positive_share"] = stats["top_day_positive_proxy"] / stats["total_positive_proxy"]
        stats["top_month_positive_share"] = stats["top_month_positive_proxy"] / stats["total_positive_proxy"]
    return variants, stats


def _public_signal_rows(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if not key.startswith("_") and not key.endswith("_num")} for row in rows]


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


def _run_variant(variant_name: str, signal_file: Path, start_name: str, start: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{variant_name}_{start_name}.log"
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
            str(signal_file),
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
        "variant": variant_name,
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


def _summarize(rows: list[dict], variant_stats: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for variant in sorted({row["variant"] for row in rows}):
        items = [row for row in rows if row["variant"] == variant]
        full = next((row for row in items if row["start_name"] == "full_20240605"), None)
        if not full or full.get("annual") is None:
            continue
        late = [_f(row["annual"]) for row in items if row["start_name"].startswith("late_") and row.get("annual") is not None]
        late = [value for value in late if not math.isnan(value)]
        out.append(
            {
                "variant": variant,
                "reason": variant_stats.get(variant, {}).get("reason"),
                "signal_rows": variant_stats.get(variant, {}).get("signal_rows"),
                "full_annual": _f(full.get("annual")),
                "full_sharpe": _f(full.get("sharpe")),
                "full_max_drawdown": _f(full.get("max_drawdown")),
                "late_min_annual": min(late) if late else None,
                "late_median_annual": float(pd.Series(late).median()) if late else None,
                "full_avg_invested_pct": full.get("avg_invested_pct"),
                "open_count": full.get("open_count"),
            }
        )
    return out


def _write_report(summary: list[dict], proxy_stats: dict, audit: dict) -> None:
    lines = [
        "# 流动性候选贡献集中度压力测试",
        "",
        "## 当前结论",
        "",
        f"测试对象：`{CASE_NAME}`。本轮不改交易规则，只对信号做去极值压力测试，并用掘金复跑。",
        "",
        "代理贡献只用于识别可能的集中来源，不作为正式收益结论；正式压力测试收益以掘金结果为准。",
        "",
        "## 代理贡献集中度",
        "",
        f"- 代理方法：`{proxy_stats['method']}`",
        f"- 代理正贡献最高股票：`{proxy_stats['top_stock']}`，占正贡献约 `{proxy_stats.get('top_stock_positive_share', 0):.2%}`",
        f"- 代理正贡献最高信号日：`{proxy_stats['top_day']}`，占正贡献约 `{proxy_stats.get('top_day_positive_share', 0):.2%}`",
        f"- 代理正贡献最高月份：`{proxy_stats['top_month']}`，占正贡献约 `{proxy_stats.get('top_month_positive_share', 0):.2%}`",
        f"- Top 1% 信号数量：`{proxy_stats['top_1pct_count']}`",
        "",
        "## 掘金压力测试",
        "",
        "| 变体 | 说明 | 信号数 | 全周期年化 | 夏普 | 最大回撤 | 近期开仓最差年化 | 近期开仓中位年化 | 开仓 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary, key=lambda item: _f(item["full_annual"]), reverse=True):
        lines.append(
            f"| {row['variant']} | {row['reason']} | {row['signal_rows']} | "
            f"{_f(row['full_annual']):.4f} | {_f(row['full_sharpe']):.4f} | {_f(row['full_max_drawdown']):.4f} | "
            f"{_f(row['late_min_annual']):.4f} | {_f(row['late_median_annual']):.4f} | {row['open_count']} |"
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
        "",
        "## 证据路径",
        "",
        f"- 代理贡献明细：`{REPORT_DIR / 'proxy_signal_contribution.csv'}`",
        f"- 掘金明细：`{REPORT_DIR / 'cases.csv'}`",
        f"- 汇总：`{REPORT_DIR / 'summary.csv'}`",
        f"- 硬过滤审计：`{REPORT_DIR / 'concentration_hard_gate_audit.json'}`",
        f"- 信号变体：`{REPORT_DIR / 'signals'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
    ])
    (REPORT_DIR / "concentration_stress_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not BASE_SIGNAL_FILE.exists():
        raise FileNotFoundError(BASE_SIGNAL_FILE)
    base_rows = _read_rows(BASE_SIGNAL_FILE)
    enriched = _attach_proxy_contribution(base_rows)
    _write_rows(REPORT_DIR / "proxy_signal_contribution.csv", _public_signal_rows(enriched))
    variants, proxy_stats = _build_variants(enriched)
    (REPORT_DIR / "proxy_concentration_summary.json").write_text(json.dumps(proxy_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    variant_stats: dict[str, dict] = {}
    signal_files: list[Path] = []
    for variant in variants:
        path = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        rows = _public_signal_rows(variant["rows"])
        _write_rows(path, rows)
        signal_files.append(path)
        variant_stats[variant["name"]] = {"reason": variant["reason"], "signal_rows": len(rows)}

    rows: list[dict] = []
    cases_path = REPORT_DIR / "cases.csv"
    if cases_path.exists():
        rows = _read_rows(cases_path)
    existing = {(row["variant"], row["start_name"]) for row in rows}
    total = len(variants) * len(STARTS)
    done = 0
    for variant in variants:
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        for start_name, start in STARTS:
            done += 1
            key = (variant["name"], start_name)
            if key in existing:
                print(f"[{done}/{total}] reuse {key}", flush=True)
                continue
            row = _run_variant(variant["name"], signal_file, start_name, start)
            rows.append(row)
            _write_rows(cases_path, rows)
            print(
                f"[{done}/{total}] {variant['name']} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )

    summary = _summarize(rows, variant_stats)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda item: _f(item["full_annual"]), reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "concentration_hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps({"proxy": proxy_stats, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, proxy_stats, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
