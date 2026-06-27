from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

import tune_strict_sync_liquidity_refill_20260624 as liq


SOURCE_DIR = liq.SOURCE_DIR / "strict_sync_liquidity_neighborhood_20260624"
REPORT_DIR = SOURCE_DIR / "formal_horizon_entry_confirmation_20260624" / "dynamic_h2_m3_entryscore_exit_20260624"
SCORE_TABLE = "score_dynamic_entry_w78_5d12_3d10_20260624"
BASE_FILTER = {"amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False}

TARGET_PCT = 0.56
HOLDING_DAYS = 2
MAX_HOLDING_DAYS = 3
MIN_HOLD = 1
CONTINUE_RATIO = 0.98
WEIGHT = {"w10d": 0.78, "w5d": 0.12, "w3d": 0.10, "w1d": 0.00}
EXIT_RATIOS = [0.95, 0.96, 0.97, 0.98]

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00", liq.BACKTEST_END),
    ("recent60_20260324", "2026-03-24 09:00:00", liq.BACKTEST_END),
    ("late_20250701", "2025-07-01 09:00:00", liq.BACKTEST_END),
    ("late_20251009", "2025-10-09 09:00:00", liq.BACKTEST_END),
    ("late_20260105", "2026-01-05 09:00:00", liq.BACKTEST_END),
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


def _entry_score(frame: pd.DataFrame) -> pd.Series:
    return (
        WEIGHT["w10d"] * frame["rank_10d"].astype(float)
        + WEIGHT["w5d"] * frame["rank_5d"].astype(float)
        + WEIGHT["w3d"] * frame["rank_3d"].astype(float)
        + WEIGHT["w1d"] * frame["rank_1d"].astype(float)
    )


def _score_prob(series: pd.Series) -> pd.Series:
    upper = max(float(series.max()), 1.0)
    return (series.astype(float) / upper).clip(lower=0.0) ** 2.0


def _ensure_score_table(base: pd.DataFrame) -> None:
    frame = base[["trade_date", "stock_code"]].copy()
    frame["entry_score"] = _entry_score(base)
    frame["pred_prob"] = _score_prob(frame["entry_score"])
    rows = list(frame[["trade_date", "stock_code", "pred_prob"]].itertuples(index=False, name=None))
    conn = sqlite3.connect(liq.SCORE_DB)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {SCORE_TABLE}")
        conn.execute(f"CREATE TABLE {SCORE_TABLE} (trade_date TEXT, stock_code TEXT, pred_prob REAL)")
        conn.executemany(f"INSERT INTO {SCORE_TABLE} (trade_date, stock_code, pred_prob) VALUES (?, ?, ?)", rows)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{SCORE_TABLE}_key ON {SCORE_TABLE}(trade_date, stock_code)")
        conn.commit()
    finally:
        conn.close()


def _case_name(exit_ratio: float) -> str:
    return f"entryscore_exit_h2m3c098_e{int(round(exit_ratio * 100)):03d}"


def _build_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], exit_ratio: float) -> tuple[Path, list[dict]]:
    frame = base.loc[liq._filter_mask(base, BASE_FILTER)].copy()
    frame["entry_score"] = _entry_score(frame)
    frame["entry_prob"] = _score_prob(frame["entry_score"])
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    rows: list[dict] = []
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
                "pred_prob": chosen["entry_prob"],
                "entry_score": chosen["entry_score"],
                "rank_1d": chosen.get("rank_1d"),
                "rank_3d": chosen.get("rank_3d"),
                "rank_5d": chosen.get("rank_5d"),
                "rank_10d": chosen.get("rank_10d"),
                "amount": chosen.get("amount"),
                "turnover_rate": chosen.get("turnover_rate"),
                "total_mv": chosen.get("total_mv"),
                "target_pct": f"{TARGET_PCT:.5f}",
                "holding_days": HOLDING_DAYS,
                "max_holding_days": MAX_HOLDING_DAYS,
                "score_exit_entry_ratio": f"{exit_ratio:.5f}",
                "score_continue_entry_ratio": f"{CONTINUE_RATIO:.5f}",
                "min_holding_days_before_score_exit": MIN_HOLD,
                "exit_score_basis": SCORE_TABLE,
            }
        )
    path = REPORT_DIR / "signals" / f"{_case_name(exit_ratio)}.csv"
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


def _run(exit_ratio: float, signal_file: Path, start_name: str, start: str, end: str) -> dict:
    name = _case_name(exit_ratio)
    log_file = REPORT_DIR / "logs" / f"{name}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(liq.BASE_ENV)
        env["GM_FORCE_SELL_MARKET_ORDER"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(exit_ratio)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(MIN_HOLD)
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(CONTINUE_RATIO)
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
            str(MAX_HOLDING_DAYS),
            "--target-position-pct",
            str(TARGET_PCT),
            "--score-db",
            str(liq.SCORE_DB),
            "--score-table",
            SCORE_TABLE,
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
        "name": name,
        "exit_ratio": exit_ratio,
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
        **_exposure_stats(log_file),
    }


def _summarize(detail: list[dict], signal_stats: dict[str, dict]) -> list[dict]:
    out = []
    for name in sorted({row["name"] for row in detail}):
        rows = [row for row in detail if row["name"] == name]
        by_start = {row["start_name"]: row for row in rows}
        late = [_f(by_start[key]["annual"]) for key in ["late_20250701", "late_20251009", "late_20260105"] if key in by_start]
        late = [value for value in late if value == value]
        first = rows[0]
        out.append(
            {
                "name": name,
                "exit_ratio": first["exit_ratio"],
                "full_annual": by_start.get("full_20240605", {}).get("annual"),
                "full_sharpe": by_start.get("full_20240605", {}).get("sharpe"),
                "full_max_drawdown": by_start.get("full_20240605", {}).get("max_drawdown"),
                "recent60_annual": by_start.get("recent60_20260324", {}).get("annual"),
                "recent60_sharpe": by_start.get("recent60_20260324", {}).get("sharpe"),
                "recent60_max_drawdown": by_start.get("recent60_20260324", {}).get("max_drawdown"),
                "late_min_annual": min(late) if late else None,
                "late_median_annual": float(pd.Series(late).median()) if late else None,
                **signal_stats.get(name, {}),
            }
        )
    return out


def _write_report(summary: list[dict], audit: dict) -> None:
    ranked = sorted(summary, key=lambda row: (_f(row["full_annual"]), _f(row["recent60_annual"])), reverse=True)
    lines = [
        "# dynamic h2_m3 入场分数一致退出实验",
        "",
        "## 当前结论",
        "",
        "本实验将入场混合分数同步写入 score table，使分数退出和延持使用同一套 `0.78*10D + 0.12*5D + 0.10*3D` 口径。目标是判断入场/退出分数不一致是否导致 recent60 弱或尾部压力。",
        "",
        "| 候选 | 退出比例 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | 近期开仓最差年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['name']} | {float(row['exit_ratio']):.2f} | {_f(row['full_annual']):.2%} | {_f(row['full_sharpe']):.2f} | "
            f"{_f(row['full_max_drawdown']):.2%} | {_f(row['recent60_annual']):.2%} | {_f(row['recent60_sharpe']):.2f} | {_f(row['late_min_annual']):.2%} |"
        )
    lines.extend([
        "",
        "## 硬过滤审计",
        "",
        f"- 审计文件数：`{audit.get('audited_files')}`",
        f"- 失败文件数：`{audit.get('failed_files')}`",
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
        f"- 分数表：`{liq.SCORE_DB}::{SCORE_TABLE}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
    ])
    (REPORT_DIR / "entryscore_exit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = liq._load_base()
    _ensure_score_table(base)
    market = liq._market_rows()
    next_date = liq._date_map(base)
    detail: list[dict] = []
    signal_files: list[Path] = []
    signal_stats: dict[str, dict] = {}
    for exit_ratio in EXIT_RATIOS:
        signal_file, rows = _build_signal(base, market, next_date, exit_ratio)
        signal_files.append(signal_file)
        name = _case_name(exit_ratio)
        signal_stats[name] = {
            "signal_rows": len(rows),
            "signal_days": len({row["signal_date"] for row in rows}),
            "latest_signal_date": max((row["signal_date"] for row in rows), default=None),
        }
        for start_name, start, end in [
            ("full_20240605", "2024-06-05 09:00:00", liq.BACKTEST_END),
            ("recent60_20260324", "2026-03-24 09:00:00", liq.BACKTEST_END),
            ("late_20250701", "2025-07-01 09:00:00", liq.BACKTEST_END),
            ("late_20251009", "2025-10-09 09:00:00", liq.BACKTEST_END),
            ("late_20260105", "2026-01-05 09:00:00", liq.BACKTEST_END),
        ]:
            row = _run(exit_ratio, signal_file, start_name, start, end)
            detail.append(row)
            print(f"{name} {start_name} annual={row['annual']} sharpe={row['sharpe']}", flush=True)
    summary = _summarize(detail, signal_stats)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_recent60.csv", sorted(summary, key=lambda row: _f(row["recent60_annual"]), reverse=True))
    audit = liq._audit_signals(signal_files)
    (REPORT_DIR / "hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
