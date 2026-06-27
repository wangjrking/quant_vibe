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

import tune_top1_soft_rank_latest_formal_20260623 as top1


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_ROOT = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
)
SOURCE_DIR = SOURCE_ROOT / "diversification_tune_20260624"
REPORT_DIR = SOURCE_DIR / "strict_sync_liquidity_refill_20260624"
BASE_CACHE = SOURCE_ROOT / "latest_formal_base_no_delist_v2.parquet"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
SCORE_DB = SOURCE_DIR / "scores_diversification.db"
SCORE_TABLE = "score_div_top1_w90_5d10_h5_e099"
STRATEGY_DIR = SOURCE_DIR / "strict_sync_strategy_snapshot_20260624"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_END = "2026-06-23 15:30:00"

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "3",
    "GM_STOP_LOSS_PCT": "none",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "0",
}

BASE_RULE = {
    "weights": {"10d": 0.90, "5d": 0.10},
    "target_pct": 0.99,
    "holding_days": 3,
    "exit_ratio": 0.97,
    "min_hold": 1,
    "gamma": 2.0,
}

FILTERS = [
    {"name": "liq_none", "amount_min": None, "total_mv_min": None, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_amt5w", "amount_min": 50000, "total_mv_min": None, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_amt10w", "amount_min": 100000, "total_mv_min": None, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_amt15w", "amount_min": 150000, "total_mv_min": None, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_mv20w", "amount_min": None, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_mv30w", "amount_min": None, "total_mv_min": 300000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_amt10w_mv20w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": None, "require_atr": False},
    {"name": "liq_amt10w_mv20w_mv200w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": 2000000, "turnover_min": None, "require_atr": False},
    {"name": "liq_amt10w_mv30w_mv200w", "amount_min": 100000, "total_mv_min": 300000, "total_mv_max": 2000000, "turnover_min": None, "require_atr": False},
    {"name": "liq_turn1_amt10w_mv20w", "amount_min": 100000, "total_mv_min": 200000, "total_mv_max": None, "turnover_min": 1.0, "require_atr": False},
    {"name": "diag_atr_nonnull", "amount_min": None, "total_mv_min": None, "total_mv_max": None, "turnover_min": None, "require_atr": True},
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


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)


def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _load_base() -> pd.DataFrame:
    if not BASE_CACHE.exists():
        raise FileNotFoundError(BASE_CACHE)
    frame = pd.read_parquet(BASE_CACHE)
    required = {
        "trade_date",
        "stock_code",
        "rank_5d",
        "rank_10d",
        "name",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"base cache missing columns: {missing}")
    for column in ["rank_5d", "rank_10d", "amount", "turnover_rate", "total_mv", "atr_qfq"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _market_rows() -> dict[str, dict[str, dict]]:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open, close,
                   limit_times, ST_TYPE AS st_type, ST_TYPE_name AS st_type_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= '20240604'
            """
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        out.setdefault(str(row["trade_date"]), {})[str(row["stock_code"])] = dict(row)
    return out


def _date_map(frame: pd.DataFrame) -> dict[str, str]:
    dates = sorted(str(value) for value in frame["trade_date"].dropna().unique())
    return {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}


def _entry_score(frame: pd.DataFrame) -> pd.Series:
    return (
        float(BASE_RULE["weights"]["10d"]) * frame["rank_10d"].astype(float)
        + float(BASE_RULE["weights"]["5d"]) * frame["rank_5d"].astype(float)
    )


def _filter_mask(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if cfg.get("amount_min") is not None:
        mask &= frame["amount"] >= float(cfg["amount_min"])
    if cfg.get("total_mv_min") is not None:
        mask &= frame["total_mv"] >= float(cfg["total_mv_min"])
    if cfg.get("total_mv_max") is not None:
        mask &= frame["total_mv"] <= float(cfg["total_mv_max"])
    if cfg.get("turnover_min") is not None:
        mask &= frame["turnover_rate"] >= float(cfg["turnover_min"])
    if cfg.get("require_atr"):
        mask &= frame["atr_qfq"].notna()
    return mask


def _candidate_availability(base: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    all_days = int(base["trade_date"].nunique())
    for cfg in FILTERS:
        mask = _filter_mask(base, cfg)
        daily = base.loc[mask].groupby("trade_date").size()
        rows.append(
            {
                "name": cfg["name"],
                "rows": int(mask.sum()),
                "days": int(daily.size),
                "missing_days": all_days - int(daily.size),
                "min_daily_candidates": int(daily.min()) if len(daily) else 0,
                "median_daily_candidates": float(daily.median()) if len(daily) else 0.0,
                "amount_min": cfg.get("amount_min"),
                "total_mv_min": cfg.get("total_mv_min"),
                "total_mv_max": cfg.get("total_mv_max"),
                "turnover_min": cfg.get("turnover_min"),
                "require_atr": cfg.get("require_atr"),
            }
        )
    return rows


def _case_name(cfg: dict) -> str:
    return f"{cfg['name']}_w90_5d10_pos99_h3_e097_mh1"


def _build_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], cfg: dict) -> tuple[Path, list[dict]]:
    frame = base.loc[_filter_mask(base, cfg)].copy()
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
                "pred_prob": top1._tailpow(float(chosen["rank_10d"]), float(BASE_RULE["gamma"]), upper),
                "entry_score": chosen["entry_score"],
                "pred_5d": chosen.get("pred_5d"),
                "pred_10d": chosen.get("pred_10d"),
                "rank_5d": chosen.get("rank_5d"),
                "rank_10d": chosen.get("rank_10d"),
                "amount": chosen.get("amount"),
                "turnover_rate": chosen.get("turnover_rate"),
                "total_mv": chosen.get("total_mv"),
                "atr_qfq": chosen.get("atr_qfq"),
                "target_pct": f"{float(BASE_RULE['target_pct']):.5f}",
                "holding_days": int(BASE_RULE["holding_days"]),
                "max_holding_days": int(BASE_RULE["holding_days"]),
                "score_exit_entry_ratio": f"{float(BASE_RULE['exit_ratio']):.5f}",
                "min_holding_days_before_score_exit": int(BASE_RULE["min_hold"]),
                "filter_name": cfg["name"],
            }
        )
    path = REPORT_DIR / "signals" / f"{_case_name(cfg)}.csv"
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


def _run_case(cfg: dict, signal_file: Path, start_name: str, start: str) -> dict:
    name = _case_name(cfg)
    log_file = REPORT_DIR / "logs" / f"{name}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(BASE_RULE["exit_ratio"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(BASE_RULE["min_hold"]))
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
            "1",
            "--holding-days",
            str(int(BASE_RULE["holding_days"])),
            "--max-holding-days",
            str(int(BASE_RULE["holding_days"])),
            "--target-position-pct",
            str(float(BASE_RULE["target_pct"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            start,
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
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "name": name,
        "filter_name": cfg["name"],
        "amount_min": cfg.get("amount_min"),
        "total_mv_min": cfg.get("total_mv_min"),
        "total_mv_max": cfg.get("total_mv_max"),
        "turnover_min": cfg.get("turnover_min"),
        "require_atr": cfg.get("require_atr"),
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
                "full_annual": annual,
                "full_sharpe": sharpe,
                "full_max_drawdown": maxdd,
                "late_min_annual": late_min,
                "late_median_annual": late_median,
                "full_avg_invested_pct": full.get("avg_invested_pct"),
                "full_ge80_ratio": full.get("ge80_ratio"),
                "full_max_active_positions": full.get("max_active_positions"),
                "open_count": full.get("open_count"),
                "signal_rows": signal_stats.get(name, {}).get("rows"),
                "signal_days": signal_stats.get(name, {}).get("signal_days"),
                "latest_signal_date": signal_stats.get(name, {}).get("latest_signal_date"),
                "latest_buy_date": signal_stats.get(name, {}).get("latest_buy_date"),
                "score": score,
            }
        )
    return out


def _signal_stats(rows: list[dict]) -> dict:
    dates = [str(row.get("signal_date") or "") for row in rows if row.get("signal_date")]
    buy_dates = [str(row.get("buy_date") or "") for row in rows if row.get("buy_date")]
    return {
        "rows": len(rows),
        "signal_days": len(set(dates)),
        "latest_signal_date": max(dates) if dates else "",
        "latest_buy_date": max(buy_dates) if buy_dates else "",
    }


def _is_open_limit_up(row: dict | None) -> bool:
    return top1._is_limit_buy(row)


def _audit_signals(signal_files: list[Path]) -> dict:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        market = {
            (str(row["trade_date"]), str(row["stock_code"])): dict(row)
            for row in conn.execute(
                """
                SELECT trade_date, stock_code, name, pre_close, open, close,
                       limit_times, ST_TYPE AS st_type, ST_TYPE_name AS st_type_name
                FROM STOCK_DAILY_DATA
                WHERE trade_date >= '20240604'
                """
            )
        }
    finally:
        conn.close()
    summary = {
        "audited_files": 0,
        "total_signal_rows": 0,
        "failed_files": 0,
        "buy_join_missing": 0,
        "bj_rows": 0,
        "signal_st_name_rows": 0,
        "buy_st_rows": 0,
        "signal_delist_name_rows": 0,
        "buy_delist_rows": 0,
        "open_limit_up_buy_rows": 0,
        "latest_signal_date": "",
        "latest_buy_date": "",
        "files": [],
    }
    for path in signal_files:
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        file_stats = {
            "file": str(path),
            "rows": len(rows),
            "buy_join_missing": 0,
            "bj_rows": 0,
            "signal_st_name_rows": 0,
            "buy_st_rows": 0,
            "signal_delist_name_rows": 0,
            "buy_delist_rows": 0,
            "open_limit_up_buy_rows": 0,
            "latest_signal_date": max((str(row.get("signal_date") or "") for row in rows), default=""),
            "latest_buy_date": max((str(row.get("buy_date") or "") for row in rows), default=""),
        }
        for row in rows:
            code = str(row.get("stock_code") or "")
            name = str(row.get("name") or "")
            buy_date = str(row.get("buy_date") or "")
            buy_row = market.get((buy_date, code))
            if code.endswith(".BJ"):
                file_stats["bj_rows"] += 1
            if name.startswith(("ST", "*ST")):
                file_stats["signal_st_name_rows"] += 1
            if "退" in name:
                file_stats["signal_delist_name_rows"] += 1
            if buy_row is None:
                file_stats["buy_join_missing"] += 1
                continue
            buy_name = str(buy_row.get("name") or "")
            if top1._is_st_like(buy_row) or buy_name.startswith(("ST", "*ST")):
                file_stats["buy_st_rows"] += 1
            if "退" in buy_name:
                file_stats["buy_delist_rows"] += 1
            if _is_open_limit_up(buy_row):
                file_stats["open_limit_up_buy_rows"] += 1
        failed = any(file_stats[key] for key in [
            "buy_join_missing",
            "bj_rows",
            "signal_st_name_rows",
            "buy_st_rows",
            "signal_delist_name_rows",
            "buy_delist_rows",
            "open_limit_up_buy_rows",
        ])
        if failed:
            summary["failed_files"] += 1
        summary["audited_files"] += 1
        summary["total_signal_rows"] += len(rows)
        for key in [
            "buy_join_missing",
            "bj_rows",
            "signal_st_name_rows",
            "buy_st_rows",
            "signal_delist_name_rows",
            "buy_delist_rows",
            "open_limit_up_buy_rows",
        ]:
            summary[key] += file_stats[key]
        summary["latest_signal_date"] = max(summary["latest_signal_date"], file_stats["latest_signal_date"])
        summary["latest_buy_date"] = max(summary["latest_buy_date"], file_stats["latest_buy_date"])
        summary["files"].append(file_stats)
    return summary


def _write_report(summary: list[dict], availability: list[dict], audit: dict) -> None:
    lines = [
        "# 严格同步流动性回补调参报告",
        "",
        "## 当前结论",
        "",
        "本轮只测试宽口径、信号日已知的流动性和市值过滤，并在每日候选池内回补 Top1；不使用行业、月份、日期排除，也不使用买入日成交额或买入日市值等未来字段。",
        "",
        "收益仍属于弱准入，参数敏感性、持续性、低路径依赖和硬过滤审计属于强准入。本报告不得单独作为生产准入通过结论。",
        "",
        "## 候选池可用性",
        "",
        "| 过滤 | 行数 | 覆盖交易日 | 缺失日 | 每日最小候选 | 每日中位候选 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in availability:
        lines.append(
            f"| {row['name']} | {row['rows']} | {row['days']} | {row['missing_days']} | "
            f"{row['min_daily_candidates']} | {row['median_daily_candidates']:.1f} |"
        )
    lines.extend([
        "",
        "## 掘金回测汇总",
        "",
        "| 策略 | 年化 | 夏普 | 最大回撤 | 近期开仓最差年化 | 近期开仓中位年化 | 平均持仓率 | 开仓次数 | 综合分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted(summary, key=lambda item: _f(item["score"]), reverse=True):
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
        f"- 候选池可用性：`{REPORT_DIR / 'candidate_availability.csv'}`",
        f"- 硬过滤审计：`{REPORT_DIR / 'liquidity_refill_hard_gate_audit.json'}`",
        f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
        f"- 信号文件：`{REPORT_DIR / 'signals'}`",
    ])
    (REPORT_DIR / "liquidity_refill_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"juejin strategy main.py not found: {STRATEGY_DIR}")
    base = _load_base()
    market = _market_rows()
    next_date = _date_map(base)
    availability = _candidate_availability(base)
    _write_rows(REPORT_DIR / "candidate_availability.csv", availability)

    signal_stats: dict[str, dict] = {}
    signal_files: list[Path] = []
    cases: list[dict] = []
    cases_path = REPORT_DIR / "cases.csv"
    if cases_path.exists():
        with cases_path.open("r", encoding="utf-8-sig", newline="") as file:
            cases = list(csv.DictReader(file))
    existing = {(row["name"], row["start_name"]) for row in cases}

    total = len(FILTERS) * len(STARTS)
    done = 0
    for cfg in FILTERS:
        signal_file, signal_rows = _build_signal(base, market, next_date, cfg)
        signal_files.append(signal_file)
        signal_stats[_case_name(cfg)] = _signal_stats(signal_rows)
        for start_name, start in STARTS:
            done += 1
            name = _case_name(cfg)
            if (name, start_name) in existing:
                print(f"[{done}/{total}] reuse {name} {start_name}", flush=True)
                continue
            row = _run_case(cfg, signal_file, start_name, start)
            cases.append(row)
            _write_rows(cases_path, cases)
            print(
                f"[{done}/{total}] {name} {start_name} "
                f"annual={row['annual']} sharpe={row['sharpe']} maxdd={row['max_drawdown']} active={row['max_active_positions']}",
                flush=True,
            )

    summary = _summarize(cases, signal_stats)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    _write_rows(REPORT_DIR / "summary_by_score.csv", sorted(summary, key=lambda row: _f(row["score"]), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(summary, key=lambda row: _f(row["full_annual"]), reverse=True))
    audit = _audit_signals(signal_files)
    (REPORT_DIR / "liquidity_refill_hard_gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, availability, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
