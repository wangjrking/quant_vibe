from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import duckdb
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"
OUT_SIGNAL_DIR = REPORT_DIR / "fullpool_refill_signals"
OUT_LOG_DIR = REPORT_DIR / "fullpool_refill_logs"
OUT_CSV = REPORT_DIR / "fullpool_refill_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "fullpool_refill_probe_20260630.json"


CASES: list[dict[str, Any]] = [
    {
        "name": "full_top1_base80_h2m3",
        "topn": 1,
        "target": 0.80,
        "target_cap": 0.80,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 1.01,
        "candidate_depth": 80,
        "amount_min": 150000,
        "total_mv_min": 300000,
    },
    {
        "name": "full_top1_gap_scale90_h2m3",
        "topn": 1,
        "target": 0.80,
        "target_cap": 0.80,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 1.01,
        "candidate_depth": 120,
        "amount_min": 150000,
        "total_mv_min": 300000,
        "scale_hot_turn": 0.90,
        "scale_new_or_no_atr": 0.90,
        "gap_nochase": True,
    },
    {
        "name": "full_top1_gap_scale75_h1m1",
        "topn": 1,
        "target": 0.80,
        "target_cap": 0.80,
        "holding_days": 1,
        "max_holding_days": 1,
        "score_exit": 0.99,
        "score_continue": 1.50,
        "candidate_depth": 120,
        "amount_min": 150000,
        "total_mv_min": 300000,
        "scale_hot_turn": 0.75,
        "scale_new_or_no_atr": 0.75,
        "gap_nochase": True,
    },
    {
        "name": "full_top1_strict_nohot_noatr_h1m1",
        "topn": 1,
        "target": 0.80,
        "target_cap": 0.80,
        "holding_days": 1,
        "max_holding_days": 1,
        "score_exit": 0.99,
        "score_continue": 1.50,
        "candidate_depth": 300,
        "amount_min": 150000,
        "total_mv_min": 300000,
        "reject_hot_turn": True,
        "reject_new_or_no_atr": True,
        "gap_nochase": True,
    },
    {
        "name": "full_top2_gap_scale75_h1m1",
        "topn": 2,
        "target": 0.50,
        "target_cap": 0.50,
        "holding_days": 1,
        "max_holding_days": 1,
        "score_exit": 0.99,
        "score_continue": 1.50,
        "candidate_depth": 160,
        "amount_min": 150000,
        "total_mv_min": 300000,
        "scale_hot_turn": 0.75,
        "scale_new_or_no_atr": 0.75,
        "gap_nochase": True,
    },
    {
        "name": "full_top3_gap_scale75_h1m1",
        "topn": 3,
        "target": 0.34,
        "target_cap": 0.34,
        "holding_days": 1,
        "max_holding_days": 1,
        "score_exit": 0.99,
        "score_continue": 1.50,
        "candidate_depth": 200,
        "amount_min": 150000,
        "total_mv_min": 300000,
        "scale_hot_turn": 0.75,
        "scale_new_or_no_atr": 0.75,
        "gap_nochase": True,
    },
]


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _parse_date(value: Any) -> datetime_module.datetime | None:
    text = str(value or "").strip()
    if len(text) < 8:
        return None
    try:
        return datetime_module.datetime.strptime(text[:8], "%Y%m%d")
    except ValueError:
        return None


def _symbol(stock_code: str) -> str:
    code = str(stock_code or "")
    if code.endswith(".SH"):
        return "SHSE." + code[:6]
    if code.endswith(".SZ"):
        return "SZSE." + code[:6]
    if code.endswith(".BJ"):
        return "BJSE." + code[:6]
    return code


def _is_risk_name(name: str) -> bool:
    text = str(name or "").strip().upper()
    return text.startswith("ST") or text.startswith("*ST") or "閫€" in text


def _list_age_days(signal_date: str, list_date: str) -> float | None:
    signal_dt = _parse_date(signal_date)
    list_dt = _parse_date(list_date)
    if signal_dt is None or list_dt is None:
        return None
    return float((signal_dt - list_dt).days)


def _is_open_limit_up(stock_code: str, buy_open: float | None, buy_pre_close: float | None) -> bool:
    if buy_open is None or buy_pre_close in (None, 0):
        return False
    code = str(stock_code or "")[:6]
    limit = 0.20 if code.startswith(("300", "301", "688", "689")) else 0.10
    return (buy_open / buy_pre_close - 1.0) >= (limit - 0.003)


def _risk_flags(row: dict[str, Any]) -> dict[str, bool]:
    pct_chg = _float(row.get("pct_chg"))
    turnover = _float(row.get("turnover_rate"))
    list_age = _float(row.get("list_age_days"))
    atr = _float(row.get("atr_qfq"))
    hot_turn = pct_chg is not None and turnover is not None and pct_chg >= 15.0 and turnover >= 15.0
    new_or_no_atr = (list_age is not None and list_age < 60.0) or atr is None
    return {"hot_turn": hot_turn, "new_or_no_atr": new_or_no_atr}


def _target_for_case(row: dict[str, Any], case: dict[str, Any]) -> tuple[float, str]:
    target = float(case["target"])
    flags = _risk_flags(row)
    reasons: list[str] = []
    hot_scale = _float(case.get("scale_hot_turn"))
    new_scale = _float(case.get("scale_new_or_no_atr"))
    if hot_scale is not None and flags["hot_turn"]:
        target *= hot_scale
        reasons.append("hot_turn_scale")
    if new_scale is not None and flags["new_or_no_atr"]:
        target *= new_scale
        reasons.append("new_or_no_atr_scale")
    gap = _float(row.get("buy_open_gap"))
    if case.get("gap_nochase") and gap is not None:
        if gap >= 0.02:
            target *= 0.50
            reasons.append("gap_up_scale")
        elif gap <= -0.08:
            target *= 0.70
            reasons.append("gap_down_scale")
    return max(0.0, min(target, float(case["target_cap"]))), "|".join(reasons)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
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


def _candidate_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    con = duckdb.connect(str(SCORE_DB), read_only=True)
    con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS market (READ_ONLY)")
    sql = f"""
    WITH td AS (
        SELECT trade_date AS signal_date,
               LEAD(trade_date) OVER (ORDER BY trade_date) AS buy_date
        FROM (
            SELECT DISTINCT trade_date
            FROM market.STOCK_DAILY_DATA
            WHERE trade_date BETWEEN '20220606' AND '20260629'
        )
    ),
    joined AS (
        SELECT
            s.trade_date AS signal_date,
            td.buy_date AS buy_date,
            s.stock_code AS stock_code,
            s.pred_prob AS pred_prob,
            m.name AS name,
            m.close AS signal_close,
            m.pct_chg AS pct_chg,
            m.amount AS amount,
            m.turnover_rate AS turnover_rate,
            m.total_mv AS total_mv,
            m.atr_qfq AS atr_qfq,
            m.list_date AS list_date,
            m.ST_TYPE AS st_type,
            m.ST_TYPE_name AS st_type_name,
            b.open AS buy_open,
            b.pre_close AS buy_pre_close,
            b.ST_TYPE AS buy_st_type,
            b.ST_TYPE_name AS buy_st_type_name,
            b.name AS buy_name
        FROM {SCORE_TABLE} s
        JOIN td ON td.signal_date = s.trade_date AND td.buy_date IS NOT NULL
        JOIN market.STOCK_DAILY_DATA m
          ON m.trade_date = s.trade_date AND m.stock_code = s.stock_code
        JOIN market.STOCK_DAILY_DATA b
          ON b.trade_date = td.buy_date AND b.stock_code = s.stock_code
        WHERE s.stock_code NOT LIKE '%.BJ'
          AND m.amount >= ?
          AND m.total_mv >= ?
          AND COALESCE(TRIM(m.ST_TYPE), '') IN ('', '0')
          AND COALESCE(TRIM(m.ST_TYPE_name), '') IN ('', '0')
          AND COALESCE(TRIM(b.ST_TYPE), '') IN ('', '0')
          AND COALESCE(TRIM(b.ST_TYPE_name), '') IN ('', '0')
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY signal_date ORDER BY pred_prob DESC, stock_code ASC) AS pre_rank
        FROM joined
    )
    SELECT *
    FROM ranked
    WHERE pre_rank <= ?
    ORDER BY signal_date, pre_rank
    """
    rows = []
    for raw in con.execute(sql, (float(case["amount_min"]), float(case["total_mv_min"]), int(case["candidate_depth"]))).fetchall():
        keys = [
            "signal_date",
            "buy_date",
            "stock_code",
            "pred_prob",
            "name",
            "signal_close",
            "pct_chg",
            "amount",
            "turnover_rate",
            "total_mv",
            "atr_qfq",
            "list_date",
            "st_type",
            "st_type_name",
            "buy_open",
            "buy_pre_close",
            "buy_st_type",
            "buy_st_type_name",
            "buy_name",
            "pre_rank",
        ]
        item = dict(zip(keys, raw))
        item["list_age_days"] = _list_age_days(item["signal_date"], item.get("list_date"))
        signal_close = _float(item.get("signal_close"))
        buy_open = _float(item.get("buy_open"))
        item["buy_open_gap"] = (buy_open / signal_close - 1.0) if buy_open is not None and signal_close not in (None, 0) else None
        rows.append(item)
    con.close()
    return rows


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    candidates = _candidate_rows(case)
    output_rows: list[dict[str, Any]] = []
    rejected = Counter()
    picked_by_date = Counter()
    for row in candidates:
        signal_date = str(row["signal_date"])
        if picked_by_date[signal_date] >= int(case["topn"]):
            continue
        name = str(row.get("name") or "")
        if _is_risk_name(name) or _is_risk_name(str(row.get("buy_name") or "")):
            rejected["risk_name"] += 1
            continue
        if _is_open_limit_up(str(row["stock_code"]), _float(row.get("buy_open")), _float(row.get("buy_pre_close"))):
            rejected["buy_open_limit_up"] += 1
            continue
        flags = _risk_flags(row)
        if case.get("reject_hot_turn") and flags["hot_turn"]:
            rejected["hot_turn"] += 1
            continue
        if case.get("reject_new_or_no_atr") and flags["new_or_no_atr"]:
            rejected["new_or_no_atr"] += 1
            continue
        target_pct, scale_reason = _target_for_case(row, case)
        picked_by_date[signal_date] += 1
        rank = picked_by_date[signal_date]
        stock_code = str(row["stock_code"])
        item = {
            "signal_date": signal_date,
            "buy_date": row["buy_date"],
            "symbol": _symbol(stock_code),
            "stock_code": stock_code,
            "name": row.get("name"),
            "rank": str(rank),
            "pred_prob": row.get("pred_prob"),
            "entry_score": row.get("pred_prob"),
            "amount": row.get("amount"),
            "turnover_rate": row.get("turnover_rate"),
            "total_mv": row.get("total_mv"),
            "atr_qfq": row.get("atr_qfq"),
            "pct_chg": row.get("pct_chg"),
            "list_age_days": row.get("list_age_days"),
            "target_pct": f"{target_pct:.5f}",
            "holding_days": str(case["holding_days"]),
            "max_holding_days": str(case["max_holding_days"]),
            "score_exit_entry_ratio": f"{float(case['score_exit']):.5f}",
            "min_holding_days_before_score_exit": "1",
            "score_continue_entry_ratio": f"{float(case['score_continue']):.5f}",
            "strategy_variant": str(case["name"]),
            "filter_name": str(case["name"]),
            "dynamic_hold_name": f"h{case['holding_days']}m{case['max_holding_days']}_c{case['score_continue']}",
            "buy_day_market_available": "True",
            "buy_day_hard_gate_complete": "True",
            "buy_day_st_rejected": "False",
            "buy_day_open_limit_up_rejected": "False",
            "buy_open_gap": "" if row.get("buy_open_gap") is None else f"{float(row['buy_open_gap']):.8f}",
            "risk_filter_scale_reason": scale_reason,
        }
        output_rows.append(item)
    output_rows.sort(key=lambda item: (item["signal_date"], int(item["rank"]), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, output_rows)
    target = int(case["topn"])
    return output, {
        "candidate_rows": len(candidates),
        "signal_rows": len(output_rows),
        "signal_days": len(picked_by_date),
        "days_below_target": sum(1 for value in picked_by_date.values() if value < target),
        "min_per_day": min(picked_by_date.values()) if picked_by_date else 0,
        "max_per_day": max(picked_by_date.values()) if picked_by_date else 0,
        "rejected_hot_turn": rejected["hot_turn"],
        "rejected_new_or_no_atr": rejected["new_or_no_atr"],
        "rejected_risk_name": rejected["risk_name"],
        "rejected_buy_open_limit_up": rejected["buy_open_limit_up"],
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": "0",
                "GM_LIGHT_STOP_LOSS_PCT": "none",
                "GM_LOG_EXPOSURE": "1",
                "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
                "GM_SYNC_POSITIONS": "1",
                "GM_CASH_BUFFER": "0.99",
                "GM_VERBOSE_TRADES": "1",
                "GM_FORCE_SELL_MARKET_ORDER": "0",
                "GM_FORCE_BUY_MARKET_ORDER": "0",
                "GM_INTRADAY_RISK_MODE": "0",
                "GM_INTRADAY_REPLACE_BUY": "0",
                "GM_EQUITY_DD_RISK_MODE": "0",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": str(case["score_exit"]),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(case["score_continue"]),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.995",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
                "GM_RESIZE_HELD_ON_SIGNAL": "0",
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
                "GM_INDEX_RISK_EXIT_MODE": "0",
                "GM_BREADTH_RISK_EXIT_MODE": "0",
            }
        )
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(int(case["topn"])),
            "--holding-days",
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["max_holding_days"])),
            "--target-position-pct",
            str(float(case["target_cap"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            "2022-06-07 09:00:00",
            "--backtest-end",
            "2026-06-29 15:30:00",
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        **case,
        **signal_meta,
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
        "note": "research-only full-pool refill probe; no production parameter change",
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

