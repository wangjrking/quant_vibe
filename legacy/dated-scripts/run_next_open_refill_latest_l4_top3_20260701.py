from __future__ import annotations

import ast
import csv
import datetime as dt
import duckdb
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stock_daily_data_route import resolve_stock_daily_duckdb_path

ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
SOURCE_REPORT = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
REPORT_DIR = DATA / "reports" / "strategy_agent_next_open_refill_latest_l4_top3_20260701"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DUCKDB = resolve_stock_daily_duckdb_path(require_exists=True)
SCORE_DB = SOURCE_REPORT / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"

OUT_SIGNAL_DIR = REPORT_DIR / "signals"
OUT_LOG_DIR = REPORT_DIR / "logs"
OUT_CSV = REPORT_DIR / "summary.csv"
OUT_JSON = REPORT_DIR / "summary.json"
OUT_DETAIL = REPORT_DIR / "detail.csv"
OUT_REPORT = REPORT_DIR / "report.md"

FULL_SLICE = ("full", "2022-06-07 09:00:00", "2026-06-29 15:30:00")
CHECK_SLICES = [
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-29 15:30:00"),
    ("recent60", "2026-04-01 09:00:00", "2026-06-29 15:30:00"),
]
ADAPTIVE_TOTAL_CAPITAL = 700000.0


BASE = {
    "name": "latest_l4_top3_pos25_cool2d20_h3m5_refill",
    "topn": 3,
    "target_cap": 0.25,
    "holding_days": 3,
    "max_holding_days": 5,
    "score_exit": 0.96,
    "score_continue": 0.97,
    "min_hold_before_score_exit": 1,
    "amount_min": 150000.0,
    "total_mv_min": 300000.0,
    "price_cap": 150.0,
    "two_day_cap": 0.20,
}


CASES: list[dict[str, Any]] = [
    {
        "name": "baseline_pool12_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "pool_depth": 12,
    },
    {
        "name": "refill_skip_u10_d9_pool12_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "pool_depth": 12,
        "skip_up_cut": 0.010,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "refill_skip_u9_d9_pool12_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "pool_depth": 12,
        "skip_up_cut": 0.009,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "refill_skip_u12_d9_pool12_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "pool_depth": 12,
        "skip_up_cut": 0.012,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "refill_skip_u10_d9_pool20_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "pool_depth": 20,
        "skip_up_cut": 0.010,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "refill_skip_u9_d9_pool20_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "pool_depth": 20,
        "skip_up_cut": 0.009,
        "skip_deep_down_cut": -0.09,
    },
]


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            try:
                return eval(payload, {"__builtins__": {}}, {"datetime": dt})
            except Exception:
                return None
    return None


def _build_candidate_pool(case: dict[str, Any]) -> list[dict[str, Any]]:
    pool_depth = int(case["pool_depth"])
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{SCORE_DB.as_posix()}' AS scoredb (READ_ONLY)")
        con.execute(f"ATTACH '{MARKET_DUCKDB.as_posix()}' AS marketdb (READ_ONLY)")
        sql = f"""
        WITH score_base AS (
            SELECT trade_date, stock_code, pred_prob
            FROM scoredb."{SCORE_TABLE}"
        ),
        market_with_prev AS (
            SELECT
                trade_date,
                stock_code,
                name,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                close,
                open,
                pre_close,
                pct_chg,
                ST_TYPE,
                ST_TYPE_name,
                limit_times,
                LAG(pct_chg, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_pct_chg
            FROM marketdb.STOCK_DAILY_DATA
        ),
        market_dates AS (
            SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA
        ),
        next_dates AS (
            SELECT
                trade_date AS signal_date,
                LEAD(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                MAX(trade_date) OVER () AS latest_market_date
            FROM market_dates
        ),
        scored AS (
            SELECT
                sb.trade_date AS signal_date,
                nd.buy_date,
                nd.latest_market_date,
                sb.stock_code,
                mwp.name,
                mwp.amount,
                mwp.turnover_rate,
                mwp.total_mv,
                mwp.atr_qfq,
                mwp.close,
                mwp.open AS signal_open,
                mwp.pre_close AS signal_pre_close,
                mwp.pct_chg,
                mwp.prev_pct_chg,
                ((1.0 + COALESCE(mwp.pct_chg, 0.0) / 100.0) * (1.0 + COALESCE(mwp.prev_pct_chg, 0.0) / 100.0) - 1.0) AS two_day_ret,
                sb.pred_prob AS entry_score
            FROM score_base sb
            INNER JOIN market_with_prev mwp
              ON sb.trade_date = mwp.trade_date
             AND sb.stock_code = mwp.stock_code
            INNER JOIN next_dates nd
              ON sb.trade_date = nd.signal_date
            WHERE nd.buy_date IS NOT NULL
              AND sb.stock_code NOT LIKE '%.BJ'
              AND COALESCE(CAST(mwp.amount AS REAL), 0.0) >= {float(BASE['amount_min'])}
              AND COALESCE(CAST(mwp.total_mv AS REAL), 0.0) >= {float(BASE['total_mv_min'])}
              AND COALESCE(CAST(mwp.close AS REAL), 0.0) <= {float(BASE['price_cap'])}
              AND ((1.0 + COALESCE(mwp.pct_chg, 0.0) / 100.0) * (1.0 + COALESCE(mwp.prev_pct_chg, 0.0) / 100.0) - 1.0) < {float(BASE['two_day_cap'])}
              AND NOT (
                    UPPER(COALESCE(mwp.name, '')) LIKE 'ST%'
                 OR UPPER(COALESCE(mwp.name, '')) LIKE '*ST%'
                 OR COALESCE(mwp.ST_TYPE_name, '') LIKE '%风险%'
                 OR (CAST(mwp.ST_TYPE AS REAL) IS NOT NULL AND CAST(mwp.ST_TYPE AS REAL) <> 0)
                 OR INSTR(COALESCE(mwp.name, ''), '退市') > 0
              )
        ),
        with_buy_checks AS (
            SELECT
                s.*,
                bm.open AS buy_open,
                bm.close AS buy_close,
                bm.pre_close AS buy_pre_close,
                bm.name AS buy_name,
                bm.ST_TYPE AS buy_st_type,
                bm.ST_TYPE_name AS buy_st_type_name,
                bm.limit_times AS buy_limit_times,
                CASE
                    WHEN bm.stock_code IS NULL THEN 0
                    WHEN bm.stock_code LIKE '%.BJ' THEN 0
                    WHEN UPPER(COALESCE(bm.name, '')) LIKE 'ST%' THEN 0
                    WHEN UPPER(COALESCE(bm.name, '')) LIKE '*ST%' THEN 0
                    WHEN COALESCE(bm.ST_TYPE_name, '') LIKE '%风险%' THEN 0
                    WHEN (CAST(bm.ST_TYPE AS REAL) IS NOT NULL AND CAST(bm.ST_TYPE AS REAL) <> 0) THEN 0
                    WHEN INSTR(COALESCE(bm.name, ''), '退市') > 0 THEN 0
                    WHEN COALESCE(CAST(bm.limit_times AS REAL), 0.0) > 0.0 THEN 0
                    WHEN CAST(bm.pre_close AS REAL) IS NULL OR CAST(bm.open AS REAL) IS NULL THEN 0
                    WHEN CAST(bm.pre_close AS REAL) <= 0 OR CAST(bm.open AS REAL) <= 0 THEN 0
                    WHEN CAST(bm.open AS REAL) >= CAST(bm.pre_close AS REAL) * (
                        1.0 + CASE WHEN substr(s.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                    ) * 0.995 THEN 0
                    WHEN CAST(bm.open AS REAL) > {float(BASE['price_cap'])} THEN 0
                    ELSE 1
                END AS buy_day_ok
            FROM scored s
            LEFT JOIN marketdb.STOCK_DAILY_DATA bm
              ON s.buy_date = bm.trade_date
             AND s.stock_code = bm.stock_code
        )
        SELECT
            signal_date,
            buy_date,
            stock_code,
            name,
            entry_score,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            close,
            pct_chg,
            prev_pct_chg,
            two_day_ret,
            buy_open,
            buy_close,
            buy_pre_close,
            buy_day_ok,
            latest_market_date,
            ROW_NUMBER() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) AS pool_rank
        FROM with_buy_checks
        WHERE buy_day_ok = 1
        """
        cur = con.execute(sql)
        cols = [item[0] for item in cur.description]
        rows = [dict(zip(cols, record)) for record in cur.fetchall()]
    finally:
        con.close()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["signal_date"])].append(row)
    out: list[dict[str, Any]] = []
    for signal_date, day_rows in grouped.items():
        day_rows.sort(key=lambda row: (-float(row["entry_score"]), str(row["stock_code"])))
        out.extend(day_rows[:pool_depth])
    out.sort(key=lambda row: (str(row["signal_date"]), int(row["pool_rank"]), str(row["stock_code"])))
    return out


def _build_candidate_pool_exact(case: dict[str, Any]) -> list[dict[str, Any]]:
    pool_depth = int(case["pool_depth"])
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{MARKET_DUCKDB.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(f"ATTACH '{SCORE_DB.as_posix()}' AS scoredb (READ_ONLY)")
        sql = f"""
        WITH score_base AS (
            SELECT trade_date, stock_code, pred_prob
            FROM scoredb."{SCORE_TABLE}"
        ),
        market_with_prev AS (
            SELECT
                trade_date,
                stock_code,
                name,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                close,
                open,
                pre_close,
                pct_chg,
                ST_TYPE,
                ST_TYPE_name,
                limit_times,
                lag(pct_chg, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_pct_chg
            FROM marketdb.STOCK_DAILY_DATA
        ),
        market_dates AS (
            SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA
        ),
        next_dates AS (
            SELECT
                trade_date AS signal_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                max(trade_date) OVER () AS latest_market_date
            FROM market_dates
        ),
        scored AS (
            SELECT
                sb.trade_date AS signal_date,
                nd.buy_date,
                nd.latest_market_date,
                sb.stock_code,
                md.name,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                md.close,
                md.pct_chg,
                md.prev_pct_chg,
                ((1 + try_cast(md.pct_chg AS DOUBLE) / 100.0)
                 * (1 + try_cast(md.prev_pct_chg AS DOUBLE) / 100.0) - 1.0) AS two_day_ret,
                sb.pred_prob AS entry_score
            FROM score_base sb
            LEFT JOIN market_with_prev md
              ON sb.trade_date = md.trade_date
             AND sb.stock_code = md.stock_code
            LEFT JOIN next_dates nd
              ON sb.trade_date = nd.signal_date
            WHERE NOT (sb.stock_code LIKE '%.BJ' OR substr(sb.stock_code, 1, 1) IN ('4', '8'))
              AND NOT (
                upper(coalesce(md.name, '')) LIKE 'ST%%'
                OR upper(coalesce(md.name, '')) LIKE '*ST%%'
                OR coalesce(md.ST_TYPE_name, '') LIKE '%椋庨櫓%'
                OR (try_cast(md.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(md.ST_TYPE AS DOUBLE) <> 0)
              )
              AND NOT (
                instr(coalesce(md.name, ''), '退市') > 0
                OR coalesce(md.name, '') LIKE '退%%'
                OR coalesce(md.name, '') LIKE '%%退'
              )
              AND coalesce(try_cast(md.limit_times AS DOUBLE), 0.0) = 0.0
              AND try_cast(md.amount AS DOUBLE) >= {float(BASE['amount_min'])}
              AND try_cast(md.total_mv AS DOUBLE) >= {float(BASE['total_mv_min'])}
              AND try_cast(md.close AS DOUBLE) <= {float(BASE['price_cap'])}
              AND coalesce(two_day_ret, -999.0) < {float(BASE['two_day_cap'])}
        ),
        with_buy_checks AS (
            SELECT
                s.*,
                bm.open AS buy_open,
                bm.close AS buy_close,
                bm.pre_close AS buy_pre_close,
                bm.name AS buy_name,
                bm.ST_TYPE AS buy_st_type,
                bm.ST_TYPE_name AS buy_st_type_name,
                bm.limit_times AS buy_limit_times,
                CASE
                    WHEN bm.stock_code IS NULL THEN 0
                    WHEN bm.stock_code LIKE '%.BJ' THEN 0
                    WHEN upper(coalesce(bm.name, '')) LIKE 'ST%%' THEN 0
                    WHEN upper(coalesce(bm.name, '')) LIKE '*ST%%' THEN 0
                    WHEN coalesce(bm.ST_TYPE_name, '') LIKE '%椋庨櫓%' THEN 0
                    WHEN try_cast(bm.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(bm.ST_TYPE AS DOUBLE) <> 0 THEN 0
                    WHEN instr(coalesce(bm.name, ''), '退市') > 0 THEN 0
                    WHEN coalesce(try_cast(bm.limit_times AS DOUBLE), 0.0) > 0.0 THEN 0
                    WHEN try_cast(bm.pre_close AS DOUBLE) IS NULL OR try_cast(bm.open AS DOUBLE) IS NULL THEN 0
                    WHEN try_cast(bm.pre_close AS DOUBLE) <= 0 OR try_cast(bm.open AS DOUBLE) <= 0 THEN 0
                    WHEN try_cast(bm.open AS DOUBLE) >= try_cast(bm.pre_close AS DOUBLE) * (
                        1.0 + CASE WHEN substr(s.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                    ) * 0.995 THEN 0
                    WHEN try_cast(bm.open AS DOUBLE) > {float(BASE['price_cap'])} THEN 0
                    ELSE 1
                END AS buy_day_ok
            FROM scored s
            LEFT JOIN marketdb.STOCK_DAILY_DATA bm
              ON s.buy_date = bm.trade_date
             AND s.stock_code = bm.stock_code
        )
        SELECT
            signal_date,
            buy_date,
            stock_code,
            name,
            entry_score,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            close,
            pct_chg,
            prev_pct_chg,
            two_day_ret,
            buy_open,
            buy_close,
            buy_pre_close,
            buy_day_ok,
            latest_market_date,
            row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) AS pool_rank
        FROM with_buy_checks
        WHERE buy_day_ok = 1
        """
        cur = con.execute(sql)
        cols = [item[0] for item in cur.description]
        rows = [dict(zip(cols, record)) for record in cur.fetchall()]
    finally:
        con.close()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["signal_date"])].append(row)
    out: list[dict[str, Any]] = []
    for signal_date, day_rows in grouped.items():
        day_rows.sort(key=lambda row: (-float(row["entry_score"]), str(row["stock_code"])))
        out.extend(day_rows[:pool_depth])
    out.sort(key=lambda row: (str(row["signal_date"]), int(row["pool_rank"]), str(row["stock_code"])))
    return out


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _gap_bucket(gap: float | None) -> str:
    if gap is None:
        return "missing_gap"
    if gap <= -0.09:
        return "deep_down9"
    if gap <= -0.02:
        return "mid_down_band"
    if gap >= 0.015:
        return "up2"
    if gap >= 0.008:
        return "up1"
    return "other"


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    pool_rows = _build_candidate_pool_exact(case)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        grouped[str(row["signal_date"])].append(row)
    out_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    gap_bucket_counts: Counter[str] = Counter()
    shortage_examples: list[str] = []
    for signal_date, day_rows in sorted(grouped.items()):
        kept: list[dict[str, Any]] = []
        for row in sorted(day_rows, key=lambda item: (-float(item["entry_score"]), str(item["stock_code"]))):
            signal_close = _float(row.get("close"))
            buy_open = _float(row.get("buy_open"))
            gap = None
            if signal_close not in (None, 0) and buy_open is not None:
                gap = buy_open / signal_close - 1.0
            gap_bucket_counts[_gap_bucket(gap)] += 1
            skip_up = _float(case.get("skip_up_cut"))
            if skip_up is not None and gap is not None and gap >= skip_up:
                reason_counts["skip_up_gap"] += 1
                continue
            skip_down = _float(case.get("skip_deep_down_cut"))
            if skip_down is not None and gap is not None and gap <= skip_down:
                reason_counts["skip_deep_down_gap"] += 1
                continue
            reason_counts["kept"] += 1
            kept.append(row)
            if len(kept) >= int(BASE["topn"]):
                break
        if len(kept) < int(BASE["topn"]):
            shortage_examples.append(f"{signal_date}:{len(kept)}")
        for idx, row in enumerate(kept, start=1):
            stock_code = str(row["stock_code"])
            symbol = f"SHSE.{stock_code[:6]}" if stock_code.endswith(".SH") else f"SZSE.{stock_code[:6]}"
            signal_close = _float(row.get("close"))
            buy_open = _float(row.get("buy_open"))
            gap = None
            if signal_close not in (None, 0) and buy_open is not None:
                gap = buy_open / signal_close - 1.0
            out_rows.append(
                {
                    "signal_date": signal_date,
                    "buy_date": str(row["buy_date"]),
                    "symbol": symbol,
                    "stock_code": stock_code,
                    "name": row.get("name", ""),
                    "rank": idx,
                    "pred_prob": f"{float(row['entry_score']):.10f}",
                    "entry_score": f"{float(row['entry_score']):.10f}",
                    "amount": row.get("amount", ""),
                    "turnover_rate": row.get("turnover_rate", ""),
                    "total_mv": row.get("total_mv", ""),
                    "atr_qfq": row.get("atr_qfq", ""),
                    "pct_chg": row.get("pct_chg", ""),
                    "prev_pct_chg": row.get("prev_pct_chg", ""),
                    "two_day_ret": row.get("two_day_ret", ""),
                    "buy_open_gap": "" if gap is None else f"{gap:.8f}",
                    "target_pct": f"{float(BASE['target_cap']):.5f}",
                    "holding_days": str(int(case["holding_days"])),
                    "max_holding_days": str(int(case["max_holding_days"])),
                    "score_exit_entry_ratio": f"{float(case['score_exit']):.5f}",
                    "min_holding_days_before_score_exit": str(int(case["min_hold_before_score_exit"])),
                    "score_continue_entry_ratio": f"{float(case['score_continue']):.5f}",
                    "signal_stop_loss_pct": "0.05000",
                    "signal_take_profit_pct": "0.08000",
                    "strategy_variant": case["name"],
                    "filter_name": "cool2d20_refill",
                    "entry_weight_name": "w72_23_05_amt150_mv30",
                    "dynamic_hold_name": f"h{case['holding_days']}m{case['max_holding_days']}",
                    "buy_day_market_available": "True",
                    "buy_day_hard_gate_complete": "True",
                    "buy_day_st_rejected": "False",
                    "buy_day_open_limit_up_rejected": "False",
                    "latest_market_date": row.get("latest_market_date", ""),
                }
            )
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, out_rows)
    day_counts = Counter(row["signal_date"] for row in out_rows)
    below_target_days = sum(1 for count in day_counts.values() if count < int(BASE["topn"]))
    return output, {
        "signal_rows": len(out_rows),
        "signal_days": len(day_counts),
        "below_target_days": below_target_days,
        "reason_counts": dict(reason_counts),
        "gap_bucket_counts": dict(gap_bucket_counts),
        "shortage_examples": shortage_examples[:20],
    }


def _run_backtest(case: dict[str, Any], signal_file: Path, tag: str, start: str, end: str) -> dict[str, Any]:
    log_file = OUT_LOG_DIR / f"{case['name']}_{tag}.log"
    order_value = ADAPTIVE_TOTAL_CAPITAL * float(BASE["target_cap"])
    env = os.environ.copy()
    env.update(
        {
            "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
            "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
            "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
            "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
            "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
            "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
            "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": f"{order_value:.2f}",
            "GM_BREADTH_RISK_EXIT_MODE": "0",
            "GM_CASH_BUFFER": "0.99",
            "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
            "GM_EQUITY_DD_RESIZE_EXISTING": "0",
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.80",
            "GM_EQUITY_DD_HARD_SCALE": "0.60",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_FORCE_SELL_MARKET_ORDER": "0",
            "GM_INDEX_RISK_EXIT_MODE": "0",
            "GM_INTRADAY_REPLACE_BUY": "0",
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_LIGHT_STOP_LOSS_PCT": "none",
            "GM_LOG_EXPOSURE": "1",
            "GM_MAX_DAILY_SELLS": "0",
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_STOP_LOSS_PCT": "0.05",
            "GM_SYNC_POSITIONS": "1",
            "GM_TAKE_PROFIT_PCT": "0.08",
            "GM_VERBOSE_TRADES": "1",
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
        str(int(BASE["topn"])),
        "--holding-days",
        str(int(case["holding_days"])),
        "--max-holding-days",
        str(int(case["max_holding_days"])),
        "--target-position-pct",
        str(float(BASE["target_cap"])),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DUCKDB),
        "--score-exit-entry-ratio",
        str(float(case["score_exit"])),
        "--score-continue-entry-ratio",
        str(float(case["score_continue"])),
        "--min-holding-days-before-score-exit",
        str(int(case["min_hold_before_score_exit"])),
        "--backtest-start",
        start,
        "--backtest-end",
        end,
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
    indicator = _extract_indicator(log_file)
    return {
        "case_name": case["name"],
        "slice": tag,
        "returncode": proc.returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def _summary_row(case: dict[str, Any], signal_meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_slice = {row["slice"]: row for row in rows}
    out = {
        "name": case["name"],
        "pool_depth": case["pool_depth"],
        "holding_days": case["holding_days"],
        "max_holding_days": case["max_holding_days"],
        "score_exit": case["score_exit"],
        "score_continue": case["score_continue"],
        "min_hold_before_score_exit": case["min_hold_before_score_exit"],
        "signal_rows": signal_meta["signal_rows"],
        "signal_days": signal_meta["signal_days"],
        "below_target_days": signal_meta["below_target_days"],
        "reason_counts": json.dumps(signal_meta["reason_counts"], ensure_ascii=False, sort_keys=True),
        "gap_bucket_counts": json.dumps(signal_meta["gap_bucket_counts"], ensure_ascii=False, sort_keys=True),
        "shortage_examples": json.dumps(signal_meta["shortage_examples"], ensure_ascii=False),
    }
    for tag in [FULL_SLICE[0], *[item[0] for item in CHECK_SLICES]]:
        item = by_slice.get(tag, {})
        out[f"{tag}_annual"] = item.get("annual")
        out[f"{tag}_sharpe"] = item.get("sharpe")
        out[f"{tag}_max_drawdown"] = item.get("max_drawdown")
        out[f"{tag}_win_ratio"] = item.get("win_ratio")
        out[f"{tag}_open_count"] = item.get("open_count")
    return out


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    def f(key: str, default: float = -9999.0) -> float:
        try:
            value = row.get(key)
            return float(value) if value not in (None, "") else default
        except Exception:
            return default

    return (f("full_annual"), f("full_sharpe"), -f("full_max_drawdown", 9999.0))


def _write_report(summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 最新 formal L4 Top3 次日开盘补位优化研究",
        "",
        "## 说明",
        "",
        "- 本轮只做 research-only 回测，不修改任何生产策略参数。",
        "- 输入为 latest formal L4 Top3 主线评分表，但不再只取原始 Top3，而是先保留更深候选池，再按 buy-day open gap 过滤后补满 Top3。",
        "- 自适应滑点订单额改为 `70W 总资产 * 25% 单票仓位 = 17.5W`。",
        "- 不启用盘中风控，只比较 open_only 口径下的次日开盘重排补位效果。",
        "",
        "## Top 结果",
        "",
        "| name | pool | full annual | full sharpe | full mdd | ytd2026 annual | recent60 annual | signal_rows | below_target_days |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows[:8]:
        lines.append(
            "| {name} | {pool} | {fa} | {fs} | {fm} | {ya} | {ra} | {sr} | {bd} |".format(
                name=row["name"],
                pool=row.get("pool_depth", ""),
                fa="" if row.get("full_annual") is None else f"{float(row['full_annual']):.6f}",
                fs="" if row.get("full_sharpe") is None else f"{float(row['full_sharpe']):.6f}",
                fm="" if row.get("full_max_drawdown") is None else f"{float(row['full_max_drawdown']):.6f}",
                ya="" if row.get("ytd2026_annual") is None else f"{float(row['ytd2026_annual']):.6f}",
                ra="" if row.get("recent60_annual") is None else f"{float(row['recent60_annual']):.6f}",
                sr=row.get("signal_rows", ""),
                bd=row.get("below_target_days", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 结论口径",
            "",
            "- 当前研究比较的是同一条 latest formal L4 -> Top3 -> open_only 主线，在 buy-day open gap 过滤后从更深候选池补位的相对变化。",
            "- 若补位后仍无法优于基线，则说明问题不在“空仓”而在分数结构本身。",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for case in CASES:
        signal_file, signal_meta = _make_signal(case)
        rows = [_run_backtest(case, signal_file, *FULL_SLICE)]
        rows.extend(_run_backtest(case, signal_file, tag, start, end) for tag, start, end in CHECK_SLICES)
        detail_rows.extend([{**row, **signal_meta, "signal_file": str(signal_file)} for row in rows])
        summary_rows.append(_summary_row(case, signal_meta, rows))
        summary_rows.sort(key=_sort_key, reverse=True)
        _write_rows(OUT_CSV, summary_rows)
        _write_rows(OUT_DETAIL, detail_rows)
        _write_json(OUT_JSON, summary_rows)
        print(
            json.dumps(
                {"case": case["name"], "full_annual": rows[0].get("annual"), "full_sharpe": rows[0].get("sharpe")},
                ensure_ascii=False,
            ),
            flush=True,
        )
    summary_rows.sort(key=_sort_key, reverse=True)
    _write_rows(OUT_CSV, summary_rows)
    _write_rows(OUT_DETAIL, detail_rows)
    _write_json(OUT_JSON, summary_rows)
    _write_report(summary_rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "case_count": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
