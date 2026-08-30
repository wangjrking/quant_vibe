from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import duckdb
from stock_daily_data_route import resolve_stock_daily_duckdb_path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_multitop_ddloose_20260629"
BASE_REPORT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_fullwindow_20260629"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DUCKDB = resolve_stock_daily_duckdb_path(require_exists=True)
SCORE_DB = BASE_REPORT_DIR / "scores" / "grid_scores.duckdb"
BASE_TABLE = "score_w72_23_05_amt150_mv30__hold3m5_c097_e096_ddtight_pos65"

SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-26 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-26 15:30:00"),
    ("recent60", "2026-03-25 09:00:00", "2026-06-26 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-26 15:30:00"),
]

CASES = [
    {"name": "top2_pos30", "topn": 2, "max_positions": 2, "target": 0.30},
    {"name": "top2_pos35", "topn": 2, "max_positions": 2, "target": 0.35},
    {"name": "top3_pos20", "topn": 3, "max_positions": 3, "target": 0.20},
    {"name": "top3_pos25", "topn": 3, "max_positions": 3, "target": 0.25},
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


def _ensure_signal(case: dict[str, Any]) -> dict[str, Any]:
    case_key = f"w72_23_05_amt150_mv30_hold3m5_pc150_ddloose_{case['name']}"
    score_table = f"score_{case_key}"
    signal_file = REPORT_DIR / "signals" / f"{case_key}.csv"
    if signal_file.exists():
        with signal_file.open("r", encoding="utf-8-sig") as file:
            return {"case_key": case_key, "score_table": score_table, "signal_file": signal_file, "signal_rows": sum(1 for _ in file) - 1}
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{MARKET_DUCKDB.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(f"ATTACH '{SCORE_DB.as_posix()}' AS scoredb")
        con.execute(f'DROP TABLE IF EXISTS scoredb."{score_table}"')
        con.execute(f'CREATE TABLE scoredb."{score_table}" AS SELECT * FROM scoredb."{BASE_TABLE}"')
        sql = f"""
        WITH market_dates AS (
            SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA
        ),
        next_dates AS (
            SELECT trade_date AS signal_date,
                   lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                   max(trade_date) OVER () AS latest_market_date
            FROM market_dates
        ),
        scored AS (
            SELECT
                s.trade_date AS signal_date,
                nd.buy_date,
                nd.latest_market_date,
                s.stock_code,
                md.name,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                s.pred_prob AS entry_score
            FROM scoredb."{BASE_TABLE}" s
            LEFT JOIN marketdb.STOCK_DAILY_DATA md
              ON s.trade_date = md.trade_date AND s.stock_code = md.stock_code
            LEFT JOIN next_dates nd
              ON s.trade_date = nd.signal_date
            WHERE NOT (s.stock_code LIKE '%.BJ' OR substr(s.stock_code, 1, 1) IN ('4', '8'))
              AND NOT (
                upper(coalesce(md.name, '')) LIKE 'ST%%'
                OR upper(coalesce(md.name, '')) LIKE '*ST%%'
                OR coalesce(md.ST_TYPE_name, '') LIKE '%风险%'
                OR (try_cast(md.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(md.ST_TYPE AS DOUBLE) <> 0)
              )
              AND NOT (
                instr(coalesce(md.name, ''), '退市') > 0
                OR coalesce(md.name, '') LIKE '退%%'
                OR coalesce(md.name, '') LIKE '%%退'
              )
              AND coalesce(try_cast(md.limit_times AS DOUBLE), 0.0) = 0.0
              AND try_cast(md.amount AS DOUBLE) >= 150000.0
              AND try_cast(md.total_mv AS DOUBLE) >= 300000.0
              AND try_cast(md.close AS DOUBLE) <= 150.0
        ),
        with_buy_checks AS (
            SELECT
                s.*,
                CASE
                    WHEN s.buy_date IS NULL THEN TRUE
                    WHEN bm.stock_code IS NULL THEN FALSE
                    WHEN upper(coalesce(bm.name, '')) LIKE 'ST%%' THEN FALSE
                    WHEN upper(coalesce(bm.name, '')) LIKE '*ST%%' THEN FALSE
                    WHEN coalesce(bm.ST_TYPE_name, '') LIKE '%风险%' THEN FALSE
                    WHEN try_cast(bm.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(bm.ST_TYPE AS DOUBLE) <> 0 THEN FALSE
                    WHEN instr(coalesce(bm.name, ''), '退市') > 0 THEN FALSE
                    WHEN coalesce(try_cast(bm.limit_times AS DOUBLE), 0.0) > 0.0 THEN FALSE
                    WHEN try_cast(bm.pre_close AS DOUBLE) IS NULL OR try_cast(bm.open AS DOUBLE) IS NULL THEN FALSE
                    WHEN try_cast(bm.pre_close AS DOUBLE) <= 0 OR try_cast(bm.open AS DOUBLE) <= 0 THEN FALSE
                    WHEN try_cast(bm.open AS DOUBLE) >= try_cast(bm.pre_close AS DOUBLE) * (
                        1.0 + CASE WHEN substr(s.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                    ) * 0.995 THEN FALSE
                    WHEN try_cast(bm.open AS DOUBLE) > 150.0 THEN FALSE
                    ELSE TRUE
                END AS buy_day_ok
            FROM scored s
            LEFT JOIN marketdb.STOCK_DAILY_DATA bm
              ON s.buy_date = bm.trade_date AND s.stock_code = bm.stock_code
        )
        SELECT
            signal_date,
            coalesce(buy_date, '') AS buy_date,
            CASE
                WHEN stock_code LIKE '%.SH' THEN 'SHSE.' || substr(stock_code, 1, 6)
                WHEN stock_code LIKE '%.SZ' THEN 'SZSE.' || substr(stock_code, 1, 6)
                WHEN substr(stock_code, 1, 1) IN ('6', '9') THEN 'SHSE.' || substr(stock_code, 1, 6)
                ELSE 'SZSE.' || substr(stock_code, 1, 6)
            END AS symbol,
            stock_code,
            name,
            row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) AS rank,
            entry_score AS pred_prob,
            entry_score,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            printf('%.5f', {case['target']}) AS target_pct,
            3 AS holding_days,
            5 AS max_holding_days,
            '0.96000' AS score_exit_entry_ratio,
            1 AS min_holding_days_before_score_exit,
            '0.97000' AS score_continue_entry_ratio,
            '0.05000' AS signal_stop_loss_pct,
            '0.08000' AS signal_take_profit_pct,
            '{case_key}' AS strategy_variant,
            'amt150000_mv300000_pc150_top{case['topn']}' AS filter_name,
            'w72_23_05_amt150_mv30' AS entry_weight_name,
            'hold3m5_c097_e096_ddloose' AS dynamic_hold_name,
            buy_day_ok AS buy_day_market_available,
            buy_day_ok AS buy_day_hard_gate_complete,
            false AS buy_day_st_rejected,
            false AS buy_day_open_limit_up_rejected,
            latest_market_date
        FROM with_buy_checks
        WHERE buy_day_ok
        QUALIFY row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) <= {case['topn']}
        ORDER BY signal_date, rank
        """
        rows = con.execute(sql).fetchdf().to_dict("records")
        _write_rows(signal_file, rows)
    finally:
        con.close()
    conn = duckdb.connect(str(SCORE_DB))
    try:
        safe_index = re.sub(r"[^A-Za-z0-9_]", "_", f"idx_{score_table}_date_code")
        conn.execute(f'CREATE INDEX IF NOT EXISTS "{safe_index}" ON "{score_table}" (trade_date, stock_code)')
    finally:
        conn.close()
    return {"case_key": case_key, "score_table": score_table, "signal_file": signal_file, "signal_rows": len(rows)}


def _env(sell_mult: str) -> dict[str, str]:
    return {
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
        "GM_INTRADAY_RISK_MODE": "1",
        "GM_INTRADAY_REPLACE_BUY": "0",
        "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
        "GM_EQUITY_DD_RISK_MODE": "1",
        "GM_EQUITY_DD_RESIZE_EXISTING": "0",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
        "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
        "GM_EQUITY_DD_SOFT_SCALE": "0.80",
        "GM_EQUITY_DD_HARD_SCALE": "0.60",
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.97",
        "GM_SCORE_EXIT_ENTRY_RATIO": "0.96",
        "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
        "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
        "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
        "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
        "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
        "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
        "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
        "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": sell_mult,
    }


def _run(meta: dict[str, Any], case: dict[str, Any], sell_mult: str, tag: str, start: str, end: str) -> dict[str, Any]:
    mult_name = sell_mult.replace(".", "p")
    log_file = REPORT_DIR / "logs" / f"{meta['case_key']}__sell{mult_name}__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(_env(sell_mult))
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir", str(STRATEGY_DIR),
            "--signal-file", str(meta["signal_file"]),
            "--log-file", str(log_file),
            "--max-positions", str(case["max_positions"]),
            "--holding-days", "3",
            "--max-holding-days", "5",
            "--target-position-pct", str(case["target"]),
            "--score-db", str(SCORE_DB),
            "--score-table", meta["score_table"],
            "--market-db", str(MARKET_DUCKDB),
            "--backtest-start", start,
            "--backtest-end", end,
            "--backtest-adjust", "none",
            "--backtest-initial-cash", "600000",
            "--backtest-slippage-ratio", "0",
            "--stop-loss-pct", "0.05",
            "--take-profit-pct", "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "case_key": meta["case_key"],
        "sell_mult": sell_mult,
        "slice": tag,
        "returncode": returncode,
        "signal_rows": meta["signal_rows"],
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def main() -> None:
    detail: list[dict[str, Any]] = []
    for case in CASES:
        meta = _ensure_signal(case)
        for sell_mult in ["0.0", "0.25"]:
            for tag, start, end in SLICES:
                detail.append(_run(meta, case, sell_mult, tag, start, end))
                _write_rows(REPORT_DIR / "multitop_detail.csv", detail)
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in detail:
        item = summary.setdefault((row["case_key"], row["sell_mult"]), {"case_key": row["case_key"], "sell_mult": row["sell_mult"], "signal_rows": row["signal_rows"]})
        for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count", "log_file"]:
            item[f"{row['slice']}_{key}"] = row.get(key)
    _write_rows(REPORT_DIR / "multitop_summary.csv", list(summary.values()))
    print("wrote", REPORT_DIR / "multitop_summary.csv")


if __name__ == "__main__":
    main()
