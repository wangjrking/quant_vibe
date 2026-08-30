from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
from prediction_manifest import load_prediction_source_manifest
from stock_daily_data_route import resolve_stock_daily_duckdb_path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DUCKDB = resolve_stock_daily_duckdb_path(require_exists=True)
SCORE_DB = REPORT_DIR / "scores" / "blend_shape_scores.duckdb"
OUT_SIGNAL_DIR = REPORT_DIR / "blend_shape_signals"
OUT_LOG_DIR = REPORT_DIR / "blend_shape_logs"
OUT_CSV = REPORT_DIR / "blend_shape_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "blend_shape_probe_20260630.json"


MANIFESTS = {
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


CASES: list[dict[str, Any]] = [
    {"name": "blend_72_23_05_g1_top5_dyn", "w10": 0.72, "w5": 0.23, "w3": 0.05, "gamma": 1.0, "topn": 5, "cap": 0.15, "target_mode": "mild"},
    {"name": "blend_80_15_05_g1p4_top5_dyn", "w10": 0.80, "w5": 0.15, "w3": 0.05, "gamma": 1.4, "topn": 5, "cap": 0.15, "target_mode": "mild"},
    {"name": "blend_80_15_05_g2_top5_dyn", "w10": 0.80, "w5": 0.15, "w3": 0.05, "gamma": 2.0, "topn": 5, "cap": 0.15, "target_mode": "mild"},
    {"name": "blend_65_30_05_g1p4_top5_dyn", "w10": 0.65, "w5": 0.30, "w3": 0.05, "gamma": 1.4, "topn": 5, "cap": 0.15, "target_mode": "mild"},
    {"name": "blend_55_35_10_g1p4_top5_dyn", "w10": 0.55, "w5": 0.35, "w3": 0.10, "gamma": 1.4, "topn": 5, "cap": 0.15, "target_mode": "mild"},
    {"name": "blend_45_45_10_g1p4_top5_dyn", "w10": 0.45, "w5": 0.45, "w3": 0.10, "gamma": 1.4, "topn": 5, "cap": 0.15, "target_mode": "mild"},
    {"name": "blend_60_25_15_g1p4_top5_dyn", "w10": 0.60, "w5": 0.25, "w3": 0.15, "gamma": 1.4, "topn": 5, "cap": 0.15, "target_mode": "mild"},
    {"name": "blend_72_23_05_g1p4_top4_t18", "w10": 0.72, "w5": 0.23, "w3": 0.05, "gamma": 1.4, "topn": 4, "cap": 0.18, "target_mode": "fixed18"},
    {"name": "blend_80_15_05_g1p4_top4_t18", "w10": 0.80, "w5": 0.15, "w3": 0.05, "gamma": 1.4, "topn": 4, "cap": 0.18, "target_mode": "fixed18"},
    {"name": "blend_65_30_05_g1p4_top4_t18", "w10": 0.65, "w5": 0.30, "w3": 0.05, "gamma": 1.4, "topn": 4, "cap": 0.18, "target_mode": "fixed18"},
    {"name": "blend_72_23_05_g1p4_top5_ma60", "w10": 0.72, "w5": 0.23, "w3": 0.05, "gamma": 1.4, "topn": 5, "cap": 0.15, "target_mode": "mild_ma60"},
    {"name": "blend_80_15_05_g1p4_top5_ma60", "w10": 0.80, "w5": 0.15, "w3": 0.05, "gamma": 1.4, "topn": 5, "cap": 0.15, "target_mode": "mild_ma60"},
]


SELL_RULE = {
    "holding_days": 1,
    "max_holding_days": 2,
    "score_exit": 0.99,
    "score_continue": 0.995,
    "min_score_exit_days": 1,
    "max_daily_sells": 1,
    "day_drop_ratio": 0.995,
}


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


def _load_manifest(label: str) -> dict[str, Any]:
    source = load_prediction_source_manifest(MANIFESTS[label], require_approved=True, allow_legacy=False)
    db_path = Path(source["db_path"])
    table = str(source["table"])
    market_db_path = Path(source["market_db_path"]) if source.get("market_db_path") else None
    if market_db_path and market_db_path.resolve() != MARKET_DUCKDB.resolve():
        raise RuntimeError(
            f"{MANIFESTS[label]} market_db_path mismatch: {market_db_path} != {MARKET_DUCKDB}"
        )
    return {"path": str(MANIFESTS[label]), "db_path": str(db_path), "table": table}


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _score_table(case: dict[str, Any]) -> str:
    return "score_" + _safe_name(case["name"])


def _target_expr(case: dict[str, Any]) -> str:
    if case["target_mode"] == "fixed18":
        return "0.18"
    base = """
        CASE
          WHEN rank_3d < 0.45 OR rank_5d < 0.45 THEN 0.09
          WHEN rank_3d < 0.65 OR rank_5d < 0.65 THEN 0.12
          ELSE 0.15
        END
    """
    if case["target_mode"] == "mild":
        return base
    if case["target_mode"] == "mild_ma60":
        return f"CASE WHEN index_close < index_ma60 THEN ({base}) * 0.4 ELSE ({base}) END"
    raise ValueError(f"unknown target_mode: {case['target_mode']}")


def _ensure_signal(case: dict[str, Any]) -> tuple[Path, str, dict[str, Any]]:
    sources = {label: _load_manifest(label) for label in ("3d", "5d", "10d")}
    table = _score_table(case)
    signal_file = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    if signal_file.exists():
        rows = list(csv.DictReader(signal_file.open("r", encoding="utf-8-sig", newline="")))
        counts = Counter(row["signal_date"] for row in rows)
        return signal_file, table, {"signal_rows": len(rows), "signal_days": len(counts), "days_below_target": sum(1 for value in counts.values() if value < int(case["topn"]))}

    OUT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    gamma = float(case["gamma"])
    rank3 = f"pow(greatest(rank_3d, 0.0), {gamma})"
    rank5 = f"pow(greatest(rank_5d, 0.0), {gamma})"
    rank10 = f"pow(greatest(rank_10d, 0.0), {gamma})"
    score_expr = f"{float(case['w10'])} * {rank10} + {float(case['w5'])} * {rank5} + {float(case['w3'])} * {rank3}"
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{Path(sources['10d']['db_path']).as_posix()}' AS l4_10d (READ_ONLY)")
        con.execute(f"ATTACH '{Path(sources['5d']['db_path']).as_posix()}' AS l4_5d (READ_ONLY)")
        con.execute(f"ATTACH '{Path(sources['3d']['db_path']).as_posix()}' AS l4_3d (READ_ONLY)")
        con.execute(f"ATTACH '{MARKET_DUCKDB.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(f"ATTACH '{SCORE_DB.as_posix()}' AS scoredb")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW joined_scores AS
            WITH base AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{sources['10d']['table']}" p10
                INNER JOIN l4_5d."{sources['5d']['table']}" p5
                    ON p10.trade_date = p5.trade_date AND p10.stock_code = p5.stock_code
                INNER JOIN l4_3d."{sources['3d']['table']}" p3
                    ON p10.trade_date = p3.trade_date AND p10.stock_code = p3.stock_code
            ),
            ranked AS (
                SELECT
                    trade_date,
                    stock_code,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS rank_3d,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS rank_5d,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS rank_10d
                FROM base
            )
            SELECT *, {score_expr} AS entry_score
            FROM ranked
            """
        )
        con.execute(f'DROP TABLE IF EXISTS scoredb."{table}"')
        con.execute(
            f"""
            CREATE TABLE scoredb."{table}" AS
            SELECT trade_date, stock_code, entry_score AS pred_prob
            FROM joined_scores
            """
        )
        selected_sql = f"""
        WITH market_with_prev AS (
            SELECT
                *,
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
        index_ma AS (
            SELECT
                trade_date,
                max(index_2000_close) AS index_close,
                avg(max(index_2000_close)) OVER (ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS index_ma60
            FROM marketdb.STOCK_DAILY_DATA
            GROUP BY trade_date
        ),
        scored AS (
            SELECT
                js.trade_date AS signal_date,
                nd.buy_date,
                nd.latest_market_date,
                js.stock_code,
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
                ix.index_close,
                ix.index_ma60,
                js.pred_3d,
                js.pred_5d,
                js.pred_10d,
                js.rank_3d,
                js.rank_5d,
                js.rank_10d,
                js.entry_score
            FROM joined_scores js
            LEFT JOIN market_with_prev md
              ON js.trade_date = md.trade_date AND js.stock_code = md.stock_code
            LEFT JOIN next_dates nd ON js.trade_date = nd.signal_date
            LEFT JOIN index_ma ix ON js.trade_date = ix.trade_date
            WHERE NOT (js.stock_code LIKE '%.BJ' OR substr(js.stock_code, 1, 1) IN ('4', '8'))
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
              AND coalesce(((1 + try_cast(md.pct_chg AS DOUBLE) / 100.0)
                 * (1 + try_cast(md.prev_pct_chg AS DOUBLE) / 100.0) - 1.0), -999.0) < 0.10
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
            pred_3d,
            pred_5d,
            pred_10d,
            rank_3d,
            rank_5d,
            rank_10d,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            pct_chg,
            prev_pct_chg,
            two_day_ret,
            printf('%.5f', {_target_expr(case)}) AS target_pct,
            {SELL_RULE['holding_days']} AS holding_days,
            {SELL_RULE['max_holding_days']} AS max_holding_days,
            printf('%.5f', {SELL_RULE['score_exit']}) AS score_exit_entry_ratio,
            {SELL_RULE['min_score_exit_days']} AS min_holding_days_before_score_exit,
            printf('%.5f', {SELL_RULE['score_continue']}) AS score_continue_entry_ratio,
            '0.05000' AS signal_stop_loss_pct,
            '0.08000' AS signal_take_profit_pct,
            '{case['name']}' AS strategy_variant,
            '{case['name']}' AS filter_name,
            'blend_shape' AS entry_weight_name,
            'daily_h1m2_e099_c0995' AS dynamic_hold_name,
            buy_day_ok AS buy_day_market_available,
            buy_day_ok AS buy_day_hard_gate_complete,
            false AS buy_day_st_rejected,
            false AS buy_day_open_limit_up_rejected,
            latest_market_date
        FROM with_buy_checks
        WHERE buy_day_ok
        QUALIFY row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) <= {int(case['topn'])}
        ORDER BY signal_date, rank
        """
        rows = con.execute(selected_sql).fetchdf().to_dict("records")
        _write_rows(signal_file, rows)
    finally:
        con.close()
    conn = duckdb.connect(str(SCORE_DB))
    try:
        index_name = re.sub(r"[^A-Za-z0-9_]", "_", f"idx_{table}_date_code")
        conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}" (trade_date, stock_code)')
    finally:
        conn.close()
    counts = Counter(str(row["signal_date"]) for row in rows)
    return signal_file, table, {"signal_rows": len(rows), "signal_days": len(counts), "days_below_target": sum(1 for value in counts.values() if value < int(case["topn"]))}


def _run(case: dict[str, Any]) -> dict[str, Any]:
    signal_file, score_table, signal_meta = _ensure_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(SELL_RULE["max_daily_sells"]),
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
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
                "GM_EQUITY_DD_SOFT_SCALE": "0.80",
                "GM_EQUITY_DD_HARD_SCALE": "0.60",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(SELL_RULE["min_score_exit_days"]),
                "GM_SCORE_EXIT_ENTRY_RATIO": str(SELL_RULE["score_exit"]),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(SELL_RULE["score_continue"]),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": str(SELL_RULE["day_drop_ratio"]),
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
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
            str(SELL_RULE["holding_days"]),
            "--max-holding-days",
            str(SELL_RULE["max_holding_days"]),
            "--target-position-pct",
            str(float(case["cap"])),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            score_table,
            "--market-db",
            str(MARKET_DUCKDB),
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
        "score_db": str(SCORE_DB),
        "score_table": score_table,
        "log_file": str(log_file),
        "intraday_risk": 0,
        "note": "research-only blend/shape probe on active formal L4; production parameters unchanged",
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case)
        rows.append(row)
        _write_rows(OUT_CSV, rows)
        OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
