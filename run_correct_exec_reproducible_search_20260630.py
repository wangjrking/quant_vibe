from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
from l1_raw_data_route import resolve_l1_raw_duckdb_path
from prediction_manifest import load_prediction_source_manifest
from stock_daily_data_route import resolve_stock_daily_duckdb_path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_correct_exec_search_20260630"
STRATEGY_DIR = REPORT_DIR / "code_snapshot_correct_exec"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DUCKDB = resolve_stock_daily_duckdb_path(require_exists=True)
L1_DAILY_DUCKDB = resolve_l1_raw_duckdb_path("daily_data", require_exists=True)
EXEC_MARKET_DB = REPORT_DIR / "exec_market_raw_open.db"
SCORE_DB = REPORT_DIR / "scores" / "correct_exec_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_correct_exec"

MANIFESTS = {
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}

ENTRY = {
    "name": "w72_23_05_amt150_mv30",
    "w10d": 0.72,
    "w5d": 0.23,
    "w3d": 0.05,
    "amount_min": 150000.0,
    "total_mv_min": 300000.0,
    "price_cap_signal_close": 150.0,
    "price_cap_buy_open": 150.0,
}

BUY_RULES = [
    {"name": "cool2d10", "two_day_cap": 0.10, "combo_cap": None, "turnover_floor": None},
    {"name": "cool2d15_turn15", "two_day_cap": None, "combo_cap": 0.15, "turnover_floor": 15.0},
    {"name": "cool2d20", "two_day_cap": 0.20, "combo_cap": None, "turnover_floor": None},
]

SELL_RULES = [
    {"name": "h1m2_e098_c099", "holding_days": 1, "max_holding_days": 2, "exit": 0.98, "cont": 0.99, "daydrop": 0.997, "max_daily_sells": 2},
    {"name": "h2m3_e097_c098", "holding_days": 2, "max_holding_days": 3, "exit": 0.97, "cont": 0.98, "daydrop": 0.995, "max_daily_sells": 1},
    {"name": "h2m3_e098_c099", "holding_days": 2, "max_holding_days": 3, "exit": 0.98, "cont": 0.99, "daydrop": 0.995, "max_daily_sells": 1},
]

TOP_CASES = [
    {"name": "top3_pos25", "topn": 3, "max_positions": 3, "target": 0.25},
    {"name": "top3_pos30", "topn": 3, "max_positions": 3, "target": 0.30},
    {"name": "top5_pos15", "topn": 5, "max_positions": 5, "target": 0.15},
    {"name": "top5_pos18", "topn": 5, "max_positions": 5, "target": 0.18},
    {"name": "top5_pos20", "topn": 5, "max_positions": 5, "target": 0.20},
]

DD_MODES = [
    {"name": "dd_off", "enabled": "0", "soft": "0.08", "hard": "0.14", "recover": "0.04", "soft_scale": "0.80", "hard_scale": "0.60"},
    {"name": "dd_loose", "enabled": "1", "soft": "0.08", "hard": "0.14", "recover": "0.04", "soft_scale": "0.80", "hard_scale": "0.60"},
]

SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-29 15:30:00"),
    ("start_202407", "2024-07-01 09:00:00", "2026-06-29 15:30:00"),
    ("start_202501", "2025-01-02 09:00:00", "2026-06-29 15:30:00"),
    ("start_202507", "2025-07-01 09:00:00", "2026-06-29 15:30:00"),
    ("start_202601", "2026-01-05 09:00:00", "2026-06-29 15:30:00"),
]
RECENT_VALIDATION_TOP_N = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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


def _load_manifest(label: str) -> dict[str, str]:
    source = load_prediction_source_manifest(MANIFESTS[label], require_approved=True, allow_legacy=False)
    return {
        "manifest": str(source["manifest_path"]),
        "db_path": str(source["db_path"]),
        "table": str(source["table"]),
    }


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
            return eval(payload, {"__builtins__": {}}, {"datetime": dt})
    return None


def _cooldown_condition(rule: dict[str, Any]) -> str:
    checks: list[str] = []
    if rule["two_day_cap"] is not None:
        checks.append(f"coalesce(two_day_ret, -999.0) < {float(rule['two_day_cap'])}")
    if rule["combo_cap"] is not None:
        checks.append(
            f"(coalesce(two_day_ret, -999.0) < {float(rule['combo_cap'])} "
            f"OR coalesce(try_cast(turnover_rate AS DOUBLE), 0.0) < {float(rule['turnover_floor'])})"
        )
    return " AND ".join(checks) if checks else "TRUE"


def _case_key(top: dict[str, Any], buy: dict[str, Any], sell: dict[str, Any], dd: dict[str, Any]) -> str:
    return f"{ENTRY['name']}_{top['name']}_{buy['name']}_{sell['name']}_{dd['name']}"


def _duckdb_table_exists(db_path: Path, table: str) -> bool:
    if not db_path.exists():
        return False
    with duckdb.connect(str(db_path), read_only=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()
    return bool(row and row[0])


def _ensure_score_and_signal(top: dict[str, Any], buy: dict[str, Any], sell: dict[str, Any], dd: dict[str, Any]) -> dict[str, Any]:
    case_key = _case_key(top, buy, sell, dd)
    signal_file = REPORT_DIR / "signals" / f"{case_key}.csv"
    if signal_file.exists():
        rows = list(csv.DictReader(signal_file.open("r", encoding="utf-8-sig", newline="")))
        return {"case_key": case_key, "signal_file": signal_file, "signal_rows": len(rows), "score_table": SCORE_TABLE}

    sources = {label: _load_manifest(label) for label in ("3d", "5d", "10d")}
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    existing = _duckdb_table_exists(SCORE_DB, SCORE_TABLE)
    con = duckdb.connect()
    try:
        for label in ("3d", "5d", "10d"):
            con.execute(f"ATTACH '{sources[label]['db_path']}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{MARKET_DUCKDB.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(f"ATTACH '{L1_DAILY_DUCKDB.as_posix()}' AS l1_daily (READ_ONLY)")
        con.execute(f"ATTACH '{SCORE_DB.as_posix()}' AS scoredb")
        if not existing:
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
                        ON p10.trade_date = p5.trade_date
                       AND p10.stock_code = p5.stock_code
                    INNER JOIN l4_3d."{sources['3d']['table']}" p3
                        ON p10.trade_date = p3.trade_date
                       AND p10.stock_code = p3.stock_code
                )
                SELECT
                    trade_date,
                    stock_code,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS rank_3d,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS rank_5d,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS rank_10d,
                    {ENTRY['w10d']} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d)
                  + {ENTRY['w5d']} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d)
                  + {ENTRY['w3d']} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS entry_score
                FROM base
                """
            )
            con.execute(f'DROP TABLE IF EXISTS scoredb."{SCORE_TABLE}"')
            con.execute(
                f"""
                CREATE TABLE scoredb."{SCORE_TABLE}" AS
                SELECT trade_date, stock_code, entry_score AS pred_prob
                FROM joined_scores
                """
            )
        else:
            con.execute(
                f"""
                CREATE OR REPLACE TEMP VIEW joined_scores AS
                SELECT
                    s.trade_date,
                    s.stock_code,
                    NULL::DOUBLE AS pred_3d,
                    NULL::DOUBLE AS pred_5d,
                    NULL::DOUBLE AS pred_10d,
                    NULL::DOUBLE AS rank_3d,
                    NULL::DOUBLE AS rank_5d,
                    NULL::DOUBLE AS rank_10d,
                    s.pred_prob AS entry_score
                FROM scoredb."{SCORE_TABLE}" s
                """
            )

        sql = f"""
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
                js.pred_3d,
                js.pred_5d,
                js.pred_10d,
                js.rank_3d,
                js.rank_5d,
                js.rank_10d,
                js.entry_score
            FROM joined_scores js
            LEFT JOIN market_with_prev md
              ON js.trade_date = md.trade_date
             AND js.stock_code = md.stock_code
            LEFT JOIN next_dates nd
              ON js.trade_date = nd.signal_date
            WHERE NOT (js.stock_code LIKE '%.BJ' OR substr(js.stock_code, 1, 1) IN ('4', '8'))
              AND NOT (
                upper(coalesce(md.name, '')) LIKE 'ST%'
                OR upper(coalesce(md.name, '')) LIKE '*ST%'
                OR coalesce(md.ST_TYPE_name, '') LIKE '%风险%'
                OR (try_cast(md.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(md.ST_TYPE AS DOUBLE) <> 0)
              )
              AND NOT (
                instr(coalesce(md.name, ''), '退市') > 0
                OR coalesce(md.name, '') LIKE '退%'
              )
              AND coalesce(try_cast(md.limit_times AS DOUBLE), 0.0) = 0.0
              AND try_cast(md.amount AS DOUBLE) >= {ENTRY['amount_min']}
              AND try_cast(md.total_mv AS DOUBLE) >= {ENTRY['total_mv_min']}
              AND try_cast(md.close AS DOUBLE) <= {ENTRY['price_cap_signal_close']}
              AND {_cooldown_condition(buy)}
        ),
        with_buy_checks AS (
            SELECT
                s.*,
                bm.name AS buy_name,
                bm.ST_TYPE AS buy_st_type,
                bm.ST_TYPE_name AS buy_st_type_name,
                bm.limit_times AS buy_limit_times,
                br.open AS buy_raw_open,
                br.pre_close AS buy_raw_pre_close,
                CASE
                    WHEN s.buy_date IS NULL THEN TRUE
                    WHEN bm.stock_code IS NULL THEN FALSE
                    WHEN br.ts_code IS NULL THEN FALSE
                    WHEN upper(coalesce(bm.name, '')) LIKE 'ST%' THEN FALSE
                    WHEN upper(coalesce(bm.name, '')) LIKE '*ST%' THEN FALSE
                    WHEN coalesce(bm.ST_TYPE_name, '') LIKE '%风险%' THEN FALSE
                    WHEN try_cast(bm.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(bm.ST_TYPE AS DOUBLE) <> 0 THEN FALSE
                    WHEN instr(coalesce(bm.name, ''), '退市') > 0 THEN FALSE
                    WHEN coalesce(try_cast(bm.limit_times AS DOUBLE), 0.0) > 0.0 THEN FALSE
                    WHEN try_cast(br.pre_close AS DOUBLE) IS NULL OR try_cast(br.open AS DOUBLE) IS NULL THEN FALSE
                    WHEN try_cast(br.pre_close AS DOUBLE) <= 0 OR try_cast(br.open AS DOUBLE) <= 0 THEN FALSE
                    WHEN try_cast(br.open AS DOUBLE) >= try_cast(br.pre_close AS DOUBLE) * (
                        1.0 + CASE WHEN substr(s.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                    ) * 0.995 THEN FALSE
                    WHEN try_cast(br.open AS DOUBLE) > {ENTRY['price_cap_buy_open']} THEN FALSE
                    ELSE TRUE
                END AS buy_day_ok
            FROM scored s
            LEFT JOIN marketdb.STOCK_DAILY_DATA bm
              ON s.buy_date = bm.trade_date
             AND s.stock_code = bm.stock_code
            LEFT JOIN l1_daily.daily_data br
              ON s.buy_date = br.trade_date
             AND s.stock_code = br.ts_code
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
            buy_raw_open,
            buy_raw_pre_close,
            printf('%.5f', {float(top['target'])}) AS target_pct,
            {int(sell['holding_days'])} AS holding_days,
            {int(sell['max_holding_days'])} AS max_holding_days,
            printf('%.5f', {float(sell['exit'])}) AS score_exit_entry_ratio,
            1 AS min_holding_days_before_score_exit,
            printf('%.5f', {float(sell['cont'])}) AS score_continue_entry_ratio,
            '0.05000' AS signal_stop_loss_pct,
            '0.08000' AS signal_take_profit_pct,
            '{case_key}' AS strategy_variant,
            '{buy['name']}' AS filter_name,
            '{ENTRY['name']}' AS entry_weight_name,
            '{sell['name']}' AS dynamic_hold_name,
            buy_day_ok AS buy_day_market_available,
            buy_day_ok AS buy_day_hard_gate_complete,
            false AS buy_day_st_rejected,
            false AS buy_day_open_limit_up_rejected,
            latest_market_date
        FROM with_buy_checks
        WHERE buy_day_ok
        QUALIFY row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) <= {int(top['topn'])}
        ORDER BY signal_date, rank
        """
        rows = con.execute(sql).fetchdf().to_dict("records")
        _write_rows(signal_file, rows)
    finally:
        con.close()

    with duckdb.connect(str(SCORE_DB)) as score_conn:
        idx = re.sub(r"[^A-Za-z0-9_]", "_", f"idx_{SCORE_TABLE}_date_code")
        score_conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{SCORE_TABLE}" (trade_date, stock_code)')

    return {"case_key": case_key, "signal_file": signal_file, "signal_rows": len(rows), "score_table": SCORE_TABLE}


def _env(sell: dict[str, Any], dd: dict[str, Any]) -> dict[str, str]:
    return {
        "GM_OPEN_DAILY_SCORE_EXIT": "1",
        "GM_MAX_DAILY_SELLS": str(sell["max_daily_sells"]),
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
        "GM_EQUITY_DD_RISK_MODE": str(dd["enabled"]),
        "GM_EQUITY_DD_RESIZE_EXISTING": "0",
        "GM_EQUITY_DD_SOFT_TRIGGER": str(dd["soft"]),
        "GM_EQUITY_DD_HARD_TRIGGER": str(dd["hard"]),
        "GM_EQUITY_DD_RECOVER_TRIGGER": str(dd["recover"]),
        "GM_EQUITY_DD_SOFT_SCALE": str(dd["soft_scale"]),
        "GM_EQUITY_DD_HARD_SCALE": str(dd["hard_scale"]),
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        "GM_SCORE_CONTINUE_ENTRY_RATIO": str(sell["cont"]),
        "GM_SCORE_EXIT_ENTRY_RATIO": str(sell["exit"]),
        "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": str(sell["daydrop"]),
        "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
        "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
        "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
        "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
        "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
        "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
        "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
        "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
    }


def _run_one(meta: dict[str, Any], top: dict[str, Any], sell: dict[str, Any], dd: dict[str, Any], slice_tag: str, start: str, end: str) -> dict[str, Any]:
    log_file = REPORT_DIR / "logs" / f"{meta['case_key']}__{slice_tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(meta["signal_file"]),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(top["max_positions"]),
            "--holding-days",
            str(sell["holding_days"]),
            "--max-holding-days",
            str(sell["max_holding_days"]),
            "--target-position-pct",
            str(top["target"]),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            str(meta["score_table"]),
            "--market-db",
            str(EXEC_MARKET_DB),
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
        env = os.environ.copy()
        env.update(_env(sell, dd))
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "case_key": meta["case_key"],
        "slice": slice_tag,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
        "command": " ".join(cmd),
        "env_delta_json": json.dumps(_env(sell, dd), ensure_ascii=False, sort_keys=True),
    }


def _sell_skip_audit(log_file: str) -> dict[str, Any]:
    path = Path(log_file)
    if not path.exists():
        return {"sell_skip_count": None}
    count = 0
    first: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "SELL_SKIP" in line:
            count += 1
            if len(first) < 5:
                first.append(line.strip())
    return {"sell_skip_count": count, "sell_skip_first": first}


def _admission(full: dict[str, Any], recent_rows: list[dict[str, Any]], signal_meta: dict[str, Any]) -> dict[str, Any]:
    annual = _float(full.get("annual")) or -999.0
    sharpe = _float(full.get("sharpe")) or -999.0
    mdd = _float(full.get("max_drawdown")) or 999.0
    open_count = int(full.get("open_count") or 0)
    recent_annuals = [_float(row.get("annual")) for row in recent_rows if row.get("returncode") == 0]
    recent_annuals = [v for v in recent_annuals if v is not None]
    recent_min = min(recent_annuals) if recent_annuals else None
    sell_skip = _sell_skip_audit(str(full.get("log_file")))
    has_recent_validation = len(recent_annuals) >= 4
    hard_pass = (
        annual >= 1.0
        and sharpe >= 1.0
        and mdd <= 0.40
        and open_count >= 100
        and has_recent_validation
        and (recent_min is not None and recent_min >= 0.0)
        and int(signal_meta.get("days_below_target") or 0) == 0
        and int(sell_skip.get("sell_skip_count") or 0) < 100
        and bool(signal_meta.get("repro_package_path"))
        and bool(signal_meta.get("repro_package_sha256_manifest"))
    )
    return {
        "admission_pass": hard_pass,
        "annual_threshold_pass": annual >= 1.0,
        "sharpe_threshold_pass": sharpe >= 1.0,
        "drawdown_threshold_pass": mdd <= 0.40,
        "trade_count_pass": open_count >= 100,
        "recent_start_min_annual": recent_min,
        "recent_validation_complete": has_recent_validation,
        "recent_start_pass": has_recent_validation and recent_min is not None and recent_min >= 0.0,
        "coverage_pass": int(signal_meta.get("days_below_target") or 0) == 0,
        "repro_package_pass": bool(signal_meta.get("repro_package_path"))
        and bool(signal_meta.get("repro_package_sha256_manifest")),
        **sell_skip,
    }


def _write_report(summary_rows: list[dict[str, Any]]) -> None:
    valid = [r for r in summary_rows if r.get("full_returncode") == 0 and r.get("full_annual") is not None]
    valid.sort(key=lambda r: (bool(r.get("admission_pass")), float(r["full_annual"]), float(r.get("full_sharpe") or 0)), reverse=True)
    lines = [
        "# correct_exec 可复现策略搜索报告",
        "",
        "## 口径",
        "",
        "- research-only，不修改生产 registry，不发布正式信号。",
        "- 输入为当前 active formal L4 3D/5D/10D DuckDB manifest。",
        "- 信号生成用 raw `daily_data.open/pre_close` 做买入日涨停门禁。",
        "- 掘金执行代码用 raw open 执行行情库，避免复权 open 与未复权 pre_close 混用。",
        "- 禁用盘中风控；启用账户持仓同步；买入状态只在账户同步确认后进入持仓状态。",
        "",
        "## 候选结果",
        "",
        "| 候选 | 准入 | 年化 | Sharpe | 最大回撤 | 近期最差年化 | 开仓 | SELL_SKIP | 信号日不足 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in valid:
        lines.append(
            "| {case} | {admit} | {annual:.2%} | {sharpe:.3f} | {mdd:.2%} | {recent:.2%} | {open_count} | {skip} | {shortage} |".format(
                case=row["case_key"],
                admit="通过" if row.get("admission_pass") else "未通过",
                annual=float(row["full_annual"]),
                sharpe=float(row.get("full_sharpe") or 0.0),
                mdd=float(row.get("full_max_drawdown") or 0.0),
                recent=float(row.get("recent_start_min_annual") or 0.0),
                open_count=row.get("full_open_count", ""),
                skip=row.get("sell_skip_count", ""),
                shortage=row.get("days_below_target", ""),
            )
        )
    if valid:
        best_pass = next((r for r in valid if r.get("admission_pass")), None)
        best = best_pass or valid[0]
        lines.extend(
            [
                "",
                "## 当前选择",
                "",
                f"- 候选：`{best['case_key']}`",
                f"- 是否通过准入：{'是' if best.get('admission_pass') else '否'}",
                f"- 年化：{float(best['full_annual']):.2%}",
                f"- Sharpe：{float(best.get('full_sharpe') or 0.0):.3f}",
                f"- 最大回撤：{float(best.get('full_max_drawdown') or 0.0):.2%}",
                f"- 信号文件：`{best['signal_file']}`",
                f"- 掘金日志：`{best['full_log_file']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 证据文件",
            "",
            f"- 汇总 CSV：`{REPORT_DIR / 'correct_exec_search_summary.csv'}`",
            f"- 明细 CSV：`{REPORT_DIR / 'correct_exec_search_detail.csv'}`",
            f"- manifest：`{REPORT_DIR / 'correct_exec_search_manifest.json'}`",
        ]
    )
    (REPORT_DIR / "correct_exec_search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [JUEJIN_PYTHON, STRATEGY_DIR / "main.py", PROD_DUCKDB, EXEC_MARKET_DB]:
        if not path.exists():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {label: _load_manifest(label) for label in ("3d", "5d", "10d")}
    run_manifest = {
        "schema_version": 1,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "research_only_no_production_change",
        "strategy_dir": str(STRATEGY_DIR),
        "strategy_main_sha256": _sha256(STRATEGY_DIR / "main.py"),
        "prod_duckdb": str(PROD_DUCKDB),
        "exec_market_db": str(EXEC_MARKET_DB),
        "exec_market_db_sha256": _sha256(EXEC_MARKET_DB),
        "score_db": str(SCORE_DB),
        "manifests": manifests,
        "entry": ENTRY,
        "top_cases": TOP_CASES,
        "buy_rules": BUY_RULES,
        "sell_rules": SELL_RULES,
        "dd_modes": DD_MODES,
        "slices": SLICES,
    }
    (REPORT_DIR / "correct_exec_search_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    if (REPORT_DIR / "correct_exec_search_summary.json").exists():
        try:
            summary_rows = json.loads((REPORT_DIR / "correct_exec_search_summary.json").read_text(encoding="utf-8"))
        except Exception:
            summary_rows = []
    done = {row.get("case_key") for row in summary_rows}

    for top in TOP_CASES:
        for buy in BUY_RULES:
            for sell in SELL_RULES:
                for dd_mode in DD_MODES:
                    case_key = _case_key(top, buy, sell, dd_mode)
                    if case_key in done:
                        continue
                    meta = _ensure_score_and_signal(top, buy, sell, dd_mode)
                    counts = Counter()
                    with Path(meta["signal_file"]).open("r", encoding="utf-8-sig", newline="") as file:
                        for row in csv.DictReader(file):
                            counts[row["signal_date"]] += 1
                    signal_meta = {
                        "signal_days": len(counts),
                        "signal_rows": meta["signal_rows"],
                        "target_topn": top["topn"],
                        "days_below_target": sum(1 for count in counts.values() if count < int(top["topn"])),
                        "min_per_day": min(counts.values()) if counts else 0,
                        "max_per_day": max(counts.values()) if counts else 0,
                        "signal_sha256": _sha256(Path(meta["signal_file"])),
                    }
                    tag, start, end = SLICES[0]
                    slice_rows = [_run_one(meta, top, sell, dd_mode, tag, start, end)]
                    detail_rows.extend(slice_rows)
                    by_slice = {row["slice"]: row for row in slice_rows}
                    full = by_slice["full"]
                    recent_rows = [row for row in slice_rows if row["slice"] != "full"]
                    admission = _admission(full, recent_rows, signal_meta)
                    out = {
                        "case_key": case_key,
                        "top_case": top["name"],
                        "topn": top["topn"],
                        "max_positions": top["max_positions"],
                        "target_pct_each": top["target"],
                        "gross_target_pct": top["target"] * top["max_positions"],
                        "buy_rule": buy["name"],
                        "sell_rule": sell["name"],
                        "dd_mode": dd_mode["name"],
                        "holding_days": sell["holding_days"],
                        "max_holding_days": sell["max_holding_days"],
                        "score_exit_ratio": sell["exit"],
                        "score_continue_ratio": sell["cont"],
                        "daydrop": sell["daydrop"],
                        "full_returncode": full.get("returncode"),
                        "full_annual": full.get("annual"),
                        "full_pnl_ratio": full.get("pnl_ratio"),
                        "full_sharpe": full.get("sharpe"),
                        "full_max_drawdown": full.get("max_drawdown"),
                        "full_win_ratio": full.get("win_ratio"),
                        "full_open_count": full.get("open_count"),
                        "full_close_count": full.get("close_count"),
                        "full_log_file": full.get("log_file"),
                        "signal_file": str(meta["signal_file"]),
                        "score_table": meta["score_table"],
                        **signal_meta,
                        **admission,
                    }
                    summary_rows.append(out)
                    _write_rows(REPORT_DIR / "correct_exec_search_detail.csv", detail_rows)
                    _write_rows(REPORT_DIR / "correct_exec_search_summary.csv", summary_rows)
                    (REPORT_DIR / "correct_exec_search_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
                    _write_report(summary_rows)
                    print(json.dumps(out, ensure_ascii=False), flush=True)

    # 第二阶段：只对全周期较好的候选补充近期开仓切片，避免把明显差的参数也跑完整准入。
    summary_by_key = {row["case_key"]: row for row in summary_rows}
    top_for_recent = sorted(
        [
            row
            for row in summary_rows
            if row.get("full_returncode") == 0
            and row.get("full_annual") is not None
            and float(row.get("full_max_drawdown") or 9.0) <= 0.45
        ],
        key=lambda row: (float(row.get("full_annual") or -999.0), float(row.get("full_sharpe") or -999.0)),
        reverse=True,
    )[:RECENT_VALIDATION_TOP_N]
    param_lookup = {
        _case_key(top, buy, sell, dd_mode): (top, buy, sell, dd_mode)
        for top in TOP_CASES
        for buy in BUY_RULES
        for sell in SELL_RULES
        for dd_mode in DD_MODES
    }
    for row in top_for_recent:
        case_key = row["case_key"]
        top, buy, sell, dd_mode = param_lookup[case_key]
        meta = {
            "case_key": case_key,
            "signal_file": Path(row["signal_file"]),
            "score_table": row["score_table"],
        }
        recent_rows = []
        for tag, start, end in SLICES[1:]:
            recent = _run_one(meta, top, sell, dd_mode, tag, start, end)
            recent_rows.append(recent)
            detail_rows.append(recent)
        full_row = {
            "returncode": row.get("full_returncode"),
            "annual": row.get("full_annual"),
            "pnl_ratio": row.get("full_pnl_ratio"),
            "sharpe": row.get("full_sharpe"),
            "max_drawdown": row.get("full_max_drawdown"),
            "win_ratio": row.get("full_win_ratio"),
            "open_count": row.get("full_open_count"),
            "close_count": row.get("full_close_count"),
            "log_file": row.get("full_log_file"),
        }
        signal_meta = {
            "days_below_target": row.get("days_below_target"),
            "signal_rows": row.get("signal_rows"),
            "signal_days": row.get("signal_days"),
        }
        row.update(_admission(full_row, recent_rows, signal_meta))
        for recent in recent_rows:
            prefix = "recent_" + recent["slice"]
            row[prefix + "_annual"] = recent.get("annual")
            row[prefix + "_sharpe"] = recent.get("sharpe")
            row[prefix + "_max_drawdown"] = recent.get("max_drawdown")
            row[prefix + "_open_count"] = recent.get("open_count")
            row[prefix + "_log_file"] = recent.get("log_file")
        summary_by_key[case_key] = row
        summary_rows = list(summary_by_key.values())
        _write_rows(REPORT_DIR / "correct_exec_search_detail.csv", detail_rows)
        _write_rows(REPORT_DIR / "correct_exec_search_summary.csv", summary_rows)
        (REPORT_DIR / "correct_exec_search_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_report(summary_rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
