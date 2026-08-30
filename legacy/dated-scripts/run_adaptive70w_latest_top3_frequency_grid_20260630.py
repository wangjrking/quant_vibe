from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
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
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"
BASE_SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"

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
    "price_cap": 150.0,
}

TOP_CASES = [
    {"name": "top3_pos25", "topn": 3, "max_positions": 3, "target": 0.25},
    {"name": "top3_pos20", "topn": 3, "max_positions": 3, "target": 0.20},
]

BUY_RULES = [
    {"name": "nocool", "two_day_cap": None, "combo_cap": None, "turnover_floor": None},
    {"name": "cool2d20", "two_day_cap": 0.20, "combo_cap": None, "turnover_floor": None},
    {"name": "cool2d18", "two_day_cap": 0.18, "combo_cap": None, "turnover_floor": None},
    {"name": "cool2d15_turn15", "two_day_cap": None, "combo_cap": 0.15, "turnover_floor": 15.0},
]

SELL_RULES = [
    {
        "name": "h3m5_e096_c097_min1",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit_ratio": 0.96,
        "score_continue_ratio": 0.97,
        "min_score_exit_days": 1,
        "day_drop_ratio": None,
    },
    {
        "name": "h2m3_e097_c098_min1",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit_ratio": 0.97,
        "score_continue_ratio": 0.98,
        "min_score_exit_days": 1,
        "day_drop_ratio": None,
    },
    {
        "name": "h1m2_e098_c099_min1",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit_ratio": 0.98,
        "score_continue_ratio": 0.99,
        "min_score_exit_days": 1,
        "day_drop_ratio": None,
    },
    {
        "name": "h3m5_e096_c097_daydrop97",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit_ratio": 0.96,
        "score_continue_ratio": 0.97,
        "min_score_exit_days": 1,
        "day_drop_ratio": 0.97,
    },
    {
        "name": "h2m3_e097_c098_daydrop97",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit_ratio": 0.97,
        "score_continue_ratio": 0.98,
        "min_score_exit_days": 1,
        "day_drop_ratio": 0.97,
    },
    {
        "name": "h3m5_e096_c097_min2",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit_ratio": 0.96,
        "score_continue_ratio": 0.97,
        "min_score_exit_days": 2,
        "day_drop_ratio": None,
    },
]

RISK_MODES = [
    {
        "name": "open_only_ddloose",
        "intraday": "0",
        "max_daily_sells": "0",
        "dd_soft": "0.08",
        "dd_hard": "0.14",
        "dd_recover": "0.04",
        "dd_soft_scale": "0.80",
        "dd_hard_scale": "0.60",
    },
    {
        "name": "intraday_ddloose",
        "intraday": "1",
        "max_daily_sells": "0",
        "dd_soft": "0.08",
        "dd_hard": "0.14",
        "dd_recover": "0.04",
        "dd_soft_scale": "0.80",
        "dd_hard_scale": "0.60",
    },
    {
        "name": "open_only_ddloose_maxsell1",
        "intraday": "0",
        "max_daily_sells": "1",
        "dd_soft": "0.08",
        "dd_hard": "0.14",
        "dd_recover": "0.04",
        "dd_soft_scale": "0.80",
        "dd_hard_scale": "0.60",
    },
]

SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-29 15:30:00"),
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


def _score_table(case_key: str) -> str:
    return "score_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in case_key)


def _score_table_ready(table: str) -> bool:
    if not SCORE_DB.exists():
        return False
    conn = duckdb.connect(str(SCORE_DB), read_only=True)
    try:
        exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            return False
        row = conn.execute(
            f'SELECT min(trade_date), max(trade_date), count(distinct trade_date) FROM "{table}"'
        ).fetchone()
        return row[0] == "20220606" and row[1] == "20260629" and int(row[2]) == 986
    finally:
        conn.close()


def _cooldown_condition(rule: dict[str, Any]) -> str:
    tests: list[str] = []
    if rule["two_day_cap"] is not None:
        tests.append(f"coalesce(two_day_ret, -999.0) < {float(rule['two_day_cap'])}")
    if rule["combo_cap"] is not None:
        tests.append(
            f"(coalesce(two_day_ret, -999.0) < {float(rule['combo_cap'])} "
            f"OR coalesce(try_cast(turnover_rate AS DOUBLE), 0.0) < {float(rule['turnover_floor'])})"
        )
    return " AND ".join(tests) if tests else "TRUE"


def _case_key(top_case: dict[str, Any], buy_rule: dict[str, Any], sell_rule: dict[str, Any]) -> str:
    return f"{ENTRY['name']}_{top_case['name']}_{buy_rule['name']}_{sell_rule['name']}"


def _ensure_score_and_signal(top_case: dict[str, Any], buy_rule: dict[str, Any], sell_rule: dict[str, Any]) -> dict[str, Any]:
    case_key = _case_key(top_case, buy_rule, sell_rule)
    score_table = BASE_SCORE_TABLE
    signal_file = REPORT_DIR / "signals" / f"{case_key}.csv"
    if signal_file.exists():
        rows = list(csv.DictReader(signal_file.open("r", encoding="utf-8-sig", newline="")))
        return {"case_key": case_key, "score_table": score_table, "signal_file": signal_file, "signal_rows": len(rows)}

    sources = {label: _load_manifest(label) for label in ("3d", "5d", "10d")}
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
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
                {float(ENTRY['w10d'])} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d)
              + {float(ENTRY['w5d'])} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d)
              + {float(ENTRY['w3d'])} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS entry_score
            FROM base
            """
        )
        if not _score_table_ready(score_table):
            con.execute(f'DROP TABLE IF EXISTS scoredb."{score_table}"')
            con.execute(
                f"""
                CREATE TABLE scoredb."{score_table}" AS
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
              AND try_cast(md.amount AS DOUBLE) >= {float(ENTRY['amount_min'])}
              AND try_cast(md.total_mv AS DOUBLE) >= {float(ENTRY['total_mv_min'])}
              AND try_cast(md.close AS DOUBLE) <= {float(ENTRY['price_cap'])}
              AND {_cooldown_condition(buy_rule)}
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
                    WHEN try_cast(bm.open AS DOUBLE) > {float(ENTRY['price_cap'])} THEN FALSE
                    ELSE TRUE
                END AS buy_day_ok
            FROM scored s
            LEFT JOIN marketdb.STOCK_DAILY_DATA bm
              ON s.buy_date = bm.trade_date
             AND s.stock_code = bm.stock_code
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
            printf('%.5f', {float(top_case['target'])}) AS target_pct,
            {int(sell_rule['holding_days'])} AS holding_days,
            {int(sell_rule['max_holding_days'])} AS max_holding_days,
            printf('%.5f', {float(sell_rule['score_exit_ratio'])}) AS score_exit_entry_ratio,
            {int(sell_rule['min_score_exit_days'])} AS min_holding_days_before_score_exit,
            printf('%.5f', {float(sell_rule['score_continue_ratio'])}) AS score_continue_entry_ratio,
            '0.05000' AS signal_stop_loss_pct,
            '0.08000' AS signal_take_profit_pct,
            '{case_key}' AS strategy_variant,
            '{buy_rule['name']}' AS filter_name,
            '{ENTRY['name']}' AS entry_weight_name,
            '{sell_rule['name']}' AS dynamic_hold_name,
            buy_day_ok AS buy_day_market_available,
            buy_day_ok AS buy_day_hard_gate_complete,
            false AS buy_day_st_rejected,
            false AS buy_day_open_limit_up_rejected,
            latest_market_date
        FROM with_buy_checks
        WHERE buy_day_ok
        QUALIFY row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) <= {int(top_case['topn'])}
        ORDER BY signal_date, rank
        """
        rows = con.execute(selected_sql).fetchdf().to_dict("records")
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


def _env(sell_rule: dict[str, Any], risk_mode: dict[str, Any]) -> dict[str, str]:
    return {
        "GM_OPEN_DAILY_SCORE_EXIT": "1",
        "GM_MAX_DAILY_SELLS": str(risk_mode["max_daily_sells"]),
        "GM_LIGHT_STOP_LOSS_PCT": "none",
        "GM_LOG_EXPOSURE": "1",
        "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
        "GM_SYNC_POSITIONS": "1",
        "GM_CASH_BUFFER": "0.99",
        "GM_VERBOSE_TRADES": "1",
        "GM_FORCE_SELL_MARKET_ORDER": "0",
        "GM_FORCE_BUY_MARKET_ORDER": "0",
        "GM_INTRADAY_RISK_MODE": str(risk_mode["intraday"]),
        "GM_INTRADAY_REPLACE_BUY": "0",
        "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
        "GM_EQUITY_DD_RISK_MODE": "1",
        "GM_EQUITY_DD_RESIZE_EXISTING": "0",
        "GM_EQUITY_DD_SOFT_TRIGGER": str(risk_mode["dd_soft"]),
        "GM_EQUITY_DD_HARD_TRIGGER": str(risk_mode["dd_hard"]),
        "GM_EQUITY_DD_RECOVER_TRIGGER": str(risk_mode["dd_recover"]),
        "GM_EQUITY_DD_SOFT_SCALE": str(risk_mode["dd_soft_scale"]),
        "GM_EQUITY_DD_HARD_SCALE": str(risk_mode["dd_hard_scale"]),
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(sell_rule["min_score_exit_days"]),
        "GM_SCORE_CONTINUE_ENTRY_RATIO": str(sell_rule["score_continue_ratio"]),
        "GM_SCORE_EXIT_ENTRY_RATIO": str(sell_rule["score_exit_ratio"]),
        "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "none"
        if sell_rule["day_drop_ratio"] is None
        else str(sell_rule["day_drop_ratio"]),
        "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
        "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
        "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
        "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
        "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
        "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
        "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
        "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
    }


def _run(
    meta: dict[str, Any],
    top_case: dict[str, Any],
    sell_rule: dict[str, Any],
    risk_mode: dict[str, Any],
    tag: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    log_file = REPORT_DIR / "logs" / f"{meta['case_key']}__{risk_mode['name']}__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(_env(sell_rule, risk_mode))
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
            str(top_case["max_positions"]),
            "--holding-days",
            str(sell_rule["holding_days"]),
            "--max-holding-days",
            str(sell_rule["max_holding_days"]),
            "--target-position-pct",
            str(top_case["target"]),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            str(meta["score_table"]),
            "--market-db",
            str(MARKET_DUCKDB),
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
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "case_key": meta["case_key"],
        "risk_mode": risk_mode["name"],
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
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    latest_meta = {
        "manifests": {label: _load_manifest(label) for label in ("3d", "5d", "10d")},
        "entry": ENTRY,
        "top_cases": TOP_CASES,
        "buy_rules": BUY_RULES,
        "sell_rules": SELL_RULES,
        "risk_modes": RISK_MODES,
        "slices": SLICES,
        "note": "research-only frequency grid; no production parameter change",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "run_manifest.json").write_text(json.dumps(latest_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for top_case in TOP_CASES:
        for buy_rule in BUY_RULES:
            for sell_rule in SELL_RULES:
                meta = _ensure_score_and_signal(top_case, buy_rule, sell_rule)
                for risk_mode in RISK_MODES:
                    rows = [
                        _run(meta, top_case, sell_rule, risk_mode, tag, start, end)
                        for tag, start, end in SLICES
                    ]
                    detail_rows.extend(rows)
                    by_slice = {row["slice"]: row for row in rows}
                    full = by_slice["full"]
                    out = {
                        "case_key": meta["case_key"],
                        "top_case": top_case["name"],
                        "target_pct_each": top_case["target"],
                        "gross_target_pct": top_case["target"] * top_case["max_positions"],
                        "buy_rule": buy_rule["name"],
                        "sell_rule": sell_rule["name"],
                        "risk_mode": risk_mode["name"],
                        "holding_days": sell_rule["holding_days"],
                        "max_holding_days": sell_rule["max_holding_days"],
                        "score_exit_ratio": sell_rule["score_exit_ratio"],
                        "score_continue_ratio": sell_rule["score_continue_ratio"],
                        "min_score_exit_days": sell_rule["min_score_exit_days"],
                        "day_drop_ratio": sell_rule["day_drop_ratio"],
                        "max_daily_sells": risk_mode["max_daily_sells"],
                        "intraday_risk": risk_mode["intraday"],
                        "signal_rows": meta["signal_rows"],
                        "signal_file": str(meta["signal_file"]),
                        "score_table": meta["score_table"],
                        "full_annual": full.get("annual"),
                        "full_pnl_ratio": full.get("pnl_ratio"),
                        "full_sharpe": full.get("sharpe"),
                        "full_max_drawdown": full.get("max_drawdown"),
                        "full_win_ratio": full.get("win_ratio"),
                        "full_open_count": full.get("open_count"),
                        "full_close_count": full.get("close_count"),
                        "full_log_file": full.get("log_file"),
                    }
                    summary_rows.append(out)
                    _write_rows(REPORT_DIR / "frequency_grid_detail.csv", detail_rows)
                    _write_rows(REPORT_DIR / "frequency_grid_summary.csv", summary_rows)
                    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
